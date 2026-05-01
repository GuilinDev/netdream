"""NetDream Kubernetes controller.

Runs as a Python process that:
  1. Fetches live service metrics from Prometheus every STEP_INTERVAL seconds.
  2. Builds a GraphObs from the cluster state.
  3. Invokes the learned world model + MPC planner to pick per-service scaling.
  4. Applies actions via the Kubernetes apps/v1 `scale` subresource.
  5. Logs each decision (state, action, realized metrics) to a JSONL file.

This is the "real-cluster" counterpart to `experiments/evaluate_planning.py`
(which runs against the Python simulator). Both share `netdream.fast_planner`.

Baselines (HPA, Random, PPO, GraphPilot) can be swapped via --agent.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

# Repo-root imports
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from envs.k8s_mesh_env import SERVICE_EDGES, SERVICE_NAMES  # noqa: E402
from netdream.cluster_state import fetch_cluster_state, to_graph_data  # noqa: E402
from netdream.dynamics_model import GNNDynamicsModel  # noqa: E402
from netdream.fast_planner import FastRandomShootingPlanner  # noqa: E402
from netdream.prometheus_client import PromClient, wait_ready  # noqa: E402


STEP_INTERVAL_S = 5.0
DEFAULT_EPISODE_STEPS = 60   # 60 × 5s = 5 minutes
MIN_REPLICAS = 1
MAX_REPLICAS = 10


@dataclass
class StepRecord:
    t: int
    wall_time: float
    state: list
    action: list
    replicas_before: list
    replicas_after: list
    frontend_qps: float
    frontend_latency_ms: float
    frontend_error_rate: float
    agent: str


@dataclass
class EpisodeLog:
    agent: str
    workload: str
    seed: int
    namespace: str
    started_at: float
    steps: list = field(default_factory=list)
    cluster: str = "microk8s"  # or "eks"


# ---------------------------------------------------------------------------
# Locust stats fetcher (lightweight HTTP poll of Locust's /stats/requests)
# ---------------------------------------------------------------------------

_locust_warn_count = 0


def fetch_locust_stats(locust_url: str, timeout: float = 3.0) -> tuple[float, float, float]:
    """Return (frontend_qps, frontend_latency_ms_p95, error_rate).

    On any error, logs a (rate-limited) warning with the reason so we can
    tell whether we're looking at "no traffic yet" vs "controller can't
    reach Locust". Falls back to (0,0,0).
    """
    from urllib.request import urlopen
    global _locust_warn_count
    try:
        with urlopen(f"{locust_url}/stats/requests", timeout=timeout) as r:
            data = json.load(r)
    except Exception as e:
        _locust_warn_count += 1
        if _locust_warn_count <= 3 or _locust_warn_count % 20 == 0:
            print(f"[locust-stats] warn ({_locust_warn_count}): {type(e).__name__}: {e}")
        return 0.0, 0.0, 0.0

    # Locust's /stats/requests returns {"stats": [...], "errors": ..., ...}.
    # The last row of stats is usually the "Aggregated" summary.
    stats = data.get("stats") or []
    if not stats:
        return 0.0, 0.0, 0.0
    # Find the Aggregated row (name == "Aggregated") or fall back to last
    agg = next((s for s in stats if s.get("name") == "Aggregated"), stats[-1])
    qps = float(agg.get("current_rps", 0.0))
    # Locust exports percentiles under different key names depending on version:
    p95 = float(
        agg.get("response_time_percentile_0.95")
        or agg.get("response_time_percentile_95")
        or agg.get("ninetieth_response_time")
        or agg.get("current_response_time_percentile_95")
        or 0.0
    )
    num_req = max(float(agg.get("num_requests", 1.0)), 1.0)
    num_fail = float(agg.get("num_failures", 0.0))
    return qps, p95, num_fail / num_req


# ---------------------------------------------------------------------------
# K8s scale subresource client (uses the in-cluster kubernetes python client)
# ---------------------------------------------------------------------------

class ScaleClient:
    def __init__(self, namespace: str, service_names: list[str]):
        from kubernetes import client, config
        try:
            config.load_kube_config()
        except Exception:
            config.load_incluster_config()
        self.apps = client.AppsV1Api()
        self.namespace = namespace
        self.service_names = service_names

    def get_replicas(self, deployment: str) -> int:
        obj = self.apps.read_namespaced_deployment_scale(deployment, self.namespace)
        return int(obj.status.replicas or 0)

    def set_replicas(self, deployment: str, n: int) -> None:
        from kubernetes import client
        n = max(MIN_REPLICAS, min(MAX_REPLICAS, int(n)))
        body = client.V1Scale(
            metadata=client.V1ObjectMeta(name=deployment, namespace=self.namespace),
            spec=client.V1ScaleSpec(replicas=n),
        )
        self.apps.patch_namespaced_deployment_scale(deployment, self.namespace, body)

    def snapshot_replicas(self) -> dict[str, int]:
        out = {}
        for svc in self.service_names:
            try:
                out[svc] = self.get_replicas(svc)
            except Exception:
                out[svc] = 0  # deployment missing (e.g., cart/currency on MicroK8s)
        return out


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class NetDreamAgent:
    def __init__(
        self,
        model_path: str,
        service_names: list[str],
        edge_index: np.ndarray,
        horizon: int = 5,
        use_safety: bool = True,
        device: str = "auto",
    ):
        # NOTE: node feature dim must match the trained checkpoint.
        #       Current k8s_best.pt uses 6-dim features (simulator convention).
        model = GNNDynamicsModel(
            node_feat_dim=6,
            action_dim=3,
            hidden_dim=128,
            num_gnn_layers=3,
            gnn_type="gat",
        )
        state_dict = torch.load(model_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)

        self.planner = FastRandomShootingPlanner(
            model=model,
            horizon=horizon,
            num_candidates=64,
            action_dim=3,
            num_nodes=len(service_names),
            safety_threshold=0.5,
            use_safety=use_safety,
            device=device,
        )
        self.service_names = service_names
        self.edge_index_np = edge_index

    def act(self, state: np.ndarray) -> np.ndarray:
        action_idx = self.planner.plan(state, self.edge_index_np)
        # Map {0, 1, 2} → {-1, 0, +1} (per-service scale delta)
        return np.asarray(action_idx, dtype=int) - 1


class PPOAgent:
    """Model-free PPO baseline trained on the simulator.

    Obs space: flat 66-dim vector (11 services × 6 features).
    Action space: Discrete(23) = {no-op, scale-up i, scale-down i}.
    Decoded back to per-service {-1, 0, +1} to match the common scaling API.
    """
    def __init__(self, ppo_path: str, n: int = 11):
        from stable_baselines3 import PPO
        self.ppo = PPO.load(ppo_path, device="cpu")
        self.n = n

    def act(self, state: np.ndarray) -> np.ndarray:
        flat = state.astype(np.float32).flatten()
        action_idx, _ = self.ppo.predict(flat, deterministic=True)
        a = int(action_idx)
        out = np.zeros(self.n, dtype=int)
        if a == 0:
            return out
        if a <= self.n:
            out[a - 1] = +1
        else:
            out[a - self.n - 1] = -1
        return out


class RandomAgent:
    def __init__(self, n: int, rng: np.random.Generator):
        self.n = n
        self.rng = rng

    def act(self, state: np.ndarray) -> np.ndarray:
        return self.rng.integers(-1, 2, size=self.n)


class HPAMirrorAgent:
    """Emulate standard K8s HPA locally: scale up if CPU > target_up, down if < target_down.

    A `min_replicas_floor` of 3 or 5 emulates "over-provisioned HPA" variants
    (each deployment pinned to a minimum floor irrespective of CPU), which
    addresses the reviewer question "why not just raise HPA's min replicas?"
    """
    def __init__(
        self,
        n: int,
        up_threshold: float = 0.7,
        down_threshold: float = 0.3,
        min_replicas_floor: int = 1,
    ):
        self.n = n
        self.up = up_threshold
        self.down = down_threshold
        self.min_floor = int(min_replicas_floor)

    def act(self, state: np.ndarray) -> np.ndarray:
        cpu = state[:, 0]
        replicas_norm = state[:, 5]  # normalized by MAX_REPLICAS=10
        action = np.zeros(self.n, dtype=int)
        action[cpu > self.up] = +1
        action[cpu < self.down] = -1
        # Enforce per-service floor: if current replicas < floor, force +1 regardless
        if self.min_floor > 1:
            current_reps = (replicas_norm * 10).round().astype(int)
            action[current_reps < self.min_floor] = +1
        return action


def make_agent(name: str, cfg: argparse.Namespace, edge_index: np.ndarray) -> object:
    n = len(cfg.service_names)
    if name == "netdream":
        return NetDreamAgent(
            model_path=cfg.model_path,
            service_names=cfg.service_names,
            edge_index=edge_index,
            horizon=cfg.horizon,
            use_safety=True,
        )
    if name == "netdream-unsafe":
        return NetDreamAgent(
            model_path=cfg.model_path,
            service_names=cfg.service_names,
            edge_index=edge_index,
            horizon=cfg.horizon,
            use_safety=False,
        )
    if name == "ppo":
        return PPOAgent(ppo_path=cfg.ppo_path, n=n)
    if name == "random":
        return RandomAgent(n=n, rng=np.random.default_rng(cfg.seed))
    if name == "hpa":
        return HPAMirrorAgent(n=n, min_replicas_floor=1)
    if name == "hpa-min3":
        return HPAMirrorAgent(n=n, min_replicas_floor=3)
    if name == "hpa-min5":
        return HPAMirrorAgent(n=n, min_replicas_floor=5)
    raise ValueError(f"Unknown agent: {name}")


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def run_episode(cfg: argparse.Namespace) -> EpisodeLog:
    prom = PromClient(cfg.prometheus_url)
    wait_ready(prom, timeout_s=30.0)
    scale_client = ScaleClient(cfg.namespace, cfg.service_names)

    edge_index_np = np.asarray(SERVICE_EDGES, dtype=np.int64).T  # [2, E]

    agent = make_agent(cfg.agent, cfg, edge_index_np)

    log = EpisodeLog(
        agent=cfg.agent,
        workload=cfg.workload,
        seed=cfg.seed,
        namespace=cfg.namespace,
        started_at=time.time(),
        cluster=cfg.cluster,
    )

    print(f"[controller] agent={cfg.agent} workload={cfg.workload} seed={cfg.seed} "
          f"steps={cfg.episode_steps} interval={STEP_INTERVAL_S}s")

    for t in range(cfg.episode_steps):
        t0 = time.time()

        # 1. Poll Locust for aggregate request metrics
        qps, p95, err = fetch_locust_stats(cfg.locust_url)

        # 2. Fetch Prometheus metrics and build state
        try:
            state = fetch_cluster_state(
                prom=prom,
                namespace=cfg.namespace,
                service_names=cfg.service_names,
                frontend_qps=qps,
                frontend_latency_ms=p95,
                frontend_error_rate=err,
            )
        except Exception as e:
            print(f"[controller] step {t}: Prometheus fetch failed: {e}")
            time.sleep(STEP_INTERVAL_S)
            continue

        # 3. Decide action
        action = agent.act(state)

        # 4. Apply scaling
        reps_before = scale_client.snapshot_replicas()
        reps_after = dict(reps_before)
        for i, svc in enumerate(cfg.service_names):
            if svc not in reps_before or reps_before[svc] == 0:
                continue  # deployment missing or disabled (e.g., cart on MicroK8s)
            target = reps_before[svc] + int(action[i])
            target = max(MIN_REPLICAS, min(MAX_REPLICAS, target))
            if target != reps_before[svc]:
                try:
                    scale_client.set_replicas(svc, target)
                    reps_after[svc] = target
                except Exception as e:
                    print(f"[controller] step {t}: scale({svc}, {target}) failed: {e}")

        # 5. Record
        rec = StepRecord(
            t=t,
            wall_time=time.time(),
            state=state.tolist(),
            action=action.tolist(),
            replicas_before=[reps_before.get(s, 0) for s in cfg.service_names],
            replicas_after=[reps_after.get(s, 0) for s in cfg.service_names],
            frontend_qps=qps,
            frontend_latency_ms=p95,
            frontend_error_rate=err,
            agent=cfg.agent,
        )
        log.steps.append(asdict(rec))

        # 6. Sleep until the next tick
        elapsed = time.time() - t0
        sleep = max(0.0, STEP_INTERVAL_S - elapsed)
        time.sleep(sleep)

        if t % 10 == 0 or t == cfg.episode_steps - 1:
            print(f"[controller] t={t:3d} qps={qps:5.1f} p95={p95:5.0f}ms err={err:.3f} "
                  f"replicas={list(reps_after.values())}")

    return log


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True,
                    choices=["netdream", "netdream-unsafe", "random",
                             "hpa", "hpa-min3", "hpa-min5", "ppo"])
    ap.add_argument("--workload", default="constant")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--namespace", default="boutique")
    ap.add_argument("--prometheus-url", default="http://localhost:9090")
    ap.add_argument("--locust-url", default="http://localhost:8089")
    ap.add_argument("--model-path", default="models/k8s_best.pt")
    ap.add_argument("--ppo-path", default="models/ppo_v2_k8s_mesh.zip")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--episode-steps", type=int, default=DEFAULT_EPISODE_STEPS)
    ap.add_argument("--cluster", default="microk8s", choices=["microk8s", "eks"])
    ap.add_argument("--out", default="results/cluster_runs")
    args = ap.parse_args()
    args.service_names = list(SERVICE_NAMES)
    return args


def main() -> None:
    cfg = parse_args()
    log = run_episode(cfg)
    out_dir = pathlib.Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{cfg.cluster}_{cfg.agent}_{cfg.workload}_seed{cfg.seed}.json"
    (out_dir / fname).write_text(json.dumps(asdict(log), indent=2))
    print(f"[controller] wrote {out_dir / fname}")


if __name__ == "__main__":
    main()
