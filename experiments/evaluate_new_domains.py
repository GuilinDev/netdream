#!/usr/bin/env python3
"""Evaluate NetDream-Safe and baselines on WaterNet-10 and EpiNet-20.

Produces numbers for Appendix tables:
  - tab:water-epinet-mse   (prediction MSE, all 4 domains)
  - tab:water-planning     (WaterNet planning results)
  - tab:epinet-planning    (EpiNet-20 planning results)
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from netdream.dynamics_model import GNNDynamicsModel
from netdream.fast_planner import FastRandomShootingPlanner
from netdream.planner import SafetyFilter
from experiments.train_dynamics import load_transitions, evaluate_multistep


# ─── Prediction MSE ──────────────────────────────────────────────────────────

def compute_mse(domain: str, node_feat_dim: int, action_dim: int) -> dict:
    model_path = f"models/{domain}_best.pt"
    data_path = f"data/{domain}_transitions.pkl"

    if not Path(model_path).exists():
        print(f"  [SKIP] {model_path} not found")
        return {}

    model = GNNDynamicsModel(
        node_feat_dim=node_feat_dim,
        action_dim=action_dim,
        hidden_dim=128,
        num_gnn_layers=3,
        num_constraints=1,
        gnn_type="gat",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.to(device)
    transitions = load_transitions(data_path)

    mse = evaluate_multistep(
        model, transitions, action_dim,
        horizons=[1, 5, 10],
        device=device,
    )
    print(f"  {domain}: 1-step MSE={mse[1]:.4e}, 5-step={mse[5]:.4e}, 10-step={mse[10]:.4e}")
    return mse


# ─── Planning evaluation helpers ─────────────────────────────────────────────

def run_episode(env, policy_fn, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    total_cost = 0.0
    violations = 0
    steps = 0

    while True:
        action = policy_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_cost -= reward   # reward is negative cost
        if np.any(obs.constraint_values > 0.5):
            violations += 1
        steps += 1
        if terminated or truncated:
            break

    return {"cost": total_cost, "violations": violations, "steps": steps}


def eval_policy(env_cls, env_kwargs, policy_fn, seeds=(42, 123, 456)) -> dict:
    results = [run_episode(env_cls(**env_kwargs), policy_fn, seed) for seed in seeds]
    costs = [r["cost"] for r in results]
    viols = [r["violations"] for r in results]
    return {
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "violations_mean": float(np.mean(viols)),
        "violations_std": float(np.std(viols)),
    }


def netdream_policy(model, num_nodes, action_dim):
    """Return a NetDream-Safe policy function (uniform random shooting, no domain-specific bias)."""
    planner = FastRandomShootingPlanner(
        model=model,
        num_candidates=64,
        horizon=5,
        action_dim=action_dim,
        num_nodes=num_nodes,
        use_safety=True,
        safety_threshold=0.5,
        noop_prob=0.0,  # uniform random — no K8s-specific no-op bias
    )

    def policy_fn(obs, env):
        x = obs.data.x.numpy()
        edge_index = obs.data.edge_index.numpy()
        action = planner.plan(x, edge_index)
        return action

    return policy_fn


# ─── WaterNet evaluation ──────────────────────────────────────────────────────

def eval_water():
    from envs.water_net_env import WaterNetEnv, PUMP_NODES, NUM_NODES

    print("\n" + "=" * 60)
    print("  WaterNet-10 Planning Evaluation")
    print("=" * 60)

    env_kwargs = {"max_steps": 96, "demand_pattern": "daily"}

    # Baseline: Do-Nothing (pumps off)
    def do_nothing(obs, env):
        a = np.zeros(env.num_nodes, dtype=int)
        return a

    # Baseline: Always-High (max pump speed)
    def always_high(obs, env):
        a = np.ones(env.num_nodes, dtype=int) * 2
        return a

    # Baseline: Greedy-Pressure (run pumps when pressure is low)
    def greedy_pressure(obs, env):
        a = np.ones(env.num_nodes, dtype=int)  # normal by default
        pressure_ok = float(obs.global_features[0]) > 0.3  # normalized pressure
        if not pressure_ok:
            for p in PUMP_NODES:
                a[p] = 2  # high speed when pressure low
        return a

    print("\n  Do-Nothing:")
    dn = eval_policy(WaterNetEnv, env_kwargs, do_nothing)
    print(f"    cost={dn['cost_mean']:.1f}±{dn['cost_std']:.1f}, "
          f"violations={dn['violations_mean']:.1f}±{dn['violations_std']:.1f}")

    print("\n  Always-High:")
    ah = eval_policy(WaterNetEnv, env_kwargs, always_high)
    print(f"    cost={ah['cost_mean']:.1f}±{ah['cost_std']:.1f}, "
          f"violations={ah['violations_mean']:.1f}±{ah['violations_std']:.1f}")

    print("\n  Greedy-Pressure:")
    gp = eval_policy(WaterNetEnv, env_kwargs, greedy_pressure)
    print(f"    cost={gp['cost_mean']:.1f}±{gp['cost_std']:.1f}, "
          f"violations={gp['violations_mean']:.1f}±{gp['violations_std']:.1f}")

    # NetDream-Safe
    model_path = "models/water_best.pt"
    if Path(model_path).exists():
        model = GNNDynamicsModel(
            node_feat_dim=5, action_dim=3, hidden_dim=128,
            num_gnn_layers=3, num_constraints=1, gnn_type="gat"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        nd_policy = netdream_policy(model, NUM_NODES, 3)
        print("\n  NetDream-Safe:")
        nd = eval_policy(WaterNetEnv, env_kwargs, nd_policy)
        print(f"    cost={nd['cost_mean']:.1f}±{nd['cost_std']:.1f}, "
              f"violations={nd['violations_mean']:.1f}±{nd['violations_std']:.1f}")
    else:
        nd = {"cost_mean": 0, "cost_std": 0, "violations_mean": 0, "violations_std": 0}
        print(f"  [SKIP] {model_path} not found")

    return {"do_nothing": dn, "always_high": ah, "greedy": gp, "netdream": nd}


# ─── Traffic evaluation ───────────────────────────────────────────────────────

def eval_traffic():
    from envs.traffic_env import TrafficSignalEnv, NUM_NODES

    print("\n" + "=" * 60)
    print("  TrafficGrid-4×4 Planning Evaluation")
    print("=" * 60)

    env_kwargs = {"max_steps": 120, "demand_type": "peak"}

    # Baseline: Fixed-Cycle (NS for 4 steps, then EW for 4 steps)
    step_counter = [0]
    def fixed_cycle(obs, env):
        phase = (step_counter[0] // 4) % 2
        step_counter[0] += 1
        return np.full(env.num_nodes, phase, dtype=int)

    # Baseline: Greedy-Clear (always NS-green if NS queue > EW, else EW-green)
    def greedy_clear(obs, env):
        x = obs.data.x.numpy()
        q_ns = x[:, 0]  # normalized NS queue
        q_ew = x[:, 1]  # normalized EW queue
        action = np.where(q_ns >= q_ew, 0, 1).astype(int)
        return action

    print("\n  Fixed-Cycle:")
    step_counter[0] = 0
    fc = eval_policy(TrafficSignalEnv, env_kwargs, fixed_cycle)
    print(f"    total_delay={fc['cost_mean']:.1f}±{fc['cost_std']:.1f}, "
          f"violations={fc['violations_mean']:.1f}±{fc['violations_std']:.1f}")

    print("\n  Greedy-Clear:")
    gc = eval_policy(TrafficSignalEnv, env_kwargs, greedy_clear)
    print(f"    total_delay={gc['cost_mean']:.1f}±{gc['cost_std']:.1f}, "
          f"violations={gc['violations_mean']:.1f}±{gc['violations_std']:.1f}")

    # NetDream-Safe
    model_path = "models/traffic_best.pt"
    if Path(model_path).exists():
        model = GNNDynamicsModel(
            node_feat_dim=6, action_dim=3, hidden_dim=128,
            num_gnn_layers=3, num_constraints=1, gnn_type="gat"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        nd_policy = netdream_policy(model, NUM_NODES, 3)
        print("\n  NetDream-Safe:")
        nd = eval_policy(TrafficSignalEnv, env_kwargs, nd_policy)
        print(f"    total_delay={nd['cost_mean']:.1f}±{nd['cost_std']:.1f}, "
              f"violations={nd['violations_mean']:.1f}±{nd['violations_std']:.1f}")
    else:
        nd = {"cost_mean": 0, "cost_std": 0, "violations_mean": 0, "violations_std": 0}
        print(f"  [SKIP] {model_path} not found")

    return {"fixed_cycle": fc, "greedy_clear": gc, "netdream": nd}


# ─── EpiNet-20 evaluation ──────────────────────────────────────────────────────

def eval_epinet():
    from envs.epinet_env import EpiNetEnv, NUM_NODES, HUB_NODES

    print("\n" + "=" * 60)
    print("  EpiNet-20 Planning Evaluation")
    print("=" * 60)

    env_kwargs = {"max_steps": 60}

    # Baseline: No-intervention
    def no_inter(obs, env):
        return np.zeros(env.num_nodes, dtype=int)

    # Baseline: Greedy-Reactive (intervene when I > threshold)
    def greedy_reactive(obs, env):
        I_vals = obs.data.x[:, 1].numpy()
        action = np.where(I_vals > 0.05, 1, 0).astype(int)
        for h in HUB_NODES:
            if I_vals[h] > 0.08:
                action[h] = 2
        return action

    # Baseline: Always-Strong
    def always_strong(obs, env):
        return np.full(env.num_nodes, 2, dtype=int)

    print("\n  No-Intervention:")
    ni = eval_policy(EpiNetEnv, env_kwargs, no_inter)
    print(f"    cost={ni['cost_mean']:.1f}±{ni['cost_std']:.1f}, "
          f"violations={ni['violations_mean']:.1f}±{ni['violations_std']:.1f}")

    print("\n  Greedy-Reactive:")
    gr = eval_policy(EpiNetEnv, env_kwargs, greedy_reactive)
    print(f"    cost={gr['cost_mean']:.1f}±{gr['cost_std']:.1f}, "
          f"violations={gr['violations_mean']:.1f}±{gr['violations_std']:.1f}")

    print("\n  Always-Strong:")
    ast = eval_policy(EpiNetEnv, env_kwargs, always_strong)
    print(f"    cost={ast['cost_mean']:.1f}±{ast['cost_std']:.1f}, "
          f"violations={ast['violations_mean']:.1f}±{ast['violations_std']:.1f}")

    # NetDream-Safe
    model_path = "models/epinet_best.pt"
    if Path(model_path).exists():
        model = GNNDynamicsModel(
            node_feat_dim=5, action_dim=3, hidden_dim=128,
            num_gnn_layers=3, num_constraints=1, gnn_type="gat"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        nd_policy = netdream_policy(model, NUM_NODES, 3)
        print("\n  NetDream-Safe:")
        nd = eval_policy(EpiNetEnv, env_kwargs, nd_policy)
        print(f"    cost={nd['cost_mean']:.1f}±{nd['cost_std']:.1f}, "
              f"violations={nd['violations_mean']:.1f}±{nd['violations_std']:.1f}")
    else:
        nd = {"cost_mean": 0, "cost_std": 0, "violations_mean": 0, "violations_std": 0}
        print(f"  [SKIP] {model_path} not found")

    return {"no_inter": ni, "greedy_reactive": gr, "always_strong": ast, "netdream": nd}


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    print("\n" + "=" * 60)
    print("  Multi-step Prediction MSE — New Domains")
    print("=" * 60)
    results["water_mse"] = compute_mse("water", node_feat_dim=5, action_dim=3)
    results["epinet_mse"] = compute_mse("epinet", node_feat_dim=5, action_dim=3)

    water_results = eval_water()
    epinet_results = eval_epinet()
    results["water_planning"] = water_results
    results["epinet_planning"] = epinet_results

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/new_domains_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved to results/new_domains_results.json")

    # Print LaTeX-ready summary
    print("\n" + "=" * 60)
    print("  LATEX TABLE NUMBERS")
    print("=" * 60)
    if results.get("water_mse"):
        w = results["water_mse"]
        print(f"  WaterNet 1-step:   {w.get(1, 0):.2e}  10-step: {w.get(10, 0):.2e}")
    if results.get("epinet_mse"):
        e = results["epinet_mse"]
        print(f"  EpiNet   1-step:   {e.get(1, 0):.2e}  10-step: {e.get(10, 0):.2e}")

    nd_w = water_results.get("netdream", {})
    gp_w = water_results.get("greedy", {})
    print(f"\n  WaterNet NetDream:  cost={nd_w.get('cost_mean',0):.1f}±{nd_w.get('cost_std',0):.1f}, "
          f"viols={nd_w.get('violations_mean',0):.1f}±{nd_w.get('violations_std',0):.1f}")
    print(f"  WaterNet Greedy:    cost={gp_w.get('cost_mean',0):.1f}, "
          f"viols={gp_w.get('violations_mean',0):.1f}")

    nd_e = epinet_results.get("netdream", {})
    gr_e = epinet_results.get("greedy_reactive", {})
    ni_e = epinet_results.get("no_inter", {})
    print(f"\n  EpiNet NetDream:    cost={nd_e.get('cost_mean',0):.1f}±{nd_e.get('cost_std',0):.1f}, "
          f"viols={nd_e.get('violations_mean',0):.1f}±{nd_e.get('violations_std',0):.1f}")
    print(f"  EpiNet Greedy-React:cost={gr_e.get('cost_mean',0):.1f}±{gr_e.get('cost_std',0):.1f}, "
          f"viols={gr_e.get('violations_mean',0):.1f}±{gr_e.get('violations_std',0):.1f}")
    print(f"  EpiNet No-Inter:    cost={ni_e.get('cost_mean',0):.1f}, "
          f"viols={ni_e.get('violations_mean',0):.1f}")
