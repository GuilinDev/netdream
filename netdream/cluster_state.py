"""Build a NetDream GraphObs from live Kubernetes cluster metrics.

State layout per service (6-dim, matching simulator):
    [0] cpu_util     — [0, 1]  avg CPU utilization over pods
    [1] mem_util     — [0, 1]  avg memory utilization over pods
    [2] qps          — [0, 1]  normalized request rate  (approx from frontend QPS × topology weight)
    [3] latency      — [0, 1]  normalized p99 latency   (approx from load generator)
    [4] error_rate   — [0, 1]  fraction of failed requests (approx)
    [5] replicas     — [0, 1]  normalized replica count, capped at MAX_REPLICAS

Approximation note (documented in paper appendix app:k8s-cluster):
    Istio is not enabled (ARM64 issues). Per-service QPS / latency / error-rate are
    not directly observable. We estimate them from the aggregate Locust-side
    measurements distributed via the static Online Boutique topology weights.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from .prometheus_client import PromClient


MAX_REPLICAS = 10
LATENCY_SCALE_MS = 500.0  # SLO threshold used for normalization


# Topology weights — proportion of frontend QPS reaching each service (rough estimate
# from Online Boutique call patterns). Frontend = 1.0 by definition.
TOPOLOGY_WEIGHTS = {
    "frontend":              1.00,
    "productcatalogservice": 0.90,
    "recommendationservice": 0.70,
    "adservice":             0.70,
    "checkoutservice":       0.20,  # only during checkout flow
    "shippingservice":       0.20,
    "paymentservice":        0.20,
    "emailservice":          0.20,
    "redis-cart":            0.40,  # accessed by frontend + checkout
    "cartservice":           0.40,
    "currencyservice":       0.50,
}


def fetch_cluster_state(
    prom: PromClient,
    namespace: str,
    service_names: list[str],
    frontend_qps: float = 0.0,
    frontend_latency_ms: float = 0.0,
    frontend_error_rate: float = 0.0,
) -> np.ndarray:
    """Return (num_services, 6) state matrix from live Prometheus metrics."""

    n = len(service_names)
    state = np.zeros((n, 6), dtype=np.float32)

    # CPU usage per pod (last 30s), averaged by deployment via label match
    cpu_q = (
        f'avg by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{namespace}",'
        f'container!="",container!="POD"}}[30s]))'
    )
    cpu_by_pod = prom.scalar_by_label(cpu_q, "pod")

    # CPU limit per container (for normalization); fall back to 1.0 if unset
    limit_q = (
        f'sum by (pod) (kube_pod_container_resource_limits{{namespace="{namespace}",'
        f'resource="cpu"}})'
    )
    limit_by_pod = prom.scalar_by_label(limit_q, "pod")

    mem_q = (
        f'avg by (pod) (container_memory_working_set_bytes{{namespace="{namespace}",'
        f'container!="",container!="POD"}})'
    )
    mem_by_pod = prom.scalar_by_label(mem_q, "pod")

    mem_limit_q = (
        f'sum by (pod) (kube_pod_container_resource_limits{{namespace="{namespace}",'
        f'resource="memory"}})'
    )
    mem_limit_by_pod = prom.scalar_by_label(mem_limit_q, "pod")

    rep_q = (
        f'kube_deployment_status_replicas{{namespace="{namespace}"}}'
    )
    replicas_by_deployment = prom.scalar_by_label(rep_q, "deployment")

    # Per-service aggregation: find all pods whose name starts with "<svc>-"
    for i, svc in enumerate(service_names):
        prefix = f"{svc}-"
        pod_cpus = [v for pod, v in cpu_by_pod.items() if pod.startswith(prefix)]
        pod_limits = [limit_by_pod.get(p, 1.0) for p in cpu_by_pod if p.startswith(prefix)]
        pod_mems = [v for pod, v in mem_by_pod.items() if pod.startswith(prefix)]
        pod_mem_limits = [mem_limit_by_pod.get(p, 512 * 1024**2)
                          for p in mem_by_pod if p.startswith(prefix)]

        cpu_util = (np.mean([c / max(lim, 1e-3) for c, lim in zip(pod_cpus, pod_limits)])
                    if pod_cpus else 0.0)
        mem_util = (np.mean([m / max(lim, 1.0) for m, lim in zip(pod_mems, pod_mem_limits)])
                    if pod_mems else 0.0)

        replicas = replicas_by_deployment.get(svc, 1.0)
        topo_w = TOPOLOGY_WEIGHTS.get(svc, 0.5)

        # Clip utilities to [0, 1] for model stability; saturated CPU is a legitimate signal
        state[i, 0] = np.clip(cpu_util, 0.0, 1.0)
        state[i, 1] = np.clip(mem_util, 0.0, 1.0)
        state[i, 2] = np.clip(topo_w * frontend_qps / 100.0, 0.0, 1.0)  # QPS approx; 100 = saturation
        state[i, 3] = np.clip(frontend_latency_ms / LATENCY_SCALE_MS, 0.0, 1.0)
        state[i, 4] = np.clip(frontend_error_rate, 0.0, 1.0)
        state[i, 5] = np.clip(replicas / MAX_REPLICAS, 0.0, 1.0)

    return state


def to_graph_data(
    state: np.ndarray,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
    device: str = "cpu",
) -> Data:
    """Convert state matrix + topology into a PyG Data object."""
    x = torch.as_tensor(state, dtype=torch.float32, device=device)
    data = Data(x=x, edge_index=edge_index.to(device))
    if edge_attr is not None:
        data.edge_attr = edge_attr.to(device)
    return data
