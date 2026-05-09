# NetDream

**Graph world models for safe Kubernetes autoscaling.**

![NetDream — 30-second explainer](assets/netdream.gif)

> 30-second walkthrough of the three modules: graph world model →
> imagination-based planner → online deployment on Kubernetes. Rendered video at
> [`assets/netdream.mp4`](assets/netdream.mp4).

NetDream learns a graph-structured dynamics model of a microservice
application, uses it for imagination-based model-predictive control,
and rejects unsafe scaling actions through a learned safety filter
before they reach the live cluster.

Trained offline on logged transitions, deployed online as a Kubernetes
controller. We evaluate on Google's Online Boutique microservice
benchmark deployed to AWS EKS, and provide cross-domain validation on
power-grid (Grid2Op IEEE 14-bus), water-distribution, and epidemic
mobility graphs.

> **Paper:** *Autoscaling by Imagination: Graph World Models for Safe
> Distributed Machine Learning Services* — `paper/main.pdf`.

---

## Repository layout

```
netdream/                  Core library
  graph_env.py             Unified GraphEnv interface (PyG-compatible)
  dynamics_model.py        GNN dynamics model (GAT / GCN / SAGE)
  planner.py               CEM-based planner + SafetyFilter
  fast_planner.py          Batched random-shooting MPC (used by paper)
  cluster_state.py         Prometheus → state-tensor estimator
  k8s_controller.py        Apply scaling via apps/v1 Deployment.scale
  prometheus_client.py     Thin Prometheus HTTP client

envs/                      Domain adapters
  k8s_mesh_env.py          11-service Online Boutique mesh
  grid2op_env.py           IEEE 14-bus power grid wrapper
  water_net_env.py         WaterNet-10 hydraulic network
  epinet_env.py            20-node SIR epidemic mobility graph
  traffic_env.py           Cell-transmission traffic grid

experiments/               Reproduction pipeline
  collect_data.py          Sample transitions from environments
  train_dynamics.py        Train GNN dynamics (--mode k8s|grid2op|...)
  evaluate_planning.py     Evaluate MPC + safety filter
  train_ppo_proper.py      PPO baseline (500K steps)
  generate_*.py            Paper figures
  aggregate_cluster_results.py   Aggregate raw EKS episode logs

infra/                     Cluster manifests
  microk8s/                Local-development cluster recipes
  eks/                     AWS EKS deployment (Terraform / Helm values)

paper/                     LaTeX source + compiled PDF
  main.tex                 Top-level entry
  sections/                Per-section .tex sources
  figures/                 Architecture diagrams + experimental plots
  references.bib           81 cited works

configs/                   YAML hyperparameter sets
data/                      Transitions (regenerate via collect_data.py)
models/                    Trained checkpoints
results/                   Aggregated benchmark results
```

---

## Reproducing the main results

```bash
# 1. Set up environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt    # torch, torch-geometric, grid2op,
                                   # gymnasium, stable-baselines3, ...

# 2. Collect transition data
python experiments/collect_data.py

# 3. Train the world model (K8s + power grid + ablation)
python experiments/train_dynamics.py --mode both --epochs 100 --gnn-type gat
python experiments/train_dynamics.py --mode ablation

# 4. Train the model-free PPO baseline
python experiments/train_ppo_proper.py

# 5. Evaluate planning + safety filter
python experiments/evaluate_planning.py

# 6. Regenerate paper figures
python experiments/generate_figures_v3.py
python experiments/generate_cluster_figures.py
```

Default hyperparameters: `H=5` planning horizon, `K=64` candidate
sequences, `τ=0.5` safety threshold, GAT 3 layers × 4 heads × 128 hidden,
86K parameters total. Full table in `paper/sections/appendix.tex`.

---

## Real-cluster deployment

The closed-loop controller in `netdream/k8s_controller.py` reads
metrics from Prometheus every 5 s, calls the trained world model, runs
the planner, and applies the selected per-service scaling action via
the Kubernetes `apps/v1` `Deployment.scale` subresource.

We have deployed this against:

- **MicroK8s 1.30** on a single Linux host (local development).
- **AWS EKS 1.30** on a 3-node `t3.medium` managed node group with
  Istio + `kube-prometheus-stack` (paper evaluation).

Cluster manifests are in `infra/`.

---

## Citing

If you build on this work, please cite the paper (that BibTeX entry will be
released with the camera-ready version).

---

## License

MIT for code (`LICENSE` file). Paper text and figures are released
under CC-BY 4.0. External assets (Online Boutique, Grid2Op,
Stable-Baselines3, PyTorch Geometric) are used under their respective
licenses.
