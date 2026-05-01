# NetDream: Supplementary Code and Data

## Structure
```
netdream/                  # Core library
  graph_env.py             # Unified GraphEnv interface
  dynamics_model.py        # GNN dynamics model (GAT/GCN/SAGE)
  fast_planner.py          # Batched random-shooting MPC + safety filter
  planner.py               # CEM-based planner (alternative)

envs/                      # Infrastructure environments
  k8s_mesh_env.py          # 11-service Kubernetes microservice mesh
  grid2op_env.py           # IEEE 14-bus power grid (Grid2Op wrapper)

experiments/               # Experiment scripts
  collect_data.py          # Collect transition data from environments
  train_dynamics.py        # Train GNN dynamics model
  evaluate_planning.py     # Evaluate planning + safety filter
  train_ppo_proper.py      # Train PPO model-free baseline (500K steps)
  grid2op_greedy_baseline.py  # Greedy reconnection baseline

models/                    # Trained model checkpoints
  k8s_best.pt              # Best K8s dynamics model (GAT)
  grid2op_best.pt          # Best Grid2Op dynamics model (GAT)

data/                      # Collected transition data
  k8s_transitions.pkl      # 48,000 K8s transitions
  grid2op_transitions.pkl  # 12,040 Grid2Op transitions
```

## Requirements
```bash
pip install torch torch-geometric grid2op gymnasium stable-baselines3 networkx numpy pandas
```

## Reproducing Results
```bash
# 1. Collect data
python experiments/collect_data.py

# 2. Train dynamics model
python experiments/train_dynamics.py --mode both

# 3. Evaluate planning
python experiments/evaluate_planning.py

# 4. Run ablation
python experiments/train_dynamics.py --mode ablation
```

## License
MIT
