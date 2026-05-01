# Transition Data

Trained models and collected transitions are not committed to this
repository because of file size. To regenerate them:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Collect transitions for each domain
python experiments/collect_data.py
```

This produces:

| File | Size | Purpose |
|------|------|---------|
| `k8s_transitions.pkl`    | ~85 MB | 48,000 K8s mesh transitions (200 episodes × 240 steps) |
| `grid2op_transitions.pkl`| ~9 MB  | 12,040 IEEE 14-bus power-grid transitions |
| `water_transitions.pkl`  | ~14 MB | 19,200 water-distribution transitions (cross-domain) |
| `epinet_transitions.pkl` | ~7 MB  | 12,000 epidemic-graph transitions (cross-domain) |

Each `.pkl` is a list of dicts with keys `obs`, `action`, `next_obs`,
`reward`, `done`, `info` plus the static graph topology.

Regeneration is deterministic given the seed; default seed is 42.
