#!/usr/bin/env python3
"""N-k adversarial contingency: simplified and robust."""

import sys, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from netdream.dynamics_model import GNNDynamicsModel
from netdream.fast_planner import FastRandomShootingPlanner
from envs.grid2op_env import Grid2OpGraphEnv


def run_contingency(env, agent_fn, n_disc, num_trials=30, attack_step=3, max_steps=30):
    """Run N-k contingency trials. Skip seeds where grid dies before attack."""
    survived = 0
    overloads_total = 0
    valid_trials = 0

    for trial in range(num_trials * 3):  # over-sample to get enough valid trials
        if valid_trials >= num_trials:
            break

        obs, _ = env.reset(seed=trial + 200)

        # Run until attack step
        alive = True
        for t in range(attack_step):
            action = np.ones(env.num_nodes, dtype=int)  # do nothing pre-attack
            obs, r, term, trunc, info = env.step(action)
            if term:
                alive = False
                break

        if not alive:
            continue  # skip this trial — grid died before we could attack

        # Force-disconnect n_disc lines
        g2op_obs = env._obs
        connected = np.where(g2op_obs.line_status)[0]
        if len(connected) < n_disc:
            continue

        rng = np.random.default_rng(trial + 5000)
        to_disc = rng.choice(connected, size=n_disc, replace=False)

        disc_ok = True
        for li in to_disc:
            act = env._env.action_space({})
            act.line_set_status = [(int(li), -1)]
            new_obs, _, done, _ = env._env.step(act)
            env._obs = new_obs
            if done:
                disc_ok = False
                break

        if not disc_ok:
            # Grid died from the disconnect itself — count as failure
            valid_trials += 1
            overloads_total += 5  # penalty
            continue

        obs = env._obs_to_graph(env._obs)
        valid_trials += 1

        # Now let agent try to recover
        trial_overloads = 0
        trial_survived = True
        for t in range(max_steps - attack_step):
            action = agent_fn(env, obs)
            obs, r, term, trunc, info = env.step(action)
            if info.get("max_rho", 0) > 1.0:
                trial_overloads += 1
            if term:
                trial_survived = False
                break

        if trial_survived:
            survived += 1
        overloads_total += trial_overloads

    survival_rate = survived / max(valid_trials, 1) * 100
    avg_overloads = overloads_total / max(valid_trials, 1)
    return survival_rate, avg_overloads, valid_trials


def do_nothing(env, obs):
    return np.ones(env.num_nodes, dtype=int)

def greedy(env, obs):
    action = np.ones(env.num_nodes, dtype=int)
    g2op_obs = env._obs
    for li in range(env._n_lines):
        if not g2op_obs.line_status[li]:
            action[g2op_obs.line_or_to_subid[li]] = 2
            break
    return action

def make_netdream(model, env):
    planner = FastRandomShootingPlanner(
        model=model, horizon=3, num_candidates=64,
        action_dim=3, num_nodes=env.num_nodes, use_safety=True)
    def fn(env_inner, obs):
        return planner.plan(obs.data.x.numpy(), obs.data.edge_index.numpy())
    return fn


if __name__ == "__main__":
    model = GNNDynamicsModel(node_feat_dim=7, action_dim=3, hidden_dim=128,
                              num_gnn_layers=3, num_constraints=1, gnn_type="gat")
    model.load_state_dict(torch.load("models/grid2op_best.pt", weights_only=True))
    model.eval()

    env = Grid2OpGraphEnv(max_steps=30)

    print("=" * 65)
    print("  N-k Adversarial Contingency Results")
    print("=" * 65)
    print(f"  {'Scenario':<12} {'Agent':<20} {'Survival%':<12} {'Overloads':<12} {'Trials'}")

    for n_disc in [1, 2]:
        for name, fn in [("Do-Nothing", do_nothing), ("Greedy", greedy), ("NetDream-Safe", make_netdream(model, env))]:
            surv, ovl, trials = run_contingency(env, fn, n_disc, num_trials=20)
            print(f"  N-{n_disc:<10} {name:<20} {surv:<12.0f} {ovl:<12.1f} {trials}")
