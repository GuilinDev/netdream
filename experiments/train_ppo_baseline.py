#!/usr/bin/env python3
"""Train PPO and DQN model-free baselines on K8s mesh for comparison."""

import sys
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.k8s_mesh_env import K8sMeshEnv


class K8sMeshGymWrapper(gym.Env):
    """Wrap K8sMeshEnv as a flat Gymnasium env for SB3."""

    def __init__(self, workload_type="variable", seed=42):
        super().__init__()
        self._env = K8sMeshEnv(workload_type=workload_type, max_steps=240, seed=seed)

        # Flat observation: all node features concatenated
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self._env.num_nodes * self._env.node_feature_dim,),
            dtype=np.float32,
        )

        # Multi-discrete: one action per service
        # SB3 DQN doesn't support MultiDiscrete, so flatten to single Discrete
        # Total actions = 3^11 is too large. Use a simpler scheme:
        # Action = (service_index * 3 + delta), cycle through services
        self.action_space = spaces.Discrete(self._env.num_nodes * self._env.action_dim)

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed)
        return obs.data.x.numpy().flatten(), info

    def step(self, action):
        # Decode: which service and what delta
        service_idx = action // self._env.action_dim
        delta = action % self._env.action_dim

        # Build full action array (all no-op except the selected service)
        full_action = np.ones(self._env.num_nodes, dtype=int)  # no-op
        full_action[service_idx] = delta

        obs, reward, terminated, truncated, info = self._env.step(full_action)
        return obs.data.x.numpy().flatten(), reward, terminated, truncated, info


def train_and_evaluate():
    seeds = [42, 123, 456, 789, 1024]
    workloads = ["constant", "variable", "bursty", "flash"]

    # Train PPO
    print("Training PPO on variable workload (50K steps)...")
    env = DummyVecEnv([lambda: K8sMeshGymWrapper("variable", 42)])
    ppo = PPO("MlpPolicy", env, verbose=0, n_steps=256, batch_size=64,
              learning_rate=3e-4, n_epochs=10, seed=42)
    ppo.learn(total_timesteps=50000)
    ppo.save("models/ppo_k8s_mesh")
    print("  PPO trained.")

    # Evaluate PPO
    print("\nEvaluating PPO:")
    for wl in workloads:
        all_violations = []
        all_costs = []
        for seed in seeds:
            eval_env = K8sMeshGymWrapper(wl, seed)
            obs, _ = eval_env.reset(seed=seed)
            total_cost, total_viol = 0, 0
            for _ in range(60):
                action, _ = ppo.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                total_cost += info.get("total_cost", 0)
                total_viol += info.get("slo_violations", 0)
                if terminated or truncated:
                    break
            all_costs.append(total_cost)
            all_violations.append(total_viol)
        print(f"  {wl:>10}: cost={np.mean(all_costs):.2f}±{np.std(all_costs):.2f}  "
              f"viol={np.mean(all_violations):.1f}±{np.std(all_violations):.1f}")


if __name__ == "__main__":
    train_and_evaluate()
