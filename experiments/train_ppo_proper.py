#!/usr/bin/env python3
"""Train PPO properly: 500K steps, tuned reward to prevent degenerate policies."""

import sys, warnings
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.k8s_mesh_env import K8sMeshEnv


class K8sMeshGymV2(gym.Env):
    """Improved K8s mesh wrapper with reward shaping to prevent degenerate policies."""

    def __init__(self, workload_type="variable", seed=42):
        super().__init__()
        self._env = K8sMeshEnv(workload_type=workload_type, max_steps=240, seed=seed)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self._env.num_nodes * self._env.node_feature_dim,),
            dtype=np.float32,
        )
        # Simplified action: select one service to scale up, one to scale down, or do nothing
        # 0 = do nothing, 1-11 = scale up service i, 12-22 = scale down service i
        self.action_space = spaces.Discrete(1 + 2 * self._env.num_nodes)

    def _decode_action(self, action):
        full_action = np.ones(self._env.num_nodes, dtype=int)  # all no-op
        if action == 0:
            pass  # do nothing
        elif action <= self._env.num_nodes:
            full_action[action - 1] = 2  # scale up
        else:
            full_action[action - self._env.num_nodes - 1] = 0  # scale down
        return full_action

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed)
        return obs.data.x.numpy().flatten(), info

    def step(self, action):
        full_action = self._decode_action(action)
        obs, reward, terminated, truncated, info = self._env.step(full_action)

        # Shaped reward: penalize both under-provisioning (SLO violations) AND
        # under-utilization (too few replicas when load is high)
        slo_penalty = info.get("slo_violations", 0) * 2.0
        cost = info.get("total_cost", 0) * 0.1

        # Bonus for maintaining replicas proportional to load
        x = obs.data.x.numpy()
        avg_cpu = np.mean(x[:, 0]) * 100  # denormalize
        avg_replicas = np.mean(x[:, 5]) * 10

        # Penalize very low replicas when CPU is high
        underprovisioning_penalty = 0.0
        if avg_cpu > 60 and avg_replicas < 2:
            underprovisioning_penalty = (avg_cpu - 60) * 0.05

        shaped_reward = -(slo_penalty + cost + underprovisioning_penalty)

        return obs.data.x.numpy().flatten(), shaped_reward, terminated, truncated, info


if __name__ == "__main__":
    seeds = [42, 123, 456, 789, 1024]
    workloads = ["constant", "variable", "bursty", "flash"]

    # Train PPO with 500K steps
    print("Training PPO v2 (500K steps, shaped reward)...")
    env = DummyVecEnv([lambda: K8sMeshGymV2("variable", 42)])
    ppo = PPO("MlpPolicy", env, verbose=0, n_steps=512, batch_size=128,
              learning_rate=3e-4, n_epochs=10, seed=42,
              ent_coef=0.01, clip_range=0.2)
    ppo.learn(total_timesteps=500000)
    ppo.save("models/ppo_v2_k8s_mesh")
    print("  PPO v2 trained (500K steps).")

    # Evaluate
    print("\nEvaluating PPO v2:")
    for wl in workloads:
        all_violations, all_costs = [], []
        for seed in seeds:
            eval_env = K8sMeshGymV2(wl, seed)
            obs, _ = eval_env.reset(seed=seed)
            total_cost, total_viol, steps = 0, 0, 0
            for _ in range(120):  # match NetDream evaluation length
                action, _ = ppo.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                total_cost += info.get("total_cost", 0)
                total_viol += info.get("slo_violations", 0)
                steps += 1
                if terminated or truncated:
                    break
            all_costs.append(total_cost)
            all_violations.append(total_viol)
        print(f"  {wl:>10}: cost={np.mean(all_costs):.2f}±{np.std(all_costs):.2f}  "
              f"viol={np.mean(all_violations):.1f}±{np.std(all_violations):.1f}")
