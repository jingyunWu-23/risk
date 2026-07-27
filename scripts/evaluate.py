import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from rarl.algos.td3 import TD3Agent
from rarl.envs.carla_env import CarlaRiskAwareEnv
from rarl.utils.config import load_config
from rarl.utils.seed import set_seed


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    checkpoint_path = resolve_project_path(args.checkpoint)
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    env = CarlaRiskAwareEnv(cfg, dry_run=args.dry_run, split="eval")
    agent = TD3Agent(env.observation_dim, env.action_dim, max_action=1.0, cfg=cfg, device=cfg["device"])
    agent.load(checkpoint_path)

    rewards = []
    collisions = []
    completions = []
    for episode in range(cfg["experiment"]["eval_episodes"]):
        obs = env.reset(seed=10000 + episode)
        done = False
        total_reward = 0.0
        final_info = {}
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, final_info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        collisions.append(float(final_info["collision"]))
        completions.append(float(final_info["route_completion"]))

    print(f"avg_reward={np.mean(rewards):.3f}")
    print(f"collision_rate={np.mean(collisions):.3f}")
    print(f"route_completion={np.mean(completions):.3f}")
    env.close()


if __name__ == "__main__":
    main()
