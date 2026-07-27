import argparse

import numpy as np

from rarl.algos.td3 import TD3Agent
from rarl.envs.carla_env import CarlaRiskAwareEnv
from rarl.utils.config import load_config
from rarl.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    env = CarlaRiskAwareEnv(cfg, dry_run=args.dry_run, split="eval")
    agent = TD3Agent(env.observation_dim, env.action_dim, max_action=1.0, cfg=cfg, device=cfg["device"])
    agent.load(args.checkpoint)

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
