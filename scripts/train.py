import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import trange

from rarl.algos.replay_buffer import ReplayBuffer
from rarl.algos.td3 import TD3Agent
from rarl.envs.carla_env import CarlaRiskAwareEnv
from rarl.utils.config import load_config
from rarl.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    env = CarlaRiskAwareEnv(cfg, dry_run=args.dry_run)
    obs_dim = env.observation_dim
    action_dim = env.action_dim
    agent = TD3Agent(obs_dim, action_dim, max_action=1.0, cfg=cfg, device=cfg["device"])
    replay_buffer = ReplayBuffer(obs_dim, action_dim, cfg["td3"]["replay_buffer_size"])

    obs = env.reset(seed=cfg["seed"])
    episode_reward = 0.0
    episode_num = 0
    episode_step = 0
    log_dir = Path(cfg["experiment"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    max_timesteps = int(cfg["experiment"]["max_timesteps"])
    for t in trange(max_timesteps, desc="train"):
        if t < cfg["td3"]["start_timesteps"]:
            action = np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
        else:
            action = agent.select_action(obs)
            action += np.random.normal(0, cfg["td3"]["expl_noise"], size=action_dim)
            action = action.clip(-1.0, 1.0)

        next_obs, reward, done, info = env.step(action)
        replay_buffer.add(obs, action, next_obs, reward, done)
        obs = next_obs
        episode_reward += reward
        episode_step += 1

        if t >= cfg["td3"]["start_timesteps"]:
            agent.train(replay_buffer)

        if done:
            print(
                f"step={t + 1} episode={episode_num} episode_step={episode_step} "
                f"reward={episode_reward:.3f} collision={info['collision']} "
                f"offroad={info['offroad']} route_completion={info['route_completion']:.3f} "
                f"risk={info['total_risk']:.3f}"
            )
            obs = env.reset(seed=cfg["seed"] + episode_num + 1)
            episode_reward = 0.0
            episode_step = 0
            episode_num += 1

        if (t + 1) % cfg["experiment"]["save_freq"] == 0:
            agent.save(log_dir / f"checkpoint_{t + 1}.pt")
            agent.save(log_dir / "latest.pt")

    agent.save(log_dir / "latest.pt")
    env.close()


if __name__ == "__main__":
    main()
