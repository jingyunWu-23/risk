"""Adapter that evaluates this project's RARL TD3 ego policy in a discrete test env."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from a.envs.action_adapter import DiscreteDrivingAction
from rarl.algos.td3 import TD3Agent
from rarl.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def is_rarl_td3_checkpoint(path: str | Path) -> bool:
    name = Path(path).name
    return name == "latest.pt" or re.fullmatch(r"checkpoint_\d+\.pt", name) is not None


class RARLTD3DiscreteAdapter:
    """Expose the RARL continuous TD3 actor through the 0-4 ego action API."""

    def __init__(
        self,
        checkpoint: str | Path,
        config_path: str | Path = PROJECT_ROOT / "configs" / "compare_3sv.yaml",
        no_cuda: bool = False,
    ):
        self.checkpoint = Path(checkpoint)
        cfg = load_config(config_path)
        if no_cuda:
            cfg["device"] = "cpu"
        self.cfg = cfg
        self.obs_dim = 8 + int(cfg["observation"]["max_surrounding_vehicles"]) * 5 + 4
        self.agent = TD3Agent(self.obs_dim, 2, max_action=1.0, cfg=cfg, device=cfg["device"])
        self.agent.load(self.checkpoint)
        self.lane_width = float(cfg.get("road", {}).get("lane_width", 3.5))
        self.route_length = float(cfg.get("risk_adapter", {}).get("route_length", 200.0))

    def select_action(self, state, deterministic: bool = True):
        del deterministic
        rarl_obs = self._adapt_25d_to_62d(state)
        continuous = self.agent.select_action(rarl_obs)
        return self._continuous_to_discrete(continuous), None, None

    def _adapt_25d_to_62d(self, state) -> np.ndarray:
        obs25 = np.asarray(state, dtype=np.float32).reshape(-1)
        if obs25.size < 25:
            obs25 = np.pad(obs25, (0, 25 - obs25.size))

        route_s = float(obs25[0]) * self.route_length
        lateral_offset = float(obs25[1]) * self.lane_width
        speed = float(obs25[2]) * 35.0
        yaw = float(obs25[3]) * np.pi
        ego = np.asarray([route_s, lateral_offset, yaw, speed, 0.0, speed, 0.0, 0.0], dtype=np.float32)

        sv_features = np.zeros((self.cfg["observation"]["max_surrounding_vehicles"], 5), dtype=np.float32)
        lane_features = [
            (obs25[6], 0.0, obs25[7]),  # same-lane front
            (-obs25[6], 0.0, 0.0),  # same-lane rear proxy
            (obs25[10], -self.lane_width, obs25[12]),  # left front
            (-obs25[11], -self.lane_width, obs25[13]),  # left rear
            (obs25[15], self.lane_width, obs25[17]),  # right front
            (-obs25[16], self.lane_width, obs25[18]),  # right rear
        ]
        for idx, (gap_norm, rel_y, dv_norm) in enumerate(lane_features[: len(sv_features)]):
            gap = float(gap_norm) * 100.0
            if abs(gap) >= 99.0:
                continue
            sv_features[idx] = np.asarray([gap, float(rel_y), float(dv_norm) * 35.0, 0.0, 0.1], dtype=np.float32)

        vehicle_risk = float(np.mean(np.clip(obs25[19:23], 0.0, 1.0)))
        road_risk = max(0.0, abs(float(obs25[1])) - 0.5)
        total_risk = road_risk + vehicle_risk
        risk = np.asarray([road_risk, vehicle_risk, total_risk, 1.0 / (1.0 + total_risk)], dtype=np.float32)
        return np.concatenate([ego, sv_features.reshape(-1), risk]).astype(np.float32)

    @staticmethod
    def _continuous_to_discrete(action) -> int:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        accel = float(action[0]) if action.size > 0 else 0.0
        steer = float(action[1]) if action.size > 1 else 0.0
        if steer < -0.25:
            return int(DiscreteDrivingAction.LANE_LEFT)
        if steer > 0.25:
            return int(DiscreteDrivingAction.LANE_RIGHT)
        if accel > 0.20:
            return int(DiscreteDrivingAction.ACCELERATE)
        if accel < -0.20:
            return int(DiscreteDrivingAction.DECELERATE)
        return int(DiscreteDrivingAction.KEEP_LANE)
