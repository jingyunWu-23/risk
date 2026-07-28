"""Lightweight runtime helpers for evaluation scripts.

This module intentionally avoids importing EgoPPO or MAPPO so external ego
policies can be evaluated without a full MARL1 ego package.
"""

from __future__ import annotations

import numpy as np

from a.envs.action_adapter import DiscreteDrivingAction


ACTION_NAMES = {action.name.lower(): int(action.value) for action in DiscreteDrivingAction}
ACTION_NAMES.update({str(int(action.value)): int(action.value) for action in DiscreteDrivingAction})


def load_hdv_policy(model_path: str):
    if not model_path:
        return None
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required to load --hdv-model.") from exc
    return PPO.load(model_path)


def hdv_actions(env, hdv_policy, fixed_action: int):
    observations = list(getattr(env, "obs_hdv_list", []))
    if not observations:
        return []
    if hdv_policy is None:
        return [int(fixed_action)] * len(observations)
    actions = []
    for obs in observations:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)[:25]
        action, _ = hdv_policy.predict(obs, deterministic=True)
        actions.append(int(action))
    return actions


def append_hdv_actions(env, base_actions, args):
    if str(getattr(args, "hdv_control_mode", "")).lower() in {"traffic_manager", "autopilot"}:
        return list(base_actions)
    return list(base_actions) + hdv_actions(
        env,
        getattr(args, "_hdv_policy", None),
        ACTION_NAMES[getattr(args, "hdv_action", "keep_lane")],
    )


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value):
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def init_episode_diagnostics():
    return {
        "ego_obstacle_risk_control_max": 0.0,
        "ego_target_lane_obstacle_risk_max": 0.0,
        "ego_obstacle_risk_steps": 0,
        "ego_obstacle_escape_available_steps": 0,
        "ego_obstacle_avoidance_active_steps": 0,
        "ego_obstacle_speed_limited_steps": 0,
        "ego_obstacle_stop_active_steps": 0,
        "ego_obstacle_takeover_active_steps": 0,
        "ego_obstacle_takeover_released_steps": 0,
        "ego_lane_obstacle_veto_steps": 0,
        "ego_lane_change_cancelled_for_obstacle_steps": 0,
        "ego_hazardous_lane_change_steps": 0,
        "ego_route_takeover_active_steps": 0,
        "ego_clear_path_recovery_active_steps": 0,
        "ego_hazardous_lane_change_penalty_sum": 0.0,
    }


def update_episode_diagnostics(diagnostics, info):
    obstacle_risk = _as_float(info.get("ego_obstacle_risk_control", 0.0))
    target_risk = _as_float(info.get("ego_target_lane_obstacle_risk", 0.0))
    diagnostics["ego_obstacle_risk_control_max"] = max(diagnostics["ego_obstacle_risk_control_max"], obstacle_risk)
    diagnostics["ego_target_lane_obstacle_risk_max"] = max(diagnostics["ego_target_lane_obstacle_risk_max"], target_risk)
    diagnostics["ego_obstacle_risk_steps"] += int(obstacle_risk > 0.0 or target_risk > 0.0)
    diagnostics["ego_obstacle_escape_available_steps"] += int(_as_bool(info.get("ego_obstacle_escape_available", False)))
    diagnostics["ego_obstacle_avoidance_active_steps"] += int(_as_bool(info.get("ego_obstacle_avoidance_active", False)))
    diagnostics["ego_obstacle_speed_limited_steps"] += int(_as_bool(info.get("ego_obstacle_speed_limited", False)))
    diagnostics["ego_obstacle_stop_active_steps"] += int(_as_bool(info.get("ego_obstacle_stop_active", False)))
    diagnostics["ego_obstacle_takeover_active_steps"] += int(_as_bool(info.get("ego_obstacle_takeover_active", False)))
    diagnostics["ego_obstacle_takeover_released_steps"] += int(_as_bool(info.get("ego_obstacle_takeover_released", False)))
    diagnostics["ego_lane_obstacle_veto_steps"] += int(_as_bool(info.get("ego_lane_obstacle_veto", False)))
    diagnostics["ego_lane_change_cancelled_for_obstacle_steps"] += int(_as_bool(info.get("ego_lane_change_cancelled_for_obstacle", False)))
    diagnostics["ego_hazardous_lane_change_steps"] += int(_as_bool(info.get("ego_hazardous_lane_change", False)))
    diagnostics["ego_route_takeover_active_steps"] += int(_as_bool(info.get("ego_route_takeover_active", False)))
    diagnostics["ego_clear_path_recovery_active_steps"] += int(_as_bool(info.get("ego_clear_path_recovery_active", False)))
    diagnostics["ego_hazardous_lane_change_penalty_sum"] += _as_float(info.get("ego_hazardous_lane_change_penalty", 0.0))


def merge_episode_diagnostics(info, diagnostics):
    merged = dict(info)
    merged.update(diagnostics)
    merged["ego_obstacle_risk_any"] = bool(diagnostics["ego_obstacle_risk_steps"] > 0)
    merged["ego_obstacle_escape_available_any"] = bool(diagnostics["ego_obstacle_escape_available_steps"] > 0)
    merged["ego_obstacle_avoidance_active_any"] = bool(diagnostics["ego_obstacle_avoidance_active_steps"] > 0)
    merged["ego_obstacle_speed_limited_any"] = bool(diagnostics["ego_obstacle_speed_limited_steps"] > 0)
    merged["ego_obstacle_stop_active_any"] = bool(diagnostics["ego_obstacle_stop_active_steps"] > 0)
    merged["ego_obstacle_takeover_active_any"] = bool(diagnostics["ego_obstacle_takeover_active_steps"] > 0)
    merged["ego_obstacle_takeover_released_any"] = bool(diagnostics["ego_obstacle_takeover_released_steps"] > 0)
    merged["ego_lane_obstacle_veto_any"] = bool(diagnostics["ego_lane_obstacle_veto_steps"] > 0)
    merged["ego_lane_change_cancelled_for_obstacle_any"] = bool(diagnostics["ego_lane_change_cancelled_for_obstacle_steps"] > 0)
    merged["ego_hazardous_lane_change_any"] = bool(diagnostics["ego_hazardous_lane_change_steps"] > 0)
    merged["ego_route_takeover_active_any"] = bool(diagnostics["ego_route_takeover_active_steps"] > 0)
    merged["ego_clear_path_recovery_active_any"] = bool(diagnostics["ego_clear_path_recovery_active_steps"] > 0)
    return merged


def episode_metrics(steps: int, episode_reward: float, reward_array, info: dict):
    agents_crash = int(info.get("adv_collision_event_count", 0)) > 0
    metrics = dict(info)
    metrics.update({
        "steps": int(steps),
        "episode_reward": float(episode_reward),
        "mean_agent_reward": float(np.mean(reward_array)) if np.size(reward_array) else 0.0,
        "route_completion": float(info.get("route_completion", 0.0)),
        "route_completion_full": float(info.get("route_completion_full", info.get("route_completion", 0.0))),
        "route_completion_distance": float(info.get("route_completion_distance", 0.0)),
        "route_length": float(info.get("route_length", 0.0)),
        "crash": bool(info.get("crash", False)),
        "agents_crash": bool(agents_crash),
        "adv_adv_collision": int(info.get("adv_adv_collision_event_count", 0)) > 0,
        "adv_ego_collision": int(info.get("adv_ego_collision_event_count", 0)) > 0,
        "adv_other_collision": int(info.get("adv_other_collision_event_count", 0)) > 0,
        "adv_recovered": int(info.get("adv_recovery_count", 0)) > 0,
        "timeout": bool(info.get("timeout", False)),
        "average_speed": float(info.get("average_speed", 0.0)),
    })
    return metrics


def diagnostic_fields():
    return [
        "ego_target_speed",
        "ego_front_gap",
        "ego_last_action",
        "ego_lane_change_failed",
        "ego_lane_change_success",
        "ego_lane_change_cooldown",
        "ego_stop_penalty_value",
        "ego_route_s",
        "ego_lateral_offset",
        "ego_heading_error",
        "ego_road_id",
        "ego_section_id",
        "ego_raw_lane_id",
        "ego_obstacle_distance",
        "ego_obstacle_lateral",
        "ego_obstacle_risk",
        "ego_obstacle_penalty",
        "ego_obstacle_escape_bonus",
        "ego_obstacle_recovery_bonus",
        "ego_obstacle_takeover_penalty",
        "ego_obstacle_takeover_release_bonus",
        "ego_obstacle_recovered",
        "ego_obstacle_low_speed_pending",
        "ego_obstacle_distance_control",
        "ego_obstacle_risk_control",
        "ego_obstacle_label",
        "ego_obstacle_lane_lateral",
        "ego_obstacle_escape_available",
        "ego_obstacle_avoidance_active",
        "ego_obstacle_avoidance_blocked_reason",
        "ego_obstacle_speed_limited",
        "ego_obstacle_stop_active",
        "ego_obstacle_takeover_active",
        "ego_obstacle_takeover_timer",
        "ego_obstacle_takeover_clear_steps",
        "ego_obstacle_takeover_released",
        "ego_obstacle_takeover_target_lane_id",
        "ego_clear_path_recovery_active",
        "ego_target_lane_obstacle_risk",
        "ego_lane_obstacle_veto",
        "ego_lane_change_cancelled_for_obstacle",
        "ego_hazardous_lane_change",
        "ego_hazardous_lane_change_risk",
        "ego_route_takeover_active",
        "ego_route_takeover_reason",
        "ego_route_current_error_deg",
        "ego_route_target_error_deg",
        "ego_hazardous_lane_change_penalty",
        "ego_obstacle_risk_control_max",
        "ego_target_lane_obstacle_risk_max",
        "ego_obstacle_risk_steps",
        "ego_obstacle_escape_available_steps",
        "ego_obstacle_avoidance_active_steps",
        "ego_obstacle_speed_limited_steps",
        "ego_obstacle_stop_active_steps",
        "ego_obstacle_takeover_active_steps",
        "ego_obstacle_takeover_released_steps",
        "ego_lane_obstacle_veto_steps",
        "ego_lane_change_cancelled_for_obstacle_steps",
        "ego_hazardous_lane_change_steps",
        "ego_route_takeover_active_steps",
        "ego_clear_path_recovery_active_steps",
        "ego_hazardous_lane_change_penalty_sum",
        "ego_obstacle_risk_any",
        "ego_obstacle_escape_available_any",
        "ego_obstacle_avoidance_active_any",
        "ego_obstacle_speed_limited_any",
        "ego_obstacle_stop_active_any",
        "ego_obstacle_takeover_active_any",
        "ego_obstacle_takeover_released_any",
        "ego_lane_obstacle_veto_any",
        "ego_lane_change_cancelled_for_obstacle_any",
        "ego_hazardous_lane_change_any",
        "ego_route_takeover_active_any",
        "ego_clear_path_recovery_active_any",
    ]


def diagnostic_row(metrics):
    return {field: metrics.get(field, "") for field in diagnostic_fields()}


def episode_log_row(episode, metrics, losses):
    return {
        "episode": episode,
        "steps": metrics.get("steps", 0),
        "episode_reward": metrics.get("episode_reward", 0.0),
        "mean_agent_reward": metrics.get("mean_agent_reward", 0.0),
        "route_completion": metrics.get("route_completion", 0.0),
        "route_completion_full": metrics.get("route_completion_full", metrics.get("route_completion", 0.0)),
        "route_completion_distance": metrics.get("route_completion_distance", 0.0),
        "route_length": metrics.get("route_length", 0.0),
        "crash": metrics.get("crash", False),
        "agents_crash": metrics.get("agents_crash", False),
        "timeout": metrics.get("timeout", False),
        **diagnostic_row(metrics),
        "actor_loss": losses.get("actor_loss", ""),
        "critic_loss": losses.get("critic_loss", ""),
        "entropy": losses.get("entropy", ""),
        "clip_fraction": losses.get("clip_fraction", ""),
        "value_error": losses.get("value_error", ""),
        "ppo_epochs": losses.get("ppo_epochs", ""),
        "samples": losses.get("samples", ""),
    }
