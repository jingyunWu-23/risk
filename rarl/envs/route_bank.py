import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteScenario:
    scenario_id: int
    town: str
    ego_spawn_index: int
    ego_route_end_index: int
    sv_spawn_indices: list
    sv_target_speeds: list
    av_target_speed: float


class RouteBank:
    """Deterministic train/eval route sampler built from CARLA spawn indices."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.town = cfg["carla"]["town"]
        self.train_size = int(cfg.get("route_bank", {}).get("train_routes", 100))
        self.eval_size = int(cfg.get("route_bank", {}).get("eval_routes", 30))
        self.spawn_count_hint = int(cfg.get("route_bank", {}).get("spawn_count_hint", 120))
        self.base_seed = int(cfg.get("route_bank", {}).get("seed", cfg.get("seed", 0)))

    def sample(self, seed=None, split="train", spawn_count=None):
        size = self.train_size if split == "train" else self.eval_size
        rng = random.Random(self.base_seed if seed is None else seed)
        scenario_id = rng.randrange(max(size, 1))
        return self.by_id(scenario_id, split=split, spawn_count=spawn_count)

    def by_id(self, scenario_id, split="train", spawn_count=None):
        spawn_count = int(spawn_count or self.spawn_count_hint)
        if spawn_count < 6:
            raise ValueError("CARLA map must expose at least 6 vehicle spawn points")

        offset = 0 if split == "train" else self.train_size
        rng = random.Random(self.base_seed + offset + int(scenario_id) * 9973)
        scenario_cfg = self._scenario_traffic_cfg()
        sv_count = rng.randint(
            int(scenario_cfg["surrounding_vehicles_min"]),
            int(scenario_cfg["surrounding_vehicles_max"]),
        )

        ego_spawn = rng.randrange(spawn_count)
        route_end = (ego_spawn + rng.randint(35, min(90, max(36, spawn_count - 1)))) % spawn_count
        candidates = [idx for idx in range(spawn_count) if idx not in {ego_spawn, route_end}]
        rng.shuffle(candidates)
        sv_spawns = candidates[:sv_count]
        sv_speeds = [
            rng.uniform(float(scenario_cfg["sv_speed_min"]), float(scenario_cfg["sv_speed_max"]))
            for _ in range(sv_count)
        ]
        av_target_speed = rng.uniform(
            float(scenario_cfg["av_target_speed_min"]),
            float(scenario_cfg["av_target_speed_max"]),
        )
        return RouteScenario(
            scenario_id=int(scenario_id),
            town=self.town,
            ego_spawn_index=ego_spawn,
            ego_route_end_index=route_end,
            sv_spawn_indices=sv_spawns,
            sv_target_speeds=sv_speeds,
            av_target_speed=av_target_speed,
        )

    def _scenario_traffic_cfg(self):
        traffic = self.cfg["traffic"]
        scenario = self.cfg["experiment"].get("scenario")
        if scenario in traffic and isinstance(traffic[scenario], dict):
            return traffic[scenario]
        return traffic
