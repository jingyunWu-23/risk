import math
import random

import numpy as np

from rarl.belief.gm_bayes import GaussianMixtureBayesianBeliefUpdater
from rarl.envs.metrics import EpisodeMetrics
from rarl.envs.route_bank import RouteBank
from rarl.risk.risk_field import TimeVaryingRiskField


class CarlaRiskAwareEnv:
    """Gym-like CARLA environment for risk-aware RL."""

    def __init__(self, cfg, dry_run=False, split="train"):
        self.cfg = cfg
        self.dry_run = dry_run
        self.split = split
        self.dt = float(cfg["carla"]["fixed_delta_seconds"])
        self.horizon = int(cfg["experiment"]["episode_horizon"])
        self.max_svs = int(cfg["observation"]["max_surrounding_vehicles"])
        self.lane_width = float(cfg.get("road", {}).get("lane_width", 3.5))
        self.num_lanes = int(cfg.get("road", {}).get("num_lanes", 3))
        self.road_width = self.lane_width * self.num_lanes
        self.episode_step = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_speed = 0.0
        self.prev_progress = 0.0
        self.route_progress = 0.0
        self.prev_location = None
        self.target_speed = 13.0
        self.route_length = 200.0
        self.route_scenario = None
        self.metrics = EpisodeMetrics()
        self.route_bank = RouteBank(cfg)
        self.belief = GaussianMixtureBayesianBeliefUpdater(
            num_modes=cfg["belief"]["num_modes"],
            process_noise=cfg["belief"]["process_noise"],
            measurement_noise=cfg["belief"]["measurement_noise"],
            min_probability=cfg["belief"]["min_mode_probability"],
        )
        self.risk_field = TimeVaryingRiskField(cfg["risk_field"])
        self.client = None
        self.world = None
        self.carla = None
        self.map = None
        self.traffic_manager = None
        self.ego_vehicle = None
        self.surrounding_vehicles = []
        self.collision_sensor = None
        self.collision_events = []
        self._actors = []
        self._original_settings = None
        self._mock_ego = None
        self._mock_svs = {}
        if not dry_run:
            self._connect_carla()

    @property
    def observation_dim(self):
        return 8 + self.max_svs * 5 + 4

    @property
    def action_dim(self):
        return 2

    def _connect_carla(self):
        try:
            import carla
        except ImportError as exc:
            raise RuntimeError("CARLA Python API is not on PYTHONPATH. Use --dry-run or export the CARLA egg path.") from exc

        self.carla = carla
        self.client = carla.Client(self.cfg["carla"]["host"], int(self.cfg["carla"]["port"]))
        self.client.set_timeout(float(self.cfg["carla"]["timeout"]))
        self.world = self.client.load_world(self.cfg["carla"]["town"])
        self.map = self.world.get_map()
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        settings.no_rendering_mode = bool(self.cfg["carla"]["no_rendering_mode"])
        self.world.apply_settings(settings)
        self.traffic_manager = self.client.get_trafficmanager(int(self.cfg["carla"]["traffic_manager_port"]))
        self.traffic_manager.set_synchronous_mode(True)
        if "global_distance_to_leading_vehicle" in self.cfg.get("traffic", {}):
            self.traffic_manager.set_global_distance_to_leading_vehicle(
                float(self.cfg["traffic"]["global_distance_to_leading_vehicle"])
            )

    def reset(self, seed=None):
        self.episode_step = 0
        self.prev_action[:] = 0.0
        self.prev_progress = 0.0
        self.route_progress = 0.0
        self.prev_location = None
        self.prev_speed = 0.0
        self.collision_events = []
        self.metrics.reset()
        self.belief.reset()
        if self.dry_run:
            self._reset_mock(seed)
        else:
            self._cleanup_actors()
            self.route_scenario = self.route_bank.sample(
                seed=seed,
                split=self.split,
                spawn_count=len(self.map.get_spawn_points()),
            )
            self.target_speed = self.route_scenario.av_target_speed
            random.seed(seed)
            np.random.seed(None if seed is None else int(seed))
            self._spawn_ego_vehicle()
            self._spawn_surrounding_vehicles()
            self.route_length = self._estimate_route_length()
            self._tick_world()
            self.prev_speed = self._read_ego_state()[5]
            self.prev_location = self.ego_vehicle.get_location()
            self.prev_progress = 0.0
            self.belief.update(self._read_surrounding_vehicle_observations())
        return self._build_observation()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        self.episode_step += 1

        previous_progress = self._route_progress()
        if self.dry_run:
            self._step_mock(action)
        else:
            self._apply_ego_action(action)
            self._tick_world()
            self._update_route_progress()
            self.belief.predict(self.dt)
            self.belief.update(self._read_surrounding_vehicle_observations())

        ego = self._read_ego_state()
        risk = self._current_risk()
        collision = self._has_collision()
        offroad = self._is_offroad()
        timeout = self.episode_step >= self.horizon
        route_completion = self._route_completion()
        done = bool(collision or offroad or timeout or route_completion >= 1.0)
        reward = self._reward(action, risk, collision, offroad, previous_progress)
        progress_delta = self._route_progress() - previous_progress
        min_distance, min_ttc = self._proximity_metrics()
        self.metrics.update(reward, progress_delta, ego[5], risk["total_risk"], min_distance, min_ttc)
        self.prev_action = action
        self.prev_speed = ego[5]
        info = {
            "collision": bool(collision),
            "collision_rate": float(collision),
            "offroad": bool(offroad),
            "offroad_rate": float(offroad),
            "timeout": bool(timeout),
            "episode_step": self.episode_step,
            "episode_length": self.episode_step,
            "route_completion": route_completion,
            "sv_count": len(self._read_surrounding_vehicle_observations()),
            "min_distance": min_distance,
            "min_ttc": min_ttc,
            **risk,
            **self.metrics.as_info(),
        }
        return self._build_observation(), reward, done, info

    def _reset_mock(self, seed=None):
        rng = np.random.default_rng(seed)
        scenario_cfg = self.route_bank._scenario_traffic_cfg()
        sv_count = int(scenario_cfg["surrounding_vehicles_min"])
        sv_count = min(max(1, sv_count), self.max_svs)
        self.target_speed = float(rng.uniform(scenario_cfg["av_target_speed_min"], scenario_cfg["av_target_speed_max"]))
        self.route_length = 200.0
        self.route_progress = 0.0
        self._mock_ego = np.array([0.0, 0.0, 0.0, self.target_speed, 0.0, self.target_speed, 0.0, 0.0], dtype=np.float32)
        self._mock_svs = {}
        for i in range(sv_count):
            lane = (i % self.num_lanes) - (self.num_lanes // 2)
            speed = float(rng.uniform(scenario_cfg["sv_speed_min"], scenario_cfg["sv_speed_max"]))
            rel_x = 18.0 + i * 10.0 + float(rng.uniform(-3.0, 3.0))
            rel_y = lane * self.lane_width
            self._mock_svs[i] = np.array([rel_x, rel_y, speed - self.target_speed, 0.0], dtype=np.float32)
        self.prev_speed = self.target_speed
        self.belief.update(self._mock_svs)

    def _step_mock(self, action):
        accel = float(action[0]) * 3.0
        steer = float(action[1])
        self._mock_ego[6] = accel
        self._mock_ego[7] = 0.0
        self._mock_ego[5] = max(0.0, self._mock_ego[5] + accel * self.dt)
        self._mock_ego[2] += steer * 0.2 * self.dt
        self._mock_ego[3] = self._mock_ego[5] * np.cos(self._mock_ego[2])
        self._mock_ego[4] = self._mock_ego[5] * np.sin(self._mock_ego[2])
        self._mock_ego[0] += self._mock_ego[3] * self.dt
        self._mock_ego[1] += self._mock_ego[4] * self.dt
        self.route_progress = max(0.0, float(self._mock_ego[0]))
        for state in self._mock_svs.values():
            state[0] += state[2] * self.dt
            state[1] += state[3] * self.dt
        self.belief.predict(self.dt)
        self.belief.update(self._mock_svs)

    def _spawn_ego_vehicle(self):
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        blueprint = self._choose_vehicle_blueprint(blueprints, preferred="vehicle.tesla.model3")
        spawn_points = self.map.get_spawn_points()
        transform = spawn_points[self.route_scenario.ego_spawn_index % len(spawn_points)]
        self.ego_vehicle = self._try_spawn_actor(blueprint, transform)
        if self.ego_vehicle is None:
            raise RuntimeError(f"Failed to spawn ego vehicle at spawn index {self.route_scenario.ego_spawn_index}")
        self._actors.append(self.ego_vehicle)
        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_bp, self.carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(lambda event: self.collision_events.append(event))
        self._actors.append(self.collision_sensor)

    def _spawn_surrounding_vehicles(self):
        self.surrounding_vehicles = []
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        spawn_points = self.map.get_spawn_points()
        for idx, speed in zip(self.route_scenario.sv_spawn_indices, self.route_scenario.sv_target_speeds):
            blueprint = self._choose_vehicle_blueprint(blueprints)
            actor = self._try_spawn_actor(blueprint, spawn_points[idx % len(spawn_points)])
            if actor is None:
                continue
            actor.set_autopilot(True, int(self.cfg["carla"]["traffic_manager_port"]))
            self.traffic_manager.vehicle_percentage_speed_difference(actor, self._speed_difference_percent(speed))
            self.traffic_manager.auto_lane_change(actor, True)
            self.traffic_manager.random_left_lanechange_percentage(
                actor, 100.0 * float(self.cfg["traffic"].get("lane_change_probability", 0.2))
            )
            self.traffic_manager.random_right_lanechange_percentage(
                actor, 100.0 * float(self.cfg["traffic"].get("lane_change_probability", 0.2))
            )
            self.surrounding_vehicles.append(actor)
            self._actors.append(actor)

    def _choose_vehicle_blueprint(self, blueprints, preferred=None):
        if preferred is not None:
            matches = [bp for bp in blueprints if bp.id == preferred]
            if matches:
                blueprint = matches[0]
            else:
                blueprint = random.choice(list(blueprints))
        else:
            blueprint = random.choice(list(blueprints))
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hero" if preferred else "autopilot")
        if blueprint.has_attribute("color"):
            colors = blueprint.get_attribute("color").recommended_values
            if colors:
                blueprint.set_attribute("color", random.choice(colors))
        return blueprint

    def _try_spawn_actor(self, blueprint, transform):
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            return actor
        for offset in (1.5, -1.5, 3.0, -3.0):
            shifted = self.carla.Transform(transform.location, transform.rotation)
            shifted.location.x += offset * math.cos(math.radians(transform.rotation.yaw))
            shifted.location.y += offset * math.sin(math.radians(transform.rotation.yaw))
            actor = self.world.try_spawn_actor(blueprint, shifted)
            if actor is not None:
                return actor
        return None

    def _speed_difference_percent(self, target_speed):
        reference_kmh = float(self.cfg["traffic"].get("traffic_manager_reference_speed_kmh", 50.0))
        target_kmh = max(1.0, float(target_speed) * 3.6)
        return 100.0 * (reference_kmh - target_kmh) / reference_kmh

    def _apply_ego_action(self, action):
        accel = float(action[0])
        throttle = max(0.0, accel)
        brake = max(0.0, -accel)
        control = self.carla.VehicleControl(
            throttle=min(1.0, throttle),
            steer=float(action[1]),
            brake=min(1.0, brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
        self.ego_vehicle.apply_control(control)

    def _tick_world(self):
        self.world.tick()

    def _cleanup_actors(self):
        for actor in reversed(self._actors):
            if actor is None:
                continue
            try:
                if hasattr(actor, "stop"):
                    actor.stop()
                actor.destroy()
            except RuntimeError:
                pass
        self._actors = []
        self.ego_vehicle = None
        self.surrounding_vehicles = []
        self.collision_sensor = None
        self.collision_events = []

    def _read_ego_state(self):
        if self.dry_run:
            return self._mock_ego.copy()
        transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        acceleration = self.ego_vehicle.get_acceleration()
        local_velocity = self._to_ego_frame(transform, velocity)
        local_accel = self._to_ego_frame(transform, acceleration)
        speed = math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)
        origin = self._route_origin()
        rel_location = self._sub_vector(transform.location, origin)
        local_position = self._to_ego_frame(self._route_origin_transform(), rel_location)
        yaw = math.radians(transform.rotation.yaw - self._route_origin_transform().rotation.yaw)
        return np.array(
            [
                local_position[0],
                local_position[1],
                self._wrap_angle(yaw),
                local_velocity[0],
                local_velocity[1],
                speed,
                local_accel[0],
                local_accel[1],
            ],
            dtype=np.float32,
        )

    def _read_surrounding_vehicle_observations(self):
        if self.dry_run:
            return {vehicle_id: state.copy() for vehicle_id, state in self._mock_svs.items()}
        ego_transform = self.ego_vehicle.get_transform()
        ego_velocity = self.ego_vehicle.get_velocity()
        observations = {}
        for actor in self.surrounding_vehicles[: self.max_svs]:
            if not actor.is_alive:
                continue
            rel_location = self._sub_vector(actor.get_location(), ego_transform.location)
            rel_velocity = self._sub_vector(actor.get_velocity(), ego_velocity)
            rel_xy = self._to_ego_frame(ego_transform, rel_location)
            rel_v = self._to_ego_frame(ego_transform, rel_velocity)
            observations[actor.id] = np.array([rel_xy[0], rel_xy[1], rel_v[0], rel_v[1]], dtype=np.float32)
        return observations

    def _to_ego_frame(self, ego_transform, vector_or_location):
        yaw = math.radians(ego_transform.rotation.yaw)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x = vector_or_location.x
        y = vector_or_location.y
        return np.array([cos_yaw * x + sin_yaw * y, -sin_yaw * x + cos_yaw * y], dtype=np.float32)

    def _current_risk(self):
        ego = self._read_ego_state()
        observations = self._read_surrounding_vehicle_observations()
        sv_modes = [self.belief.get_vehicle_modes(vehicle_id) for vehicle_id in sorted(observations.keys())]
        return self.risk_field.total_risk(
            np.zeros(2, dtype=np.float32),
            lane_offset=ego[1],
            lane_width=self.road_width,
            sv_modes=sv_modes,
        )

    def _build_observation(self):
        risk = self._current_risk()
        risk_vec = np.array(
            [risk["road_risk"], risk["vehicle_risk"], risk["total_risk"], risk["risk_margin"]],
            dtype=np.float32,
        )
        return np.concatenate(
            [self._read_ego_state(), self.belief.as_features(self.max_svs).reshape(-1), risk_vec]
        ).astype(np.float32)

    def _reward(self, action, risk, collision, offroad, previous_progress):
        cfg = self.cfg["reward"]
        ego = self._read_ego_state()
        progress_delta = self._route_progress() - previous_progress
        speed_term = -abs(ego[5] - self.target_speed) / max(self.target_speed, 1e-3)
        smoothness = np.linalg.norm(action - self.prev_action)
        reward = cfg["progress_weight"] * progress_delta
        reward += cfg["speed_weight"] * speed_term
        reward -= cfg["risk_weight"] * risk["total_risk"]
        reward -= cfg["collision_penalty"] * float(collision)
        reward -= cfg["offroad_penalty"] * float(offroad)
        reward -= cfg["action_smoothness_weight"] * smoothness
        return float(reward)

    def _route_progress(self):
        if self.dry_run:
            return max(0.0, float(self._mock_ego[0]))
        return float(self.route_progress)

    def _update_route_progress(self):
        current_location = self.ego_vehicle.get_location()
        if self.prev_location is None:
            self.prev_location = current_location
            return
        delta = self._sub_vector(current_location, self.prev_location)
        transform = self.ego_vehicle.get_transform()
        yaw = math.radians(transform.rotation.yaw)
        forward_progress = delta.x * math.cos(yaw) + delta.y * math.sin(yaw)
        self.route_progress += max(0.0, float(forward_progress))
        self.prev_location = current_location

    def _route_completion(self):
        progress = max(0.0, self._route_progress())
        return float(np.clip(progress / max(self.route_length, 1.0), 0.0, 1.0))

    def _route_origin(self):
        return self.map.get_spawn_points()[self.route_scenario.ego_spawn_index].location

    def _route_origin_transform(self):
        return self.map.get_spawn_points()[self.route_scenario.ego_spawn_index]

    def _estimate_route_length(self):
        spawn_points = self.map.get_spawn_points()
        start = spawn_points[self.route_scenario.ego_spawn_index].location
        end = spawn_points[self.route_scenario.ego_route_end_index].location
        delta = self._sub_vector(end, start)
        euclidean = math.sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z)
        return max(50.0, euclidean)

    def _has_collision(self):
        if self.dry_run:
            min_distance, _ = self._proximity_metrics()
            return min_distance < 2.0
        return len(self.collision_events) > 0

    def _is_offroad(self):
        ego = self._read_ego_state()
        return abs(float(ego[1])) > 0.5 * self.road_width

    def _proximity_metrics(self):
        observations = self._read_surrounding_vehicle_observations()
        min_distance = math.inf
        min_ttc = math.inf
        for obs in observations.values():
            rel_pos = np.asarray(obs[:2], dtype=np.float32)
            rel_vel = np.asarray(obs[2:4], dtype=np.float32)
            distance = float(np.linalg.norm(rel_pos))
            min_distance = min(min_distance, distance)
            closing_speed = -float(np.dot(rel_pos, rel_vel)) / max(distance, 1e-3)
            if closing_speed > 1e-3:
                min_ttc = min(min_ttc, distance / closing_speed)
        return (
            -1.0 if math.isinf(min_distance) else float(min_distance),
            -1.0 if math.isinf(min_ttc) else float(min_ttc),
        )

    def _wrap_angle(self, angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _sub_vector(self, lhs, rhs):
        if self.carla is None:
            return lhs - rhs
        return self.carla.Vector3D(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z)

    def close(self):
        if self.world is not None:
            self._cleanup_actors()
            if self.traffic_manager is not None:
                self.traffic_manager.set_synchronous_mode(False)
            if self._original_settings is not None:
                self.world.apply_settings(self._original_settings)
            else:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
