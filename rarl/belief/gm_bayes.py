from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianMode:
    mean: np.ndarray
    covariance: np.ndarray
    probability: float


class GaussianMixtureBayesianBeliefUpdater:
    """Tracks multimodal motion belief for surrounding vehicles."""

    def __init__(self, num_modes, process_noise, measurement_noise, min_probability):
        self.num_modes = num_modes
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.min_probability = min_probability
        self.mode_lateral_velocity = np.linspace(-1.5, 1.5, num_modes, dtype=np.float32)
        if num_modes == 7:
            self.mode_lateral_velocity = np.asarray([0.0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5], dtype=np.float32)
        self.transition = self._build_transition_matrix(num_modes)
        self._beliefs = {}

    def reset(self):
        self._beliefs.clear()

    def initialize_vehicle(self, vehicle_id, state):
        mean = np.asarray(state[:4], dtype=np.float32)
        covariance = np.eye(4, dtype=np.float32) * self.measurement_noise
        probability = 1.0 / self.num_modes
        modes = []
        for mode_idx in range(self.num_modes):
            mode_mean = mean.copy()
            mode_mean[3] += self.mode_lateral_velocity[mode_idx]
            modes.append(GaussianMode(mode_mean, covariance.copy(), probability))
        self._beliefs[vehicle_id] = modes

    def predict(self, dt):
        transition = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        q = np.eye(4, dtype=np.float32) * self.process_noise
        for modes in self._beliefs.values():
            previous_means = [mode.mean.copy() for mode in modes]
            previous_covariances = [mode.covariance.copy() for mode in modes]
            previous_probabilities = np.asarray([mode.probability for mode in modes], dtype=np.float32)
            mixed_probabilities = previous_probabilities @ self.transition
            for mode_idx, mode in enumerate(modes):
                transfer_weights = previous_probabilities * self.transition[:, mode_idx]
                transfer_weights /= max(float(transfer_weights.sum()), self.min_probability)
                mixed_mean = sum(weight * mean for weight, mean in zip(transfer_weights, previous_means))
                mixed_covariance = np.zeros((4, 4), dtype=np.float32)
                for weight, mean, covariance in zip(transfer_weights, previous_means, previous_covariances):
                    delta = (mean - mixed_mean).reshape(4, 1)
                    mixed_covariance += weight * (covariance + delta @ delta.T)
                mode.mean = mixed_mean
                mode.covariance = mixed_covariance
                mode.probability = float(max(mixed_probabilities[mode_idx], self.min_probability))
                mode.mean = transition @ mode.mean
                mode.mean[3] += 0.1 * self.mode_lateral_velocity[mode_idx] * dt
                mode.covariance = transition @ mode.covariance @ transition.T + q
            total = sum(mode.probability for mode in modes)
            for mode in modes:
                mode.probability = float(mode.probability / max(total, self.min_probability))

    def update(self, observations):
        for vehicle_id, obs in observations.items():
            if vehicle_id not in self._beliefs:
                self.initialize_vehicle(vehicle_id, obs)
            measurement = np.asarray(obs[:4], dtype=np.float32)
            likelihoods = []
            for mode in self._beliefs[vehicle_id]:
                innovation = measurement - mode.mean
                r = np.eye(4, dtype=np.float32) * self.measurement_noise
                s = mode.covariance + r
                gain = mode.covariance @ np.linalg.inv(s)
                mode.mean = mode.mean + gain @ innovation
                mode.covariance = (np.eye(4, dtype=np.float32) - gain) @ mode.covariance
                likelihood = np.exp(-0.5 * innovation @ np.linalg.inv(s) @ innovation)
                lateral_velocity_error = abs(float(measurement[3] - mode.mean[3]))
                maneuver_score = np.exp(-0.5 * lateral_velocity_error)
                likelihood *= maneuver_score
                likelihoods.append(max(float(likelihood), self.min_probability))

            probs = np.asarray([m.probability for m in self._beliefs[vehicle_id]]) * np.asarray(likelihoods)
            probs = np.maximum(probs, self.min_probability)
            probs = probs / probs.sum()
            for mode, prob in zip(self._beliefs[vehicle_id], probs):
                mode.probability = float(prob)

    def get_vehicle_modes(self, vehicle_id):
        return self._beliefs.get(vehicle_id, [])

    def as_features(self, max_vehicles):
        features = []
        for vehicle_id in sorted(self._beliefs.keys())[:max_vehicles]:
            modes = self._beliefs[vehicle_id]
            weighted_mean = sum(mode.probability * mode.mean for mode in modes)
            uncertainty = sum(mode.probability * np.trace(mode.covariance) for mode in modes)
            features.append(np.concatenate([weighted_mean, [uncertainty]], dtype=np.float32))
        while len(features) < max_vehicles:
            features.append(np.zeros(5, dtype=np.float32))
        return np.asarray(features, dtype=np.float32)

    def _build_transition_matrix(self, num_modes):
        transition = np.full((num_modes, num_modes), 0.05 / max(1, num_modes - 1), dtype=np.float32)
        np.fill_diagonal(transition, 0.95)
        return transition / transition.sum(axis=1, keepdims=True)
