import math


class EpisodeMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.episode_reward = 0.0
        self.distance = 0.0
        self.speed_sum = 0.0
        self.speed_count = 0
        self.min_distance = math.inf
        self.min_ttc = math.inf
        self.risk_sum = 0.0
        self.risk_count = 0

    def update(self, reward, progress_delta, speed, total_risk, min_distance, min_ttc):
        self.episode_reward += float(reward)
        self.distance += max(0.0, float(progress_delta))
        self.speed_sum += float(speed)
        self.speed_count += 1
        self.risk_sum += float(total_risk)
        self.risk_count += 1
        if min_distance is not None:
            self.min_distance = min(self.min_distance, float(min_distance))
        if min_ttc is not None and math.isfinite(float(min_ttc)):
            self.min_ttc = min(self.min_ttc, float(min_ttc))

    def as_info(self):
        return {
            "episode_reward": self.episode_reward,
            "avg_speed": self.speed_sum / max(1, self.speed_count),
            "min_distance": -1.0 if math.isinf(self.min_distance) else self.min_distance,
            "min_ttc": -1.0 if math.isinf(self.min_ttc) else self.min_ttc,
            "avg_total_risk": self.risk_sum / max(1, self.risk_count),
        }
