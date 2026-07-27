import math

import numpy as np


class TimeVaryingRiskField:
    """Road, lane-center, and multimodal surrounding-vehicle risk field."""

    def __init__(self, cfg):
        self.cfg = cfg

    def road_risk(self, lane_offset, lane_width):
        margin = float(self.cfg.get("boundary_safety_margin", 0.4))
        boundary_gap = 0.5 * float(lane_width) - abs(float(lane_offset))
        denominator = max(boundary_gap, margin)
        boundary = float(self.cfg["road_weight"]) / (denominator * denominator)

        lane_width_single = float(self.cfg.get("lane_width", 3.5))
        nearest_center = round(float(lane_offset) / lane_width_single) * lane_width_single
        center_decay = float(self.cfg.get("center_decay", 0.8))
        center_weight = float(self.cfg.get("center_weight", 0.1))
        center = center_weight * (1.0 - math.exp(-center_decay * abs(float(lane_offset) - nearest_center)))
        return boundary + center

    def vehicle_risk(self, ego_xy, sv_modes):
        ego_xy = np.asarray(ego_xy, dtype=np.float32)
        risk = 0.0
        for modes in sv_modes:
            for mode in modes:
                mean = np.asarray(mode.mean[:2], dtype=np.float32)
                covariance = np.asarray(mode.covariance[:2, :2], dtype=np.float32)
                uncertainty = max(0.0, float(np.trace(covariance)))
                longitudinal_axis = float(self.cfg.get("vehicle_length", 4.8))
                lateral_axis = float(self.cfg.get("vehicle_width", 2.0))
                scale = math.sqrt(max(1e-6, -2.0 * math.log(max(1e-6, 1.0 - min(0.999, mode.probability)))))
                axis_x = longitudinal_axis + float(self.cfg.get("uncertainty_weight", 1.0)) * uncertainty
                axis_y = lateral_axis + 0.5 * float(self.cfg.get("uncertainty_weight", 1.0)) * uncertainty
                axis_x = max(axis_x * scale, 1e-3)
                axis_y = max(axis_y * scale, 1e-3)
                delta = ego_xy - mean
                ellipse_value = (delta[0] / axis_x) ** 2 + (delta[1] / axis_y) ** 2
                if ellipse_value <= 1.0:
                    mode_risk = float(self.cfg["vehicle_weight"])
                else:
                    distance_to_region = math.sqrt(ellipse_value) - 1.0
                    mode_risk = float(self.cfg["vehicle_weight"]) * math.exp(
                        -float(self.cfg.get("vehicle_decay", 1.0)) * distance_to_region
                    )
                risk += mode.probability * mode_risk
        return risk

    def total_risk(self, ego_xy, lane_offset, lane_width, sv_modes):
        road = self.road_risk(lane_offset, lane_width)
        vehicle = self.vehicle_risk(ego_xy, sv_modes)
        total = road + vehicle
        return {
            "road_risk": float(road),
            "vehicle_risk": float(vehicle),
            "total_risk": float(total),
            "risk_margin": float(1.0 / (1.0 + total)),
        }
