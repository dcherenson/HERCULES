"""Deterministic moving-target motion and target-centered slot geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np


def wrap_angle(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def route_basis(heading: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return route-forward and route-left unit vectors in NED XY."""

    forward = np.array([np.cos(float(heading)), np.sin(float(heading))], dtype=float)
    left = np.array([-forward[1], forward[0]], dtype=float)
    return forward, left


def target_center_before_goal(start: np.ndarray, goal: np.ndarray, fraction: float = 0.7) -> np.ndarray:
    """Place a target pattern on the requested fraction of the route."""

    start = np.asarray(start, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    value = float(np.clip(fraction, 0.0, 1.0))
    return start + value * (goal - start)


@dataclass
class FigureEightTargetController:
    """Follow a fixed route-frame Gerono figure-eight with a Husky command."""

    center: np.ndarray
    route_heading: float
    longitudinal_span: float = 10.0
    lateral_span: float = 8.0
    speed: float = 1.5
    sample_count: int = 64
    waypoint_radius: float = 1.0
    heading_gain: float = 2.0
    max_yaw_rate: float = 1.5
    index: int = 0

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(3)
        self.sample_count = max(8, int(self.sample_count))
        self._points = self._build_points()
        # Start on a lateral lobe, avoiding the crossing point at startup.
        self.index = int(round(self.sample_count * 0.125)) % self.sample_count

    def _build_points(self) -> np.ndarray:
        forward, left = route_basis(self.route_heading)
        theta = np.linspace(0.0, 2.0 * np.pi, self.sample_count, endpoint=False)
        offsets = (
            0.5 * float(self.longitudinal_span) * np.sin(theta)[:, None] * forward[None, :]
            + 0.5 * float(self.lateral_span) * np.sin(2.0 * theta)[:, None] * left[None, :]
        )
        points = np.repeat(self.center[None, :], self.sample_count, axis=0)
        points[:, :2] += offsets
        return points

    @property
    def points(self) -> np.ndarray:
        return self._points.copy()

    @property
    def phase(self) -> float:
        return float(2.0 * np.pi * self.index / self.sample_count)

    def reference(self, lookahead: int = 2) -> np.ndarray:
        return self._points[(self.index + int(lookahead)) % self.sample_count].copy()

    def update(self, position: np.ndarray, yaw: float, dt: float) -> np.ndarray:
        """Return AirSim model-level ``[speed, yaw_rate]`` for the target."""

        position = np.asarray(position, dtype=float).reshape(3)
        while np.linalg.norm(position[:2] - self._points[self.index, :2]) <= self.waypoint_radius:
            self.index = (self.index + 1) % self.sample_count
            if self.index == 0:
                break
        target = self.reference()
        delta = target[:2] - position[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            return np.zeros(2, dtype=float)
        desired_heading = float(np.arctan2(delta[1], delta[0]))
        error = wrap_angle(desired_heading - float(yaw))
        speed = min(float(self.speed), distance)
        speed *= max(0.25, np.cos(error))
        yaw_rate = np.clip(float(self.heading_gain) * error, -float(self.max_yaw_rate), float(self.max_yaw_rate))
        return np.array([speed, yaw_rate], dtype=float)

    def diagnostics(self) -> Dict[str, object]:
        return {
            "type": "gerono_figure_eight",
            "center": self.center.tolist(),
            "route_heading": float(self.route_heading),
            "longitudinal_span": float(self.longitudinal_span),
            "lateral_span": float(self.lateral_span),
            "speed": float(self.speed),
            "sample_count": int(self.sample_count),
            "phase": float(self.phase),
            "index": int(self.index),
        }


UAV_TARGET_SLOTS: Mapping[str, np.ndarray] = {
    "Drone1": np.array([-2.0, -2.0, 0.0]),
    "Drone2": np.array([2.0, -2.0, 0.0]),
    "SimpleFlight": np.array([0.0, 0.0, 0.0]),
    "Drone4": np.array([-2.0, 2.0, 0.0]),
    "Drone5": np.array([2.0, 2.0, 0.0]),
}


def target_centered_slot(
    agent_id: str,
    vehicle_type: str,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    fallback_heading: float,
    uav_altitude: float,
    target_z: Optional[float] = None,
    ugv_circumradius: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return a target-centered position, feed-forward velocity, and heading."""

    target_position = np.asarray(target_position, dtype=float).reshape(3)
    target_velocity = np.asarray(target_velocity, dtype=float).reshape(3)
    planar_speed = float(np.linalg.norm(target_velocity[:2]))
    heading = float(np.arctan2(target_velocity[1], target_velocity[0])) if planar_speed > 0.2 else float(fallback_heading)
    forward, _ = route_basis(heading)

    if vehicle_type == "drone":
        offset = np.asarray(UAV_TARGET_SLOTS.get(agent_id, np.zeros(3)), dtype=float).copy()
        offset[:2] = np.array([
            np.cos(heading) * offset[0] - np.sin(heading) * offset[1],
            np.sin(heading) * offset[0] + np.cos(heading) * offset[1],
        ])
        position = target_position + offset
        position[2] = float(uav_altitude)
        velocity = target_velocity.copy()
        velocity[2] = 0.0
        return position, velocity, heading

    # The three vertices sum to zero, so the target lies at the triangle
    # centroid. Husky1 is the forward vertex and the followers are +/-120 deg.
    phase_by_agent = {"Husky1": 0.0, "Husky2": 2.0 * np.pi / 3.0, "Husky3": -2.0 * np.pi / 3.0}
    phase = phase_by_agent.get(agent_id, 0.0)
    angle = heading + phase
    position = target_position.copy()
    position[:2] += float(ugv_circumradius) * np.array([np.cos(angle), np.sin(angle)])
    position[2] = float(target_z if target_z is not None else target_position[2])
    velocity = target_velocity.copy()
    velocity[2] = 0.0
    return position, velocity, heading

