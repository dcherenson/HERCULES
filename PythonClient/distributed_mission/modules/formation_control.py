"""Leader-follower nominal formation controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .cbf import AgentState


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class FormationConfig:
    leader_id: str = "Husky1"
    position_gain: float = 0.6
    velocity_gain: float = 2.0
    max_speed: float = 2.0
    uav_max_acceleration: float = 4.0
    ugv_max_acceleration: float = 2.0
    ugv_max_yaw_rate: float = 1.0
    ugv_heading_gain: float = 1.5
    uav_altitude: float = -5.0
    slots: Dict[str, np.ndarray] = field(default_factory=lambda: {
        # Husky1 is the front vertex of an equilateral triangle with side
        # length 4 m. The two followers sit behind it, symmetrically.
        "Husky1": np.array([0.0, 0.0, 0.0]),
        "Husky2": np.array([-2.0 * np.sqrt(3.0), -2.0, 0.0]),
        "Husky3": np.array([-2.0 * np.sqrt(3.0), 2.0, 0.0]),
        # Five UAVs form a 4 m x 4 m box: four corners plus its center.
        "Drone1": np.array([-2.0, -2.0, -4.0]),
        "Drone2": np.array([2.0, -2.0, -4.0]),
        "SimpleFlight": np.array([0.0, 0.0, -4.0]),
        "Drone4": np.array([-2.0, 2.0, -4.0]),
        "Drone5": np.array([2.0, 2.0, -4.0]),
    })

    def __post_init__(self) -> None:
        self.slots = {key: np.asarray(value, dtype=float).reshape(3) for key, value in self.slots.items()}


class FormationController:
    """Compute nominal model-level controls from a fixed leader formation."""

    def __init__(self, config: Optional[FormationConfig] = None):
        self.config = config or FormationConfig()

    def reference_for(self, agent_id: str, states: Mapping[str, AgentState]) -> Tuple[np.ndarray, np.ndarray, float]:
        if self.config.leader_id not in states:
            raise KeyError("leader state is required for formation references")
        leader = states[self.config.leader_id]
        offset = self.config.slots.get(agent_id, np.zeros(3))
        rotation = np.array([
            [np.cos(leader.yaw), -np.sin(leader.yaw), 0.0],
            [np.sin(leader.yaw), np.cos(leader.yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        rotated_offset = rotation @ offset
        position = leader.position + rotated_offset
        angular_velocity = np.array([0.0, 0.0, leader.yaw_rate])
        velocity = leader.velocity + np.cross(angular_velocity, rotated_offset)
        # Keep UAV altitude in the global AirSim NED frame. Husky body origins
        # sit above the ground, so deriving UAV Z from the leader causes a
        # vertical reference mismatch and visible bobbing.
        if offset[2] != 0.0:
            position[2] = self.config.uav_altitude
            velocity[2] = 0.0
        return position, velocity, leader.yaw

    def nominal_control(
        self,
        agent: AgentState,
        states: Mapping[str, AgentState],
        mission_goal: np.ndarray,
    ) -> np.ndarray:
        """Return acceleration for Wang/UAV or [speed, yaw_rate] for Mestres UGV.

        The caller selects the model by passing ``unicycle=True``. The
        controller itself remains independent of the CBF method.
        """
        if agent.agent_id == self.config.leader_id:
            target_position = np.asarray(mission_goal, dtype=float)
            target_velocity = np.zeros(3)
        else:
            target_position, target_velocity, _ = self.reference_for(agent.agent_id, states)

        position_error = target_position - agent.position
        desired_velocity = target_velocity + self.config.position_gain * position_error
        speed = float(np.linalg.norm(desired_velocity))
        if speed > self.config.max_speed:
            desired_velocity = desired_velocity * (self.config.max_speed / speed)

        return self.config.velocity_gain * (desired_velocity - agent.velocity)

    def nominal_unicycle_control(
        self,
        agent: AgentState,
        states: Mapping[str, AgentState],
        mission_goal: np.ndarray,
    ) -> np.ndarray:
        if agent.agent_id == self.config.leader_id:
            target = np.asarray(mission_goal, dtype=float)[:2]
        else:
            target, _, _ = self.reference_for(agent.agent_id, states)
            target = target[:2]
        delta = target - agent.position[:2]
        distance = float(np.linalg.norm(delta))
        desired_heading = float(np.arctan2(delta[1], delta[0])) if distance > 1e-9 else agent.yaw
        heading_error = wrap_angle(desired_heading - agent.yaw)
        speed = min(self.config.max_speed, self.config.position_gain * distance)
        speed *= max(0.0, np.cos(heading_error))
        yaw_rate = np.clip(self.config.ugv_heading_gain * heading_error, -self.config.ugv_max_yaw_rate, self.config.ugv_max_yaw_rate)
        return np.array([speed, yaw_rate], dtype=float)

    def metrics(self, states: Mapping[str, AgentState]) -> Dict[str, float]:
        if self.config.leader_id not in states:
            return {}
        errors = []
        for agent_id, state in states.items():
            if agent_id == self.config.leader_id or agent_id not in self.config.slots:
                continue
            reference, _, _ = self.reference_for(agent_id, states)
            errors.append(float(np.linalg.norm(state.position - reference)))
        if not errors:
            return {"formation_rms_error": 0.0, "formation_max_error": 0.0}
        return {
            "formation_rms_error": float(np.sqrt(np.mean(np.square(errors)))),
            "formation_max_error": float(max(errors)),
        }
