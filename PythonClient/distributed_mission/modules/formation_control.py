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
    # Optional course-level waypoint. This is deliberately a single static
    # waypoint; no online waypoint generation is performed by formation
    # control or perception.
    intermediate_waypoint: Optional[np.ndarray] = None
    waypoint_radius: float = 2.5
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
        if self.intermediate_waypoint is not None:
            self.intermediate_waypoint = np.asarray(self.intermediate_waypoint, dtype=float).reshape(3)


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
        # During the one fixed course-waypoint approach, UAVs use that point
        # as a virtual formation center. This starts the planned bypass before
        # the UGV leader reaches the obstacle row; Husky followers continue to
        # track the real leader and retain their own formation.
        virtual_center = leader.position
        virtual_velocity = leader.velocity
        waypoint = self.config.intermediate_waypoint
        if (
            agent_id != self.config.leader_id
            and waypoint is not None
            and np.linalg.norm(leader.position[:2] - waypoint[:2]) > self.config.waypoint_radius
        ):
            virtual_center = waypoint
            virtual_velocity = np.zeros(3)
        position = virtual_center + rotated_offset
        angular_velocity = np.array([0.0, 0.0, leader.yaw_rate])
        velocity = virtual_velocity + np.cross(angular_velocity, rotated_offset)
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
            target_position = self._leader_target(agent, mission_goal)
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
            target = self._leader_target(agent, mission_goal)[:2]
        else:
            target, _, _ = self.reference_for(agent.agent_id, states)
            target = target[:2]
        delta = target - agent.position[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= 2.0:
            return np.zeros(2, dtype=float)
        desired_heading = float(np.arctan2(delta[1], delta[0])) if distance > 1e-9 else agent.yaw
        heading_error = wrap_angle(desired_heading - agent.yaw)
        speed = min(self.config.max_speed, self.config.position_gain * distance)
        # Keep a small forward command while turning. AirSim's car cannot
        # rotate in place, so an exact zero here creates a deadlock after the
        # fixed course waypoint when the goal is behind the current heading.
        if agent.agent_id == self.config.leader_id and distance > 1.5:
            alignment = max(0.25, np.cos(heading_error))
        else:
            alignment = max(0.0, np.cos(heading_error))
        speed *= alignment
        yaw_rate = np.clip(self.config.ugv_heading_gain * heading_error, -self.config.ugv_max_yaw_rate, self.config.ugv_max_yaw_rate)
        return np.array([speed, yaw_rate], dtype=float)

    def _leader_target(self, agent: AgentState, mission_goal: np.ndarray) -> np.ndarray:
        target = np.asarray(mission_goal, dtype=float)
        waypoint = self.config.intermediate_waypoint
        if waypoint is not None and np.linalg.norm(agent.position[:2] - waypoint[:2]) > self.config.waypoint_radius:
            return waypoint.copy()
        return target.copy()

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
