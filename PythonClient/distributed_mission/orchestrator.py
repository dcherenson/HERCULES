#!/usr/bin/env python3
"""Run the heterogeneous distributed CBF mission in AirSim."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, ".."))

from agent import Agent
from modules.cbf import AgentState, CBFConfig, ObstacleProxy
from modules.formation_control import FormationConfig, FormationController
from modules.obstacle_detection import (
    ObstacleDetector,
    PerceptionConfig,
    decode_depth_response,
    estimate_ground_z,
    truth_obstacle_proxies,
)
from modules.obstacle_course import course_from_tuples, load_course
from modules.perception_diagnostics import PerceptionTraceStore
from modules.target_motion import (
    FigureEightTargetController,
    target_center_before_goal,
)
from modules.target_observation import MissionTimeMapper, TargetObservationWorker, truth_target_measurement
from modules.target_tracking import (
    DistributedTargetTracking,
    TargetMeasurement,
    TARGET_CBF_SIGMA_MULTIPLIER,
)
from modules.video_recording import (
    AirSimFrameRecorder,
    FollowCameraController,
    analyze_camera_alignment,
    render_recordings,
)
from modules.mission_plots import load_mission_records
from simulation.airsim_runtime import AsyncJsonlWriter, AirSimFacade, AirSimLaunchConfig, AirSimLauncher


CONTROL_DT = 0.1
SIMULATION_STEPS = 100
COMMUNICATION_RANGE_METERS = 10.0
UAV_ALTITUDE_CEILING_METERS = 10.0
RURAL_TARGET_CAMERA_RIGHT_OFFSET_METERS = 5.0
# Move the RuralAustralia target route back toward the robot launch point,
# measured along the resolved start-to-goal route.  The whole fixed
# figure-eight is translated because it is anchored at this position.
RURAL_TARGET_TOWARD_ROBOTS_METERS = 5.0
# Fixed startup phase whose first figure-eight segment moves toward the
# right-hand side of the route-aligned chase-camera image.
TARGET_START_SAMPLE_INDEX = 5
# The CPHusky adapter receives the normalized unicycle yaw-rate command.
UGV_STEERING_SCALE = 1.0
# These are the three cameras that the AirSim multirotor pawn actually
# registers.  Keeping the fan limited to existing cameras avoids changing the
# vehicle model or adding a second perception sensor configuration.  The
# response pose for each camera is used, so the installed mount orientations
# remain authoritative.
UAV_OBSTACLE_CAMERAS = ("front_center", "front_left", "front_right")

# A short staggered through-gap course: the first central opening is near
# y=0 and the second is shifted to y=1, so the team must turn instead of
# driving straight through a symmetric wall. The conservative CBF corridors
# are slightly narrower than the nominal formations, requiring modest
# compression. Heights vary across both rows and a floating block tests UAV
# altitude changes. All centers/scales use AirSim NED.
BLOCK_COURSE = (
    # First row: central gap centered at y=0.
    (7.5, -10.0, -2.0, (2.0, 3.0, 4.0)),   # low ground block
    (7.5, -5.5, -4.0, (2.0, 3.0, 8.0)),    # tall ground block
    (7.5, 5.5, -3.0, (2.0, 3.0, 6.0)),     # medium ground block
    (7.5, 10.0, -1.0, (2.0, 3.0, 2.0)),    # low ground block
    # Second row: central gap shifted to y=1, forcing a turn.
    (12.5, -9.0, -2.0, (2.0, 3.0, 4.0)),   # low ground block
    (12.5, -4.5, -4.0, (2.0, 3.0, 8.0)),   # tall ground block
    (12.5, 6.5, -3.0, (2.0, 3.0, 6.0)),    # medium ground block
    (12.5, 10.5, -1.0, (2.0, 3.0, 2.0)),   # low ground block
    # A floating block occupies the staggered UAV passage at nominal altitude.
    (10.5, 0.0, -6.5, (1.0, 0.8, 2.0)),
)


def target_start_anchor_for_map(
    start: np.ndarray,
    goal: np.ndarray,
    map_name: str,
    ground_z: float,
    route_heading: float,
) -> np.ndarray:
    """Return the map-specific XY anchor for the spawned tracking target.

    RuralAustralia places the target one quarter along the initial route and
    shifts it 5 m to the camera-right side of the route. In FlyingCPP the
    target starts at the resolved mission goal instead. The target is a ground
    vehicle, so its physical Z coordinate is always the calibrated
    vehicle-ground height rather than the goal marker's Z.
    """

    fraction = 0.25 if str(map_name) == "rural_australia" else 1.0
    anchor = target_center_before_goal(start, goal, fraction)
    if str(map_name) == "rural_australia":
        # For an AirSim/NED camera looking along the route, local +Y (image
        # right) is [-sin(h), cos(h)].
        camera_right = np.array([
            -np.sin(float(route_heading)),
            np.cos(float(route_heading)),
            0.0,
        ])
        anchor += RURAL_TARGET_CAMERA_RIGHT_OFFSET_METERS * camera_right
        route_forward = np.array([
            np.cos(float(route_heading)),
            np.sin(float(route_heading)),
            0.0,
        ])
        anchor -= RURAL_TARGET_TOWARD_ROBOTS_METERS * route_forward
    anchor[2] = float(ground_z)
    return anchor


def build_adjacency_matrix(positions: Dict[str, np.ndarray], comm_range: float) -> Dict[str, List[str]]:
    ids = list(positions)
    adjacency = {agent_id: [] for agent_id in ids}
    for index, first in enumerate(ids):
        for second in ids[index + 1:]:
            if np.linalg.norm(positions[first] - positions[second]) <= comm_range:
                adjacency[first].append(second)
                adjacency[second].append(first)
    return adjacency


def ugv_speed_control_inputs(
    desired_speed: float,
    measured_velocity: np.ndarray,
    base_brake: float = 0.0,
) -> Tuple[float, float]:
    """Convert a model-level forward speed into bounded CarControls inputs.

    AirSim's car API accepts throttle, not velocity. Keeping this conversion
    at the simulator boundary prevents a valid unicycle speed command from
    becoming an open-loop acceleration request. The gain is intentionally
    small because the CPHusky dynamics retain momentum between 100 ms calls;
    no CBF or nominal-control equation is changed.
    """

    desired = max(0.0, float(desired_speed))
    velocity = np.asarray(measured_velocity, dtype=float).reshape(-1)
    measured = float(np.linalg.norm(velocity[:2])) if len(velocity) >= 2 else 0.0
    measured = max(0.0, measured) if np.isfinite(measured) else 0.0
    brake = float(np.clip(base_brake, 0.0, 1.0))
    if brake >= 1.0 or desired < 0.05:
        return 0.0, max(brake, 1.0 if desired < 0.05 else 0.0)
    speed_error = desired - measured
    throttle = float(np.clip(0.02 * desired + 0.10 * max(0.0, speed_error), 0.0, 0.08))
    if measured > desired + 0.10:
        brake = max(brake, float(np.clip((measured - desired - 0.10) / 0.50, 0.0, 1.0)))
        throttle = 0.0
    return throttle, brake


def _startup_position_clearance(
    candidate: np.ndarray,
    proxies: Sequence[ObstacleProxy],
    vehicle_type: str,
    vehicle_radius: float,
    obstacle_margin: float,
) -> float:
    """Return clearance from truth proxies in the vehicle's modeled space."""

    candidate = np.asarray(candidate, dtype=float).reshape(3)
    if not proxies:
        return float("inf")
    if vehicle_type == "drone":
        distances = [
            float(np.linalg.norm(candidate - proxy.center))
            - float(proxy.radius)
            - float(vehicle_radius)
            - float(obstacle_margin)
            for proxy in proxies
        ]
    else:
        distances = [
            float(np.linalg.norm(candidate[:2] - proxy.center[:2]))
            - float(proxy.radius)
            - float(vehicle_radius)
            - float(obstacle_margin)
            for proxy in proxies
        ]
    return min(distances)


def safe_target_centered_startup_positions(
    initial_positions: Mapping[str, np.ndarray],
    agent_names: Sequence[str],
    target_position: np.ndarray,
    truth_obstacles: Sequence[Dict[str, Any]],
    vehicle_type: str,
    vehicle_radius: float,
    obstacle_margin: float = 0.0,
    minimum_clearance: float = 0.15,
) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, np.ndarray, np.ndarray]]]:
    """Keep target-centered startup slots out of known static proxies.

    This is a startup-only placement check. It preserves each UGV's distance
    from the target when that ring is feasible, and searches a fixed angular
    grid for the smallest deterministic change that is clear of the truth
    geometry and already selected same-type agents. If a ring is blocked, the
    search expands it in 0.5 m increments. The running target-centered nominal
    controller is not changed; the CBF remains responsible for subsequent
    deviations as the target moves.

    UAV slots use 3-D spherical clearance against the vertical truth slices;
    UGV slots use the planar clearance used by their unicycle CBF.
    """

    positions = {
        name: np.asarray(value, dtype=float).reshape(3).copy()
        for name, value in initial_positions.items()
    }
    target = np.asarray(target_position, dtype=float).reshape(3)
    proxies = truth_obstacle_proxies(
        truth_obstacles,
        vehicle_type,
        timestamp=0.0,
        vehicle_z=float(target[2]) if vehicle_type != "drone" else None,
        vehicle_radius=float(vehicle_radius),
    )
    if not proxies:
        return positions, []

    changes: List[Tuple[str, np.ndarray, np.ndarray]] = []
    selected: List[np.ndarray] = []
    separation = 2.0 * float(vehicle_radius) + max(0.0, float(obstacle_margin))
    # Search in nearest-angle order. Including both signs avoids treating a
    # -5 degree correction as a 355 degree correction when choosing a slot.
    offset_degrees = [0.0]
    for degree in range(5, 360, 5):
        offset_degrees.extend((float(degree), -float(degree)))
    offsets = np.deg2rad(offset_degrees)
    for name in agent_names:
        desired = positions.get(name)
        if desired is None:
            continue
        relative = desired[:2] - target[:2]
        desired_ring_radius = float(np.linalg.norm(relative))
        desired_angle = float(np.arctan2(relative[1], relative[0])) if desired_ring_radius > 1e-9 else 0.0
        if desired_ring_radius > 0.1:
            candidate_radii = [desired_ring_radius + 0.5 * step for step in range(0, 17)]
        else:
            # The center UAV normally remains at the target-centered origin;
            # only move it if that exact point is occupied by a truth proxy.
            candidate_radii = [0.5 * step for step in range(0, 17)]
        candidates = []
        for candidate_radius in candidate_radii:
            for offset in offsets:
                angle = desired_angle + float(offset)
                candidate = desired.copy()
                candidate[:2] = target[:2] + candidate_radius * np.array([np.cos(angle), np.sin(angle)])
                obstacle_clearance = _startup_position_clearance(
                    candidate, proxies, vehicle_type, vehicle_radius, obstacle_margin
                )
                agent_clearance = min(
                    [
                        float(np.linalg.norm(candidate[:2] - other[:2])) - separation
                        for other in selected
                    ]
                    or [float("inf")]
                )
                candidates.append((candidate_radius, offset, obstacle_clearance, agent_clearance, candidate))
        valid = [
            item for item in candidates
            if item[2] >= float(minimum_clearance) and item[3] >= 0.0
        ]
        if valid:
            # Preserve the nominal radius/center whenever possible, then
            # minimize angular disturbance. Radius expansion is only used if
            # the desired ring has no clear candidate.
            choice = min(valid, key=lambda item: (item[0] - desired_ring_radius if desired_ring_radius > 0.1 else item[0], abs(float(item[1]))))
        else:
            choice = max(candidates, key=lambda item: (min(item[2], item[3]), -abs(float(item[1]))))
        candidate = choice[4]
        if not np.allclose(candidate[:2], desired[:2]):
            changes.append((name, desired.copy(), candidate.copy()))
        positions[name] = candidate
        selected.append(candidate)
    return positions, changes


def safe_target_ugv_startup_positions(
    initial_positions: Mapping[str, np.ndarray],
    ugv_names: Sequence[str],
    target_position: np.ndarray,
    target_ugv_circumradius: float,
    truth_obstacles: Sequence[Dict[str, Any]],
    ugv_radius: float,
    obstacle_margin: float = 0.0,
    minimum_clearance: float = 0.15,
) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, np.ndarray, np.ndarray]]]:
    """Backward-compatible UGV wrapper for target-centered startup checks."""

    del target_ugv_circumradius  # The requested ring is already in positions.
    return safe_target_centered_startup_positions(
        initial_positions,
        ugv_names,
        target_position,
        truth_obstacles,
        "ugv",
        ugv_radius,
        obstacle_margin=obstacle_margin,
        minimum_clearance=minimum_clearance,
    )


def safe_target_start_index(
    target_controller: FigureEightTargetController,
    truth_obstacles: Sequence[Dict[str, Any]],
    target_radius: float,
    minimum_clearance: float = 0.15,
) -> Tuple[int, float]:
    """Choose the nearest clear sample on the fixed target route.

    The target is not part of the CBF controller, so it needs a valid initial
    pose before the deterministic figure-eight command is applied. This only
    changes the initial phase of the existing fixed route; it does not add
    waypoints or perform online replanning.
    """

    points = target_controller.points
    if len(points) == 0:
        return int(target_controller.index), float("inf")
    preferred = int(target_controller.index) % len(points)
    clearances = []
    for index, point in enumerate(points):
        proxies = truth_obstacle_proxies(
            truth_obstacles,
            "ugv",
            timestamp=0.0,
            vehicle_z=float(point[2]),
            vehicle_radius=float(target_radius),
        )
        clearances.append(_startup_position_clearance(point, proxies, "ugv", target_radius, 0.0))
    for offset in range(len(points)):
        index = (preferred + offset) % len(points)
        if clearances[index] >= float(minimum_clearance):
            return index, float(clearances[index])
    best = max(range(len(points)), key=lambda index: clearances[index])
    return int(best), float(clearances[best])


def filter_agent_body_obstacle_proxies(
    proxies: Sequence[ObstacleProxy],
    states: Mapping[str, Any],
    vehicle_radii: Mapping[str, float],
    exclusion_margin: float = 0.5,
) -> Tuple[List[ObstacleProxy], int]:
    """Remove perception clusters whose centers are controlled-agent bodies.

    Pairwise same-type CBF constraints already handle controlled agents. A
    depth camera or LiDAR return from a vehicle must not be added a second
    time as a static obstacle, especially not as the ego vehicle itself. The
    filter only removes perception proxies close to a currently known agent;
    explicit truth and target proxies are retained for their separate CBF
    roles.
    """

    agent_positions: List[Tuple[np.ndarray, float]] = []
    for name, state in states.items():
        if isinstance(state, Mapping):
            position = state.get("position")
        else:
            position = getattr(state, "position", None)
        if position is None:
            continue
        candidate = np.asarray(position, dtype=float).reshape(-1)
        if len(candidate) < 3 or not np.all(np.isfinite(candidate[:3])):
            continue
        radius = float(vehicle_radii.get(name, 1.0))
        agent_positions.append((candidate[:3], max(0.0, radius) + max(0.0, float(exclusion_margin))))

    retained: List[ObstacleProxy] = []
    rejected = 0
    for proxy in proxies:
        source = str(proxy.source or "")
        if source.startswith("truth") or source == "target_tracking":
            retained.append(proxy)
            continue
        center = np.asarray(proxy.center, dtype=float).reshape(-1)
        if len(center) >= 3 and np.all(np.isfinite(center[:3])):
            if any(float(np.linalg.norm(center[:3] - position)) <= threshold for position, threshold in agent_positions):
                rejected += 1
                continue
        retained.append(proxy)
    return retained, rejected


def target_obstacle_proxy_for_agent(
    vehicle_type: str,
    target_name: str,
    target_estimate: Mapping[str, Any],
    target_z: float,
    target_radius: float,
    now: float,
) -> Optional[ObstacleProxy]:
    """Build the moving-target CBF proxy for ground vehicles only.

    Target estimates remain available to every agent for formation control.
    The target is a CBF obstacle only for UGVs; UAVs are airborne and should
    not spend barrier authority avoiding the target's planar footprint.
    """
    if vehicle_type != "ugv" or not target_estimate.get("active", False):
        return None

    target_covariance = np.asarray(
        target_estimate.get("covariance", np.eye(2)), dtype=float
    ).reshape((2, 2))
    target_covariance = 0.5 * (target_covariance + target_covariance.T)
    try:
        target_sigma = float(np.sqrt(max(0.0, np.max(np.linalg.eigvalsh(target_covariance)))))
    except np.linalg.LinAlgError:
        target_sigma = 0.0

    target_center = np.array([
        float(target_estimate["position"][0]),
        float(target_estimate["position"][1]),
        float(target_z),
    ])
    target_velocity = np.array([
        float(target_estimate.get("velocity", [0.0, 0.0])[0]),
        float(target_estimate.get("velocity", [0.0, 0.0])[1]),
        0.0,
    ])
    return ObstacleProxy(
        "target_" + target_name,
        target_center,
        target_radius + TARGET_CBF_SIGMA_MULTIPLIER * target_sigma,
        "target_tracking",
        float(target_estimate.get("timestamp", now)),
        0,
        True,
        target_velocity,
    )


def cbf_sensor_valid_for_target(
    vehicle_type: str,
    static_sensor_valid: bool,
    target_proxy: Optional[ObstacleProxy],
) -> bool:
    """Keep an independently valid moving-target barrier enabled.

    Static obstacle perception and target tracking have separate freshness
    sources. A stale LiDAR frame must not bypass the target CBF for a UGV
    whose DRWT estimate is active; the existing CBF module still receives the
    cached static proxies and applies its normal fail-safe behavior when
    neither source is valid. UAVs never use the target as a CBF obstacle.
    """

    if static_sensor_valid:
        return True
    return vehicle_type == "ugv" and target_proxy is not None


def _retime_target_measurement(
    measurement: TargetMeasurement,
    time_mapper: MissionTimeMapper,
) -> TargetMeasurement:
    """Copy an asynchronous capture into the controller's time domain."""

    capture_wall_timestamp = float(measurement.timestamp)
    metadata = dict(measurement.metadata)
    metadata.setdefault("capture_wall_timestamp", capture_wall_timestamp)
    return TargetMeasurement(
        target_id=measurement.target_id,
        position=measurement.position.copy(),
        covariance=measurement.covariance.copy(),
        timestamp=time_mapper.mission_timestamp(capture_wall_timestamp),
        valid=measurement.valid,
        source=measurement.source,
        capture_id=measurement.capture_id,
        sensor=measurement.sensor,
        visible=measurement.visible,
        metadata=metadata,
    )


def _quaternion_matrix(quaternion: Any) -> np.ndarray:
    w = float(getattr(quaternion, "w_val", 1.0))
    x = float(getattr(quaternion, "x_val", 0.0))
    y = float(getattr(quaternion, "y_val", 0.0))
    z = float(getattr(quaternion, "z_val", 0.0))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _vector3(value: Any) -> np.ndarray:
    return np.array([value.x_val, value.y_val, value.z_val], dtype=float)


def _quaternion_values(quaternion: Any) -> List[float]:
    return [
        float(getattr(quaternion, "w_val", 1.0)),
        float(getattr(quaternion, "x_val", 0.0)),
        float(getattr(quaternion, "y_val", 0.0)),
        float(getattr(quaternion, "z_val", 0.0)),
    ]


def _rotation_quaternion(rotation: np.ndarray) -> List[float]:
    """Convert a rotation matrix to an AirSim-compatible [w, x, y, z]."""

    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1e-12))
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1e-12))
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1e-12))
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    values = np.asarray([w, x, y, z], dtype=float)
    values /= max(np.linalg.norm(values), 1e-12)
    return values.tolist()


def compose_sensor_world_pose(
    vehicle_position: np.ndarray,
    vehicle_rotation: np.ndarray,
    sensor_position: np.ndarray,
    sensor_rotation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose a sensor-relative pose with the vehicle world pose."""

    vehicle_position = np.asarray(vehicle_position, dtype=float)
    vehicle_rotation = np.asarray(vehicle_rotation, dtype=float)
    sensor_position = np.asarray(sensor_position, dtype=float)
    sensor_rotation = np.asarray(sensor_rotation, dtype=float)
    return vehicle_position + vehicle_rotation @ sensor_position, vehicle_rotation @ sensor_rotation


def camera_response_world_pose(
    response_position: np.ndarray,
    response_rotation: np.ndarray,
    actor_position: Optional[np.ndarray],
    kinematics_position: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Convert an AirSim image-response pose into world NED.

    In Hero mode an image response pose is expressed in the vehicle's
    configured starting-point frame.  That frame can be translated relative
    to the shared Unreal world when vehicles are placed from settings.json.
    The response orientation is already expressed in the inertial axes; only
    the frame translation is needed here.  Keeping the fallback explicit is
    useful for older AirSim builds that omit the response pose.
    """

    position = np.asarray(response_position, dtype=float).reshape(3)
    rotation = np.asarray(response_rotation, dtype=float).reshape((3, 3))
    if actor_position is not None and kinematics_position is not None:
        actor = np.asarray(actor_position, dtype=float).reshape(3)
        kinematics = np.asarray(kinematics_position, dtype=float).reshape(3)
        if np.all(np.isfinite(actor)) and np.all(np.isfinite(kinematics)):
            return position + actor - kinematics, rotation, "world_ned_from_vehicle_start_frame"
    return position, rotation, "response_pose_frame_unknown"


def _yaw_quaternion(yaw: float) -> List[float]:
    return [float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))]


def rotate_xy_right(point: np.ndarray) -> np.ndarray:
    """Rotate an NED mission point 90 degrees left in the world view.

    AirSim uses NED coordinates, so a visual left turn from the original
    +X heading is a -90 degree yaw: ``(x, y) -> (y, -x)``. Z is unchanged.
    """

    rotated = np.asarray(point, dtype=float).reshape(3).copy()
    rotated[:2] = np.asarray([rotated[1], -rotated[0]], dtype=float)
    return rotated


def rotate_xy_heading(point: np.ndarray, heading: float) -> np.ndarray:
    """Rotate a formation/course offset into a world heading."""

    rotated = np.asarray(point, dtype=float).reshape(3).copy()
    cosine = np.cos(float(heading))
    sine = np.sin(float(heading))
    x, y = rotated[:2]
    rotated[:2] = [cosine * x - sine * y, sine * x + cosine * y]
    return rotated


def startup_formation_offset(point: np.ndarray, map_name: str, mission_objective: str, heading: float) -> np.ndarray:
    """Return the startup slot offset for the selected mission objective.

    The course geometry is already rotated into the selected map frame before
    this helper is called.  The vehicle slots therefore use the resolved
    start-to-goal heading for every objective.  This is important for the
    fixed-goal backward-compatible path: the positions and the formation
    controller's body-frame references must share the same heading or the
    followers will immediately turn around to repair their slots.
    """

    del map_name, mission_objective
    return rotate_xy_heading(point, heading)


def heading_to_goal(start: np.ndarray, goal: np.ndarray, fallback: float = 0.0) -> float:
    """Return the planar heading from ``start`` to ``goal`` in NED radians."""

    delta = np.asarray(goal, dtype=float).reshape(3)[:2] - np.asarray(start, dtype=float).reshape(3)[:2]
    if not np.all(np.isfinite(delta)) or np.linalg.norm(delta) <= 1e-9:
        return float(fallback)
    return float(np.arctan2(delta[1], delta[0]))


def _scene_object_position(facade: AirSimFacade, object_name: str) -> Optional[np.ndarray]:
    """Read one named Unreal actor pose as a world-NED position."""

    try:
        pose = facade.multirotor.simGetObjectPose(object_name, True)
        position = getattr(pose, "position", None)
        if position is None:
            return None
        candidate = _vector3(position)
        return candidate if np.all(np.isfinite(candidate)) else None
    except Exception:
        return None


def resolve_goal_actor(
    facade: AirSimFacade,
    fallback_goal: np.ndarray,
    actor_name: Optional[str] = None,
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Resolve an Unreal actor to use as the mission goal.

    A user-created actor normally receives an automatically generated name
    such as ``Actor_0``.  With ``actor_name`` omitted, only those generic
    actor names are considered and the candidate must be substantially
    opposite the existing RuralAustralia fallback route.  This avoids
    accidentally selecting map infrastructure.  An explicit name bypasses
    the directional gate and works on either map.
    """

    if actor_name:
        position = _scene_object_position(facade, actor_name)
        return (actor_name, position) if position is not None else (None, None)

    fallback = np.asarray(fallback_goal, dtype=float).reshape(3)
    fallback_xy = fallback[:2]
    fallback_norm = float(np.linalg.norm(fallback_xy))
    if fallback_norm <= 1e-9:
        return None, None
    fallback_direction = fallback_xy / fallback_norm
    try:
        candidates = facade.multirotor.simListSceneObjects(r"^Actor(?:_\d+)?$") or []
    except Exception:
        return None, None

    scored = []
    for candidate_name in candidates:
        name = str(candidate_name)
        position = _scene_object_position(facade, name)
        if position is None:
            continue
        distance = float(np.linalg.norm(position[:2]))
        if distance <= max(10.0, 0.5 * fallback_norm):
            continue
        opposite = float(np.dot(position[:2] / distance, -fallback_direction))
        if opposite < 0.5:
            continue
        # Prefer a clear opposite-side actor, then the one furthest from the
        # launch point. This remains deterministic if more than one generic
        # actor is present in a map.
        scored.append((opposite, distance, name, position))
    if not scored:
        return None, None
    _, _, name, position = max(scored, key=lambda item: (item[0], item[1], item[2]))
    return name, position


def camera_pose_for_goal(
    start: np.ndarray,
    goal: np.ndarray,
    horizontal_distance: float,
    camera_z: float,
    top_down: bool = False,
) -> Tuple[np.ndarray, float]:
    """Place the external camera behind the route and point it at the goal.

    The returned yaw is in degrees because CameraDirector uses Unreal degree
    fields, while all vehicle headings in the controller remain radians.
    """

    start = np.asarray(start, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    heading = heading_to_goal(start, goal)
    distance = max(float(horizontal_distance), 1.0)
    direction = np.asarray([np.cos(heading), np.sin(heading)])
    position = start.copy()
    position[:2] -= distance * direction
    position[2] = float(camera_z)
    # With a top-down camera, the yaw is rotated by 180 degrees so travel is
    # away from the camera on screen. In the normal view, yaw points along the
    # route toward the detected goal.
    yaw = heading + (np.pi if top_down else 0.0)
    yaw_degrees = float((np.degrees(yaw) + 180.0) % 360.0 - 180.0)
    return position, yaw_degrees


def startup_pose_position(
    name: str,
    initial_positions: Dict[str, np.ndarray],
    newly_spawned: set,
    vehicle_origins: Dict[str, np.ndarray],
) -> np.ndarray:
    """Convert a desired world pose to the correct AirSim vehicle-frame pose."""

    position = np.asarray(initial_positions[name], dtype=float)
    if name in newly_spawned:
        return position.copy()
    return position - np.asarray(vehicle_origins.get(name, np.zeros(3)), dtype=float)


def startup_heading_pose_position(
    name: str,
    initial_positions: Dict[str, np.ndarray],
    newly_spawned: set,
    vehicle_origins: Dict[str, np.ndarray],
) -> np.ndarray:
    """Return a pose translation that changes startup heading only.

    ``simAddVehicle`` has already placed a runtime-spawned pawn at its
    requested world position.  AirSim interprets a later
    ``simSetVehiclePose`` for that pawn in its spawn frame, so sending the
    world translation again applies the formation offset twice.  A zero
    translation preserves the spawn location while still allowing the
    startup stabilization pass to restore the route-aligned yaw.  Configured
    vehicles retain the existing frame-origin conversion.
    """

    if name in newly_spawned:
        return np.zeros(3, dtype=float)
    return startup_pose_position(name, initial_positions, newly_spawned, vehicle_origins)


def rotate_course_left(course: Dict[str, Any]) -> Dict[str, Any]:
    """Return a course with its XY geometry rotated 90 degrees left."""

    rotated = dict(course)
    rotated["goal"] = rotate_xy_right(course["goal"]).tolist()
    rotated["waypoints"] = [rotate_xy_right(point).tolist() for point in course.get("waypoints", [])]
    rotated_obstacles = []
    for obstacle in course.get("obstacles", []):
        rotated_obstacle = dict(obstacle)
        rotated_obstacle["center"] = rotate_xy_right(obstacle["center"]).tolist()
        if str(obstacle.get("shape", "box")).lower() == "box":
            dimensions = np.asarray(obstacle["dimensions"], dtype=float)
            rotated_obstacle["dimensions"] = [float(dimensions[1]), float(dimensions[0]), float(dimensions[2])]
        rotated_obstacles.append(rotated_obstacle)
    rotated["obstacles"] = rotated_obstacles
    return rotated


def camera_director_for_map(
    map_name: str, camera_x: float, camera_y: float, camera_height: float
) -> Tuple[Tuple[float, float, float], float]:
    """Return an optional top-down camera pose aligned behind the mission."""

    camera_xy = np.asarray([camera_x, camera_y], dtype=float)
    if map_name == "rural_australia":
        # The route turns from +X to -Y. Put the camera on the rear (+Y)
        # side while retaining the user-selected camera offset magnitude.
        camera_xy = np.asarray([-camera_y, camera_x], dtype=float)
        # CameraDirector rotation fields are Unreal degrees; vehicle yaw is
        # represented in radians elsewhere in the mission.
        # With a top-down pitch, yaw controls the screen's north/up direction
        # through the gimbal singularity. +90 degrees makes the -Y travel
        # direction appear away from the rear-side camera.
        camera_yaw = 90.0
    else:
        camera_yaw = 0.0
    return (float(camera_xy[0]), float(camera_xy[1]), -abs(float(camera_height))), float(camera_yaw)


def effective_obstacle_margin(map_name: str, requested: Optional[float]) -> float:
    """Select the existing CBF margin, with no extra map-specific default."""

    if requested is not None:
        return float(requested)
    return 0.0


def map_ground_z_offset(map_name: str) -> float:
    """Return the known NED ground-level correction for each map."""

    # RuralAustralia Example 1 is approximately 2 m lower than the
    # FlyingCPP mission reference. NED down is positive, so ground-referenced
    # points move from z=-1 to z=+1. UAV flight altitude remains independent.
    return 2.0 if map_name == "rural_australia" else 0.0


def map_vehicle_ground_z(map_name: str) -> float:
    """Return the settled AirSim body-center height for a CPHusky.

    These are map-frame body poses, not terrain heights.  The CPHusky actor
    is initially placed above the landscape and allowed to settle under its
    wheel physics.  RuralAustralia has uneven terrain and a positive NED
    spawn height can place the skid body below the local collision surface,
    where Chaos lets it fall through the terrain.  Starting above the surface
    avoids that invalid-body path; the startup settle check holds the vehicle
    before the mission clock begins.
    """

    return -1.0 if map_name == "rural_australia" else 0.7


def map_uav_altitude_floor(map_name: str, requested: Optional[float] = None) -> float:
    """Return the UAV NED floor one metre above the calibrated map ground."""

    if requested is not None:
        return float(requested)
    # One metre above a surface is one metre more negative in NED.
    return map_ground_z_offset(map_name) - 1.0


def map_uav_altitude_ceiling(map_name: str, requested: Optional[float] = None) -> float:
    """Return the minimum permitted UAV NED Z for an altitude ceiling.

    AirSim uses NED coordinates, so a ceiling ``h`` metres above the
    calibrated ground reference is ``ground_z - h``.  The returned value is
    the lowest permitted NED Z: UAV positions must remain greater than or
    equal to it to avoid climbing above the ceiling.
    """

    altitude = UAV_ALTITUDE_CEILING_METERS if requested is None else float(requested)
    if altitude <= 0.0:
        raise ValueError("UAV altitude ceiling must be positive")
    return map_ground_z_offset(map_name) - altitude


def clamp_uav_velocity_to_altitude_ceiling(
    velocity: np.ndarray,
    position_z: float,
    ceiling_z: float,
    dt: float,
    velocity_limit: float,
) -> np.ndarray:
    """Prevent one commanded UAV step from crossing the NED altitude ceiling.

    The CBF still produces the model-level safe acceleration.  This final
    simulator-boundary guard handles an actuator command that would otherwise
    continue climbing after repeated cycles.  Positive NED Z is downward, so
    the guard raises the commanded vertical velocity only when needed to
    remain at or below the configured physical altitude.
    """

    result = np.asarray(velocity, dtype=float).reshape(-1).copy()
    if len(result) < 3 or not np.isfinite(position_z) or not np.isfinite(ceiling_z):
        return result
    if dt <= 0.0 or velocity_limit <= 0.0:
        raise ValueError("dt and velocity_limit must be positive")
    minimum_step_velocity = (float(ceiling_z) - float(position_z)) / float(dt)
    result[2] = max(float(result[2]), minimum_step_velocity)
    result[2] = float(np.clip(result[2], -float(velocity_limit), float(velocity_limit)))
    return result


def shift_course_z(course: Dict[str, Any], offset: float) -> Dict[str, Any]:
    """Shift ground-referenced course geometry vertically in NED."""

    if abs(float(offset)) <= 1e-12:
        return course
    shifted = dict(course)
    shifted["goal"] = np.asarray(course["goal"], dtype=float).copy()
    shifted["goal"][2] += float(offset)
    shifted["goal"] = shifted["goal"].tolist()
    shifted["waypoints"] = []
    for point in course.get("waypoints", []):
        value = np.asarray(point, dtype=float).copy()
        value[2] += float(offset)
        shifted["waypoints"].append(value.tolist())
    shifted_obstacles = []
    for obstacle in course.get("obstacles", []):
        adjusted = dict(obstacle)
        center = np.asarray(obstacle["center"], dtype=float).copy()
        center[2] += float(offset)
        adjusted["center"] = center.tolist()
        shifted_obstacles.append(adjusted)
    shifted["obstacles"] = shifted_obstacles
    return shifted


def _capture_obstacles(
    facade: AirSimFacade,
    detector: ObstacleDetector,
    name: str,
    vehicle_type: str,
    state: Dict[str, Any],
    now: float,
    camera_fovs: Dict[str, float],
) -> Tuple[List[ObstacleProxy], bool, Dict[str, Any], Dict[str, Any]]:
    try:
        if vehicle_type == "drone":
            # A single front camera leaves a moving formation blind to
            # obstacles that enter from the side.  The multirotor pawn already
            # exposes this small three-camera fan; merge the world-frame point
            # clouds before the unchanged detector/clustering stage.  If an
            # older AirSim build rejects one of the optional side cameras,
            # retain the established front_center-only behavior.
            camera_names = list(UAV_OBSTACLE_CAMERAS)
            requests = [
                facade.airsim.ImageRequest(camera_name, facade.airsim.ImageType.DepthPerspective, True, False)
                for camera_name in camera_names
            ]
            try:
                responses = facade.multirotor.simGetImages(requests, vehicle_name=name)
            except Exception:
                camera_names = ["front_center"]
                responses = facade.multirotor.simGetImages([
                    facade.airsim.ImageRequest("front_center", facade.airsim.ImageType.DepthPerspective, True, False)
                ], vehicle_name=name)
            if not responses:
                return [], False, {}, {}
            capture_time = time.time()
            sensor_point_sets = []
            world_point_sets = []
            camera_entries = []
            for camera_name, response in zip(camera_names, responses):
                depth = decode_depth_response(response)
                if depth.size == 0 or depth.ndim != 2 or depth.shape[1] == 0:
                    continue
                fov_key = "{}:{}".format(name, camera_name)
                if fov_key not in camera_fovs:
                    camera = facade.multirotor.simGetCameraInfo(camera_name, vehicle_name=name)
                    camera_fovs[fov_key] = float(camera.fov) * np.pi / 180.0
                horizontal_fov = camera_fovs[fov_key]
                response_position = getattr(response, "camera_position", None)
                response_orientation = getattr(response, "camera_orientation", None)
                reported_camera_position = _vector3(response_position) if response_position is not None else state["position"]
                reported_camera_orientation = _quaternion_matrix(response_orientation) if response_orientation is not None else np.eye(3)
                camera_position, camera_orientation, camera_position_frame = camera_response_world_pose(
                    reported_camera_position,
                    reported_camera_orientation,
                    state.get("actor_position"),
                    state.get("kinematics_position"),
                )
                sensor_points = detector.depth_to_sensor(depth, horizontal_fov, stride=detector.config.depth_stride)
                points = sensor_points @ camera_orientation.T + camera_position
                sensor_point_sets.append(sensor_points)
                world_point_sets.append(points)
                vertical_fov = 2.0 * np.arctan(
                    np.tan(horizontal_fov / 2.0) * depth.shape[0] / depth.shape[1]
                )
                camera_entries.append({
                    "name": camera_name,
                    "position": camera_position.tolist(),
                    "orientation_quaternion": _quaternion_values(response_orientation) if response_orientation is not None else _rotation_quaternion(camera_orientation),
                    "horizontal_fov_deg": float(np.degrees(horizontal_fov)),
                    "vertical_fov_deg": float(np.degrees(vertical_fov)),
                    "image_width": int(depth.shape[1]),
                    "image_height": int(depth.shape[0]),
                    "position_frame": camera_position_frame,
                    "reported_position": reported_camera_position.tolist(),
                })
            if not world_point_sets:
                return [], False, {}, {}
            sensor_points = np.vstack(sensor_point_sets)
            points = np.vstack(world_point_sets)
            primary = camera_entries[0]
            sensor_view = {
                "sensor_type": "uav_camera",
                "camera_names": [entry["name"] for entry in camera_entries],
                "cameras": camera_entries,
                "position": primary["position"],
                "orientation_quaternion": primary["orientation_quaternion"],
                "horizontal_fov_deg": primary["horizontal_fov_deg"],
                "vertical_fov_deg": primary["vertical_fov_deg"],
                "range_m": float(detector.config.max_range),
                "capture_timestamp": float(capture_time),
                "position_frame": primary["position_frame"],
                "point_frame": "per_camera_local_ned",
                "vehicle_position": np.asarray(state["position"], dtype=float).tolist(),
                "reported_position": primary["reported_position"],
                "reported_position_frame": "vehicle_start_frame_ned",
                "vehicle_frame_origin": (
                    (np.asarray(state["actor_position"], dtype=float) - np.asarray(state["kinematics_position"], dtype=float)).tolist()
                    if state.get("actor_position") is not None and state.get("kinematics_position") is not None else None
                ),
            }
            # FlyingCPP's map ground is around NED z=2 rather than z=0. A
            # fixed z=0 rejection both leaves the background floor in the
            # cloud and can remove the top face of a course block. Estimate
            # the visible horizontal ground return in the world cloud.
            estimated_ground_z = estimate_ground_z(
                points, state["position"], search_below=2.0, search_above=12.0,
                min_range=detector.config.min_range, max_range=detector.config.max_range,
                min_separation_above_ego=2.0,
            )
            obstacles, diagnostics = detector.detect_with_diagnostics(
                points, state["position"], capture_time, "depth_" + name, False, ground_z=estimated_ground_z
            )
            if estimated_ground_z is not None:
                diagnostics.stage_counts["estimated_ground_z_mm"] = int(round(estimated_ground_z * 1000.0))
            return obstacles, True, sensor_view, {
                "sensor_points": sensor_points,
                "world_points": points,
                "diagnostics": diagnostics,
            }

        # Hero exposes Husky sensor APIs on the car service, matching
        # start_formation.py. The facade still owns this single hidden handle.
        lidar = facade.car.getLidarData(lidar_name="Lidar1", vehicle_name=name)
        raw_values = np.asarray(lidar.point_cloud, dtype=float)
        if raw_values.size < 3:
            raw = np.empty((0, 3), dtype=float)
        else:
            usable_size = raw_values.size - raw_values.size % 3
            raw = raw_values[:usable_size].reshape((-1, 3))
        raw_sensor_points = raw.copy()
        raw_count = len(raw)
        lidar_pose = getattr(lidar, "pose", None)
        vehicle_position = np.asarray(state["position"], dtype=float)
        vehicle_orientation = _quaternion_matrix(getattr(state.get("kinematics"), "orientation", None))
        if lidar_pose is not None and getattr(lidar_pose, "position", None) is not None:
            relative_position = _vector3(lidar_pose.position)
            relative_orientation = _quaternion_matrix(lidar_pose.orientation)
            sensor_position, sensor_rotation = compose_sensor_world_pose(
                vehicle_position, vehicle_orientation, relative_position, relative_orientation
            )
        else:
            sensor_position = vehicle_position
            sensor_rotation = vehicle_orientation
        sensor_orientation = _rotation_quaternion(sensor_rotation)
        capture_time = time.time()
        sensor_view = {
            "sensor_type": "ugv_lidar",
            "position": sensor_position.tolist(),
            "orientation_quaternion": sensor_orientation,
            "horizontal_fov_deg": 360.0,
            "vertical_fov_deg": 20.0,
            "vertical_fov_lower_deg": -10.0,
            "vertical_fov_upper_deg": 10.0,
            "range_m": float(detector.config.max_range),
            "capture_timestamp": float(capture_time),
            "position_frame": "world_ned",
            "point_frame": "lidar_local_ned",
            "vehicle_position": vehicle_position.tolist(),
            "vehicle_orientation_quaternion": _quaternion_values(getattr(state.get("kinematics"), "orientation", None)),
        }
        finite_nonzero = np.all(np.isfinite(raw), axis=1) & (np.linalg.norm(raw, axis=1) > 1e-6)
        zero_returns = int(np.sum(~finite_nonzero))
        raw = raw[finite_nonzero]
        if len(raw) > detector.config.max_points:
            raw = raw[np.linspace(0, len(raw) - 1, detector.config.max_points, dtype=int)]
        points = raw @ sensor_rotation.T + sensor_position
        # The map ground is not necessarily NED z=0. Estimate its local
        # height from the dense LiDAR return so the planar detector does not
        # turn the ground mesh into large obstacle proxies.
        estimated_ground_z = estimate_ground_z(
            points, vehicle_position, min_range=detector.config.min_range,
            max_range=detector.config.max_range,
        )
        obstacles, diagnostics = detector.detect_with_diagnostics(
            points, vehicle_position, capture_time, "lidar_" + name, True, ground_z=estimated_ground_z
        )
        if estimated_ground_z is not None:
            diagnostics.stage_counts["estimated_ground_z_mm"] = int(round(estimated_ground_z * 1000.0))
        diagnostics.stage_counts["raw_sensor_points"] = raw_count
        diagnostics.stage_counts["finite_nonzero_points"] = len(raw)
        diagnostics.stage_counts["zero_returns"] = zero_returns
        diagnostics.stage_counts["zero_returns_entered_detector"] = 0
        return obstacles, True, sensor_view, {
            "sensor_points": raw_sensor_points,
            "world_points": points,
            "diagnostics": diagnostics,
        }
    except Exception:
        return [], False, {}, {}


def _pose(airsim: Any, position: np.ndarray, yaw: float = 0.0) -> Any:
    return airsim.Pose(airsim.Vector3r(*position.tolist()), airsim.to_quaternion(0, 0, yaw))


def _camera_pose(
    airsim: Any, position: np.ndarray, yaw_degrees: float, pitch_degrees: float, roll_degrees: float = 0.0
) -> Any:
    """Build a CameraDirector pose from NED position and Unreal angles."""

    return airsim.Pose(
        airsim.Vector3r(*np.asarray(position, dtype=float).tolist()),
        airsim.to_quaternion(
            np.radians(float(roll_degrees)), np.radians(float(pitch_degrees)), np.radians(float(yaw_degrees))
        ),
    )


def _camera_pose_quaternion(airsim: Any, position: np.ndarray, quaternion: Sequence[float]) -> Any:
    """Build an AirSim pose from a world-NED position and [w,x,y,z]."""

    values = np.asarray(quaternion, dtype=float).reshape(4)
    return airsim.Pose(
        airsim.Vector3r(*np.asarray(position, dtype=float).tolist()),
        airsim.Quaternionr(float(values[1]), float(values[2]), float(values[3]), float(values[0])),
    )


def _plot_route_markers(facade: AirSimFacade, waypoints: List[np.ndarray]) -> None:
    """Draw fixed mission route markers in Unreal when AirSim supports it."""

    if not waypoints:
        return
    try:
        points = [facade.airsim.Vector3r(*np.asarray(point, dtype=float).tolist()) for point in waypoints]
        facade.multirotor.simPlotPoints(
            points, color_rgba=[1.0, 0.2, 0.0, 1.0], size=20.0,
            duration=-1.0, is_persistent=True,
        )
        if len(points) > 1:
            facade.multirotor.simPlotLineStrip(
                points, color_rgba=[1.0, 0.6, 0.0, 1.0], thickness=8.0,
                duration=-1.0, is_persistent=True,
            )
    except Exception as error:
        # Route visualization is optional and must never prevent a mission.
        print("Warning: Unreal route markers unavailable: {}".format(error), flush=True)


def _spawn_block_course(
    facade: AirSimFacade, airsim: Any, obstacles: List[Dict[str, Any]]
) -> Tuple[List[str], List[Dict[str, Any]]]:
    names = []
    truth = []
    try:
        listed_assets = facade.multirotor.simListAssets()
        available_assets = {str(asset) for asset in listed_assets} if listed_assets else None
    except Exception:
        # Older AirSim builds may not expose asset enumeration. The plugin
        # still performs its own null-safe lookup and the spawn exception is
        # handled below.
        available_assets = None
    # Cleanup is limited to this prefix so user-created scene objects survive.
    for index, obstacle in enumerate(obstacles):
        name = "distributed_cbf_block_{}".format(index)
        try:
            # AirSim may reuse an already-running Unreal world between
            # missions. Remove only the prior objects owned by this
            # orchestrator before recording fresh truth geometry.
            facade.delete_object(name)
        except Exception:
            pass
        center = np.asarray(obstacle["center"], dtype=float)
        requested_shape = str(obstacle.get("shape", "box")).lower()
        shape = requested_shape
        if shape == "sphere":
            radius = float(obstacle["radius"])
            asset_name = obstacle.get("asset_name", "SM_Sphere")
            scale = airsim.Vector3r(2.0 * radius, 2.0 * radius, 2.0 * radius)
        else:
            dimensions = np.asarray(obstacle["dimensions"], dtype=float)
            asset_name = obstacle.get("asset_name", "1M_Cube_Chamfer")
            scale = airsim.Vector3r(*dimensions.tolist())
        if available_assets and asset_name not in available_assets:
            fallback_asset = "1M_Cube_Chamfer"
            if shape == "sphere" and fallback_asset in available_assets:
                # Keep the requested sphere's diameter as the fallback box
                # dimensions so truth geometry still matches what spawned.
                print("Warning: asset {!r} unavailable; spawning sphere {!r} as a box fallback".format(
                    asset_name, name
                ), flush=True)
                shape = "box"
                dimensions = np.full(3, 2.0 * radius, dtype=float)
                asset_name = fallback_asset
                scale = airsim.Vector3r(*dimensions.tolist())
            else:
                print("Warning: asset {!r} unavailable; skipping {!r}".format(asset_name, name), flush=True)
                continue
        pose = _pose(airsim, center)
        try:
            if facade.spawn_object(name, asset_name, pose, scale):
                names.append(name)
                # Record the pose Unreal actually assigned. This catches
                # asset/world-origin surprises in the perception report and
                # keeps the truth overlay tied to the spawned actor.
                actual_center = center.copy()
                try:
                    actual_pose = facade.multirotor.simGetObjectPose(name, True)
                    actual_position = getattr(actual_pose, "position", None)
                    if actual_position is not None:
                        candidate = _vector3(actual_position)
                        if np.all(np.isfinite(candidate)):
                            actual_center = candidate
                except Exception:
                    pass
                truth_entry = {
                    "id": name,
                    "source_id": obstacle.get("id", name),
                    "shape": shape,
                    "center": actual_center.tolist(),
                }
                if shape == "sphere":
                    truth_entry["radius"] = radius
                else:
                    truth_entry["dimensions"] = dimensions.tolist()
                if requested_shape != shape:
                    truth_entry["requested_shape"] = requested_shape
                truth.append(truth_entry)
        except Exception as error:
            print("Warning: failed to spawn {} {!r}: {}".format(shape, name, error), flush=True)
    return names, truth


def _is_ground_object(object_name: str) -> bool:
    """Identify common Unreal ground actors for UGV contact filtering."""

    normalized = str(object_name or "").strip().lower()
    return any(token in normalized for token in ("ground", "landscape", "terrain", "floor"))


def run_artifact_stem(cbf_method: str, map_name: str, run_timestamp: int) -> str:
    """Return the shared stem used by all artifacts from one mission run."""

    return "{}_{}_{}".format(str(cbf_method).strip(), str(map_name).strip(), int(run_timestamp))


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-mode", choices=("visible", "headless", "existing"), default="visible")
    parser.add_argument("--map", dest="map_name", choices=("flyingcpp", "rural_australia"), default="flyingcpp")
    parser.add_argument("--cbf-method", choices=("mestres", "wang"), default="mestres")
    parser.add_argument("--steps", type=int, default=SIMULATION_STEPS)
    parser.add_argument("--dt", type=float, default=CONTROL_DT)
    parser.add_argument("--timing-mode", choices=("realtime", "stepped"), default="realtime")
    parser.add_argument("--uav-altitude", type=float, default=-5.0, help="UAV hover altitude in AirSim NED coordinates")
    parser.add_argument(
        "--initial-heading-offset-deg", type=float, default=0.0,
        help="offset the route-aligned startup heading; useful for map-specific orientation experiments",
    )
    parser.add_argument(
        "--uav-altitude-floor", type=float, default=None,
        help="maximum UAV NED Z; defaults to 1 m above the calibrated map ground",
    )
    parser.add_argument(
        "--uav-altitude-ceiling", type=float, default=UAV_ALTITUDE_CEILING_METERS,
        help="maximum UAV altitude above calibrated ground in metres (default: 10)",
    )
    parser.add_argument("--camera-height", type=float, default=30.0, help="top-down override camera height above the NED origin")
    parser.add_argument("--camera-x", type=float, default=6.0)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument(
        "--top-down-camera",
        action="store_true",
        help="opt in to the launch-time external CameraDirector top-down view",
    )
    parser.add_argument(
        "--no-top-down-camera",
        action="store_true",
        help="backward-compatible explicit disable for the top-down camera override",
    )
    parser.add_argument("--animation-fps", type=float, default=None, help="post-run MP4 frame rate; defaults to 1/dt")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--record-video", action="store_true", help="record chase and selected UAV/UGV camera videos")
    parser.add_argument("--record-uav", default="Drone1", help="UAV name used for chase and FPV recording")
    parser.add_argument("--record-ugv", default="Husky1", help="UGV name used for FPV recording")
    parser.add_argument("--video-resolution", nargs=2, type=int, default=[1280, 720], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument(
        "--video-fps", type=float, default=30.0,
        help="source capture rate for the chase and selected FPV streams",
    )
    parser.add_argument("--gif-height", type=int, default=540)
    parser.add_argument("--gif-fps", type=float, default=10.0)
    parser.add_argument("--playback-speed", type=float, default=2.0, help="post-run media playback speed relative to mission time")
    parser.add_argument("--keep-recording-frames", action="store_true")
    parser.add_argument("--sensor-rate", type=float, default=2.5, help="per-agent obstacle perception rate")
    parser.add_argument("--sensor-stale-after", type=float, default=None, help="maximum cached sensor age; defaults to one sensor period plus 50 ms")
    parser.add_argument("--top-n-obstacles", type=int, default=5)
    parser.add_argument("--uncertainty-radius", type=float, default=0.0)
    parser.add_argument("--uav-radius", type=float, default=1.0)
    parser.add_argument("--ugv-radius", type=float, default=1.25)
    parser.add_argument(
        "--obstacle-margin", type=float, default=None,
        help="additional CBF obstacle clearance margin; defaults to 0 m",
    )
    parser.add_argument("--uav-velocity-limit", type=float, default=3.0)
    parser.add_argument("--ugv-speed-limit", type=float, default=3.0)
    parser.add_argument("--uav-acceleration-limit", type=float, default=6.0)
    parser.add_argument("--ugv-acceleration-limit", type=float, default=3.0)
    parser.add_argument("--nominal-speed", type=float, default=1.0, help="existing formation nominal speed limit")
    parser.add_argument(
        "--leader-nominal-speed", type=float, default=1.0,
        help="nominal leader speed limit; followers retain --nominal-speed for catch-up",
    )
    parser.add_argument("--nominal-position-gain", type=float, default=0.5)
    parser.add_argument("--nominal-velocity-gain", type=float, default=3.0)
    parser.add_argument("--ugv-heading-gain", type=float, default=1.0)
    parser.add_argument("--ugv-max-yaw-rate", type=float, default=1.0)
    parser.add_argument(
        "--ugv-lookahead-distance", type=float, default=0.1,
        help="existing UGV CBF control-point lookahead; tune without changing the CBF equations",
    )
    parser.add_argument("--k1", type=float, default=2.0)
    parser.add_argument("--k2", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--communication-range", type=float, default=COMMUNICATION_RANGE_METERS)
    parser.add_argument("--multirotor-port", type=int, default=41451)
    parser.add_argument("--car-port", type=int, default=41452)
    parser.add_argument("--uproject-path", default=None)
    parser.add_argument("--settings-path", default=None, help="AirSim settings.json to copy for launch-time camera overrides")
    parser.add_argument(
        "--goal-actor", default=None,
        help="exact Unreal actor name to use as the goal; RuralAustralia auto-detects a generic opposite-side Actor_* by default",
    )
    parser.add_argument("--resx", type=int, default=800)
    parser.add_argument("--resy", type=int, default=600)
    parser.add_argument("--no-spawn-obstacles", action="store_true")
    parser.add_argument("--target-name", default="Target1")
    parser.add_argument("--target-observation-source", choices=("camera", "truth"), default="camera")
    parser.add_argument(
        "--target-camera-preconfigured", action="store_true",
        help="allow camera target tracking with --launch-mode existing when target_bottom/front_center are already configured",
    )
    parser.add_argument("--mission-objective", choices=("track-target", "fixed-goal"), default="track-target")
    parser.add_argument("--tracking-rate", type=float, default=4.0)
    parser.add_argument("--tracking-window", type=float, default=5.0)
    parser.add_argument("--tracking-admm-rho", type=float, default=1.0)
    parser.add_argument("--tracking-admm-max-iterations", type=int, default=20)
    parser.add_argument("--tracking-admm-tolerance", type=float, default=1e-3)
    parser.add_argument("--tracking-process-noise", type=float, default=0.20)
    parser.add_argument("--tracking-measurement-std", type=float, default=0.25)
    parser.add_argument("--target-sensing-range", type=float, default=100.0)
    parser.add_argument(
        "--target-speed", type=float, default=0.5,
        help="target figure-eight speed; lower values make the physical target easier to track",
    )
    parser.add_argument("--target-pattern-length", type=float, default=10.0)
    parser.add_argument("--target-pattern-width", type=float, default=8.0)
    # Five metres keeps the target-centered triangle close to the target while
    # leaving the forward Husky clearance from the pre-existing FlyingCPP map
    # mesh. The formation geometry is unchanged; this remains tunable from
    # the command line for other maps and course layouts.
    parser.add_argument("--target-ugv-circumradius", type=float, default=5.0)
    parser.add_argument(
        "--obstacle-course",
        default=None,
        help="JSON course from tools/obstacle_course_editor.py; otherwise use the built-in FlyingCPP course",
    )
    parser.add_argument(
        "--use-truth-obstacles",
        action="store_true",
        help="use successfully spawned obstacle boxes as fixed CBF test proxies instead of sensor detections",
    )
    parser.add_argument("--debug-dir", default=os.path.join(current_dir, "debug_runs"))
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--unreal-editor-path", default=AirSimLaunchConfig().unreal_editor_path)
    return parser.parse_args(argv)


def main(argv: List[str] = None) -> int:
    args = parse_args(argv)
    track_target = args.mission_objective == "track-target"
    if args.record_video and args.launch_mode == "existing":
        print("Error: --record-video requires --launch-mode visible or headless so recording cameras can be configured before launch.", file=sys.stderr)
        return 2
    if (
        track_target
        and
        args.target_observation_source == "camera"
        and args.launch_mode == "existing"
        and not args.target_camera_preconfigured
    ):
        print(
            "Error: camera target tracking requires --launch-mode visible/headless, "
            "or --target-camera-preconfigured for an existing launch.",
            file=sys.stderr,
        )
        return 2
    names = ["Drone1", "Drone2", "SimpleFlight", "Drone4", "Drone5", "Husky1", "Husky2", "Husky3"]
    uav_names = names[:5]
    ugv_names = names[5:]
    types = {name: "drone" for name in uav_names}
    types.update({name: "ugv" for name in ugv_names})
    target_name = str(args.target_name)
    if not target_name or target_name in names:
        print("Error: --target-name must be nonempty and distinct from controlled agents", file=sys.stderr)
        return 2
    if track_target:
        print("Mission objective: track-target; {} will be spawned and tracked.".format(target_name), flush=True)
    else:
        print(
            "Mission objective: fixed-goal; target tracking is disabled and {} will not be spawned.".format(target_name),
            flush=True,
        )
    if args.record_video:
        if args.record_uav not in names or types[args.record_uav] != "drone":
            print("Error: --record-uav must name a configured UAV", file=sys.stderr)
            return 2
        if args.record_ugv not in names or types[args.record_ugv] != "ugv":
            print("Error: --record-ugv must name a configured UGV", file=sys.stderr)
            return 2
        if args.video_resolution[0] <= 0 or args.video_resolution[1] <= 0 or args.video_fps <= 0 or args.gif_height <= 0 or args.gif_fps <= 0 or args.playback_speed <= 0:
            print("Error: video dimensions and rates must be positive", file=sys.stderr)
            return 2
    if (
        args.nominal_speed <= 0.0 or args.nominal_position_gain <= 0.0
        or args.nominal_velocity_gain <= 0.0 or args.ugv_heading_gain < 0.0
        or args.ugv_max_yaw_rate <= 0.0 or args.ugv_lookahead_distance <= 0.0
        or args.leader_nominal_speed <= 0.0
    ):
        print("Error: nominal gains, yaw rate, speed, and formation scale must be positive", file=sys.stderr)
        return 2
    if (
        args.tracking_rate <= 0.0 or args.tracking_window <= 0.0
        or args.tracking_admm_rho <= 0.0 or args.tracking_admm_max_iterations <= 0
        or args.tracking_admm_tolerance <= 0.0 or args.tracking_process_noise < 0.0
        or args.tracking_measurement_std <= 0.0 or args.target_sensing_range <= 0.0
        or args.target_speed <= 0.0 or args.target_pattern_length <= 0.0
        or args.target_pattern_width <= 0.0 or args.target_ugv_circumradius <= 0.0
    ):
        print("Error: target tracking and target motion parameters must be positive", file=sys.stderr)
        return 2
    if args.uav_altitude_ceiling <= 0.0:
        print("Error: --uav-altitude-ceiling must be positive", file=sys.stderr)
        return 2
    top_down_camera = args.top_down_camera and not args.no_top_down_camera
    try:
        course = load_course(args.obstacle_course) if args.obstacle_course else course_from_tuples(BLOCK_COURSE)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print("Error: unable to load obstacle course: {}".format(error), file=sys.stderr)
        return 2
    initial_yaw = -np.pi / 2.0 if args.map_name == "rural_australia" else 0.0
    ground_z_offset = map_ground_z_offset(args.map_name)
    uav_altitude_floor = map_uav_altitude_floor(args.map_name, args.uav_altitude_floor)
    uav_altitude_ceiling = map_uav_altitude_ceiling(args.map_name, args.uav_altitude_ceiling)
    print(
        "UAV altitude floor: z={:.3f} NED ({})".format(
            uav_altitude_floor,
            "1 m above calibrated ground" if args.uav_altitude_floor is None else "command-line override",
        ),
        flush=True,
    )
    print(
        "UAV altitude ceiling: {:.1f} m above calibrated ground (minimum z={:.3f} NED)".format(
            args.uav_altitude_ceiling, uav_altitude_ceiling
        ),
        flush=True,
    )
    if args.map_name == "rural_australia":
        course = rotate_course_left(course)
    course = shift_course_z(course, ground_z_offset)
    # Keep the existing CBF obstacle-margin parameter explicit. The default
    # remains zero until a map-specific collision calibration is available;
    # the command line can still tune it without changing the CBF equations.
    obstacle_margin = effective_obstacle_margin(args.map_name, args.obstacle_margin)
    camera_position = None
    camera_yaw = 0.0
    if top_down_camera:
        camera_position, camera_yaw = camera_director_for_map(
            args.map_name, args.camera_x, args.camera_y, args.camera_height
        )
    os.makedirs(args.debug_dir, exist_ok=True)
    run_stamp = int(time.time())
    # Put the selected map in the run stem so every derived artifact (plots,
    # animations, videos, manifests, and perception diagnostics) is
    # self-identifying when several maps are stored in one debug directory.
    artifact_stem = run_artifact_stem(args.cbf_method, args.map_name, run_stamp)
    log_path = os.path.join(args.debug_dir, artifact_stem + ".jsonl")
    recording_folder = os.path.join(args.debug_dir, "{}_recording_frames".format(os.path.splitext(os.path.basename(log_path))[0]))
    launch_config = AirSimLaunchConfig(
        launch_mode=args.launch_mode,
        map_name=args.map_name,
        unreal_editor_path=args.unreal_editor_path,
        uproject_path=args.uproject_path,
        multirotor_port=args.multirotor_port,
        car_port=args.car_port,
        resolution=(args.resx, args.resy),
        startup_timeout=args.startup_timeout,
        settings_path=args.settings_path,
        camera_director_position=camera_position,
        camera_director_yaw=camera_yaw,
        rotate_camera_director=args.map_name == "rural_australia",
        record_video=args.record_video,
        record_uav=args.record_uav if args.record_video else None,
        record_ugv=args.record_ugv if args.record_video else None,
        video_resolution=tuple(args.video_resolution),
        video_fps=args.video_fps,
        recording_folder=recording_folder if args.record_video else None,
        target_tracking=track_target and args.target_observation_source == "camera",
        target_uav_camera="target_bottom",
    )
    launcher = AirSimLauncher(launch_config)
    facade = AirSimFacade(launch_config)
    blocks: List[str] = []
    true_obstacles: List[Dict[str, Any]] = []
    target_controller: Optional[FigureEightTargetController] = None
    target_observation_worker: Optional[TargetObservationWorker] = None
    target_capture_ids: Dict[str, Optional[str]] = {name: None for name in names}
    target_time_mapper = MissionTimeMapper()
    target_rng = np.random.default_rng(7)
    target_start_position = None
    target_start_yaw = initial_yaw
    # Course waypoints are retained for explicit fixed-goal route experiments;
    # target tracking uses the deterministic target-centered objective and no
    # online recovery waypoint generation.
    intermediate_waypoint = None
    formation = FormationController(FormationConfig(
        uav_altitude=args.uav_altitude,
        max_speed=args.nominal_speed,
        leader_max_speed=args.leader_nominal_speed,
        position_gain=args.nominal_position_gain,
        velocity_gain=args.nominal_velocity_gain,
        ugv_heading_gain=args.ugv_heading_gain,
        ugv_max_yaw_rate=args.ugv_max_yaw_rate,
        intermediate_waypoint=intermediate_waypoint,
        waypoint_radius=3.0,
    ))
    cbf_config = CBFConfig(
        method=args.cbf_method,
        uncertainty_radius=args.uncertainty_radius,
        uav_radius=args.uav_radius,
        ugv_radius=args.ugv_radius,
        obstacle_margin=obstacle_margin,
        uav_velocity_limit=args.uav_velocity_limit,
        ugv_speed_limit=args.ugv_speed_limit,
        uav_acceleration_limit=args.uav_acceleration_limit,
        ugv_acceleration_limit=args.ugv_acceleration_limit,
        k1=args.k1,
        k2=args.k2,
        alpha=args.alpha,
        uav_altitude_floor=uav_altitude_floor,
    )
    vehicle_radii = {
        name: cbf_config.uav_radius if types[name] == "drone" else cbf_config.ugv_radius
        for name in names
    }
    # Wang's double-integrator formulation is retained for UAVs. The AirSim
    # Husky cannot realize a planar acceleration vector directly, so Wang
    # mode uses the already-supported unicycle CBF model for UGVs, as allowed
    # by the initial implementation scope.
    # The physical Husky cannot rotate in place. A shorter existing unicycle
    # lookahead keeps the safety control point from entering a surface-based
    # LiDAR proxy before the car has completed its turn.
    ugv_cbf_config = replace(
        cbf_config, method="mestres", lookahead_distance=args.ugv_lookahead_distance
    )
    agents = {
        name: Agent(
            name,
            types[name],
            ugv_cbf_config if types[name] == "ugv" else cbf_config,
            tracking_config={
                "window_seconds": args.tracking_window,
                "process_noise_spectral_density": args.tracking_process_noise,
                "measurement_std": args.tracking_measurement_std,
                "rho": args.tracking_admm_rho,
                "max_iterations": args.tracking_admm_max_iterations,
                "tolerance": args.tracking_admm_tolerance,
            },
        )
        for name in names
    }
    tracking = DistributedTargetTracking(
        {name: agents[name].tracking_module for name in names},
        max_iterations=args.tracking_admm_max_iterations,
        tolerance=args.tracking_admm_tolerance,
    )
    # Depth cameras observe partial obstacle surfaces, so their spherical
    # proxy needs a little more geometric padding to cover the unseen edge of
    # the surface. LiDAR already observes the planar footprint directly and
    # keeps the tighter fit used for UGVs.
    detectors = {
        "drone": ObstacleDetector(PerceptionConfig(
            top_n=args.top_n_obstacles,
            cluster_min_samples=20 if args.map_name == "rural_australia" else 8,
            min_proxy_points=32 if args.map_name == "rural_australia" else 1,
            fit_padding=0.25 if args.map_name == "rural_australia" else 0.75,
            max_proxy_radius=1.0 if args.map_name == "rural_australia" else 2.0,
        )),
        "ugv": ObstacleDetector(PerceptionConfig(
            top_n=args.top_n_obstacles,
            planar_surface_offset=0.0 if args.map_name == "rural_australia" else 0.75,
            # RuralAustralia foliage is a sparse collection of small LiDAR
            # returns rather than a dense planar mesh. Keep FlyingCPP's
            # stricter clustering unchanged while allowing those returns to
            # form local obstacle patches on the RuralAustralia map.
            cluster_eps=0.85 if args.map_name == "rural_australia" else 0.65,
            cluster_min_samples=3 if args.map_name == "rural_australia" else 8,
            min_proxy_points=32 if args.map_name == "rural_australia" else 1,
            max_proxy_radius=1.0 if args.map_name == "rural_australia" else 2.0,
            fit_padding=0.35 if args.map_name == "rural_australia" else 0.0,
            ground_band=0.15 if args.map_name == "rural_australia" else 0.25,
            planar_use_nearest_surface=args.map_name == "rural_australia",
            rank_by_surface_distance=args.map_name == "rural_australia",
        )),
    }
    goal = np.asarray(course["goal"], dtype=float)
    goal_actor_name: Optional[str] = None
    route_markers = [np.asarray(point, dtype=float) for point in course.get("waypoints", [])]
    route_markers.extend([goal])
    writer = AsyncJsonlWriter(log_path)
    perception_trace = PerceptionTraceStore()
    perception_sidecar_path = os.path.join(args.debug_dir, "{}_perception_points.npz".format(os.path.splitext(os.path.basename(log_path))[0]))
    capture_sequence = 0
    camera_fovs: Dict[str, float] = {}
    active_collisions = set()
    setup_paused = False
    recording_started = False
    frame_recorder = None
    follow_camera = None
    recording_camera_base = None

    try:
        launcher.launch()
        facade.connect()
        facade.multirotor.simRunConsoleCommand("DisableAllScreenMessages")
        if args.launch_mode == "existing" and (top_down_camera or args.map_name == "rural_australia"):
            print(
                "Warning: --launch-mode existing cannot apply launch-time CameraDirector settings; "
                "runtime goal-camera alignment will still be attempted.",
                flush=True,
            )

        # Freeze physics while all actors are placed. This prevents a
        # settings-defined Husky from beginning to fall or skid before its
        # formation pose and brake command have been applied.
        facade.pause(True)
        setup_paused = True

        fallback_goal = goal.copy()
        if args.map_name == "rural_australia" or args.goal_actor:
            detected_name, detected_position = resolve_goal_actor(
                facade, fallback_goal, actor_name=args.goal_actor
            )
            if detected_position is not None:
                goal_actor_name = detected_name
                goal = detected_position
                route_markers = [np.asarray(point, dtype=float) for point in course.get("waypoints", [])]
                route_markers.append(goal)
                print(
                    "Goal actor resolved: {} at NED ({:.3f}, {:.3f}, {:.3f})".format(
                        goal_actor_name, goal[0], goal[1], goal[2]
                    ),
                    flush=True,
                )
            elif args.goal_actor:
                print(
                    "Warning: requested goal actor {!r} was not found; using course goal {}".format(
                        args.goal_actor, fallback_goal.tolist()
                    ),
                    flush=True,
                )
            elif args.map_name == "rural_australia":
                print(
                    "Warning: no opposite-side generic Unreal actor was found; using course goal {}".format(
                        fallback_goal.tolist()
                    ),
                    flush=True,
                )

        # The formation and its initial body frame follow the actual goal
        # direction. This keeps the requested box/triangle geometry intact
        # while avoiding a hard-coded RuralAustralia heading.
        initial_yaw = heading_to_goal(np.zeros(3), goal, fallback=initial_yaw)
        initial_yaw += float(np.deg2rad(args.initial_heading_offset_deg))

        if track_target:
            target_ground_z = map_vehicle_ground_z(args.map_name)
            target_center = target_start_anchor_for_map(
                np.zeros(3), goal, args.map_name, target_ground_z, initial_yaw
            )
            target_controller = FigureEightTargetController(
                center=target_center,
                route_heading=initial_yaw,
                longitudinal_span=args.target_pattern_length,
                lateral_span=args.target_pattern_width,
                speed=args.target_speed,
                direction=1,
            )
            target_controller.index = TARGET_START_SAMPLE_INDEX % target_controller.sample_count
            # Keep the route's deliberate lateral-lobe startup while placing
            # the physical target at the map-specific anchor: one quarter
            # along RuralAustralia's route and at the resolved goal in
            # FlyingCPP.
            target_controller.place_start_at(target_center)
            if args.map_name == "flyingcpp" and not args.no_spawn_obstacles:
                target_index, target_start_clearance = safe_target_start_index(
                    target_controller,
                    course["obstacles"],
                    cbf_config.ugv_radius,
                    minimum_clearance=0.75,
                )
                if target_index != target_controller.index:
                    print(
                        "Target startup phase advanced from sample {} to {} to clear spawned geometry "
                        "({:.2f} m clearance)".format(
                            target_controller.index, target_index, target_start_clearance
                        ),
                        flush=True,
                    )
                    target_controller.index = target_index
            target_start_position = target_controller.points[target_controller.index]
            target_start_yaw = heading_to_goal(
                target_start_position,
                target_controller.reference(1),
                fallback=initial_yaw + np.pi,
            )

        # Reorient only the external map camera. Vehicle-mounted camera poses
        # remain untouched because they are used by obstacle perception.
        current_camera_pose = facade.external_camera_pose()
        camera_pitch = -25.0
        if current_camera_pose is not None and getattr(current_camera_pose, "position", None) is not None:
            current_camera_position = _vector3(current_camera_pose.position)
            current_camera_distance = float(np.linalg.norm(current_camera_position[:2]))
            current_camera_z = float(current_camera_position[2])
            if not top_down_camera and hasattr(facade.airsim, "quaternion_to_euler_angles"):
                try:
                    orientation = getattr(current_camera_pose, "orientation")
                    _, camera_pitch_rad, _ = facade.airsim.quaternion_to_euler_angles(orientation)
                    if np.isfinite(camera_pitch_rad):
                        camera_pitch = float(np.degrees(camera_pitch_rad))
                except Exception:
                    pass
        else:
            camera_reference = np.asarray(camera_position, dtype=float) if camera_position is not None else np.asarray([-20.0, 0.0])
            current_camera_distance = float(np.linalg.norm(camera_reference[:2]))
            current_camera_z = -abs(args.camera_height) if top_down_camera else -15.0
        if not np.isfinite(current_camera_distance) or current_camera_distance < 1.0:
            current_camera_distance = 6.0 if top_down_camera else 20.0
        if not np.isfinite(current_camera_z):
            current_camera_z = -abs(args.camera_height) if top_down_camera else -15.0
        camera_target_position, camera_target_yaw = camera_pose_for_goal(
            np.zeros(3), goal, current_camera_distance, current_camera_z, top_down=top_down_camera
        )
        try:
            facade.set_external_camera_pose(
                _camera_pose(
                    facade.airsim, camera_target_position, camera_target_yaw,
                    pitch_degrees=-90.0 if top_down_camera else camera_pitch,
                    roll_degrees=0.0,
                )
            )
            print(
                "External camera aligned to {} (NED {}, yaw {:.1f} deg, roll 0.0 deg)".format(
                    goal_actor_name or "mission goal", np.round(camera_target_position, 3).tolist(), camera_target_yaw
                ),
                flush=True,
            )
        except Exception as error:
            print("Warning: external camera could not be aligned to goal: {}".format(error), flush=True)

        uav_start_position = np.array([0.0, 0.0, -1.0])
        ugv_start_position = np.array([0.0, 0.0, map_vehicle_ground_z(args.map_name)])
        if track_target:
            # Target tracking starts at the common launch center, using the
            # established compact, non-overlapping formation offsets. Literal
            # collocation of the physics bodies makes the AirSim takeoff and
            # CBF safety constraints infeasible; this keeps the launch point
            # common without introducing a startup collision or a large
            # target-centered ring. The estimator then drives the formation to
            # the moving target slots after startup.
            initial_positions = {
                name: (ugv_start_position if types[name] == "ugv" else uav_start_position) + (
                    rotate_xy_heading(formation.config.slots[name], initial_yaw)
                )
                for name in names
            }
        else:
            initial_positions = {
                name: (ugv_start_position if types[name] == "ugv" else uav_start_position) + (
                    startup_formation_offset(
                        formation.config.slots[name],
                        args.map_name,
                        args.mission_objective,
                        initial_yaw,
                    )
                )
                for name in names
            }
        newly_spawned = set()
        vehicle_origins = {}
        target_newly_spawned = False
        for name in names:
            if types[name] == "ugv":
                # Clear residual CarControls state when an Unreal world is
                # reused between missions. This is initialization, not a
                # runtime recovery/reset heuristic.
                # Managed launches defer CPHusky creation until this paused
                # setup block, so there is no vehicle to command yet.
                if facade.has_vehicle(name):
                    facade.stop_ugv(name)
        for name in names:
            initial_pose = _pose(facade.airsim, initial_positions[name], initial_yaw)
            if facade.spawn_vehicle(name, types[name], initial_pose):
                newly_spawned.add(name)
        if track_target:
            target_newly_spawned = facade.spawn_vehicle(
                target_name,
                "ugv",
                _pose(facade.airsim, target_start_position, target_start_yaw),
            )
        # simAddVehicle already applies the requested pose. Applying it again
        # would add the formation offset a second time in Unreal. Only
        # settings-defined vehicles, which were not spawned above, need an
        # explicit pose update.
        for name in names:
            if name not in newly_spawned:
                origin = facade.vehicle_frame_origin(name)
                if origin is None:
                    origin = np.zeros(3)
                vehicle_origins[name] = origin
                facade.set_vehicle_pose(
                    name,
                    _pose(facade.airsim, startup_pose_position(name, initial_positions, newly_spawned, vehicle_origins), initial_yaw),
                )
        for name in names:
            facade.enable(name, types[name], True)
        if track_target:
            facade.enable(target_name, "ugv", True)
        for name in names:
            if types[name] == "ugv":
                facade.stop_ugv(name)
        if track_target:
            facade.stop_ugv(target_name)
        if args.map_name == "flyingcpp" and not args.no_spawn_obstacles:
            # Create the course before starting vehicle motion so obstacle
            # creation is not perceived as a delayed post-takeoff stage.
            blocks, true_obstacles = _spawn_block_course(facade, facade.airsim, course["obstacles"])

        if track_target:
            # Runtime-spawned targets are reusable between runs, but always get
            # the same deterministic starting pose while the world is paused.
            # This second application is intentional: this fork can honor the
            # position passed to simAddVehicle while retaining the old CPHusky
            # yaw. Reapplying the identical pose fixes the startup orientation
            # without resetting the world.
            facade.set_vehicle_pose(
                target_name,
                _pose(facade.airsim, target_start_position, target_start_yaw),
            )
            facade.stop_ugv(target_name)

        _plot_route_markers(facade, route_markers)

        # Unreal may remain paused after vehicle/object setup. Release the
        # world for the UAV takeoff sequence; UGV controls are already held by
        # their startup handbrakes and will be posed again below.
        facade.pause(False)

        takeoff_futures = [facade.multirotor.takeoffAsync(vehicle_name=name) for name in uav_names]
        for future in takeoff_futures:
            future.join()

        altitude_futures = []
        for name in uav_names:
            # moveToZAsync is less sensitive to the vehicle's collision-model
            # origin than a full-pose command and avoids starting the CBF loop
            # while a drone is still below the ground plane.
            altitude_futures.append(facade.multirotor.moveToZAsync(args.uav_altitude, 3.0, vehicle_name=name))
        for future in altitude_futures:
            future.join()
        # Hero's SimpleFlight controller can report completion of moveToZ
        # before the physics body has converged when it was just teleported.
        # A short position-hold command gives the estimator a settled,
        # physically valid starting state without affecting mission timing.
        hold_futures = [
            facade.multirotor.moveByVelocityZAsync(
                0.0, 0.0, args.uav_altitude, 1.0, vehicle_name=name
            )
            for name in uav_names
        ]
        for future in hold_futures:
            future.join()
        for name in uav_names:
            facade.multirotor.hoverAsync(vehicle_name=name).join()

        # Stabilize settings-defined UGV poses only after UAV takeoff. Letting
        # such a Husky fall on RuralAustralia before the UAV startup is
        # complete is nondeterministic: a wheel can catch the terrain and
        # rotate the body upside down. Runtime-spawned Huskies are already at
        # their requested world position; they receive a heading-only pose
        # update here, while settings-defined vehicles receive the existing
        # frame-corrected position and heading.
        facade.pause(True)
        setup_paused = True
        for name in ugv_names:
            # Runtime-spawned vehicles need a heading-only correction here:
            # their position was already applied by simAddVehicle, while a
            # repeated world translation would duplicate the road offset.
            # Settings-defined vehicles retain the existing frame correction.
            facade.set_vehicle_pose(
                name,
                _pose(
                    facade.airsim,
                    startup_heading_pose_position(name, initial_positions, newly_spawned, vehicle_origins),
                    initial_yaw,
                ),
            )
            facade.stop_ugv(name)
        facade.pause(False)
        setup_paused = False
        time.sleep(0.05)

        # A failed registration must not be allowed to become a multi-agent
        # CBF failure: all drones at one origin would make the initial pair
        # constraints genuinely infeasible. Retry the formation placement a
        # few times while still in startup, before logging mission steps.
        for attempt in range(3):
            settled_states = {name: facade.state(name) for name in names}
            xy_error = max(
                float(np.linalg.norm(settled_states[name]["position"][:2] - initial_positions[name][:2]))
                for name in names
            )
            altitude_error = max(
                abs(float(settled_states[name]["position"][2]) - float(args.uav_altitude))
                for name in uav_names
            )
            startup_error = max(xy_error, altitude_error)
            if xy_error <= 1.0 and altitude_error <= 1.0:
                break
            print("Warning: startup formation error {:.3f} m; reapplying poses (attempt {})".format(
                startup_error, attempt + 1
            ), flush=True)
            if xy_error > 1.0:
                facade.pause(True)
                setup_paused = True
                for name in names:
                    if name not in newly_spawned:
                        facade.set_vehicle_pose(
                            name,
                            _pose(facade.airsim, startup_pose_position(name, initial_positions, newly_spawned, vehicle_origins), initial_yaw),
                        )
                facade.pause(False)
                setup_paused = False
            if altitude_error > 1.0:
                altitude_futures = [
                    facade.multirotor.moveToZAsync(args.uav_altitude, 3.0, vehicle_name=name)
                    for name in uav_names
                ]
                for future in altitude_futures:
                    future.join()
                hold_futures = [
                    facade.multirotor.moveByVelocityZAsync(
                        0.0, 0.0, args.uav_altitude, 1.0, vehicle_name=name
                    )
                    for name in uav_names
                ]
                for future in hold_futures:
                    future.join()
            time.sleep(0.25)

        if track_target and args.target_observation_source == "camera":
            target_observation_worker = TargetObservationWorker(
                facade.airsim,
                args.multirotor_port,
                {name: "target_bottom" for name in uav_names} | {name: "front_center" for name in ugv_names},
                target_id=target_name,
                target_actor_pattern=target_name + "*",
                sensing_range=args.target_sensing_range,
                measurement_std=args.tracking_measurement_std,
                # AirSim's DepthPerspective median over the named Husky ROI
                # is already close to the actor center in this environment.
                # Keep the physical radius in the CBF proxy below, but do not
                # add it a second time when back-projecting the measurement.
                target_radius=0.0,
                rate_hz=args.tracking_rate,
            )
            target_observation_worker.start()
            print(
                "Target tracking camera capture started at {:.2f} Hz per agent (UAV target_bottom, UGV front_center)".format(
                    args.tracking_rate
                ),
                flush=True,
            )

        if args.timing_mode == "stepped":
            facade.pause(True)

        sensor_order = list(names)
        sensor_cursor = 0
        latest_obstacles: Dict[str, List[ObstacleProxy]] = {name: [] for name in names}
        sensor_period = 1.0 / max(args.sensor_rate, 1e-3)
        latest_sensor_views: Dict[str, Dict[str, Any]] = {name: {} for name in names}
        last_perception: Dict[str, float] = {}
        tracking_period = 1.0 / max(args.tracking_rate, 1e-3)
        next_tracking_time = 0.0
        last_tracking_result: Dict[str, Any] = {"estimates": {}, "iterations": 0, "residual": 0.0, "handoffs": []}
        tracking_measurement_records: Dict[str, Dict[str, Any]] = {name: {} for name in names}
        truth_control_obstacles = {
            name: truth_obstacle_proxies(
                true_obstacles,
                types[name],
                vehicle_z=initial_positions[name][2] if types[name] == "ugv" else None,
                vehicle_radius=cbf_config.ugv_radius if types[name] == "ugv" else 0.0,
            )
            for name in names
        }
        if args.use_truth_obstacles:
            if not true_obstacles:
                print(
                    "Warning: --use-truth-obstacles requested, but no orchestrator truth boxes were spawned; "
                    "the CBF will receive an empty truth obstacle set.",
                    flush=True,
                )
            latest_obstacles = truth_control_obstacles
            latest_sensor_views = {
                name: {
                    "sensor_type": "truth_obstacle_geometry",
                    "position_frame": "world_ned",
                    "point_frame": "world_ned",
                    "capture_timestamp": time.time(),
                    "capture_id": "truth_obstacles",
                    "age": 0.0,
                    "valid": True,
                    "obstacle_count": len(truth_control_obstacles[name]),
                }
                for name in names
            }
            last_perception = {name: time.time() for name in names}
        for detector in detectors.values():
            if args.sensor_stale_after is not None:
                detector.config.stale_after = args.sensor_stale_after
            else:
                # Each agent is captured by a team-wide round-robin scheduler;
                # allow two delayed cycles while the bounded age margin
                # expands the cached proxy. This avoids dropping all local
                # obstacles for a brief RPC/sensor delay, while a much older
                # observation is still rejected by the existing stale-sensor
                # fail-safe.
                detector.config.stale_after = max(
                    detector.config.stale_after, 3.0 * sensor_period + args.dt
                )
        # Warm all sensors before starting the real-time deadline clock. The
        # first depth frame is intentionally expensive and must not become a
        # control-cycle deadline miss.
        if not args.use_truth_obstacles:
            warmup_states = {name: facade.state(name) for name in names}
            warmup_time = time.time()
            for name in names:
                obstacles, sensor_valid, sensor_view, trace = _capture_obstacles(facade, detectors[types[name]], name, types[name], warmup_states[name], warmup_time, camera_fovs)
                if sensor_valid:
                    capture_id = "capture_{:06d}_{}".format(capture_sequence, name)
                    capture_sequence += 1
                    perception_trace.add(
                        capture_id, name, sensor_view.get("sensor_type", "unknown"),
                        trace.get("sensor_points", np.empty((0, 3))), trace.get("world_points", np.empty((0, 3))),
                        trace.get("diagnostics"),
                    )
                    sensor_view["capture_id"] = capture_id
                    latest_obstacles[name] = obstacles
                    latest_sensor_views[name] = sensor_view
                    last_perception[name] = float(sensor_view.get("capture_timestamp", time.time()))

        if args.record_video:
            follow_camera = FollowCameraController(initial_yaw, aspect=float(args.video_resolution[0]) / float(args.video_resolution[1]))

            def update_recording_camera(
                position_map: Dict[str, np.ndarray],
            ) -> Dict[str, Any]:
                nonlocal recording_camera_base
                # Include the moving target in the tracked volume, while
                # keeping it out of the ordinary formation/controller set.
                chase_camera = follow_camera.update(list(position_map.values()), args.dt)
                world_position = np.asarray(chase_camera["world_position"], dtype=float)
                world_orientation = chase_camera["orientation_quaternion"]
                measured = {}
                if recording_camera_base is None:
                    try:
                        # simGetCameraInfo() routes through
                        # PIPCamera::getCameraInfo(), which is unsafe for an
                        # External camera in this AirSim fork. The image API
                        # returns the pose without that crashing path.
                        response = facade.multirotor.simGetImages([
                            facade.airsim.ImageRequest(
                                "mission_follow", facade.airsim.ImageType.Scene, False, True
                            )
                        ], vehicle_name=args.record_uav)
                        response_position = getattr(response[0], "camera_position", None) if response else None
                        if response_position is not None:
                            recording_camera_base = _vector3(response_position)
                    except Exception:
                        recording_camera_base = np.zeros(3)
                command_position = world_position - (
                    recording_camera_base if recording_camera_base is not None else np.zeros(3)
                )
                try:
                    # mission_follow is an External camera and is therefore
                    # controlled in Unreal-world NED, not Drone1-local NED.
                    facade.set_recording_camera_pose(
                        "mission_follow",
                        _camera_pose_quaternion(facade.airsim, command_position, world_orientation),
                        args.record_uav,
                    )
                    # Keep the visible map camera synchronized with the
                    # recorded chase stream when running with a viewport.
                    facade.set_external_camera_pose(
                        _camera_pose_quaternion(facade.airsim, world_position, world_orientation)
                    )
                except Exception as error:
                    print("Warning: chase camera update failed: {}".format(error), flush=True)
                return {
                    "chase_camera": dict(
                        chase_camera,
                        control_frame="unreal_world_ned",
                        command_world_position=command_position.tolist(),
                        camera_base_offset=recording_camera_base.tolist()
                        if recording_camera_base is not None else None,
                        measured=measured,
                        capture_camera="mission_follow",
                        vehicle_name=args.record_uav,
                    )
                }

            try:
                # Position the additional camera before starting capture so
                # its first frame is already a chase frame.  The dedicated
                # worker uses AirSim SceneCapture requests because this fork's
                # native recorder can report active without writing PNGs.
                initial_recording_positions = dict(initial_positions)
                if track_target:
                    initial_recording_positions[target_name] = target_start_position
                update_recording_camera(initial_recording_positions)
                frame_recorder = AirSimFrameRecorder(
                    facade.airsim,
                    args.multirotor_port,
                    [
                        (args.record_uav, "mission_follow"),
                        (args.record_uav, "front_center"),
                        (args.record_ugv, "front_center"),
                    ],
                    recording_folder,
                    args.video_fps,
                )
                frame_recorder.start()
                recording_started = True
                print(
                    "Mission recording started: chase {}, UAV FPV {}, UGV FPV {}".format(
                        args.record_uav, args.record_uav, args.record_ugv
                    ),
                    flush=True,
                )
            except Exception as error:
                print("Warning: unable to start AirSim frame capture: {}".format(error), flush=True)

        next_deadline = time.monotonic()
        formation_convergence_time = None
        for step in range(args.steps):
                cycle_start = time.monotonic()
                phase_start = cycle_start
                phase_ms = {}
                wall_now = time.time()
                now = float(step * args.dt)
                # Tracking dynamics use the fixed mission clock.  Capture
                # metadata retains wall_now/capture timestamps for diagnosis.
                target_time_mapper.update(wall_now, now)
                raw_states = {name: facade.state(name) for name in names}
                target_state = None
                if track_target:
                    raw_target_state = facade.state(target_name)
                    target_state = AgentState(
                        target_name,
                        raw_target_state["position"],
                        raw_target_state["velocity"],
                        raw_target_state.get("yaw", target_start_yaw),
                        vehicle_type="target",
                        timestamp=now,
                        yaw_rate=raw_target_state.get("yaw_rate", 0.0),
                    )
                phase_ms["state"] = (time.monotonic() - phase_start) * 1000.0
                states = {
                    name: AgentState(name, value["position"], value["velocity"], value["yaw"], vehicle_type=types[name], timestamp=now, yaw_rate=value.get("yaw_rate", 0.0))
                    for name, value in raw_states.items()
                }
                positions = {name: value.position for name, value in states.items()}
                adjacency = build_adjacency_matrix(positions, args.communication_range)
                tracking_communication_links = sorted(
                    [sorted((name, neighbor)) for name in names for neighbor in adjacency[name]
                     if name < neighbor]
                )
                safety_communication_links = sorted(
                    [sorted((name, neighbor)) for name in names for neighbor in adjacency[name]
                     if types[neighbor] == types[name] and name < neighbor]
                )
                recording_data = {}
                if args.record_video and follow_camera is not None and recording_started:
                    recording_positions = dict(positions)
                    if track_target and target_state is not None:
                        recording_positions[target_name] = target_state.position
                    recording_data = update_recording_camera(recording_positions)
                sensor_data = {
                    name: dict(raw_states[name], obstacle_points=None, skip_target_tracking=True)
                    for name in names
                }

                outbound = {}
                for name, agent in agents.items():
                    outbound[name] = agent.compute_step(sensor_data[name], {"localization": {}, "tracking": {}})
                phase_ms["estimate"] = (time.monotonic() - phase_start) * 1000.0 - phase_ms["state"]

                predicted_targets = {name: {} for name in names}
                target_command = None
                if track_target:
                    target_command = target_controller.update(target_state.position, target_state.yaw, args.dt)
                    # Consume each asynchronous camera capture at most once. A
                    # missing camera observation remains a missing measurement for
                    # this DRWT epoch; cached control cycles never masquerade as
                    # fresh sensor data.
                    if args.target_observation_source == "camera" and target_observation_worker is not None:
                        latest_target_observations = target_observation_worker.snapshot()
                    else:
                        latest_target_observations = {}
                    if step * args.dt + 1e-9 >= next_tracking_time:
                        epoch_measurements: Dict[str, Dict[str, Optional[TargetMeasurement]]] = {}
                        for name in names:
                            measurement = None
                            if args.target_observation_source == "truth":
                                measurement = truth_target_measurement(
                                    target_name,
                                    target_state.position,
                                    states[name].position,
                                    now,
                                    args.tracking_measurement_std,
                                    args.target_sensing_range,
                                    target_rng,
                                    "target_truth_{:06d}_{}".format(step, name),
                                )
                            else:
                                candidate = latest_target_observations.get(name)
                                capture_id = candidate.capture_id if candidate is not None else None
                                if capture_id is not None and capture_id != target_capture_ids.get(name):
                                    target_capture_ids[name] = capture_id
                                    measurement = (
                                        _retime_target_measurement(candidate, target_time_mapper)
                                        if candidate.valid else None
                                    )
                            epoch_measurements[name] = {target_name: measurement} if measurement is not None else {}
                            if measurement is not None:
                                tracking_measurement_records[name] = dict(
                                    measurement.to_dict(),
                                    age=max(0.0, float(now) - float(measurement.timestamp)),
                                )
                            elif args.target_observation_source == "camera":
                                candidate = latest_target_observations.get(name)
                                retimed_candidate = (
                                    _retime_target_measurement(candidate, target_time_mapper)
                                    if candidate is not None else None
                                )
                                tracking_measurement_records[name] = dict(
                                    retimed_candidate.to_dict(),
                                    age=max(0.0, float(now) - float(retimed_candidate.timestamp)),
                                ) if retimed_candidate is not None else {
                                    "target_id": target_name, "valid": False, "source": "camera", "capture_id": None,
                                }
                        last_tracking_result = tracking.update(now, epoch_measurements, adjacency)
                        tracking.perform_handoffs(now, adjacency)
                        next_tracking_time += tracking_period
                    predicted_targets = tracking.predicted_estimates(now)
                for name, agent in agents.items():
                    agent.set_target_estimates(predicted_targets.get(name, {}))

                # Formation control consumes the estimator interface. Its
                # source is selected explicitly by --target-observation-source.
                estimated_states = {
                    name: AgentState(
                        name,
                        agents[name].local_state_estimate["estimated_position"],
                        agents[name].local_state_estimate["estimated_velocity"],
                        agents[name].local_state_estimate.get("yaw", 0.0),
                        vehicle_type=types[name],
                        timestamp=now,
                        yaw_rate=raw_states[name].get("yaw_rate", 0.0),
                    )
                    for name in names
                }
                formation_metrics = formation.metrics(estimated_states)
                leader_goal_xy_distance = float(np.linalg.norm(
                    estimated_states[formation.config.leader_id].position[:2] - goal[:2]
                ))
                formation_converged_2m_xy = bool(
                    leader_goal_xy_distance <= 2.0
                    and formation_metrics.get("formation_xy_max_error", float("inf")) <= 2.0
                )
                if formation_converged_2m_xy and formation_convergence_time is None:
                    formation_convergence_time = float(step * args.dt)
                    print(
                        "Formation converged in XY at mission t={:.2f} s "
                        "(leader-goal {:.2f} m, max slot error {:.2f} m)".format(
                            formation_convergence_time,
                            leader_goal_xy_distance,
                            formation_metrics["formation_xy_max_error"],
                        ),
                        flush=True,
                    )

                for name, agent in agents.items():
                    if args.mission_objective == "track-target":
                        target_estimate = agent.target_estimate.get(target_name, {})
                        if types[name] == "ugv":
                            nominal = formation.target_nominal_unicycle_control(
                                estimated_states[name],
                                target_estimate,
                                initial_yaw,
                                target_z=target_state.position[2],
                                ugv_circumradius=args.target_ugv_circumradius,
                            )
                        else:
                            nominal = formation.target_nominal_control(
                                estimated_states[name],
                                target_estimate,
                                initial_yaw,
                                target_z=target_state.position[2],
                                ugv_circumradius=args.target_ugv_circumradius,
                            )
                    elif types[name] == "ugv":
                        nominal = formation.nominal_unicycle_control(estimated_states[name], estimated_states, goal)
                    else:
                        nominal = formation.nominal_control(estimated_states[name], estimated_states, goal)
                    agent.set_nominal_control(nominal)

                # ``sensor_rate`` is per agent. The round-robin scheduler must
                # budget that rate across the whole team; dividing by the
                # sensor period under-captured each agent by a factor of the
                # team size.
                if not args.use_truth_obstacles:
                    capture_count = max(1, int(np.ceil(len(names) * args.dt * args.sensor_rate)))
                    capture_names = [sensor_order[(sensor_cursor + index) % len(sensor_order)] for index in range(capture_count)]
                    sensor_cursor = (sensor_cursor + capture_count) % len(sensor_order)
                    for name in capture_names:
                        obstacles, sensor_valid, sensor_view, trace = _capture_obstacles(facade, detectors[types[name]], name, types[name], raw_states[name], wall_now, camera_fovs)
                        if sensor_valid:
                            capture_id = "capture_{:06d}_{}".format(capture_sequence, name)
                            capture_sequence += 1
                            perception_trace.add(
                                capture_id, name, sensor_view.get("sensor_type", "unknown"),
                                trace.get("sensor_points", np.empty((0, 3))), trace.get("world_points", np.empty((0, 3))),
                                trace.get("diagnostics"),
                            )
                            sensor_view["capture_id"] = capture_id
                            latest_obstacles[name] = obstacles
                            latest_sensor_views[name] = sensor_view
                            last_perception[name] = float(sensor_view.get("capture_timestamp", time.time()))
                phase_ms["perception"] = (time.monotonic() - phase_start) * 1000.0 - sum(phase_ms.values())

                safe_commands = {}
                obstacle_records = {}
                for name, agent in agents.items():
                    neighbor_messages = [outbound[neighbor]["localization"] for neighbor in adjacency[name] if types[neighbor] == types[name]]
                    age = 0.0 if args.use_truth_obstacles else max(0.0, wall_now - last_perception.get(name, -float("inf")))
                    sensor_valid = True if args.use_truth_obstacles else bool(age <= detectors[types[name]].config.stale_after)
                    age_margin = 0.0
                    if sensor_valid:
                        speed_limit = cbf_config.uav_velocity_limit if types[name] == "drone" else cbf_config.ugv_speed_limit
                        acceleration_limit = cbf_config.uav_acceleration_limit if types[name] == "drone" else cbf_config.ugv_acceleration_limit
                        # The detector already bounds each geometric proxy.
                        # Do not let delayed RPC timing inflate that bounded
                        # local surface into a map-sized wall. The raw age is
                        # retained in JSONL for diagnosis; only the applied
                        # CBF proxy radius is capped here.
                        uncapped_age_margin = speed_limit * age + 0.5 * acceleration_limit * age * age
                        age_margin = min(
                            uncapped_age_margin,
                            (0.25 if args.map_name == "rural_australia" else 0.5)
                            * float(detectors[types[name]].config.max_proxy_radius),
                        )
                    perception_obstacles = [ObstacleProxy(
                        item.obstacle_id, item.center, item.radius + age_margin, item.source,
                        item.timestamp, item.point_count, item.is_planar
                    ) for item in latest_obstacles[name]]
                    perception_obstacles, rejected_agent_proxies = filter_agent_body_obstacle_proxies(
                        perception_obstacles,
                        states,
                        vehicle_radii,
                    ) if not args.use_truth_obstacles else (perception_obstacles, 0)
                    obstacles = list(perception_obstacles)
                    target_proxy = None
                    if track_target and target_state is not None:
                        target_estimate = agents[name].target_estimate.get(target_name, {})
                        target_proxy = target_obstacle_proxy_for_agent(
                            types[name],
                            target_name,
                            target_estimate,
                            target_state.position[2],
                            cbf_config.ugv_radius,
                            now,
                        )
                        if target_proxy is not None:
                            obstacles.append(target_proxy)
                    cbf_sensor_valid = cbf_sensor_valid_for_target(
                        types[name], sensor_valid, target_proxy
                    )
                    obstacle_records[name] = {
                        "sensor_valid": sensor_valid,
                        "cbf_sensor_valid": cbf_sensor_valid,
                        "age": age,
                        "age_margin": age_margin,
                        "perception_proxy_count": len(perception_obstacles) + rejected_agent_proxies,
                        "rejected_agent_proxy_count": rejected_agent_proxies,
                        "count": len(obstacles),
                        "sensor_view": dict(latest_sensor_views[name], age=age, valid=sensor_valid),
                        "proxies": [{"id": item.obstacle_id, "center": item.center.tolist(), "radius": item.radius, "source": item.source, "points": item.point_count, "velocity": item.velocity.tolist() if item.velocity is not None else None} for item in obstacles],
                    }
                    safe_commands[name] = agent.control_step(
                        neighbor_messages, obstacles, cbf_sensor_valid
                    )
                phase_ms["cbf"] = (time.monotonic() - phase_start) * 1000.0 - sum(phase_ms.values())

                for name, command in safe_commands.items():
                    if types[name] == "drone":
                        velocity = states[name].velocity + args.dt * command
                        velocity = np.clip(velocity, -cbf_config.uav_velocity_limit, cbf_config.uav_velocity_limit)
                        velocity = clamp_uav_velocity_to_altitude_ceiling(
                            velocity,
                            states[name].position[2],
                            uav_altitude_ceiling,
                            args.dt,
                            cbf_config.uav_velocity_limit,
                        )
                        facade.command_uav(name, velocity, args.dt)
                    elif types[name] == "ugv":
                        turn_speed = float(command[0])
                        turn_rate = float(command[1])
                        # A physical AirSim car cannot realize the ideal
                        # unicycle command [0, yaw_rate] by rotating in place.
                        # Preserve the CBF output while using a small crawl
                        # speed only when it requests a turn at zero speed.
                        ugv_brake = 0.0
                        if turn_speed < 0.05 and abs(turn_rate) > 0.05:
                            turn_speed = min(0.3, cbf_config.ugv_speed_limit)
                        elif turn_speed < 0.05:
                            # Zero throttle alone lets the physical Husky
                            # coast through a settled goal/formation slot.
                            # Braking here is only the simulator actuation
                            # conversion; the nominal and CBF commands are
                            # unchanged.
                            ugv_brake = 1.0
                        ugv_throttle, ugv_brake = ugv_speed_control_inputs(
                            turn_speed,
                            states[name].velocity,
                            base_brake=ugv_brake,
                        )
                        facade.command_ugv(
                            name,
                            turn_speed,
                            UGV_STEERING_SCALE * turn_rate / cbf_config.ugv_yaw_rate_limit,
                            args.dt,
                            brake=ugv_brake,
                            throttle=ugv_throttle,
                        )
                    else:
                        desired_velocity = states[name].velocity[:2] + args.dt * command[:2]
                        speed = float(np.linalg.norm(desired_velocity))
                        heading = float(np.arctan2(desired_velocity[1], desired_velocity[0])) if speed > 1e-6 else states[name].yaw
                        heading_error = float((heading - states[name].yaw + np.pi) % (2.0 * np.pi) - np.pi)
                        steering = np.clip(heading_error / np.pi, -1.0, 1.0)

                        # Wang's UGV CBF output is a planar acceleration.  A
                        # car cannot realize that vector directly, so use its
                        # magnitude as the forward-speed request and its
                        # direction as the steering request.  Do not project
                        # the speed onto the current heading: doing so brakes
                        # whenever a fixed waypoint is more than 90 degrees
                        # away and prevents the car from turning toward it.
                        command_speed = speed
                        forward = np.array([np.cos(states[name].yaw), np.sin(states[name].yaw)])
                        longitudinal_acceleration = float(np.dot(command[:2], forward))
                        acceleration_limit = max(cbf_config.ugv_acceleration_limit, 1e-6)
                        lateral_acceleration = float(np.dot(command[:2], np.array([-forward[1], forward[0]])))
                        command_magnitude = float(np.linalg.norm(command[:2]))
                        # The car cannot realize lateral acceleration directly;
                        # steering supplies that component, so retain throttle
                        # for turn-dominated commands.  Apply brake only when
                        # the command is clearly a longitudinal deceleration.
                        turn_dominated = abs(lateral_acceleration) >= abs(longitudinal_acceleration)
                        throttle = command_magnitude / acceleration_limit if turn_dominated else max(0.0, longitudinal_acceleration / acceleration_limit)
                        needs_braking = longitudinal_acceleration < -0.05 and not turn_dominated
                        brake = min(1.0, max(0.0, -longitudinal_acceleration / acceleration_limit))
                        if command_magnitude < 0.05:
                            brake = 1.0
                        facade.command_ugv(
                            name,
                            command_speed,
                            steering,
                            args.dt,
                            brake=brake if needs_braking else 0.0,
                            throttle=throttle,
                        )
                if track_target:
                    # Target1 is deliberately outside the distributed CBF and
                    # formation controllers. It follows its fixed startup
                    # figure-eight route with the deterministic UGV command.
                    target_speed = float(target_command[0])
                    target_yaw_rate = float(target_command[1])
                    target_steering = np.clip(
                        UGV_STEERING_SCALE * target_yaw_rate / max(cbf_config.ugv_yaw_rate_limit, 1e-6),
                        -1.0,
                        1.0,
                    )
                    # CarControls accepts throttle rather than a speed command.
                    # Close the small target-speed loop here so the deterministic
                    # fixed route is not turned into an uncontrolled acceleration
                    # experiment. The target remains outside the distributed
                    # agent controller; this only makes its prescribed motion
                    # match FigureEightTargetController.speed.
                    target_actuation_speed = target_speed
                    target_throttle, target_brake = ugv_speed_control_inputs(
                        target_actuation_speed,
                        target_state.velocity,
                    )
                    if target_actuation_speed > 0.01 and target_brake < 1.0:
                        # A CPHusky needs a small nonzero throttle to overcome
                        # static friction while turning. Keep this floor below
                        # the ordinary UGV crawl floor so the physical target
                        # does not outrun its requested figure-eight speed.
                        target_throttle = max(float(target_throttle), 0.02)
                    facade.command_ugv(
                        target_name,
                        target_actuation_speed,
                        target_steering,
                        args.dt,
                        brake=float(target_brake),
                        throttle=float(target_throttle),
                    )
                phase_ms["actuation"] = (time.monotonic() - phase_start) * 1000.0 - sum(phase_ms.values())

                collision_records = {}
                collision_names = names + ([target_name] if track_target else [])
                collision_types = dict(types)
                if track_target:
                    collision_types[target_name] = "target"
                for name in collision_names:
                    collision = facade.collision_info(name)
                    ignored_ground = (
                        collision_types[name] in ("ugv", "target")
                        and collision["has_collided"]
                        and _is_ground_object(collision.get("object_name", ""))
                    )
                    ignored_initial = step == 0 and collision["has_collided"]
                    collision["ignored_ground"] = ignored_ground
                    collision["ignored_initial"] = ignored_initial
                    collision["relevant"] = collision["has_collided"] and not ignored_ground and not ignored_initial
                    collision_records[name] = collision
                    if not collision["relevant"]:
                        active_collisions = {item for item in active_collisions if item[0] != name}
                        continue
                    collision_key = (name, collision.get("object_name", ""), collision.get("object_id", -1))
                    if collision_key not in active_collisions:
                        object_name = collision.get("object_name") or "unknown object"
                        depth = collision.get("penetration_depth", 0.0)
                        print(
                            "WARNING: COLLISION at mission t={:.2f} s ({}) for {} with {} "
                            "(penetration {:.3f} m)".format(
                                step * args.dt,
                                datetime.now().astimezone().isoformat(timespec="seconds"),
                                name,
                                object_name,
                                depth,
                            ),
                            flush=True,
                        )
                        active_collisions.add(collision_key)

                cycle_ms = (time.monotonic() - cycle_start) * 1000.0
                deadline_miss = cycle_ms > args.dt * 1000.0
                target_record = None
                target_tracking_record = {
                    "enabled": track_target,
                    "observation_source": args.target_observation_source if track_target else None,
                    "target_id": target_name if track_target else None,
                    "agents": {
                        name: {
                            "measurement": tracking_measurement_records.get(name, {}) if track_target else {},
                            "estimate": predicted_targets.get(name, {}).get(target_name, {}) if track_target else {},
                            "active": bool(
                                predicted_targets.get(name, {}).get(target_name, {}).get("active", False)
                            ) if track_target else False,
                        }
                        for name in names
                    },
                    "iterations": int(last_tracking_result.get("iterations", 0)) if track_target else 0,
                    "consensus_residual": float(last_tracking_result.get("residual", 0.0)) if track_target else 0.0,
                    "handoffs": [
                        {
                            key: value
                            for key, value in handoff.items()
                            if key not in ("information_matrix", "information_vector")
                        }
                        for handoff in (last_tracking_result.get("handoffs") or [])
                    ] if track_target else [],
                }
                if track_target and target_state is not None:
                    target_record = {
                        "name": target_name,
                        "position": target_state.position.tolist(),
                        "velocity": target_state.velocity.tolist(),
                        "yaw": float(target_state.yaw),
                        "command": np.asarray(target_command).tolist(),
                        "phase": float(target_controller.phase),
                        "pattern": target_controller.diagnostics(),
                        "collision": collision_records.get(target_name, {}),
                    }
                record = {
                    "step": step,
                    "dt": args.dt,
                    "timestamp": now,
                    "wall_timestamp": wall_now,
                    "method": args.cbf_method,
                    "goal": goal.tolist(),
                    "goal_actor": goal_actor_name,
                    "ground_z_offset": float(ground_z_offset),
                    "uav_altitude_floor": float(uav_altitude_floor),
                    "uav_altitude_ceiling": float(uav_altitude_ceiling),
                    "route_markers": [point.tolist() for point in route_markers],
                    "vehicle_types": types,
                    "vehicle_radii": vehicle_radii,
                    "formation": dict(
                        formation_metrics,
                        leader_goal_xy_distance=leader_goal_xy_distance,
                        converged_2m_xy=formation_converged_2m_xy,
                        convergence_time=formation_convergence_time,
                    ),
                    "states": {
                        name: {
                            "position": states[name].position.tolist(),
                            "velocity": states[name].velocity.tolist(),
                            "yaw": float(states[name].yaw),
                            "yaw_rate": float(states[name].yaw_rate),
                            "kinematics_position": raw_states[name].get("kinematics_position", states[name].position).tolist(),
                            "actor_position": (
                                raw_states[name]["actor_position"].tolist()
                                if raw_states[name].get("actor_position") is not None else None
                            ),
                        }
                        for name in names
                    },
                    "commands": {name: np.asarray(command).tolist() for name, command in safe_commands.items()},
                    "target": target_record,
                    "target_truth": target_record,
                    "targets": {target_name: target_record} if target_record is not None else {},
                    "target_tracking": target_tracking_record,
                    "obstacles": obstacle_records,
                    "true_obstacles": true_obstacles,
                    "collisions": collision_records,
                    # The legacy field is retained as the all-type graph;
                    # consumers that need safety-only edges use the explicit
                    # safety field below.
                    "communication_links": tracking_communication_links,
                    "tracking_communication_links": tracking_communication_links,
                    "safety_communication_links": safety_communication_links,
                    "recording": recording_data,
                    "cbf": {name: agents[name].last_cbf_result.__dict__ if agents[name].last_cbf_result else {} for name in names},
                    "timing": {"cycle_ms": cycle_ms, "deadline_miss": deadline_miss, "timing_mode": args.timing_mode, "phase_ms": phase_ms},
                }
                writer.write(record)
                if args.timing_mode == "stepped":
                    facade.continue_for(args.dt)
                else:
                    next_deadline += args.dt
                    time.sleep(max(0.0, next_deadline - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if setup_paused:
                facade.pause(False)
                setup_paused = False
            if recording_started:
                try:
                    recording_stats = frame_recorder.stop() if frame_recorder is not None else {}
                    recording_started = False
                    print(
                        "Mission recording stopped ({} captures, {} capture errors)".format(
                            recording_stats.get("captures", 0), recording_stats.get("errors", 0)
                        ),
                        flush=True,
                    )
                except Exception as error:
                    print("Warning: unable to stop AirSim frame capture: {}".format(error), flush=True)
            if target_observation_worker is not None:
                try:
                    target_observation_worker.stop()
                except Exception as error:
                    print("Warning: unable to stop target observation worker: {}".format(error), flush=True)
            close_vehicles = [(name, types[name]) for name in names]
            if track_target:
                close_vehicles.append((target_name, "ugv"))
            facade.close(close_vehicles)
            for block in blocks:
                try:
                    facade.delete_object(block)
                except Exception:
                    pass
            if args.timing_mode == "stepped":
                facade.pause(False)
        except Exception:
            pass
        launcher.cleanup()
        writer.close()
    try:
        perception_trace.save(perception_sidecar_path)
        print("Perception trace generated: {}".format(perception_sidecar_path))
    except Exception as error:
        print("Warning: perception trace generation failed: {}".format(error))
    plot_paths = []
    try:
        # Plotting is deliberately post-run so matplotlib and file rendering
        # never consume time from the real-time control loop.
        from modules.mission_plots import generate_mission_plots

        plot_paths = generate_mission_plots(
            log_path,
            args.debug_dir,
            vehicle_radii,
            animation_fps=args.animation_fps,
            include_animation=not args.no_animation,
            playback_speed=args.playback_speed,
        )
        if args.record_video:
            plot_paths.extend(analyze_camera_alignment(
                load_mission_records(log_path), recording_folder, args.debug_dir,
                os.path.splitext(os.path.basename(log_path))[0], vehicle=args.record_uav,
            ))
            plot_paths.extend(render_recordings(
                recording_folder,
                args.debug_dir,
                os.path.splitext(os.path.basename(log_path))[0],
                load_mission_records(log_path),
                args.record_uav,
                args.record_ugv,
                width=args.video_resolution[0],
                height=args.video_resolution[1],
                fps=args.video_fps,
                gif_height=args.gif_height,
                gif_fps=args.gif_fps,
                keep_frames=args.keep_recording_frames,
                capture_stats=(frame_recorder.capture_count, frame_recorder.error_count)
                if frame_recorder is not None else None,
                playback_speed=args.playback_speed,
                map_name=args.map_name,
            ))
        from modules.perception_diagnostics import generate_perception_diagnostics

        plot_paths.extend(generate_perception_diagnostics(log_path, perception_sidecar_path, args.debug_dir))
    except Exception as error:
        print("Warning: post-run plot generation failed: {}".format(error))
    print("Mission complete; diagnostics: {}".format(log_path))
    for plot_path in plot_paths:
        print("Plot generated: {}".format(plot_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
