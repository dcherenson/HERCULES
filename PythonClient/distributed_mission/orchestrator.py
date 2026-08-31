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
from typing import Any, Dict, List, Optional, Tuple

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
from simulation.airsim_runtime import AsyncJsonlWriter, AirSimFacade, AirSimLaunchConfig, AirSimLauncher


CONTROL_DT = 0.1
SIMULATION_STEPS = 100
COMMUNICATION_RANGE_METERS = 30.0

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


def build_adjacency_matrix(positions: Dict[str, np.ndarray], comm_range: float) -> Dict[str, List[str]]:
    ids = list(positions)
    adjacency = {agent_id: [] for agent_id in ids}
    for index, first in enumerate(ids):
        for second in ids[index + 1:]:
            if np.linalg.norm(positions[first] - positions[second]) <= comm_range:
                adjacency[first].append(second)
                adjacency[second].append(first)
    return adjacency


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
    """Select the existing CBF margin, with a calibrated Rural default."""

    if requested is not None:
        return float(requested)
    return 1.0 if map_name == "rural_australia" else 0.0


def map_ground_z_offset(map_name: str) -> float:
    """Return the known NED ground-level correction for each map."""

    # RuralAustralia Example 1 is approximately 2 m lower than the
    # FlyingCPP mission reference. NED down is positive, so ground-referenced
    # points move from z=-1 to z=+1. UAV flight altitude remains independent.
    return 2.0 if map_name == "rural_australia" else 0.0


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
            request = facade.airsim.ImageRequest("front_center", facade.airsim.ImageType.DepthPerspective, True, False)
            responses = facade.multirotor.simGetImages([request], vehicle_name=name)
            if not responses:
                return [], False, {}, {}
            response = responses[0]
            depth = decode_depth_response(response)
            if name not in camera_fovs:
                camera = facade.multirotor.simGetCameraInfo("front_center", vehicle_name=name)
                camera_fovs[name] = float(camera.fov) * np.pi / 180.0
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
            horizontal_fov = camera_fovs[name]
            vertical_fov = 2.0 * np.arctan(np.tan(horizontal_fov / 2.0) * depth.shape[0] / depth.shape[1])
            capture_time = time.time()
            sensor_points = detector.depth_to_sensor(depth, horizontal_fov, stride=detector.config.depth_stride)
            points = sensor_points @ camera_orientation.T + camera_position
            sensor_view = {
                "sensor_type": "uav_camera",
                "position": camera_position.tolist(),
                "orientation_quaternion": _quaternion_values(response_orientation) if response_orientation is not None else _rotation_quaternion(camera_orientation),
                "horizontal_fov_deg": float(np.degrees(horizontal_fov)),
                "vertical_fov_deg": float(np.degrees(vertical_fov)),
                "range_m": float(detector.config.max_range),
                "capture_timestamp": float(capture_time),
                "position_frame": camera_position_frame,
                "point_frame": "camera_local_ned",
                "vehicle_position": np.asarray(state["position"], dtype=float).tolist(),
                "reported_position": reported_camera_position.tolist(),
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


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-mode", choices=("visible", "headless", "existing"), default="visible")
    parser.add_argument("--map", dest="map_name", choices=("flyingcpp", "rural_australia"), default="flyingcpp")
    parser.add_argument("--cbf-method", choices=("mestres", "wang"), default="mestres")
    parser.add_argument("--steps", type=int, default=SIMULATION_STEPS)
    parser.add_argument("--dt", type=float, default=CONTROL_DT)
    parser.add_argument("--timing-mode", choices=("realtime", "stepped"), default="realtime")
    parser.add_argument("--uav-altitude", type=float, default=-5.0, help="UAV hover altitude in AirSim NED coordinates")
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
    parser.add_argument("--sensor-rate", type=float, default=2.5, help="per-agent obstacle perception rate")
    parser.add_argument("--sensor-stale-after", type=float, default=None, help="maximum cached sensor age; defaults to one sensor period plus 50 ms")
    parser.add_argument("--top-n-obstacles", type=int, default=5)
    parser.add_argument("--uncertainty-radius", type=float, default=0.0)
    parser.add_argument("--uav-radius", type=float, default=1.0)
    parser.add_argument("--ugv-radius", type=float, default=1.25)
    parser.add_argument(
        "--obstacle-margin", type=float, default=None,
        help="additional CBF obstacle clearance margin; defaults to 1.0 m on RuralAustralia and 0 m elsewhere",
    )
    parser.add_argument("--uav-velocity-limit", type=float, default=2.0)
    parser.add_argument("--ugv-speed-limit", type=float, default=2.0)
    parser.add_argument("--uav-acceleration-limit", type=float, default=4.0)
    parser.add_argument("--ugv-acceleration-limit", type=float, default=2.0)
    parser.add_argument("--nominal-speed", type=float, default=2.0, help="existing formation nominal speed limit")
    parser.add_argument("--k1", type=float, default=2.0)
    parser.add_argument("--k2", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--communication-range", type=float, default=COMMUNICATION_RANGE_METERS)
    parser.add_argument("--multirotor-port", type=int, default=41451)
    parser.add_argument("--car-port", type=int, default=41452)
    parser.add_argument("--uproject-path", default=None)
    parser.add_argument("--settings-path", default=None, help="AirSim settings.json to copy for launch-time camera overrides")
    parser.add_argument("--resx", type=int, default=800)
    parser.add_argument("--resy", type=int, default=600)
    parser.add_argument("--no-spawn-obstacles", action="store_true")
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
    top_down_camera = args.top_down_camera and not args.no_top_down_camera
    try:
        course = load_course(args.obstacle_course) if args.obstacle_course else course_from_tuples(BLOCK_COURSE)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print("Error: unable to load obstacle course: {}".format(error), file=sys.stderr)
        return 2
    initial_yaw = -np.pi / 2.0 if args.map_name == "rural_australia" else 0.0
    ground_z_offset = map_ground_z_offset(args.map_name)
    if args.map_name == "rural_australia":
        course = rotate_course_left(course)
    course = shift_course_z(course, ground_z_offset)
    # RuralAustralia's foliage returns represent a visible surface, while
    # the collision body can extend beyond that sparse surface. Use the
    # existing CBF obstacle-margin parameter as a map calibration only; the
    # explicit command-line value always wins, including --obstacle-margin 0.
    obstacle_margin = effective_obstacle_margin(args.map_name, args.obstacle_margin)
    if args.map_name == "rural_australia" and args.obstacle_margin is None:
        print("RuralAustralia perception calibration: using 1.0 m obstacle margin", flush=True)
    camera_position = None
    camera_yaw = 0.0
    if top_down_camera:
        camera_position, camera_yaw = camera_director_for_map(
            args.map_name, args.camera_x, args.camera_y, args.camera_height
        )
    os.makedirs(args.debug_dir, exist_ok=True)
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
    )
    launcher = AirSimLauncher(launch_config)
    facade = AirSimFacade(launch_config)
    blocks: List[str] = []
    true_obstacles: List[Dict[str, Any]] = []
    names = ["Drone1", "Drone2", "SimpleFlight", "Drone4", "Drone5", "Husky1", "Husky2", "Husky3"]
    types = {name: "drone" for name in names[:5]}
    types.update({name: "ugv" for name in names[5:]})
    # The through-gap experiment intentionally has no bypass waypoint. The
    # nominal controller follows the leader and formation offsets; CBF
    # constraints provide the small deformation needed at the central gap.
    intermediate_waypoint = None
    formation = FormationController(FormationConfig(
        uav_altitude=args.uav_altitude,
        max_speed=args.nominal_speed,
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
    ugv_cbf_config = replace(cbf_config, method="mestres", lookahead_distance=0.5)
    agents = {
        name: Agent(name, types[name], ugv_cbf_config if types[name] == "ugv" else cbf_config)
        for name in names
    }
    # Depth cameras observe partial obstacle surfaces, so their spherical
    # proxy needs a little more geometric padding to cover the unseen edge of
    # the surface. LiDAR already observes the planar footprint directly and
    # keeps the tighter fit used for UGVs.
    detectors = {
        "drone": ObstacleDetector(PerceptionConfig(top_n=args.top_n_obstacles, fit_padding=0.75)),
        "ugv": ObstacleDetector(PerceptionConfig(
            top_n=args.top_n_obstacles,
            planar_surface_offset=0.0 if args.map_name == "rural_australia" else 0.75,
            # RuralAustralia foliage is a sparse collection of small LiDAR
            # returns rather than a dense planar mesh. Keep FlyingCPP's
            # stricter clustering unchanged while allowing those returns to
            # form local obstacle patches on the RuralAustralia map.
            cluster_eps=0.85 if args.map_name == "rural_australia" else 0.65,
            cluster_min_samples=3 if args.map_name == "rural_australia" else 8,
            fit_padding=0.35 if args.map_name == "rural_australia" else 0.0,
            ground_band=0.15 if args.map_name == "rural_australia" else 0.25,
            planar_use_nearest_surface=args.map_name == "rural_australia",
            rank_by_surface_distance=args.map_name == "rural_australia",
        )),
    }
    goal = np.asarray(course["goal"], dtype=float)
    route_markers = [np.asarray(point, dtype=float) for point in course.get("waypoints", [])]
    route_markers.extend([goal])
    log_path = os.path.join(args.debug_dir, "{}_{}.jsonl".format(args.cbf_method, int(time.time())))
    writer = AsyncJsonlWriter(log_path)
    perception_trace = PerceptionTraceStore()
    perception_sidecar_path = os.path.join(args.debug_dir, "{}_perception_points.npz".format(os.path.splitext(os.path.basename(log_path))[0]))
    capture_sequence = 0
    camera_fovs: Dict[str, float] = {}
    active_collisions = set()

    try:
        launcher.launch()
        facade.connect()
        facade.multirotor.simRunConsoleCommand("DisableAllScreenMessages")
        if args.launch_mode == "existing" and (top_down_camera or args.map_name == "rural_australia"):
            print("Warning: --launch-mode existing cannot apply the Rural/camera CameraDirector override; "
                  "configure it before launching Unreal.", flush=True)

        uav_start_position = np.array([0.0, 0.0, -1.0])
        ugv_start_position = np.array([0.0, 0.0, -1.0 + ground_z_offset])
        initial_positions = {
            name: (ugv_start_position if types[name] == "ugv" else uav_start_position) + (
                rotate_xy_right(formation.config.slots[name])
                if args.map_name == "rural_australia" else formation.config.slots[name]
            )
            for name in names
        }
        newly_spawned = set()
        vehicle_origins = {}
        for name in names:
            if types[name] == "ugv":
                # Clear residual CarControls state when an Unreal world is
                # reused between missions. This is initialization, not a
                # runtime recovery/reset heuristic.
                facade.stop_ugv(name)
        for name in names:
            initial_pose = _pose(facade.airsim, initial_positions[name], initial_yaw)
            if facade.spawn_vehicle(name, types[name], initial_pose):
                newly_spawned.add(name)
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
                facade.set_vehicle_pose(name, _pose(facade.airsim, initial_positions[name] - origin, initial_yaw))
        for name in names:
            facade.enable(name, types[name], True)
        for name in names:
            if types[name] == "ugv":
                facade.stop_ugv(name)
        time.sleep(0.2)

        if args.map_name == "flyingcpp" and not args.no_spawn_obstacles:
            # Create the course before starting vehicle motion so obstacle
            # creation is not perceived as a delayed post-takeoff stage.
            blocks, true_obstacles = _spawn_block_course(facade, facade.airsim, course["obstacles"])

        _plot_route_markers(facade, route_markers)

        takeoff_futures = [facade.multirotor.takeoffAsync(vehicle_name=name) for name in names[:5]]
        for future in takeoff_futures:
            future.join()

        altitude_futures = []
        for name in names[:5]:
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
            for name in names[:5]
        ]
        for future in hold_futures:
            future.join()
        for name in names[:5]:
            facade.multirotor.hoverAsync(vehicle_name=name).join()

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
                for name in names[:5]
            )
            startup_error = max(xy_error, altitude_error)
            if xy_error <= 1.0 and altitude_error <= 1.0:
                break
            print("Warning: startup formation error {:.3f} m; reapplying poses (attempt {})".format(
                startup_error, attempt + 1
            ), flush=True)
            if xy_error > 1.0:
                for name in names:
                    facade.set_vehicle_pose(name, _pose(facade.airsim, initial_positions[name], initial_yaw))
            if altitude_error > 1.0:
                altitude_futures = [
                    facade.multirotor.moveToZAsync(args.uav_altitude, 3.0, vehicle_name=name)
                    for name in names[:5]
                ]
                for future in altitude_futures:
                    future.join()
                hold_futures = [
                    facade.multirotor.moveByVelocityZAsync(
                        0.0, 0.0, args.uav_altitude, 1.0, vehicle_name=name
                    )
                    for name in names[:5]
                ]
                for future in hold_futures:
                    future.join()
            time.sleep(0.25)

        if args.timing_mode == "stepped":
            facade.pause(True)

        sensor_order = list(names)
        sensor_cursor = 0
        latest_obstacles: Dict[str, List[ObstacleProxy]] = {name: [] for name in names}
        sensor_period = 1.0 / max(args.sensor_rate, 1e-3)
        latest_sensor_views: Dict[str, Dict[str, Any]] = {name: {} for name in names}
        last_perception: Dict[str, float] = {}
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
                # allow one delayed cycle while the age margin expands the
                # cached proxy. This avoids dropping all local obstacles for a
                # brief RPC/sensor delay, while a much older observation is
                # still rejected by the existing stale-sensor fail-safe.
                detector.config.stale_after = max(
                    detector.config.stale_after, 2.0 * sensor_period + args.dt
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

        next_deadline = time.monotonic()
        for step in range(args.steps):
                cycle_start = time.monotonic()
                phase_start = cycle_start
                phase_ms = {}
                now = time.time()
                raw_states = {name: facade.state(name) for name in names}
                phase_ms["state"] = (time.monotonic() - phase_start) * 1000.0
                states = {
                    name: AgentState(name, value["position"], value["velocity"], value["yaw"], vehicle_type=types[name], timestamp=now, yaw_rate=value.get("yaw_rate", 0.0))
                    for name, value in raw_states.items()
                }
                positions = {name: value.position for name, value in states.items()}
                adjacency = build_adjacency_matrix(positions, args.communication_range)
                sensor_data = {
                    name: dict(raw_states[name], obstacle_points=None)
                    for name in names
                }

                outbound = {}
                for name, agent in agents.items():
                    outbound[name] = agent.compute_step(sensor_data[name], {"localization": {}, "tracking": {}})
                phase_ms["estimate"] = (time.monotonic() - phase_start) * 1000.0 - phase_ms["state"]

                # Formation control consumes the estimator interface. The
                # estimator is truth-backed in v1 and can later be replaced.
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

                for name, agent in agents.items():
                    if types[name] == "ugv":
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
                        obstacles, sensor_valid, sensor_view, trace = _capture_obstacles(facade, detectors[types[name]], name, types[name], raw_states[name], now, camera_fovs)
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
                    age = 0.0 if args.use_truth_obstacles else max(0.0, now - last_perception.get(name, -float("inf")))
                    sensor_valid = True if args.use_truth_obstacles else bool(age <= detectors[types[name]].config.stale_after)
                    age_margin = 0.0
                    if sensor_valid:
                        speed_limit = cbf_config.uav_velocity_limit if types[name] == "drone" else cbf_config.ugv_speed_limit
                        acceleration_limit = cbf_config.uav_acceleration_limit if types[name] == "drone" else cbf_config.ugv_acceleration_limit
                        age_margin = speed_limit * age + 0.5 * acceleration_limit * age * age
                    obstacles = [ObstacleProxy(
                        item.obstacle_id, item.center, item.radius + age_margin, item.source,
                        item.timestamp, item.point_count, item.is_planar
                    ) for item in latest_obstacles[name]]
                    obstacle_records[name] = {
                        "sensor_valid": sensor_valid,
                        "age": age,
                        "age_margin": age_margin,
                        "count": len(obstacles),
                        "sensor_view": dict(latest_sensor_views[name], age=age, valid=sensor_valid),
                        "proxies": [{"id": item.obstacle_id, "center": item.center.tolist(), "radius": item.radius, "source": item.source, "points": item.point_count} for item in obstacles],
                    }
                    safe_commands[name] = agent.control_step(neighbor_messages, obstacles, sensor_valid)
                phase_ms["cbf"] = (time.monotonic() - phase_start) * 1000.0 - sum(phase_ms.values())

                for name, command in safe_commands.items():
                    if types[name] == "drone":
                        velocity = states[name].velocity + args.dt * command
                        velocity = np.clip(velocity, -cbf_config.uav_velocity_limit, cbf_config.uav_velocity_limit)
                        facade.command_uav(name, velocity, args.dt)
                    elif types[name] == "ugv":
                        turn_speed = float(command[0])
                        # A physical AirSim car cannot realize the ideal
                        # unicycle command [0, yaw_rate] by rotating in place.
                        # Preserve the CBF output while using a small crawl
                        # speed only when it requests a turn at zero speed.
                        if turn_speed < 0.05 and abs(float(command[1])) > 0.05:
                            turn_speed = min(0.3, cbf_config.ugv_speed_limit)
                        facade.command_ugv(name, turn_speed, command[1] / cbf_config.ugv_yaw_rate_limit, args.dt)
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
                        facade.command_ugv(
                            name,
                            command_speed,
                            steering,
                            args.dt,
                            brake=brake if needs_braking else 0.0,
                            throttle=throttle,
                        )
                phase_ms["actuation"] = (time.monotonic() - phase_start) * 1000.0 - sum(phase_ms.values())

                collision_records = {}
                for name in names:
                    collision = facade.collision_info(name)
                    ignored_ground = (
                        types[name] == "ugv"
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
                record = {
                    "step": step,
                    "dt": args.dt,
                    "timestamp": now,
                    "method": args.cbf_method,
                    "goal": goal.tolist(),
                    "ground_z_offset": float(ground_z_offset),
                    "route_markers": [point.tolist() for point in route_markers],
                    "vehicle_types": types,
                    "vehicle_radii": vehicle_radii,
                    "formation": formation.metrics(estimated_states),
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
                    "obstacles": obstacle_records,
                    "true_obstacles": true_obstacles,
                    "collisions": collision_records,
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
            facade.close([(name, types[name]) for name in names])
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
        )
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
