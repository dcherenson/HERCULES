#!/usr/bin/env python3
"""Run the heterogeneous distributed CBF mission in AirSim."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, ".."))

from agent import Agent
from modules.cbf import AgentState, CBFConfig, ObstacleProxy
from modules.formation_control import FormationConfig, FormationController
from modules.obstacle_detection import ObstacleDetector, PerceptionConfig, decode_depth_response
from modules.perception_diagnostics import PerceptionTraceStore
from simulation.airsim_runtime import AsyncJsonlWriter, AirSimFacade, AirSimLaunchConfig, AirSimLauncher


CONTROL_DT = 0.1
SIMULATION_STEPS = 100
COMMUNICATION_RANGE_METERS = 30.0

# Narrow lateral passages make the 4 m-wide UAV box split apart. The
# blocks have deliberately different heights, so the UAVs must also change
# altitude while crossing the course. All centers/scales use AirSim NED.
BLOCK_COURSE = (
    (5.0, -5.0, -2.0, (2.0, 3.0, 4.0)),  # low ground block
    (5.0, 0.0, -4.0, (2.0, 3.0, 8.0)),   # tall ground block
    (5.0, 5.0, -3.0, (2.0, 3.0, 6.0)),   # medium ground block
    # Floating blocks occupy the two passages at different heights.
    (8.0, -2.5, -5.5, (2.0, 1.5, 2.0)),
    (8.0, 2.5, -7.0, (2.0, 1.5, 2.0)),
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


def _yaw_quaternion(yaw: float) -> List[float]:
    return [float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))]


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
            camera_position = _vector3(response_position) if response_position is not None else state["position"]
            camera_orientation = _quaternion_matrix(response_orientation) if response_orientation is not None else np.eye(3)
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
                "position_frame": "world_ned",
                "point_frame": "camera_local_ned",
                "vehicle_position": np.asarray(state["position"], dtype=float).tolist(),
            }
            obstacles, diagnostics = detector.detect_with_diagnostics(
                points, state["position"], capture_time, "depth_" + name, False, ground_z=0.0
            )
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
        obstacles, diagnostics = detector.detect_with_diagnostics(
            points, vehicle_position, capture_time, "lidar_" + name, True, ground_z=0.0
        )
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


def _pose(airsim: Any, position: np.ndarray) -> Any:
    return airsim.Pose(airsim.Vector3r(*position.tolist()), airsim.to_quaternion(0, 0, 0))


def _spawn_block_course(facade: AirSimFacade, airsim: Any) -> Tuple[List[str], List[Dict[str, Any]]]:
    names = []
    truth = []
    # Cleanup is limited to this prefix so user-created scene objects survive.
    for index, (x, y, z, dimensions) in enumerate(BLOCK_COURSE):
        name = "distributed_cbf_block_{}".format(index)
        pose = _pose(airsim, np.array([x, y, z]))
        scale = airsim.Vector3r(*dimensions)
        try:
            if facade.spawn_object(name, "1M_Cube_Chamfer", pose, scale):
                names.append(name)
                truth.append({"id": name, "shape": "box", "center": [x, y, z], "dimensions": list(dimensions)})
        except Exception:
            pass
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
    parser.add_argument("--camera-height", type=float, default=30.0, help="top-down external camera height above the NED origin")
    parser.add_argument("--camera-x", type=float, default=6.0)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--no-top-down-camera", action="store_true")
    parser.add_argument("--animation-fps", type=float, default=None, help="post-run MP4 frame rate; defaults to 1/dt")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--sensor-rate", type=float, default=2.5, help="per-agent obstacle perception rate")
    parser.add_argument("--sensor-stale-after", type=float, default=None, help="maximum cached sensor age; defaults to one sensor period plus 50 ms")
    parser.add_argument("--top-n-obstacles", type=int, default=5)
    parser.add_argument("--uncertainty-radius", type=float, default=0.5)
    parser.add_argument("--uav-radius", type=float, default=1.0)
    parser.add_argument("--ugv-radius", type=float, default=1.25)
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
    parser.add_argument("--debug-dir", default=os.path.join(current_dir, "debug_runs"))
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--unreal-editor-path", default=AirSimLaunchConfig().unreal_editor_path)
    return parser.parse_args(argv)


def main(argv: List[str] = None) -> int:
    args = parse_args(argv)
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
        camera_director_position=None if args.no_top_down_camera else (args.camera_x, args.camera_y, -abs(args.camera_height)),
    )
    launcher = AirSimLauncher(launch_config)
    facade = AirSimFacade(launch_config)
    blocks: List[str] = []
    true_obstacles: List[Dict[str, Any]] = []
    names = ["Drone1", "Drone2", "SimpleFlight", "Drone4", "Drone5", "Husky1", "Husky2", "Husky3"]
    types = {name: "drone" for name in names[:5]}
    types.update({name: "ugv" for name in names[5:]})
    formation = FormationController(FormationConfig(uav_altitude=args.uav_altitude))
    cbf_config = CBFConfig(
        method=args.cbf_method,
        uncertainty_radius=args.uncertainty_radius,
        uav_radius=args.uav_radius,
        ugv_radius=args.ugv_radius,
        k1=args.k1,
        k2=args.k2,
        alpha=args.alpha,
    )
    vehicle_radii = {
        name: cbf_config.uav_radius if types[name] == "drone" else cbf_config.ugv_radius
        for name in names
    }
    agents = {name: Agent(name, types[name], cbf_config) for name in names}
    detector = ObstacleDetector(PerceptionConfig(top_n=args.top_n_obstacles))
    goal = np.array([16.0, 0.0, -1.0])
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
        if args.launch_mode == "existing" and not args.no_top_down_camera:
            print("Warning: --launch-mode existing cannot apply the top-down CameraDirector override; "
                  "configure it before launching Unreal.", flush=True)

        leader_position = np.array([0.0, 0.0, -1.0])
        initial_positions = {
            name: leader_position + formation.config.slots[name]
            for name in names
        }
        for name in names:
            initial_pose = _pose(facade.airsim, initial_positions[name])
            facade.spawn_vehicle(name, types[name], initial_pose)
            # Settings-defined actors may already exist, so spawning alone is
            # not enough to establish the requested formation.
            facade.set_vehicle_pose(name, initial_pose)
            facade.enable(name, types[name], True)

        if args.map_name == "flyingcpp" and not args.no_spawn_obstacles:
            # Create the course before starting vehicle motion so obstacle
            # creation is not perceived as a delayed post-takeoff stage.
            blocks, true_obstacles = _spawn_block_course(facade, facade.airsim)

        takeoff_futures = [facade.multirotor.takeoffAsync(vehicle_name=name) for name in names[:5]]
        for future in takeoff_futures:
            future.join()

        altitude_futures = []
        for name in names[:5]:
            target = initial_positions[name].copy()
            target[2] = args.uav_altitude
            altitude_futures.append(
                facade.multirotor.moveToPositionAsync(*target.tolist(), velocity=3.0, vehicle_name=name)
            )
        for future in altitude_futures:
            future.join()

        if args.timing_mode == "stepped":
            facade.pause(True)

        sensor_order = list(names)
        sensor_cursor = 0
        latest_obstacles: Dict[str, List[ObstacleProxy]] = {name: [] for name in names}
        latest_sensor_views: Dict[str, Dict[str, Any]] = {name: {} for name in names}
        last_perception: Dict[str, float] = {}
        sensor_period = 1.0 / max(args.sensor_rate, 1e-3)
        if args.sensor_stale_after is not None:
            detector.config.stale_after = args.sensor_stale_after
        else:
            detector.config.stale_after = max(detector.config.stale_after, sensor_period + args.dt * 0.5)
        # Warm all sensors before starting the real-time deadline clock. The
        # first depth frame is intentionally expensive and must not become a
        # control-cycle deadline miss.
        warmup_states = {name: facade.state(name) for name in names}
        warmup_time = time.time()
        for name in names:
            obstacles, sensor_valid, sensor_view, trace = _capture_obstacles(facade, detector, name, types[name], warmup_states[name], warmup_time, camera_fovs)
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
                    if types[name] == "ugv" and args.cbf_method == "mestres":
                        nominal = formation.nominal_unicycle_control(estimated_states[name], estimated_states, goal)
                    else:
                        nominal = formation.nominal_control(estimated_states[name], estimated_states, goal)
                    agent.set_nominal_control(nominal)

                capture_count = max(1, int(np.ceil(len(names) * args.dt / sensor_period)))
                capture_names = [sensor_order[(sensor_cursor + index) % len(sensor_order)] for index in range(capture_count)]
                sensor_cursor = (sensor_cursor + capture_count) % len(sensor_order)
                for name in capture_names:
                    obstacles, sensor_valid, sensor_view, trace = _capture_obstacles(facade, detector, name, types[name], raw_states[name], now, camera_fovs)
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
                    age = max(0.0, now - last_perception.get(name, -float("inf")))
                    sensor_valid = bool(age <= detector.config.stale_after)
                    age_margin = 0.0
                    if sensor_valid:
                        speed_limit = cbf_config.uav_velocity_limit if types[name] == "drone" else cbf_config.ugv_speed_limit
                        acceleration_limit = cbf_config.uav_acceleration_limit if types[name] == "drone" else cbf_config.ugv_acceleration_limit
                        age_margin = speed_limit * age + 0.5 * acceleration_limit * age * age
                    obstacles = [ObstacleProxy(item.obstacle_id, item.center, item.radius + age_margin, item.source, item.timestamp, item.point_count, item.is_planar) for item in latest_obstacles[name]]
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
                    elif args.cbf_method == "mestres":
                        facade.command_ugv(name, command[0], command[1] / cbf_config.ugv_yaw_rate_limit, args.dt)
                    else:
                        desired_velocity = states[name].velocity[:2] + args.dt * command[:2]
                        speed = float(np.linalg.norm(desired_velocity))
                        heading = float(np.arctan2(desired_velocity[1], desired_velocity[0])) if speed > 1e-6 else states[name].yaw
                        steering = np.clip((heading - states[name].yaw) / np.pi, -1.0, 1.0)
                        facade.command_ugv(name, speed, steering, args.dt)
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
                    "vehicle_types": types,
                    "vehicle_radii": vehicle_radii,
                    "formation": formation.metrics(estimated_states),
                    "states": {name: {"position": states[name].position.tolist(), "velocity": states[name].velocity.tolist()} for name in names},
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
