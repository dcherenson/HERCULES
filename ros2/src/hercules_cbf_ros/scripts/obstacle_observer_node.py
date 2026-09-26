#!/usr/bin/env python3
"""Asynchronous AirSim obstacle proxy observer for the Phase 3 replay path.

The detector and capture helpers are imported from the Python mission oracle;
the ROS node only schedules immutable snapshots and never publishes commands.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PointStamped
from hercules_interfaces.msg import GroundTruthState, ObstacleProxy, ObstacleProxyArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header


DEFAULT_UAV_AGENTS = ("Drone1", "Drone2", "SimpleFlight")
DEFAULT_UGV_AGENTS = ("Husky1", "Husky2", "Husky3")
DEFAULT_CONTROLLED_AGENTS = DEFAULT_UAV_AGENTS + DEFAULT_UGV_AGENTS
AGENTS = list(DEFAULT_CONTROLLED_AGENTS)


def _configured_agents(value, parameter_name: str, default) -> Tuple[str, ...]:
    """Validate a configured vehicle-name array while preserving its order."""
    values = default if value is None else value
    if isinstance(values, str):
        raise ValueError(f"{parameter_name} must be a list of vehicle names")
    names = tuple(str(agent).strip() for agent in values)
    if not names or any(not agent for agent in names):
        raise ValueError(f"{parameter_name} must contain at least one nonempty name")
    if len(set(names)) != len(names):
        raise ValueError(f"{parameter_name} must not contain duplicate names")
    return names


def _source_modules():
    roots = [Path(os.environ.get("HERCULES_REPO_ROOT", "/workspaces/hercules")),
             Path(__file__).resolve().parents[4]]
    for root in roots:
        for path in (root / "PythonClient", root / "PythonClient" / "distributed_mission"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
    from modules.obstacle_detection import (  # pylint: disable=import-outside-toplevel
        ObstacleDetector, PerceptionConfig,
    )
    from orchestrator import (  # pylint: disable=import-outside-toplevel
        _capture_obstacles,
        filter_agent_body_obstacle_proxies,
    )
    from simulation.airsim_runtime import AirSimFacade, AirSimLaunchConfig  # pylint: disable=import-outside-toplevel
    return (ObstacleDetector, PerceptionConfig, _capture_obstacles,
            filter_agent_body_obstacle_proxies, AirSimFacade, AirSimLaunchConfig)


def _time_message(value: float) -> Time:
    result = Time()
    result.sec = int(np.floor(value))
    result.nanosec = int(round((value - result.sec) * 1e9))
    if result.nanosec >= 1_000_000_000:
        result.sec += 1
        result.nanosec -= 1_000_000_000
    return result


def proxy_message(proxy) -> ObstacleProxy:
    """Convert an oracle proxy without changing order or geometry."""
    result = ObstacleProxy()
    result.proxy_id = str(proxy.obstacle_id)
    result.source = str(proxy.source)
    result.center = [float(value) for value in np.asarray(proxy.center, dtype=float).reshape(3)]
    result.radius = max(0.0, float(proxy.radius))
    velocity = getattr(proxy, "velocity", None)
    result.has_velocity = velocity is not None
    if velocity is not None:
        result.velocity = [float(value) for value in np.asarray(velocity, dtype=float).reshape(3)]
    result.stamp = _time_message(float(getattr(proxy, "timestamp", 0.0)))
    result.point_count = int(getattr(proxy, "point_count", 0))
    result.is_planar = bool(getattr(proxy, "is_planar", False))
    return result


def snapshot_message(agent_id: str, capture_id: str, source: str,
                     success: bool, valid: bool, capture_stamp: float,
                     proxies, clock_domain: str = "wall") -> ObstacleProxyArray:
    """Build the transport message used by the cache and replay tests."""
    result = ObstacleProxyArray()
    result.header = Header(frame_id="airsim_world_ned", stamp=_time_message(time.time()))
    result.agent_id = agent_id
    result.capture_id = capture_id
    result.acquisition_success = bool(success)
    result.valid = bool(valid)
    result.capture_stamp = _time_message(float(capture_stamp))
    result.clock_domain = clock_domain
    result.source = source
    result.proxies = [proxy_message(proxy) for proxy in proxies]
    return result


def _finite_vector(values, size: int) -> Optional[np.ndarray]:
    candidate = np.asarray(values, dtype=float).reshape(-1)
    if candidate.size != size or not np.all(np.isfinite(candidate)):
        return None
    return candidate.copy()


def _kinematics_from_state(message: GroundTruthState):
    """Build the tiny AirSim kinematics shape expected by the oracle.

    The ROS state adapter is the canonical source of orientation.  The
    detector helper only needs ``kinematics.orientation`` for LiDAR pose
    composition, so avoid manufacturing a second frame conversion here.
    """
    orientation = _finite_vector(message.orientation, 4)
    if orientation is None or np.linalg.norm(orientation) < 1e-9:
        orientation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        orientation /= np.linalg.norm(orientation)
    quaternion = SimpleNamespace(
        w_val=float(orientation[0]), x_val=float(orientation[1]),
        y_val=float(orientation[2]), z_val=float(orientation[3]),
    )
    return SimpleNamespace(orientation=quaternion)


def _capture_state(facade, agent: str, message: GroundTruthState,
                   origin: np.ndarray) -> Dict[str, object]:
    """Combine canonical ROS pose data with AirSim frame-origin metadata."""
    position = _finite_vector(message.position, 3)
    velocity = _finite_vector(message.velocity, 3)
    if position is None or velocity is None:
        raise ValueError("canonical state contains a nonfinite position or velocity")
    origin = _finite_vector(origin, 3)
    if origin is None:
        raise ValueError("calibrated origin is unavailable or nonfinite")
    state: Dict[str, object] = {
        "position": position,
        "velocity": velocity,
        "yaw": float(message.yaw),
        "actor_position": position.copy(),
        "kinematics_position": position - origin,
        "kinematics": _kinematics_from_state(message),
    }

    # The canonical ROS state is already in world NED.  The Python capture
    # helper expects actor_position in that frame and kinematics_position in
    # the vehicle's local start frame, whose translation is the calibrated
    # origin.  Keep this deterministic rather than mixing in a second RPC
    # read that can race the state topic.
    return state


def _capture_worker_pool() -> ThreadPoolExecutor:
    """Create the single thread that owns all AirSim perception clients.

    The msgpack/Tornado client used by AirSim is thread-affine.  A lock around
    calls is insufficient when a facade is constructed on one pool worker and
    later reused on another, so client construction and every capture must run
    on the same dedicated worker.
    """
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="cbf-perception")


class ObstacleObserverNode(Node):
    def __init__(self) -> None:
        super().__init__("obstacle_observer")
        self.declare_parameter("source", "perception")
        self.declare_parameter("sensor_rate", 2.5)
        self.declare_parameter("stale_after", 1.3)
        self.declare_parameter("uav_agents", list(DEFAULT_UAV_AGENTS))
        self.declare_parameter("ugv_agents", list(DEFAULT_UGV_AGENTS))
        self.declare_parameter("host_ip", "127.0.0.1")
        self.declare_parameter("rpc_port", 41451)
        self.declare_parameter("car_port", 41452)
        self.declare_parameter("uav_radius", 1.0)
        self.declare_parameter("ugv_radius", 1.25)
        self.declare_parameter("body_exclusion_margin", 0.5)
        self.declare_parameter("map_name", "rural_australia")
        self.uav_agents = _configured_agents(
            self.get_parameter("uav_agents").value, "uav_agents", DEFAULT_UAV_AGENTS)
        self.ugv_agents = _configured_agents(
            self.get_parameter("ugv_agents").value, "ugv_agents", DEFAULT_UGV_AGENTS)
        if set(self.uav_agents).intersection(self.ugv_agents):
            raise ValueError("uav_agents and ugv_agents must be disjoint")
        self.agents = self.uav_agents + self.ugv_agents
        self.source = str(self.get_parameter("source").value)
        if self.source not in ("perception", "truth", "none"):
            raise ValueError("source must be perception, truth, or none")
        self.sensor_rate = max(0.1, float(self.get_parameter("sensor_rate").value))
        # ``Node.publishers`` is an rclpy-managed read-only property; keep
        # the observer's handles under a private name instead.
        self.proxy_publishers = {agent: self.create_publisher(
            ObstacleProxyArray, f"/hercules_mission/obstacles/{agent}", 10) for agent in self.agents}
        self.states: Dict[str, GroundTruthState] = {}
        self.origins: Dict[str, np.ndarray] = {}
        # ``Node.subscriptions`` is also managed by rclpy and read-only.
        self.state_subscriptions = [self.create_subscription(
            GroundTruthState, f"/hercules_mission/ground_truth/{agent}",
            lambda message, agent=agent: self.states.__setitem__(agent, message), 10)
            for agent in self.agents]
        origin_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.origin_subscriptions = [self.create_subscription(
            PointStamped, f"/hercules_mission/calibrated_origin/{agent}",
            lambda message, agent=agent: self.origins.__setitem__(
                agent, np.asarray([message.point.x, message.point.y, message.point.z], dtype=float)),
            origin_qos)
            for agent in self.agents]
        # ``Node.executor`` is an rclpy-managed read-only property.  Keep all
        # AirSim client construction and use on one dedicated thread: the
        # msgpack/Tornado client is thread-affine, so a lock alone does not
        # make a facade safe to reuse across a multi-worker pool.
        self.worker_pool = _capture_worker_pool()
        self.capture_lock = threading.Lock()
        self.futures = {}
        (detector_type, config_type, capture_type, filter_type,
         facade_type, launch_type) = _source_modules()
        self.capture = capture_type
        self.filter_agent_body_obstacle_proxies = filter_type
        # AirSim RPC clients are mutable and are not safe to share across the
        # worker pool.  Keep the constructor types/config here and create one
        # facade (and its clients) inside each capture worker.
        self.facade_type = facade_type
        self.launch_type = launch_type
        self.host_ip = str(self.get_parameter("host_ip").value or "127.0.0.1")
        self.rpc_port = int(self.get_parameter("rpc_port").value)
        self.car_port = int(self.get_parameter("car_port").value)
        self.uav_radius = float(self.get_parameter("uav_radius").value)
        self.ugv_radius = float(self.get_parameter("ugv_radius").value)
        self.body_exclusion_margin = float(self.get_parameter("body_exclusion_margin").value)
        if (not np.isfinite(self.uav_radius) or not np.isfinite(self.ugv_radius) or
                not np.isfinite(self.body_exclusion_margin) or
                self.uav_radius < 0.0 or self.ugv_radius < 0.0 or
                self.body_exclusion_margin < 0.0):
            raise ValueError("body radii and exclusion margin must be finite and nonnegative")
        self.map_name = str(self.get_parameter("map_name").value).strip().lower()
        stale_after = float(self.get_parameter("stale_after").value)

        # Keep the ROS perception path on the same tuned detector profiles as
        # the Python orchestrator.  The RuralAustralia point cloud is sparse
        # foliage/terrain data and needs different clustering and proxy
        # fitting thresholds from the dense FlyingCPP scene.
        def detector_config(agent: str):
            if self.map_name == "rural_australia":
                if agent in self.uav_agents:
                    return config_type(
                        top_n=5,
                        cluster_min_samples=20,
                        min_proxy_points=32,
                        fit_padding=0.25,
                        max_proxy_radius=1.0,
                        stale_after=stale_after,
                    )
                return config_type(
                    top_n=5,
                    planar_surface_offset=0.0,
                    cluster_eps=0.85,
                    cluster_min_samples=3,
                    min_proxy_points=32,
                    max_proxy_radius=1.0,
                    fit_padding=0.35,
                    ground_band=0.15,
                    planar_use_nearest_surface=True,
                    rank_by_surface_distance=True,
                    stale_after=stale_after,
                )
            return config_type(stale_after=stale_after)
        self.detectors = {
            agent: detector_type(detector_config(agent))
            for agent in self.agents
        }
        # A capture future is serialized per agent, and all futures run on the
        # dedicated worker, so one facade per agent is safe to reuse.  Creating
        # a new AirSim facade on every sensor tick leaks RPC client resources in
        # the AirSim Python bindings and eventually exhausts the host when a
        # launch is left unattended.
        self.facades = {}
        # The camera FOV is static for a mission.  Keep the cache at node scope
        # so each UAV camera makes at most one simGetCameraInfo RPC.
        self.camera_fovs: Dict[str, float] = {}
        self.timer = self.create_timer(1.0 / self.sensor_rate, self.schedule)

    def destroy_node(self):
        self.timer.cancel()
        self.worker_pool.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()

    def schedule(self) -> None:
        if self.source in ("none", "truth"):
            now = time.time()
            for agent in self.agents:
                self.proxy_publishers[agent].publish(snapshot_message(
                    agent, f"empty_{int(now * 1e6)}_{agent}", self.source,
                    True, True, now, []))
            return
        for agent, state in list(self.states.items()):
            origin = self.origins.get(agent)
            if origin is None or origin.shape != (3,) or not np.all(np.isfinite(origin)):
                # Perception frames are undefined until the latched origin
                # arrives; do not emit a plausible-looking misframed sample.
                continue
            if agent not in self.proxy_publishers or agent in self.futures and not self.futures[agent].done():
                continue
            self.futures[agent] = self.worker_pool.submit(self.capture_one, agent, state)
            self.futures[agent].add_done_callback(lambda future, agent=agent: self.publish_result(agent, future))

    def capture_one(self, agent: str, message: GroundTruthState) -> Tuple[ObstacleProxyArray, bool]:
        try:
            with self.capture_lock:
                facade = self.facades.get(agent)
                if facade is None:
                    facade = self.facade_type(self.launch_type(
                        launch_mode="existing", host=self.host_ip,
                        multirotor_port=self.rpc_port, car_port=self.car_port))
                    facade.connect()
                    self.facades[agent] = facade
                state = _capture_state(facade, agent, message, self.origins.get(agent))
                proxies, valid, sensor_view, trace = self.capture(
                    facade, self.detectors[agent], agent,
                    "drone" if message.vehicle_type == "drone" else "ugv",
                    state, time.time(), self.camera_fovs)
            if valid:
                states = dict(self.states)
                vehicle_radii = {
                    name: self.uav_radius
                    if str(getattr(value, "vehicle_type", "")).lower() == "drone"
                    else self.ugv_radius
                    for name, value in states.items()
                }
                proxies, _ = self.filter_agent_body_obstacle_proxies(
                    proxies, states, vehicle_radii,
                    exclusion_margin=self.body_exclusion_margin,
                )
            capture_id = str(trace.get("capture_id", "")) or f"capture_{time.time_ns()}_{agent}"
            capture_stamp = float(sensor_view.get("capture_timestamp", time.time()))
            return snapshot_message(agent, capture_id, self.source, True, valid,
                                    capture_stamp, proxies), True
        except Exception as error:  # worker failures are transport data, not node failure
            self.get_logger().warning(f"obstacle capture failed for {agent}: {error}")
            now = time.time()
            return snapshot_message(agent, f"failed_{time.time_ns()}_{agent}", self.source,
                                    False, False, now, []), False

    def publish_result(self, agent: str, future) -> None:
        try:
            message, _ = future.result()
            self.proxy_publishers[agent].publish(message)
        except Exception as error:
            self.get_logger().warning(f"obstacle worker result failed for {agent}: {error}")


def main() -> None:
    rclpy.init()
    node = ObstacleObserverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
