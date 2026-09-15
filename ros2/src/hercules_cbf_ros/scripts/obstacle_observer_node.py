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
import time
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from hercules_interfaces.msg import GroundTruthState, ObstacleProxy, ObstacleProxyArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Header


AGENTS = ["Drone1", "Drone2", "SimpleFlight", "Drone4", "Drone5",
          "Husky1", "Husky2", "Husky3"]


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
    from orchestrator import _capture_obstacles  # pylint: disable=import-outside-toplevel
    from simulation.airsim_runtime import AirSimFacade, AirSimLaunchConfig  # pylint: disable=import-outside-toplevel
    return ObstacleDetector, PerceptionConfig, _capture_obstacles, AirSimFacade, AirSimLaunchConfig


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


class ObstacleObserverNode(Node):
    def __init__(self) -> None:
        super().__init__("obstacle_observer")
        self.declare_parameter("source", "perception")
        self.declare_parameter("sensor_rate", 2.5)
        self.declare_parameter("stale_after", 1.3)
        self.declare_parameter("rpc_port", 41451)
        self.declare_parameter("map_name", "rural_australia")
        self.source = str(self.get_parameter("source").value)
        if self.source not in ("perception", "truth", "none"):
            raise ValueError("source must be perception, truth, or none")
        self.sensor_rate = max(0.1, float(self.get_parameter("sensor_rate").value))
        # ``Node.publishers`` is an rclpy-managed read-only property; keep
        # the observer's handles under a private name instead.
        self.proxy_publishers = {agent: self.create_publisher(
            ObstacleProxyArray, f"/hercules_mission/obstacles/{agent}", 10) for agent in AGENTS}
        self.states: Dict[str, GroundTruthState] = {}
        # ``Node.subscriptions`` is also managed by rclpy and read-only.
        self.state_subscriptions = [self.create_subscription(
            GroundTruthState, f"/hercules_mission/ground_truth/{agent}",
            lambda message, agent=agent: self.states.__setitem__(agent, message), 10)
            for agent in AGENTS]
        # ``Node.executor`` is an rclpy-managed read-only property.
        self.worker_pool = ThreadPoolExecutor(max_workers=len(AGENTS), thread_name_prefix="cbf-perception")
        self.futures = {}
        detector_type, config_type, capture_type, facade_type, launch_type = _source_modules()
        self.capture = capture_type
        # AirSim RPC clients are mutable and are not safe to share across the
        # worker pool.  Keep the constructor types/config here and create one
        # facade (and its clients) inside each capture worker.
        self.facade_type = facade_type
        self.launch_type = launch_type
        self.rpc_port = int(self.get_parameter("rpc_port").value)
        self.map_name = str(self.get_parameter("map_name").value).strip().lower()
        stale_after = float(self.get_parameter("stale_after").value)

        # Keep the ROS perception path on the same tuned detector profiles as
        # the Python orchestrator.  The RuralAustralia point cloud is sparse
        # foliage/terrain data and needs different clustering and proxy
        # fitting thresholds from the dense FlyingCPP scene.
        def detector_config(agent: str):
            if self.map_name == "rural_australia":
                if agent.startswith("Drone") or agent == "SimpleFlight":
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
            for agent in AGENTS
        }
        # A capture future is serialized per agent, so one facade per agent is
        # safe to reuse.  Constructing a new AirSim facade on every sensor
        # tick leaks RPC client resources in the AirSim Python bindings and
        # eventually exhausts the host when a launch is left unattended.
        self.facades = {}
        self.timer = self.create_timer(1.0 / self.sensor_rate, self.schedule)

    def destroy_node(self):
        self.timer.cancel()
        self.worker_pool.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()

    def schedule(self) -> None:
        if self.source == "none":
            now = time.time()
            for agent in AGENTS:
                self.proxy_publishers[agent].publish(snapshot_message(
                    agent, f"empty_{int(now * 1e6)}_{agent}", "none", True, True, now, []))
            return
        for agent, state in list(self.states.items()):
            if agent not in self.proxy_publishers or agent in self.futures and not self.futures[agent].done():
                continue
            self.futures[agent] = self.worker_pool.submit(self.capture_one, agent, state)
            self.futures[agent].add_done_callback(lambda future, agent=agent: self.publish_result(agent, future))

    def capture_one(self, agent: str, message: GroundTruthState) -> Tuple[ObstacleProxyArray, bool]:
        try:
            state = {
                "position": np.asarray(message.position, dtype=float),
                "velocity": np.asarray(message.velocity, dtype=float),
                "yaw": float(message.yaw),
                "actor_position": np.asarray(message.position, dtype=float),
                "kinematics_position": np.asarray(message.position, dtype=float),
            }
            facade = self.facades.get(agent)
            if facade is None:
                facade = self.facade_type(self.launch_type(
                    launch_mode="existing", multirotor_port=self.rpc_port))
                facade.connect()
                self.facades[agent] = facade
            proxies, valid, _, trace = self.capture(
                facade, self.detectors[agent], agent,
                "drone" if message.vehicle_type == "drone" else "ugv",
                state, time.time(), {})
            capture_id = str(trace.get("capture_id", "")) or f"capture_{time.time_ns()}_{agent}"
            return snapshot_message(agent, capture_id, self.source, True, valid, time.time(), proxies), True
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
