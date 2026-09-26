#!/usr/bin/env python3
"""ROS wrapper around the existing asynchronous Python target observer."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Dict

import numpy as np
import rclpy
from hercules_interfaces.msg import (
    GroundTruthState,
    TargetMeasurement as RosTargetMeasurement,
    TargetObservationDiagnostics,
    TrackingEpoch,
)
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException


UAV_AGENTS = ["Drone1", "Drone2", "SimpleFlight"]
UGV_AGENTS = ["Husky1", "Husky2", "Husky3"]
AGENTS = UAV_AGENTS + UGV_AGENTS
CAMERAS = {name: "target_bottom" for name in UAV_AGENTS} | {
    name: "front_center" for name in UGV_AGENTS
}


def _source_modules():
    roots = [Path(os.environ.get("HERCULES_REPO_ROOT", "/workspaces/hercules"))]
    roots.append(Path(__file__).resolve().parents[4])
    for root in roots:
        mission = str(root / "PythonClient" / "distributed_mission")
        client = str(root / "PythonClient")
        for path in (client, mission):
            if path not in sys.path:
                sys.path.insert(0, path)
    from modules.target_observation import (  # pylint: disable=import-outside-toplevel
        MissionTimeMapper,
        TargetObservationWorker,
        truth_target_measurement,
    )
    return MissionTimeMapper, TargetObservationWorker, truth_target_measurement


def _airsim_module():
    import hercules_cosysairsim as airsim  # pylint: disable=import-outside-toplevel
    return airsim


def _seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _time_message(value: float):
    from builtin_interfaces.msg import Time  # pylint: disable=import-outside-toplevel
    result = Time()
    result.sec = int(np.floor(value))
    result.nanosec = int(round((value - result.sec) * 1e9))
    if result.nanosec == 1_000_000_000:
        result.sec += 1
        result.nanosec = 0
    return result


class TargetObserverNode(Node):
    def __init__(self) -> None:
        super().__init__("target_observer")
        self.declare_parameter("observation_source", "camera")
        self.declare_parameter("target_id", "Target1")
        self.declare_parameter("host_ip", "127.0.0.1")
        self.declare_parameter("rpc_port", 41451)
        self.declare_parameter("drone_port", int(self.get_parameter("rpc_port").value))
        self.declare_parameter("ugv_port", 41452)
        self.declare_parameter("target_sensing_range", 100.0)
        self.declare_parameter("tracking_measurement_std", 0.25)
        self.declare_parameter("tracking_rate", 4.0)
        self.declare_parameter("truth_seed", 7)
        self.declare_parameter("truth_rng_mode", "seeded")
        self.source = str(self.get_parameter("observation_source").value)
        self.target_id = str(self.get_parameter("target_id").value)
        if self.source not in ("truth", "camera"):
            raise ValueError("observation_source must be truth or camera")
        self.sensing_range = float(self.get_parameter("target_sensing_range").value)
        self.measurement_std = float(self.get_parameter("tracking_measurement_std").value)
        self.host_ip = str(self.get_parameter("host_ip").value or "127.0.0.1")
        self.drone_port = int(self.get_parameter("drone_port").value)
        self.ugv_port = int(self.get_parameter("ugv_port").value)
        self.truth_rng_mode = str(self.get_parameter("truth_rng_mode").value).strip().lower()
        if self.truth_rng_mode in ("deterministic", "seeded"):
            self.truth_rng_mode = "seeded"
        elif self.truth_rng_mode in ("python_global", "python-global", "global"):
            self.truth_rng_mode = "python_global"
        elif self.truth_rng_mode in ("random", "nondeterministic"):
            self.truth_rng_mode = "random"
        else:
            raise ValueError("truth_rng_mode must be seeded, python_global, or random")
        self.measurement_publishers = {
            agent: self.create_publisher(
                RosTargetMeasurement,
                f"/hercules_tracking/{agent}/{self.target_id}/measurement",
                20,
            )
            for agent in AGENTS
        }
        self.diagnostics_publisher = self.create_publisher(
            TargetObservationDiagnostics, "/hercules_tracking/observation_diagnostics", 10
        )
        self.states: Dict[str, GroundTruthState] = {}
        self.state_subscriptions = [
            self.create_subscription(
                GroundTruthState,
                f"/hercules_mission/ground_truth/{name}",
                lambda message, name=name: self.states.__setitem__(name, message),
                10,
            )
            for name in AGENTS + [self.target_id]
        ]
        self.epoch_subscription = self.create_subscription(
            TrackingEpoch, "/hercules_tracking/epoch", self.on_epoch, 20
        )
        mapper_type, worker_type, truth_function = _source_modules()
        self.mapper = mapper_type()
        self.truth_function = truth_function
        seed = int(self.get_parameter("truth_seed").value)
        if self.truth_rng_mode == "python_global":
            shared_rng = np.random.default_rng(seed)
            self.rng = {name: shared_rng for name in AGENTS}
        else:
            self.rng = {
                name: np.random.default_rng(seed + 1009 * index)
                if self.truth_rng_mode == "seeded" else np.random.default_rng()
                for index, name in enumerate(AGENTS)
            }
        self.worker = None
        self.published_capture_ids: Dict[str, str] = {}
        self.truth_capture_count = 0
        self.truth_visible_count = 0
        self.truth_invalid_count = 0
        self.started_wall = time.time()
        if self.source == "camera":
            self.worker = worker_type(
                _airsim_module(),
                self.drone_port,
                CAMERAS,
                host=self.host_ip,
                endpoint_ports={
                    **{name: self.drone_port for name in UAV_AGENTS},
                    **{name: self.ugv_port for name in UGV_AGENTS},
                },
                target_id=self.target_id,
                target_actor_pattern=self.target_id + "*",
                sensing_range=self.sensing_range,
                measurement_std=self.measurement_std,
                target_radius=0.0,
                rate_hz=float(self.get_parameter("tracking_rate").value),
                detection_filter_patterns=[self.target_id + "*"] + [agent + "*" for agent in AGENTS],
            )
            self.worker.start()
            self.poll_timer = self.create_timer(0.01, self.publish_camera_samples)
        self.diagnostics_timer = self.create_timer(1.0, self.publish_diagnostics)
        self.get_logger().info(
            f"target observation source={self.source}; rng={self.truth_rng_mode}; "
            f"host={self.host_ip}; cameras={CAMERAS if self.source == 'camera' else 'truth'}"
        )

    def destroy_node(self):
        if self.worker is not None:
            self.worker.stop()
        return super().destroy_node()

    def on_epoch(self, epoch: TrackingEpoch) -> None:
        if epoch.target_id != self.target_id:
            return
        mission_time = _seconds(epoch.stamp)
        self.mapper.update(time.time(), mission_time)
        if self.source != "truth":
            return
        target = self.states.get(self.target_id)
        if target is None:
            return
        target_position = np.asarray(target.position, dtype=float)
        for agent in AGENTS:
            state = self.states.get(agent)
            if state is None:
                continue
            capture_id = f"target_truth_{epoch.epoch_id:06d}_{agent}"
            value = self.truth_function(
                self.target_id,
                target_position,
                np.asarray(state.position, dtype=float),
                mission_time,
                self.measurement_std,
                self.sensing_range,
                self.rng[agent],
                capture_id,
            )
            if value is None:
                self.publish_invalid(agent, capture_id, mission_time, "truth")
                self.truth_invalid_count += 1
            else:
                self.publish_value(agent, value, mission_time, time.time())
                self.truth_visible_count += 1
            self.truth_capture_count += 1

    def publish_value(self, agent, value, mission_time: float, capture_wall_time: float) -> None:
        message = RosTargetMeasurement()
        message.target_id = self.target_id
        message.stamp = _time_message(mission_time)
        message.capture_stamp = _time_message(float(capture_wall_time))
        message.receipt_stamp = self.get_clock().now().to_msg()
        message.position = [float(value.position[0]), float(value.position[1])]
        message.covariance = np.asarray(value.covariance, dtype=float).reshape(-1).tolist()
        message.valid = bool(value.valid)
        message.source_id = str(agent)
        message.capture_id = str(value.capture_id or "")
        message.sensor_id = str(value.sensor or "")
        message.visible = bool(value.visible)
        self.measurement_publishers[agent].publish(message)

    def publish_invalid(self, agent: str, capture_id: str, mission_time: float, sensor: str) -> None:
        message = RosTargetMeasurement()
        message.target_id = self.target_id
        message.stamp = _time_message(mission_time)
        message.capture_stamp = _time_message(time.time())
        message.receipt_stamp = self.get_clock().now().to_msg()
        message.position = [0.0, 0.0]
        message.covariance = [self.measurement_std ** 2, 0.0, 0.0, self.measurement_std ** 2]
        message.valid = False
        message.source_id = agent
        message.capture_id = capture_id
        message.sensor_id = sensor
        message.visible = False
        self.measurement_publishers[agent].publish(message)

    def publish_camera_samples(self) -> None:
        if self.worker is None or not self.mapper.samples:
            return
        for agent, value in self.worker.snapshot().items():
            capture_id = str(value.capture_id or "")
            if not capture_id or self.published_capture_ids.get(agent) == capture_id:
                continue
            self.published_capture_ids[agent] = capture_id
            mission_time = self.mapper.mission_timestamp(float(value.timestamp))
            self.publish_value(agent, value, mission_time, float(value.timestamp))

    def publish_diagnostics(self) -> None:
        values = self.worker.diagnostics() if self.worker is not None else {
            "captures": self.truth_capture_count,
            "visible": self.truth_visible_count,
            "invalid": self.truth_invalid_count,
            "errors": 0,
            "mean_capture_rate_hz": self.truth_capture_count / max(1e-9, time.time() - self.started_wall),
            "capture_interval_p95_sec": 0.0,
            "capture_interval_max_sec": 0.0,
            "mean_rpc_duration_sec": 0.0,
            "max_rpc_duration_sec": 0.0,
        }
        message = TargetObservationDiagnostics()
        message.observation_source = self.source
        message.stamp = self.get_clock().now().to_msg()
        message.camera_captures = int(values["captures"])
        message.visible_detections = int(values["visible"])
        message.invalid_detections = int(values["invalid"])
        message.rpc_errors = int(values["errors"])
        message.mean_capture_rate_hz = float(values["mean_capture_rate_hz"])
        message.capture_interval_p95_sec = float(values["capture_interval_p95_sec"])
        message.capture_interval_max_sec = float(values["capture_interval_max_sec"])
        message.mean_image_rpc_duration_sec = float(values["mean_rpc_duration_sec"])
        message.max_image_rpc_duration_sec = float(values["max_rpc_duration_sec"])
        self.diagnostics_publisher.publish(message)


def main() -> None:
    rclpy.init()
    node = TargetObserverNode()
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
