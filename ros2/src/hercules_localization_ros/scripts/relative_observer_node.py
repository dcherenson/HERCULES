#!/usr/bin/env python3
"""Publish planar peer measurements for cooperative localization.

The default ``truth`` source is deterministic for CI/reproduction.  The
``camera`` source uses AirSim detection metadata for association and the
localization-front/front-center depth images for geometric measurements.
"""
from math import atan2, cos, hypot, pi, sin
import os
from pathlib import Path
import sys

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from hercules_interfaces.msg import GroundTruthState, PlanarRelativeMeasurement


UAV_AGENTS = ["Drone1", "Drone2", "SimpleFlight"]
UGV_AGENTS = ["Husky1", "Husky2", "Husky3"]
AGENTS = UAV_AGENTS + UGV_AGENTS


def _relative_worker_type():
    roots = [Path(os.environ.get("HERCULES_REPO_ROOT", "/workspaces/hercules")),
             Path(__file__).resolve().parents[4]]
    for root in roots:
        for path in (root / "PythonClient", root / "PythonClient" / "distributed_mission"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
    from modules.relative_observation import RelativeObservationWorker
    return RelativeObservationWorker


def wrap_angle(value: float) -> float:
    return (value + pi) % (2.0 * pi) - pi


class RelativeObserver(Node):
    def __init__(self) -> None:
        super().__init__("localization_relative_observer")
        self.declare_parameter("observation_source", "truth")
        self.declare_parameter("host_ip", "127.0.0.1")
        self.declare_parameter("drone_port", 41451)
        self.declare_parameter("ugv_port", 41452)
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("sensing_range", 100.0)
        self.declare_parameter("range_std", 0.25)
        self.declare_parameter("bearing_std_rad", pi / 180.0)
        # ``localization_front`` is the forward-facing UAV camera used by the
        # cooperative-localization geometry.  The legacy top-down
        # ``target_bottom`` camera is deliberately opt-in: it is useful for a
        # deployment that has no forward camera, but is not a peer-observation
        # default.
        self.declare_parameter("uav_camera", "localization_front")
        self.declare_parameter("uav_camera_fallback", "")
        self.rate_hz = max(1e-3, float(self.get_parameter("rate_hz").value))
        self.observation_source = str(self.get_parameter("observation_source").value).strip().lower()
        if self.observation_source not in ("truth", "camera"):
            raise ValueError("observation_source must be truth or camera")
        self.sensing_range = max(0.1, float(self.get_parameter("sensing_range").value))
        self.range_std = max(0.01, float(self.get_parameter("range_std").value))
        self.bearing_std = max(1e-5, float(self.get_parameter("bearing_std_rad").value))
        uav_camera = str(self.get_parameter("uav_camera").value or "localization_front").strip()
        uav_camera = uav_camera or "localization_front"
        uav_camera_fallback = str(self.get_parameter("uav_camera_fallback").value or "").strip()
        self.states = {}
        self.sequence = 0
        self.publisher = self.create_publisher(
            PlanarRelativeMeasurement, "/hercules_localization/relative", 100)
        self.worker = None
        self.published_capture_ids = set()
        if self.observation_source == "camera":
            RelativeObservationWorker = _relative_worker_type()
            try:
                import hercules_cosysairsim as airsim
                uav_cameras = uav_camera
                if uav_camera_fallback and uav_camera_fallback != uav_camera:
                    uav_cameras = (uav_camera, uav_camera_fallback)
                self.worker = RelativeObservationWorker(
                    airsim_module=airsim,
                    port=int(self.get_parameter("drone_port").value),
                    agent_cameras={agent: (uav_cameras
                                           if agent in UAV_AGENTS else "front_center")
                                   for agent in AGENTS},
                    agent_ids=AGENTS,
                    sensing_range=self.sensing_range,
                    range_std=self.range_std,
                    bearing_std_rad=self.bearing_std,
                    rate_hz=self.rate_hz,
                    host=str(self.get_parameter("host_ip").value or "127.0.0.1"),
                    endpoint_ports={**{agent: int(self.get_parameter("drone_port").value)
                                      for agent in UAV_AGENTS},
                                    **{agent: int(self.get_parameter("ugv_port").value)
                                       for agent in UGV_AGENTS}},
                )
                self.worker.start()
                self.camera_timer = self.create_timer(0.01, self.publish_camera_measurements)
            except ImportError as error:
                raise RuntimeError("camera localization requires hercules_cosysairsim") from error
        # ``Node.subscriptions`` is a read-only rclpy introspection property.
        # Keep explicit references under our own name so the subscriptions are
        # retained for the lifetime of this observer.
        self.state_subscriptions = [
            self.create_subscription(
                GroundTruthState,
                f"/hercules_mission/ground_truth/{agent}",
                lambda msg, agent=agent: self.states.__setitem__(agent, msg),
                10,
            ) for agent in AGENTS
        ]
        self.timer = self.create_timer(1.0 / self.rate_hz, self.publish_measurements)

    def destroy_node(self):
        if self.worker is not None:
            self.worker.stop()
        return super().destroy_node()

    def publish_measurements(self) -> None:
        if self.observation_source != "truth":
            return
        if len(self.states) < 2:
            return
        self.sequence += 1
        stamp = self.get_clock().now().to_msg()
        for observer in AGENTS:
            source = self.states.get(observer)
            if source is None or not source.valid:
                continue
            sx, sy = float(source.position[0]), float(source.position[1])
            yaw = float(source.yaw)
            for observed in AGENTS:
                if observed == observer:
                    continue
                target = self.states.get(observed)
                if target is None or not target.valid:
                    continue
                dx = float(target.position[0]) - sx
                dy = float(target.position[1]) - sy
                distance = hypot(dx, dy)
                if distance <= 1e-6 or distance > self.sensing_range:
                    continue
                body_x = cos(yaw) * dx + sin(yaw) * dy
                body_y = -sin(yaw) * dx + cos(yaw) * dy
                message = PlanarRelativeMeasurement()
                message.header.stamp = stamp
                message.header.frame_id = "airsim_body_ned"
                message.source_id = observer
                message.target_id = observed
                message.observer_id = observer
                message.observed_id = observed
                message.sensor_id = "localization_relative"
                message.capture_id = f"relative_{self.sequence:08d}_{observer}_{observed}"
                message.frame_id = "airsim_body_ned"
                message.relative_position = [body_x, body_y]
                message.relative_velocity = [
                    cos(yaw) * (float(target.velocity[0]) - float(source.velocity[0])) +
                    sin(yaw) * (float(target.velocity[1]) - float(source.velocity[1])),
                    -sin(yaw) * (float(target.velocity[0]) - float(source.velocity[0])) +
                    cos(yaw) * (float(target.velocity[1]) - float(source.velocity[1])),
                ]
                message.range = distance
                message.bearing = wrap_angle(atan2(body_y, body_x))
                message.timestamp = float(stamp.sec) + 1e-9 * float(stamp.nanosec)
                message.sequence = self.sequence
                message.covariance = [
                    self.range_std * self.range_std, 0.0,
                    0.0, self.bearing_std * self.bearing_std,
                ]
                message.valid = True
                self.publisher.publish(message)

    def publish_camera_measurements(self) -> None:
        if self.worker is None:
            return
        for value in self.worker.snapshot().values():
            if not value.valid or value.capture_id in self.published_capture_ids:
                continue
            self.published_capture_ids.add(value.capture_id)
            message = PlanarRelativeMeasurement()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "airsim_body_ned"
            message.source_id = value.observer_id
            message.target_id = value.observed_id
            message.observer_id = value.observer_id
            message.observed_id = value.observed_id
            message.sensor_id = value.sensor
            message.capture_id = value.capture_id
            message.frame_id = "airsim_body_ned"
            message.relative_position = [float(item) for item in value.relative_position]
            message.relative_velocity = [0.0, 0.0]
            message.covariance = [float(item) for item in value.covariance.reshape(-1)]
            message.range = float(value.range)
            message.bearing = float(value.bearing)
            message.timestamp = float(value.timestamp)
            message.sequence = len(self.published_capture_ids)
            message.valid = True
            self.publisher.publish(message)

def main() -> None:
    rclpy.init()
    node = RelativeObserver()
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
