#!/usr/bin/env python3
"""Validation/calibration boundary: publish direct AirSim world-NED poses."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node


def _import_airsim():
    roots = [os.environ.get("HERCULES_REPO_ROOT", "/workspaces/hercules")]
    roots.append(str(Path(__file__).resolve().parents[4]))
    for root in roots:
        site_packages = str(Path(root) / "herculesvenv" / "lib" / "python3.10" / "site-packages")
        if site_packages not in sys.path:
            sys.path.insert(0, site_packages)
        client_dir = str(Path(root) / "PythonClient")
        if client_dir not in sys.path:
            sys.path.insert(0, client_dir)
    import hercules_cosysairsim as airsim  # pylint: disable=import-outside-toplevel
    return airsim


class DirectPoseBridge(Node):
    def __init__(self) -> None:
        super().__init__("mission_direct_pose_bridge")
        self.declare_parameter("vehicle_names", [
            "Drone1", "Drone2", "SimpleFlight", "Drone4", "Drone5",
            "Husky1", "Husky2", "Husky3", "Target1",
        ])
        self.declare_parameter("rpc_host", "127.0.0.1")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("startup_timeout_sec", 30.0)
        self.names = list(self.get_parameter("vehicle_names").value)
        if not self.names:
            raise RuntimeError("vehicle_names is required")
        airsim = _import_airsim()
        self.client = airsim.VehicleClient(
            ip=str(self.get_parameter("rpc_host").value), port=41451
        )
        self.client.confirmConnection()
        available = set(self.client.listVehicles())
        missing = sorted(set(self.names) - available)
        if missing:
            raise RuntimeError(f"AirSim is missing required vehicles: {missing}")
        self.pose_publishers = {
            name: self.create_publisher(
                PointStamped, f"/hercules_mission/direct_pose/{name}", 10
            )
            for name in self.names
        }
        rate = float(self.get_parameter("publish_rate_hz").value)
        self.timer = self.create_timer(1.0 / rate, self.publish)
        self.get_logger().info("direct AirSim pose bridge verified all nine vehicles")

    def publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for name, publisher in self.pose_publishers.items():
            pose = self.client.simGetObjectPose(name, True)
            point = pose.position
            if any(not float(value) == float(value) for value in (point.x_val, point.y_val, point.z_val)):
                continue
            message = PointStamped()
            message.header.stamp = stamp
            message.header.frame_id = "airsim_world_ned"
            message.point.x = float(point.x_val)
            message.point.y = float(point.y_val)
            message.point.z = float(point.z_val)
            publisher.publish(message)


def main() -> None:
    rclpy.init()
    try:
        rclpy.spin(DirectPoseBridge())
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
