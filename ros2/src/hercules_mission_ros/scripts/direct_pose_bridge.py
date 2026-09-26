#!/usr/bin/env python3
"""Validation/calibration boundary: publish direct AirSim world-NED poses."""

from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException


def _import_airsim():
    roots = [os.environ.get("HERCULES_REPO_ROOT", "/workspaces/hercules")]
    roots.append(str(Path(__file__).resolve().parents[4]))
    for root in roots:
        client_dir = str(Path(root) / "PythonClient")
        if client_dir not in sys.path:
            sys.path.insert(0, client_dir)
    import hercules_cosysairsim as airsim  # pylint: disable=import-outside-toplevel
    return airsim


def _resolve_rpc_host(host: str) -> str:
    value = str(host or "127.0.0.1")
    if value in {"host.docker.internal", "docker.for.mac.host.internal"}:
        try:
            addresses = socket.getaddrinfo(value, None, socket.AF_INET, socket.SOCK_STREAM)
            if addresses:
                return str(addresses[0][4][0])
        except OSError:
            pass
    return value


class DirectPoseBridge(Node):
    def __init__(self) -> None:
        super().__init__("mission_direct_pose_bridge")
        self.declare_parameter("vehicle_names", [
            "Drone1", "Drone2", "SimpleFlight",
            "Husky1", "Husky2", "Husky3", "Target1",
        ])
        self.declare_parameter("rpc_host", "127.0.0.1")
        self.declare_parameter("rpc_port", 41451)
        self.declare_parameter("car_port", 41452)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("startup_timeout_sec", 30.0)
        self.names = list(self.get_parameter("vehicle_names").value)
        if not self.names:
            raise RuntimeError("vehicle_names is required")
        airsim = _import_airsim()
        host = _resolve_rpc_host(str(self.get_parameter("rpc_host").value or "127.0.0.1"))
        drone_port = int(self.get_parameter("rpc_port").value)
        car_port = int(self.get_parameter("car_port").value)
        # ``Node.clients`` is a read-only rclpy-managed attribute on some
        # distributions; keep the AirSim endpoint handles private.
        # ``Node._clients`` is an rclpy executor-owned collection; do not
        # shadow it with AirSim handles or the executor will iterate port
        # integers as if they were ROS service clients.
        startup_timeout = max(1.0, float(self.get_parameter("startup_timeout_sec").value))
        deadline = time.monotonic() + startup_timeout
        self._airsim_clients = {}
        available = set()
        last_error = None
        while time.monotonic() < deadline:
            clients = {}
            try:
                for port in dict.fromkeys((drone_port, car_port)):
                    client = airsim.VehicleClient(ip=host, port=port)
                    client.confirmConnection()
                    clients[port] = client
                    available.update(client.listVehicles())
                self._airsim_clients = clients
                break
            except Exception as error:
                last_error = error
                for client in clients.values():
                    close = getattr(client, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            pass
                available.clear()
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        if not self._airsim_clients:
            raise RuntimeError(
                f"AirSim RPC endpoints {host}:{drone_port}/{car_port} were not ready "
                f"within {startup_timeout:.1f}s: {last_error}"
            )
        missing = sorted(set(self.names) - available)
        if missing:
            raise RuntimeError(f"AirSim is missing required vehicles: {missing}")
        self.vehicle_clients = {
            name: self._airsim_clients[
                car_port if name.lower().startswith(("husky", "ugv", "car", "target"))
                else drone_port
            ]
            for name in self.names
        }
        self.pose_publishers = {
            name: self.create_publisher(
                PointStamped, f"/hercules_mission/direct_pose/{name}", 10
            )
            for name in self.names
        }
        rate = float(self.get_parameter("publish_rate_hz").value)
        self.timer = self.create_timer(1.0 / rate, self.publish)
        self.get_logger().info(f"direct AirSim pose bridge verified all {len(self.names)} vehicles")

    def publish(self) -> None:
        # A timer callback can already be queued while launch is tearing down
        # the node.  Avoid touching publishers after the rclpy context has
        # been invalidated (which otherwise produces a noisy traceback during
        # an otherwise clean Ctrl-C shutdown).
        if not rclpy.ok():
            return
        stamp = self.get_clock().now().to_msg()
        for name, publisher in self.pose_publishers.items():
            try:
                pose = self.vehicle_clients[name].simGetObjectPose(name, True)
            except Exception:
                continue
            point = pose.position
            if any(not float(value) == float(value) for value in (point.x_val, point.y_val, point.z_val)):
                continue
            message = PointStamped()
            message.header.stamp = stamp
            message.header.frame_id = "airsim_world_ned"
            message.point.x = float(point.x_val)
            message.point.y = float(point.y_val)
            message.point.z = float(point.z_val)
            try:
                publisher.publish(message)
            except Exception:
                if not rclpy.ok():
                    return


def main() -> None:
    rclpy.init()
    node = DirectPoseBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
