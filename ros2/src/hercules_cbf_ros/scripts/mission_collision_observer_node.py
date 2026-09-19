#!/usr/bin/env python3
"""Read-only AirSim collision observer with event de-duplication."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Dict, Set

import rclpy
from hercules_interfaces.msg import MissionCollision
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


VEHICLES = ["Drone1", "Drone2", "SimpleFlight", "Drone4", "Drone5",
            "Husky1", "Husky2", "Husky3", "Target1"]


def _source_modules():
    roots = [Path(os.environ.get("HERCULES_REPO_ROOT", "/workspaces/hercules")),
             Path(__file__).resolve().parents[4]]
    for root in roots:
        for path in (root / "PythonClient", root / "PythonClient" / "distributed_mission"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
    from simulation.airsim_runtime import AirSimFacade, AirSimLaunchConfig  # pylint: disable=import-outside-toplevel
    return AirSimFacade, AirSimLaunchConfig


def collision_event_id(vehicle: str, record: dict) -> str:
    return "{}:{}:{}:{}".format(vehicle, record.get("object_name", ""),
                                 record.get("object_id", -1),
                                 record.get("time_stamp", 0.0))


class MissionCollisionObserverNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_collision_observer")
        self.declare_parameter("poll_rate", 10.0)
        self.declare_parameter("host_ip", "127.0.0.1")
        self.declare_parameter("rpc_port", 41451)
        self.declare_parameter("car_port", 41452)
        self.poll_rate = max(1.0, float(self.get_parameter("poll_rate").value))
        self.host_ip = str(self.get_parameter("host_ip").value or "127.0.0.1")
        self.rpc_port = int(self.get_parameter("rpc_port").value)
        self.car_port = int(self.get_parameter("car_port").value)
        self.publisher = self.create_publisher(MissionCollision, "/hercules_mission/collisions", 20)
        facade_type, config_type = _source_modules()
        self.facade = facade_type(config_type(
            launch_mode="existing", host=self.host_ip,
            multirotor_port=self.rpc_port, car_port=self.car_port))
        self.connected = False
        self.active_events: Set[str] = set()
        self.timer = self.create_timer(1.0 / self.poll_rate, self.poll)

    def destroy_node(self):
        self.timer.cancel()
        return super().destroy_node()

    def poll(self) -> None:
        if not self.connected:
            try:
                self.facade.connect()
                self.connected = True
            except Exception as error:
                self.get_logger().warning(f"collision observer connection unavailable: {error}")
                return
        current: Set[str] = set()
        for vehicle in VEHICLES:
            try:
                record = self.facade.collision_info(vehicle)
            except Exception as error:
                record = {"available": False, "has_collided": False, "error": str(error)}
            available = bool(record.get("available", False))
            collided = bool(record.get("has_collided", False))
            event_id = collision_event_id(vehicle, record) if collided else ""
            relevant = available and collided and bool(record.get("object_name", ""))
            if relevant:
                current.add(event_id)
            if relevant and event_id in self.active_events:
                continue
            message = MissionCollision()
            message.header.stamp = self.get_clock().now().to_msg()
            message.vehicle_name = vehicle
            message.available = available
            message.has_collided = collided
            message.relevant = relevant
            message.object_name = str(record.get("object_name", ""))
            message.object_id = int(record.get("object_id", -1))
            message.penetration_depth = float(record.get("penetration_depth", 0.0))
            message.simulator_timestamp = float(record.get("time_stamp", 0.0))
            message.object_position = [float(value) for value in record.get("object_position", [])]
            message.event_id = event_id
            message.poll_status = "ok" if available else str(record.get("error", "unavailable"))
            self.publisher.publish(message)
        self.active_events = current


def main() -> None:
    rclpy.init()
    node = MissionCollisionObserverNode()
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
