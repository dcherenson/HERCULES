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


DEFAULT_UAV_AGENTS = ("Drone1", "Drone2", "SimpleFlight")
DEFAULT_UGV_AGENTS = ("Husky1", "Husky2", "Husky3")
DEFAULT_CONTROLLED_AGENTS = DEFAULT_UAV_AGENTS + DEFAULT_UGV_AGENTS
DEFAULT_TARGET_AGENTS = ("Target1",)
DEFAULT_VEHICLES = DEFAULT_CONTROLLED_AGENTS + DEFAULT_TARGET_AGENTS
VEHICLES = list(DEFAULT_VEHICLES)


def _configured_agents(value, parameter_name: str, default) -> tuple[str, ...]:
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
    from simulation.airsim_runtime import AirSimFacade, AirSimLaunchConfig  # pylint: disable=import-outside-toplevel
    return AirSimFacade, AirSimLaunchConfig


def collision_event_id(vehicle: str, record: dict) -> str:
    return "{}:{}:{}:{}".format(vehicle, record.get("object_name", ""),
                                 record.get("object_id", -1),
                                 record.get("time_stamp", 0.0))


def _is_ground_object(object_name: str) -> bool:
    """Identify Unreal ground actors for the UGV/target contact filter.

    The Python mission keeps the raw AirSim collision record but excludes
    terrain contact from its actionable collision metric. Keep this narrow
    name-based rule at the observer boundary so UAV terrain contacts remain
    visible and all object names/timestamps still reach the mission log.
    """
    normalized = str(object_name or "").strip().lower()
    return any(token in normalized for token in ("ground", "landscape", "terrain", "floor"))


def collision_is_relevant(vehicle: str, record: dict,
                          ugv_agents=DEFAULT_UGV_AGENTS,
                          target_agents=DEFAULT_TARGET_AGENTS) -> bool:
    """Return the Python-compatible actionable-collision classification.

    Ground contacts are ignored only for configured UGVs and targets. The
    observer deliberately does not suppress first-sample events because it
    does not know when a simulator reset or mission phase occurred.
    """
    available = bool(record.get("available", False))
    collided = bool(record.get("has_collided", False))
    object_name = str(record.get("object_name", ""))
    if not (available and collided and object_name):
        return False
    ground_vehicle = vehicle in set(ugv_agents) or vehicle in set(target_agents)
    return not (ground_vehicle and _is_ground_object(object_name))


class MissionCollisionObserverNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_collision_observer")
        self.declare_parameter("poll_rate", 10.0)
        self.declare_parameter("uav_agents", list(DEFAULT_UAV_AGENTS))
        self.declare_parameter("ugv_agents", list(DEFAULT_UGV_AGENTS))
        self.declare_parameter("target_agents", list(DEFAULT_TARGET_AGENTS))
        self.declare_parameter("host_ip", "127.0.0.1")
        self.declare_parameter("rpc_port", 41451)
        self.declare_parameter("car_port", 41452)
        self.uav_agents = _configured_agents(
            self.get_parameter("uav_agents").value, "uav_agents", DEFAULT_UAV_AGENTS)
        self.ugv_agents = _configured_agents(
            self.get_parameter("ugv_agents").value, "ugv_agents", DEFAULT_UGV_AGENTS)
        self.target_agents = _configured_agents(
            self.get_parameter("target_agents").value, "target_agents", DEFAULT_TARGET_AGENTS)
        controlled = self.uav_agents + self.ugv_agents
        if set(self.uav_agents).intersection(self.ugv_agents):
            raise ValueError("uav_agents and ugv_agents must be disjoint")
        if set(controlled).intersection(self.target_agents):
            raise ValueError("target_agents must be separate from controlled agents")
        self.vehicles = controlled + self.target_agents
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
        for vehicle in self.vehicles:
            try:
                record = self.facade.collision_info(vehicle)
            except Exception as error:
                record = {"available": False, "has_collided": False, "error": str(error)}
            available = bool(record.get("available", False))
            collided = bool(record.get("has_collided", False))
            event_id = collision_event_id(vehicle, record) if collided else ""
            relevant = collision_is_relevant(
                vehicle, record, self.ugv_agents, self.target_agents)
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
