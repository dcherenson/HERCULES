import importlib.util
from pathlib import Path
import sys
import types


def _module(monkeypatch):
    """Load the pure collision helpers without requiring a ROS installation."""
    class Node:
        pass

    rclpy = types.ModuleType("rclpy")
    rclpy.executors = types.ModuleType("rclpy.executors")
    rclpy.executors.ExternalShutdownException = type(
        "ExternalShutdownException", (Exception,), {})
    rclpy.node = types.ModuleType("rclpy.node")
    rclpy.node.Node = Node
    interfaces = types.ModuleType("hercules_interfaces")
    interfaces.msg = types.ModuleType("hercules_interfaces.msg")
    interfaces.msg.MissionCollision = type("MissionCollision", (), {})
    for name, value in {
        "rclpy": rclpy,
        "rclpy.executors": rclpy.executors,
        "rclpy.node": rclpy.node,
        "hercules_interfaces": interfaces,
        "hercules_interfaces.msg": interfaces.msg,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    path = Path(__file__).parents[1] / "scripts" / "mission_collision_observer_node.py"
    spec = importlib.util.spec_from_file_location("mission_collision_observer_node", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collision_event_id_is_stable_for_deduplication(monkeypatch):
    module = _module(monkeypatch)
    record = {"object_name": "Box", "object_id": 4, "time_stamp": 123.5}
    assert module.collision_event_id("Drone1", record) == module.collision_event_id("Drone1", record)
    assert module.collision_event_id("Drone1", record) != module.collision_event_id("Husky1", record)


def test_ground_contacts_match_python_filter_without_hiding_uav_contacts(monkeypatch):
    module = _module(monkeypatch)
    ground = {"available": True, "has_collided": True, "object_name": "Landscape_1"}
    obstacle = {"available": True, "has_collided": True, "object_name": "InstancedFoliageActor_0"}

    assert not module.collision_is_relevant("Husky2", ground)
    assert not module.collision_is_relevant("Target1", ground)
    assert module.collision_is_relevant("Drone1", ground)
    assert module.collision_is_relevant("Husky2", obstacle)
    assert not module.collision_is_relevant(
        "Husky2", {**ground, "has_collided": False})
