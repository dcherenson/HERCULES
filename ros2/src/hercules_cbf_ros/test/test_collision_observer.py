import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "mission_collision_observer_node.py"
    spec = importlib.util.spec_from_file_location("mission_collision_observer_node", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collision_event_id_is_stable_for_deduplication():
    module = _module()
    record = {"object_name": "Box", "object_id": 4, "time_stamp": 123.5}
    assert module.collision_event_id("Drone1", record) == module.collision_event_id("Drone1", record)
    assert module.collision_event_id("Drone1", record) != module.collision_event_id("Husky1", record)
