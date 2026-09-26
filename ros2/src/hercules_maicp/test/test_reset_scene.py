from __future__ import annotations

import math
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hercules_maicp.reset_scene import (  # noqa: E402
    CONTROLLED_AGENTS,
    COLUMN_PLANAR_RADIUS,
    scene_plan,
)


def test_scene_plan_is_seeded_and_case_independent_for_agent_jitters():
    collision = scene_plan(123, "collision")
    tracking = scene_plan(123, "tracking")
    localization = scene_plan(123, "localization")

    assert collision["agents"] == tracking["agents"] == localization["agents"]
    assert collision["target"]["position"] == [-8.0, 0.0, -1.0]
    assert tracking["target"]["position"] == [2.0, 6.0, -1.0]
    assert localization["target"]["position"] == [-8.0, 0.0, -1.0]


def test_scene_plan_contains_six_agents_and_bounded_jitters():
    plan = scene_plan(9, "localization")

    assert tuple(plan["agents"]) == CONTROLLED_AGENTS
    assert plan["collision_geometry"] == []
    for agent in plan["agents"].values():
        assert all(abs(value) <= 0.1 for value in agent["jitter_xy"])
    assert {agent["class"] for agent in plan["agents"].values()} == {"uav", "ugv"}


def test_collision_geometry_exposes_controller_geometry():
    plan = scene_plan(0, "collision")

    assert len(plan["collision_geometry"]) == 5
    assert math.isclose(plan["column_planar_radius"], COLUMN_PLANAR_RADIUS)
    columns = [item for item in plan["collision_geometry"] if item["kind"] == "column"]
    walls = [item for item in plan["collision_geometry"] if item["kind"] == "wall"]
    assert [item["center"] for item in columns] == [
        [2.5, -9.0, -4.0],
        [2.5, 0.0, -4.0],
        [2.5, 9.0, -4.0],
    ]
    assert {item["inner_face_y"] for item in walls} == {-14.75, 14.75}
    assert all(item["collision_enabled"] and not item["physics_enabled"] for item in plan["collision_geometry"])
