"""Regression checks for the configured CBF observer vehicle topology."""

import ast
from pathlib import Path
import re


PACKAGE_ROOT = Path(__file__).parents[1]


def _literal_constants(path: Path):
    """Read simple tuple/list constants without importing ROS dependencies."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        for target in statement.targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                values[target.id] = ast.literal_eval(statement.value)
            except (ValueError, TypeError):
                if (isinstance(statement.value, ast.BinOp)
                        and isinstance(statement.value.op, ast.Add)
                        and isinstance(statement.value.left, ast.Name)
                        and isinstance(statement.value.right, ast.Name)
                        and statement.value.left.id in values
                        and statement.value.right.id in values):
                    values[target.id] = (
                        values[statement.value.left.id]
                        + values[statement.value.right.id]
                    )
    return values


def _yaml_flow_list(text: str, key: str):
    match = re.search(rf"^\s+{re.escape(key)}:\s*\[([^]]*)\]\s*$", text, re.MULTILINE)
    assert match, f"missing {key} in rural_perception.yaml"
    return tuple(part.strip() for part in match.group(1).split(",") if part.strip())


def test_obstacle_observer_defaults_have_three_uavs_and_three_ugvs():
    values = _literal_constants(PACKAGE_ROOT / "scripts" / "obstacle_observer_node.py")
    assert values["DEFAULT_UAV_AGENTS"] == ("Drone1", "Drone2", "SimpleFlight")
    assert values["DEFAULT_UGV_AGENTS"] == ("Husky1", "Husky2", "Husky3")
    assert values["DEFAULT_CONTROLLED_AGENTS"] == (
        "Drone1", "Drone2", "SimpleFlight", "Husky1", "Husky2", "Husky3"
    )
    assert "Target1" not in values["DEFAULT_CONTROLLED_AGENTS"]


def test_collision_observer_keeps_target_separate_from_controlled_fleet():
    values = _literal_constants(PACKAGE_ROOT / "scripts" / "mission_collision_observer_node.py")
    assert values["DEFAULT_CONTROLLED_AGENTS"] == (
        "Drone1", "Drone2", "SimpleFlight", "Husky1", "Husky2", "Husky3"
    )
    assert values["DEFAULT_TARGET_AGENTS"] == ("Target1",)
    assert values["DEFAULT_VEHICLES"] == values["DEFAULT_CONTROLLED_AGENTS"] + values["DEFAULT_TARGET_AGENTS"]


def test_ros_defaults_match_configured_fleet():
    config = (PACKAGE_ROOT / "config" / "rural_perception.yaml").read_text(encoding="utf-8")
    expected_uavs = ("Drone1", "Drone2", "SimpleFlight")
    expected_ugvs = ("Husky1", "Husky2", "Husky3")
    assert _yaml_flow_list(config, "uav_agents") == expected_uavs
    assert _yaml_flow_list(config, "ugv_agents") == expected_ugvs
    assert _yaml_flow_list(config, "target_agents") == ("Target1",)
