import json
import os
import pathlib
import subprocess
import sys

import numpy as np


SOURCE = pathlib.Path(os.environ["HERCULES_SOURCE_DIR"]).resolve()
sys.path.insert(0, str(SOURCE))

from modules.cbf import AgentState  # noqa: E402
from modules.formation_control import FormationConfig, FormationController  # noqa: E402
from modules.target_motion import FigureEightTargetController, target_centered_slot  # noqa: E402
from orchestrator import target_start_anchor_for_map  # noqa: E402


def flatten(value):
    return np.asarray(value, dtype=float).reshape(-1).tolist()


def baseline_controller():
    start = np.zeros(3)
    goal = np.array([20.0, 8.0, -1.0])
    heading = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]) + np.pi / 2.0)
    center = target_start_anchor_for_map(start, goal, "rural_australia", -1.0, heading)
    controller = FigureEightTargetController(
        center=center,
        route_heading=heading,
        longitudinal_span=10.0,
        lateral_span=8.0,
        speed=0.10,
        sample_count=64,
        waypoint_radius=1.0,
        heading_gain=2.0,
        max_yaw_rate=1.5,
        minimum_alignment=0.75,
        direction=1,
    )
    controller.index = 5
    controller.place_start_at(center)
    return controller


def figure_geometry_reference():
    controller = baseline_controller()
    return {
        "center": flatten(controller.center),
        "points": flatten(controller.points),
        "index": int(controller.index),
        "phase": float(controller.phase),
        "references": [
            {"lookahead": lookahead, "value": flatten(controller.reference(lookahead))}
            for lookahead in (-3, 0, 1, 2, 7)
        ],
    }


def figure_dynamics_reference():
    controller = baseline_controller()
    points = controller.points
    position_indices = (5, 6, 12, 14, 2, 3)
    yaws = (0.0, 0.5, -2.8, 2.5, 0.1, -1.1)
    deltas = (0.1, 0.25, 0.05, 0.1, 0.2, 0.1)
    output = []
    for point_index, yaw, dt in zip(position_indices, yaws, deltas):
        command = controller.update(points[point_index], yaw, dt)
        output.append({
            "index": int(controller.index),
            "phase": float(controller.phase),
            "reference": flatten(controller.reference()),
            "command": flatten(command),
        })
    return output


def slot_reference():
    target_position = np.array([10.0, 20.0, -1.0])
    velocities = (
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 3.0, 0.0]),
    )
    agents = (
        "Drone1", "Drone2", "SimpleFlight", "Drone4", "Drone5",
        "Husky1", "Husky2", "Husky3",
    )
    output = []
    for case, velocity in enumerate(velocities):
        for agent in agents:
            vehicle_type = "ugv" if agent.startswith("Husky") else "drone"
            position, feedforward, heading = target_centered_slot(
                agent, vehicle_type, target_position, velocity, 0.6, -5.0,
                target_z=-1.0, ugv_circumradius=5.0,
            )
            output.append({
                "case": case,
                "agent": agent,
                "position": flatten(position),
                "velocity": flatten(feedforward),
                "heading": float(heading),
            })
    return output


def formation_controller():
    return FormationController(FormationConfig(
        uav_altitude=-5.0,
        max_speed=1.0,
        leader_max_speed=1.0,
        position_gain=0.5,
        velocity_gain=3.0,
        ugv_heading_gain=1.0,
        ugv_max_yaw_rate=1.0,
    ))


def state(agent_id, position, velocity, yaw, vehicle_type):
    return AgentState(
        agent_id, np.asarray(position, dtype=float), np.asarray(velocity, dtype=float),
        yaw=float(yaw), vehicle_type=vehicle_type,
    )


def control_reference():
    controller = formation_controller()
    active = {"position": [10.0, 20.0], "velocity": [0.5, 0.25], "active": True}
    inactive = {"position": [10.0, 20.0], "velocity": [0.5, 0.25], "active": False}
    fallback = 0.4
    output = [
        {"case": "uav_ordinary", "command": flatten(controller.target_nominal_control(
            state("Drone1", [8.0, 17.0, -4.0], [0.2, -0.1, 0.3], 0.0, "drone"),
            active, fallback, target_z=-1.0, ugv_circumradius=5.0,
        ))},
        {"case": "uav_speed_saturation", "command": flatten(controller.target_nominal_control(
            state("Drone5", [-20.0, 40.0, -20.0], [-1.0, 0.5, 0.0], 0.0, "drone"),
            active, fallback, target_z=-1.0, ugv_circumradius=5.0,
        ))},
        {"case": "uav_inactive", "command": flatten(controller.target_nominal_control(
            state("SimpleFlight", [2.0, 3.0, -5.0], [1.0, 1.0, 0.0], 0.0, "drone"),
            inactive, fallback, target_z=-1.0, ugv_circumradius=5.0,
        ))},
        {"case": "ugv_ordinary", "command": flatten(controller.target_nominal_unicycle_control(
            state("Husky1", [12.0, 18.0, -1.0], [0.0, 0.0, 0.0], 0.2, "ugv"),
            active, fallback, target_z=-1.0, ugv_circumradius=5.0,
        ))},
        {"case": "ugv_yaw_saturation", "command": flatten(controller.target_nominal_unicycle_control(
            state("Husky2", [-5.0, 5.0, -1.0], [0.0, 0.0, 0.0], 3.0, "ugv"),
            active, fallback, target_z=-1.0, ugv_circumradius=5.0,
        ))},
    ]
    hold_position, _, _ = target_centered_slot(
        "Husky3", "ugv", [10.0, 20.0, -1.0], [0.5, 0.25, 0.0],
        fallback, -5.0, target_z=-1.0, ugv_circumradius=5.0,
    )
    output.extend([
        {"case": "ugv_hold", "command": flatten(controller.target_nominal_unicycle_control(
            state("Husky3", hold_position + [0.3, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "ugv"),
            active, fallback, target_z=-1.0, ugv_circumradius=5.0,
        ))},
        {"case": "ugv_inactive", "command": flatten(controller.target_nominal_unicycle_control(
            state("Husky1", [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], 0.0, "ugv"),
            inactive, fallback, target_z=-1.0, ugv_circumradius=5.0,
        ))},
    ])
    return output


def assert_tree(actual, expected, path, maxima):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key, value in expected.items():
            assert_tree(actual[key], value, f"{path}.{key}", maxima)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        if expected and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in expected):
            error = float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))))
            maxima[path] = error
            assert error <= 1e-10, f"{path}: {error}"
        else:
            for index, value in enumerate(expected):
                assert_tree(actual[index], value, f"{path}[{index}]", maxima)
    elif isinstance(expected, bool):
        assert actual is expected, path
    elif isinstance(expected, (int, float)):
        error = abs(float(actual) - float(expected))
        maxima[path] = error
        assert error <= 1e-10, f"{path}: {error}"
    else:
        assert actual == expected, path


def test_python_cpp_nominal_mission_equivalence():
    actual = json.loads(subprocess.check_output([os.environ["MISSION_PARITY_EXE"]], text=True))
    expected = {
        "figure_geometry": figure_geometry_reference(),
        "figure_dynamics": figure_dynamics_reference(),
        "slots": slot_reference(),
        "controls": control_reference(),
    }
    maxima = {}
    assert_tree(actual, expected, "mission", maxima)
    summary = {
        "figure_geometry_max": max(
            value for key, value in maxima.items() if ".figure_geometry" in key
        ),
        "figure_dynamics_max": max(
            value for key, value in maxima.items() if ".figure_dynamics" in key
        ),
        "slot_max": max(value for key, value in maxima.items() if ".slots" in key),
        "control_max": max(value for key, value in maxima.items() if ".controls" in key),
        "all_max": max(maxima.values()),
    }
    pathlib.Path("mission_parity_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("MISSION_PARITY_MAX " + json.dumps(summary, sort_keys=True))
