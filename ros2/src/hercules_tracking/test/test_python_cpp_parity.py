import json
import os
import pathlib
import subprocess
import sys

import numpy as np


SOURCE = pathlib.Path(os.environ["HERCULES_SOURCE_DIR"]).resolve()
sys.path.insert(0, str(SOURCE))

from modules.target_tracking import (  # noqa: E402
    DistributedTargetTracking,
    TargetMeasurement,
    TargetTrackingModule,
    constant_acceleration_process_noise,
    constant_velocity_transition,
)


def measurement(target, x, y, timestamp, variance=0.25):
    return TargetMeasurement(target, np.array([x, y]), np.eye(2) * variance, timestamp)


def flatten(value):
    return np.asarray(value, dtype=float).reshape(-1).tolist()


def estimate(value):
    return {
        "position": flatten(value["position"]),
        "velocity": flatten(value["velocity"]),
        "covariance": flatten(value["covariance"]),
        "state_covariance": flatten(value["state_covariance"]),
        "timestamp": float(value["timestamp"]),
        "active": bool(value["active"]),
    }


def model_reference():
    return [
        {
            "dt": dt,
            "transition": flatten(constant_velocity_transition(dt)),
            "process": flatten(constant_acceleration_process_noise(dt, 0.7)),
        }
        for dt in (0.0, 0.5, 1.3, -0.7)
    ]


def local_reference():
    module = TargetTrackingModule("local")
    module.begin_epoch(0.0, {"Target1": measurement("Target1", 1.0, -2.0, 0.0)}, 2)
    module.begin_epoch(0.7, {}, 2)
    module.begin_epoch(1.6, {"Target1": measurement("Target1", 2.1, -1.1, 1.4)}, 2)
    track = module.tracks["Target1"]
    return {
        "times": list(track.times),
        "information_shape": list(track.information.shape),
        "information": flatten(track.information),
        "information_vector": flatten(track.information_vector),
        "trajectory": flatten(track.x),
        "full_covariance": flatten(track.covariance),
        "estimate": estimate(track.finalize()),
    }


def line_graph():
    return {
        "Drone1": ["Husky1"],
        "Husky1": ["Drone1", "Husky2"],
        "Husky2": ["Husky1"],
    }


def manual_rounds(agent_ids, observations, graph, max_iterations):
    modules = {
        name: TargetTrackingModule(name, max_iterations=max_iterations, tolerance=0.0)
        for name in agent_ids
    }
    target_ids = sorted({target for values in observations.values() for target in values})
    for name, module in modules.items():
        module.begin_epoch(0.0, observations.get(name, {}), len(modules))
    for _ in range(max(1, len(modules))):
        changed = False
        announcements = {name: module.messages() for name, module in modules.items()}
        for name, module in modules.items():
            if all(target in module.tracks for target in target_ids):
                continue
            inbound = []
            for neighbor in graph.get(name, []):
                inbound.extend(announcements.get(neighbor, {}).values())
            before = set(module.tracks)
            module.seed_from_messages(inbound, 0.0)
            changed |= set(module.tracks) != before
        if not changed:
            break
    history = []
    residual = float("inf")
    for _ in range(max_iterations):
        messages = {name: module.messages() for name, module in modules.items()}
        residual = 0.0
        for name, module in modules.items():
            inbound = {}
            for neighbor in graph.get(name, []):
                for target, message_value in messages.get(neighbor, {}).items():
                    inbound.setdefault(target, []).append(message_value)
            residual = max(residual, module.consensus_round(inbound))
        history.append({
            name: {
                "trajectory": flatten(module.tracks["Target1"].x),
                "dual": flatten(module.tracks["Target1"].dual),
                "residual": float(module.tracks["Target1"].consensus_residual),
            }
            for name, module in modules.items()
        })
        if residual <= 0.0:
            break
    return history, residual


def network_reference(agent_ids, observations, graph, max_iterations):
    history, residual = manual_rounds(agent_ids, observations, graph, max_iterations)
    modules = {
        name: TargetTrackingModule(name, max_iterations=max_iterations, tolerance=0.0)
        for name in agent_ids
    }
    network = DistributedTargetTracking(modules, max_iterations=max_iterations, tolerance=0.0)
    result = network.update(0.0, observations, graph)
    return {
        "iterations": int(result["iterations"]),
        "residual": float(residual),
        "rounds": history,
        "estimates": {
            name: estimate(result["estimates"][name]["Target1"])
            for name in agent_ids
        },
    }


def async_reference():
    module = TargetTrackingModule("a")
    module.begin_epoch(10.0, {"Target1": measurement("Target1", 4.0, -3.0, 9.25)})
    module.begin_epoch(10.5, {})
    track = module.tracks["Target1"]
    return {
        "times": list(track.times),
        "information": flatten(track.information),
        "trajectory": flatten(track.x),
    }


def handoff_reference():
    modules = {
        name: TargetTrackingModule(name, window_seconds=1.0)
        for name in ("a", "b")
    }
    network = DistributedTargetTracking(modules, max_iterations=4, tolerance=0.0)
    graph = {"a": ["b"], "b": ["a"]}
    network.update(0.0, {"a": {"Target1": measurement("Target1", 3.0, 4.0, 0.0)}, "b": {}}, graph)
    network.update(1.0, {"a": {}, "b": {}}, graph)
    handoffs = network.perform_handoffs(1.0, graph)
    receiver = modules["b"].tracks["Target1"]
    return {
        "count": len(handoffs),
        "matrix": flatten(handoffs[0]["information_matrix"]),
        "vector": flatten(handoffs[0]["information_vector"]),
        "receiver_information": flatten(receiver.information),
        "receiver_vector": flatten(receiver.information_vector),
        "receiver_state": flatten(receiver.x),
        "receiver_active": bool(receiver.active),
    }


def assert_numeric_tree(actual, expected, tolerance, path, maxima):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            assert_numeric_tree(actual[key], expected[key], tolerance, f"{path}.{key}", maxima)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        if expected and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in expected):
            error = float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))))
            maxima[path] = error
            assert error <= tolerance, f"{path}: {error} > {tolerance}"
        else:
            for index, item in enumerate(expected):
                assert_numeric_tree(actual[index], item, tolerance, f"{path}[{index}]", maxima)
    elif isinstance(expected, bool):
        assert actual is expected, path
    elif isinstance(expected, (int, float)):
        error = abs(float(actual) - float(expected))
        maxima[path] = error
        assert error <= tolerance, f"{path}: {error} > {tolerance}"
    else:
        assert actual == expected, path


def test_python_cpp_numerical_equivalence():
    output = subprocess.check_output([os.environ["TRACKING_PARITY_EXE"]], text=True)
    actual = json.loads(output)
    connected_observations = {
        "Drone1": {"Target1": measurement("Target1", 0.0, 0.0, 0.0)},
        "Husky1": {"Target1": measurement("Target1", 1.0, 0.0, 0.0)},
        "Husky2": {},
    }
    disconnected_observations = {
        "a": {"Target1": measurement("Target1", 0.0, 0.0, 0.0)},
        "b": {"Target1": measurement("Target1", 10.0, 0.0, 0.0)},
    }
    expected = {
        "models": model_reference(),
        "local": local_reference(),
        "connected": network_reference(
            ["Drone1", "Husky1", "Husky2"], connected_observations, line_graph(), 8
        ),
        "disconnected": network_reference(
            ["a", "b"], disconnected_observations, {"a": [], "b": []}, 4
        ),
        "async": async_reference(),
        "handoff": handoff_reference(),
    }
    maxima = {}
    assert_numeric_tree(actual["models"], expected["models"], 1e-12, "models", maxima)
    for case in ("local", "connected", "disconnected", "async", "handoff"):
        assert_numeric_tree(actual[case], expected[case], 1e-7, case, maxima)
    summary = {
        "model_max": max(value for key, value in maxima.items() if key.startswith("models")),
        "state_information_covariance_max": max(
            value for key, value in maxima.items() if not key.startswith("models") and "residual" not in key
        ),
        "consensus_residual_max": max(
            (value for key, value in maxima.items() if "residual" in key), default=0.0
        ),
        "all_max": max(maxima.values()),
    }
    pathlib.Path("parity_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PARITY_MAX " + json.dumps(summary, sort_keys=True))
