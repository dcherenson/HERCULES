import numpy as np

from modules.target_tracking import (
    DistributedTargetTracking,
    TargetMeasurement,
    TargetTrackingModule,
    assemble_window_information,
    constant_acceleration_process_noise,
    constant_velocity_transition,
    dense_information_solution,
    position_measurement_matrix,
    solve_block_tridiagonal,
)


def measurement(target_id, position, timestamp, variance=0.25):
    return TargetMeasurement(target_id, position, np.eye(2) * variance, timestamp)


def test_constant_velocity_model_and_white_acceleration_noise():
    assert np.allclose(constant_velocity_transition(0.5), [
        [1.0, 0.0, 0.5, 0.0],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    assert np.allclose(constant_acceleration_process_noise(2.0, 1.0), [
        [8.0 / 3.0, 0.0, 2.0, 0.0],
        [0.0, 8.0 / 3.0, 0.0, 2.0],
        [2.0, 0.0, 2.0, 0.0],
        [0.0, 2.0, 0.0, 2.0],
    ])
    assert np.array_equal(position_measurement_matrix(), np.eye(2, 4))


def test_algorithm_two_block_cholesky_matches_dense_equation_15():
    times = [0.0, 0.25, 0.5]
    measurements = [
        measurement("Target1", [1.0, -2.0], 0.0),
        None,
        measurement("Target1", [1.5, -1.5], 0.5),
    ]
    information, vector = assemble_window_information(
        times, measurements, np.zeros(4), np.eye(4) * 4.0, process_noise_spectral_density=0.5
    )
    block_solution = solve_block_tridiagonal(information, vector)
    dense_solution = dense_information_solution(information, vector)
    assert np.allclose(block_solution, dense_solution, atol=1e-8)


def test_distributed_admm_reaches_consensus_and_keeps_missing_samples():
    modules = {
        name: TargetTrackingModule(name, max_iterations=20, tolerance=1e-3)
        for name in ("Drone1", "Husky1", "Husky2")
    }
    tracker = DistributedTargetTracking(modules, max_iterations=20, tolerance=1e-3)
    graph = {
        "Drone1": ["Husky1"],
        "Husky1": ["Drone1", "Husky2"],
        "Husky2": ["Husky1"],
    }
    result = tracker.update(
        0.0,
        {
            "Drone1": {"Target1": measurement("Target1", [0.0, 0.0], 0.0)},
            "Husky1": {"Target1": measurement("Target1", [1.0, 0.0], 0.0)},
            "Husky2": {},
        },
        graph,
    )
    estimates = tracker.predicted_estimates(0.0)
    values = [estimates[name]["Target1"]["position"][0] for name in modules]
    assert result["active_targets"] == ["Target1"]
    assert max(values) - min(values) < 0.02
    assert 0.2 < np.mean(values) < 0.8

    tracker.update(
        0.25,
        {name: {} for name in modules},
        graph,
    )
    assert len(modules["Drone1"].tracks["Target1"].times) == 2


def test_disconnected_components_do_not_exchange_measurements():
    modules = {name: TargetTrackingModule(name, max_iterations=20) for name in ("a", "b")}
    tracker = DistributedTargetTracking(modules, max_iterations=20)
    result = tracker.update(
        0.0,
        {
            "a": {"Target1": measurement("Target1", [0.0, 0.0], 0.0)},
            "b": {"Target1": measurement("Target1", [10.0, 0.0], 0.0)},
        },
        {"a": [], "b": []},
    )
    estimates = tracker.predicted_estimates(0.0)
    assert abs(estimates["a"]["Target1"]["position"][0] - estimates["b"]["Target1"]["position"][0]) > 1.0
    assert result["iterations"] == 1


def test_multi_target_states_are_independent():
    module = TargetTrackingModule("Drone1", max_iterations=10)
    module.begin_epoch(0.0, {
        "Target1": measurement("Target1", [1.0, 0.0], 0.0),
        "Target2": measurement("Target2", [20.0, 0.0], 0.0),
    })
    estimates = module.predicted_estimates(0.0)
    assert set(estimates) == {"Target1", "Target2"}
    assert estimates["Target1"]["position"][0] < estimates["Target2"]["position"][0]


def test_handoff_is_information_keyed_and_sustains_receiver():
    modules = {name: TargetTrackingModule(name, window_seconds=5.0, max_iterations=10) for name in ("a", "b")}
    tracker = DistributedTargetTracking(modules, max_iterations=10)
    graph = {"a": ["b"], "b": ["a"]}
    tracker.update(0.0, {"a": {"Target1": measurement("Target1", [3.0, 4.0], 0.0)}, "b": {}}, graph)
    result = tracker.update(5.0, {"a": {}, "b": {}}, graph)
    handoffs = tracker.perform_handoffs(5.0, graph)
    result["handoffs"] = handoffs
    assert result["handoffs"]
    assert result["handoffs"][0]["target_id"] == "Target1"
    assert result["handoffs"][0]["receiver_id"] == "b"
    assert modules["b"].tracks["Target1"].active


def test_connected_silent_receiver_stays_active_without_covariance_inflation():
    modules = {
        name: TargetTrackingModule(name, window_seconds=2.0, max_iterations=10)
        for name in ("Drone1", "Husky1")
    }
    tracker = DistributedTargetTracking(modules, max_iterations=10)
    graph = {"Drone1": ["Husky1"], "Husky1": ["Drone1"]}

    for epoch in range(7):
        timestamp = float(epoch) * 0.5
        tracker.update(
            timestamp,
            {
                "Drone1": {"Target1": measurement("Target1", [timestamp, 0.0], timestamp)},
                "Husky1": {},
            },
            graph,
        )

    estimate = tracker.predicted_estimates(3.0)["Husky1"]["Target1"]
    assert estimate["active"]
    assert np.max(np.linalg.eigvalsh(np.asarray(estimate["covariance"]))) < 10.0
