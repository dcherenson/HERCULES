"""Focused numerical tests for the ROS-independent MAICP core."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hercules_maicp.config import ClassConfig, MAICPConfig, paper_config
from hercules_maicp.core import (
    calibrate_round,
    exact_order_statistic,
    flood_tagged,
    graph_diameter,
    local_pinball_step,
    mission_score,
    pinball_gradient,
    pinball_order_statistic,
    solve_margin_update,
    standard_split_cp,
    validate_graph,
)


def _small_config() -> MAICPConfig:
    classes = {
        "uav": ClassConfig(name="uav", configured_kappa=0.6),
        "ugv": ClassConfig(name="ugv", configured_kappa=0.6),
    }
    return paper_config(
        calibration_missions=20,
        deployment_missions=20,
        K_S=0,
        K_q=0,
        classes=classes,
    )


def test_paper_defaults_are_explicit_and_class_ranks_are_n_minus_ell() -> None:
    config = paper_config()
    assert (config.outer_replicates, config.rounds) == (3, 5)
    assert (config.calibration_missions, config.deployment_missions) == (200, 200)
    assert (config.K_S, config.K_q, config.admm_steps, config.simulation_steps) == (
        100,
        100,
        50,
        100,
    )
    assert math.isclose(config.mission_duration_sec, 10.0)
    assert math.isclose(config.pinball_gamma_0, 0.05)
    assert math.isclose(config.pinball_exponent, 0.75)
    assert math.isclose(config.proximal_weight, 0.25)
    assert set(config.classes) == {"uav", "ugv"}
    assert all(cls.n == 3 and cls.ell == 1 for cls in config.classes.values())
    assert all(cls.mission_rank == 2 for cls in config.classes.values())
    assert all(math.isclose(cls.initial_margin, 0.2) for cls in config.classes.values())
    assert all(math.isclose(cls.configured_kappa, 0.6) for cls in config.classes.values())
    changed = paper_config(alpha=0.2, delta_cal=0.95)
    assert all(math.isclose(cls.alpha, 0.2) for cls in changed.classes.values())
    assert all(math.isclose(cls.delta_cal, 0.95) for cls in changed.classes.values())


def test_graph_validation_rejects_bad_topology_and_reports_diameter() -> None:
    graph = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}}
    assert graph_diameter(graph) == 2
    assert validate_graph(graph) == graph
    with pytest.raises(ValueError, match="connected"):
        validate_graph({"a": set(), "b": set()})
    with pytest.raises(ValueError, match="undirected"):
        validate_graph({"a": {"b"}, "b": set()})
    with pytest.raises(ValueError, match="self"):
        validate_graph({"a": {"a"}})


def test_tagged_flooding_preserves_empty_owners_and_rejects_duplicate_tags() -> None:
    graph = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}}
    flooded, rounds = flood_tagged(
        graph,
        {"a": {"x": 1.0}, "b": {}, "c": {"y": 2.0}},
    )
    assert rounds == 2
    assert all(set(values) == {"x", "y"} for values in flooded.values())
    with pytest.raises(ValueError, match="multiple owners"):
        flood_tagged(graph, {"a": {"x": 1.0}, "b": {"x": 1.0}, "c": {}})


def test_pinball_uses_strict_ties_and_local_step() -> None:
    # At q=2 the value equal to q is not counted in the strict subgradient.
    assert pinball_gradient([1.0, 2.0, 3.0], 2.0, 0.5) == pytest.approx(-0.5)
    assert local_pinball_step([1.0, 2.0, 3.0], 2.0, 2.0, 0.5, 0.1) == pytest.approx(2.05)


@pytest.mark.parametrize("iterations", [0, 1, 8])
def test_finite_recovery_is_an_upper_bound_even_before_pinball_converges(iterations: int) -> None:
    graph = {
        "a": {"b"},
        "b": {"a", "c"},
        "c": {"b", "d"},
        "d": {"c", "e"},
        "e": {"d"},
    }
    stores = {
        "a": {"a0": 1.0},
        "b": {},
        "c": {"c0": 3.0},
        "d": {"d0": 2.0},
        "e": {"e0": 2.0},
    }
    result = pinball_order_statistic(stores, rank=3, iterations=iterations, graph=graph)
    assert result.value >= exact_order_statistic(stores, 3)
    assert result.sample_count == 4
    assert result.pinball_history.shape == (iterations + 1, len(graph))


def test_order_statistic_allows_empty_owners_and_counts_ties_by_tag() -> None:
    graph = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}}
    stores = {"a": {"left": 2.0}, "b": {}, "c": {"right": 2.0, "high": 5.0}}
    result = pinball_order_statistic(stores, rank=2, iterations=0, graph=graph)
    assert result.value == 2.0
    assert exact_order_statistic(stores, 2) == 2.0
    assert mission_score({"r1": 2.0, "r2": 2.0, "r3": 5.0}, 2) == 2.0


def test_calibration_preserves_robot_id_ownership_and_rejects_wrong_class_count() -> None:
    config = _small_config()
    graph = {"uav0": {"uav1"}, "uav1": {"uav0"}}
    data = {
        "uav": [
            {"uav0": 0.1, "uav1": 0.2, "uav2": 0.3}
            for _ in range(config.calibration_missions)
        ],
        "ugv": [
            [0.1, 0.2, 0.3]
            for _ in range(config.calibration_missions)
        ],
    }
    with pytest.raises(ValueError, match="outside the graph"):
        calibrate_round("maicp", None, data, config, graph=graph)
    bad = dict(data)
    bad["uav"] = [{"uav0": 0.1, "uav1": 0.2} for _ in range(config.calibration_missions)]
    with pytest.raises(ValueError, match="exactly 3"):
        calibrate_round("centralized", None, bad, config, graph=graph)


def test_split_cp_uses_m_plus_one_without_clipping() -> None:
    threshold, rank = standard_split_cp([1.0, 2.0, 3.0], alpha=0.1)
    assert rank == 4
    assert threshold == float("inf")
    threshold, rank = standard_split_cp([1.0, 2.0, 3.0, 4.0], alpha=0.25)
    assert rank == 4
    assert threshold == 4.0


def test_coupled_margin_update_is_feasible_and_uses_scaled_distance() -> None:
    config = _small_config()
    result = solve_margin_update(
        {"uav": 0.2, "ugv": 0.2},
        {"uav": 0.1, "ugv": 0.12},
        {"uav": 0.6, "ugv": 0.6},
        config,
    )
    assert result.status == "optimal"
    assert result.certified is False  # configured kappa is not a certificate
    assert all(result.margins[key] >= result.thresholds[key] for key in result.margins)
    assert all(slack >= -2e-6 for slack in result.constraint_slacks.values())
    expected_delta = np.linalg.norm(
        [result.margins[key] - 0.2 for key in ("uav", "ugv")]
    )
    assert result.normalized_change == pytest.approx(expected_delta, abs=1e-5)


def test_margin_update_fails_explicitly_when_infeasible() -> None:
    config = paper_config(
        classes={
            "uav": ClassConfig(name="uav", margin_max=0.3),
            "ugv": ClassConfig(name="ugv", margin_max=0.3),
        }
    )
    with pytest.raises(RuntimeError, match="infeasible"):
        solve_margin_update(
            {"uav": 0.2, "ugv": 0.2},
            {"uav": 0.29, "ugv": 0.29},
            {"uav": 0.6, "ugv": 0.6},
            config,
        )


def test_calibration_methods_and_ros_callback_share_the_same_data_contract() -> None:
    config = _small_config()
    data = {
        "uav": [[0.05 + 0.001 * i, 0.06 + 0.001 * i, 0.07 + 0.001 * i] for i in range(20)],
        "ugv": [[0.04 + 0.001 * i, 0.05 + 0.001 * i, 0.06 + 0.001 * i] for i in range(20)],
    }
    graph = {"a": {"b"}, "b": {"a"}}
    original = {key: [list(mission) for mission in missions] for key, missions in data.items()}
    results = {
        method: calibrate_round(method, None, data, config, graph=graph)
        for method in ("maicp", "centralized", "unshifted", "nominal", "fixed_margin")
    }
    assert results["nominal"].margins == {"uav": 0.0, "ugv": 0.0}
    assert results["fixed_margin"].margins == results["fixed_margin"].thresholds
    assert all(results["maicp"].thresholds[key] >= results["centralized"].thresholds[key] for key in data)
    assert results["unshifted"].kappa == {"uav": 0.0, "ugv": 0.0}
    assert data == original  # calibration data are not consumed or mutated

    calls: list[tuple[int, int]] = []

    def callback(stores, rank, iterations, callback_config):
        calls.append((rank, iterations))
        return exact_order_statistic(stores, rank)

    callback_result = calibrate_round(
        "maicp", None, data, config, graph=graph, order_statistic=callback
    )
    assert calls
    assert callback_result.thresholds["uav"] <= 1.0


def test_margin_update_callback_is_replicated_only_for_distributed_methods() -> None:
    config = _small_config()
    data = {
        "uav": [[0.05 + 0.001 * i, 0.06 + 0.001 * i, 0.07 + 0.001 * i] for i in range(20)],
        "ugv": [[0.04 + 0.001 * i, 0.05 + 0.001 * i, 0.06 + 0.001 * i] for i in range(20)],
    }
    graph = {"a": {"b"}, "b": {"a"}}
    calls: list[tuple[dict, dict, dict]] = []

    def callback(current, thresholds, kappa, callback_config):
        calls.append((dict(current), dict(thresholds), dict(kappa)))
        return solve_margin_update(current, thresholds, kappa, callback_config)

    centralized = calibrate_round(
        "centralized", None, data, config, graph=graph, margin_update=callback
    )
    assert calls == []
    maicp = calibrate_round(
        "maicp", None, data, config, graph=graph, margin_update=callback
    )
    unshifted = calibrate_round(
        "unshifted", None, data, config, graph=graph, margin_update=callback
    )
    assert len(calls) == 2
    assert calls[0][2] == {"uav": 0.6, "ugv": 0.6}
    assert calls[1][2] == {"uav": 0.0, "ugv": 0.0}
    assert maicp.margin_update is not None
    assert unshifted.margin_update is not None
    assert maicp.margin_update.replica_count == 1
    assert unshifted.margin_update.replica_count == 1
    assert centralized.margin_update is not None
