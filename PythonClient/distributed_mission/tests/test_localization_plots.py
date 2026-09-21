import json

import numpy as np

from modules.localization_plots import (
    compute_localization_metrics,
    extract_localization_series,
    has_localization_estimates,
    localization_log_entry,
    plot_localization_truth_vs_estimate,
    render_localization_plots,
)


def _record(step, estimate=True):
    truth = {
        "position": [float(step), 2.0 * float(step), -1.0],
        "velocity": [1.0, 2.0, 0.0],
        "yaw": np.pi - 0.1 * step,
        "yaw_rate": 0.1,
    }
    record = {
        "step": step,
        "dt": 0.1,
        "timestamp": 0.1 * step,
        "states": {"Drone1": dict(truth)},
    }
    if estimate:
        record["localization"] = {
            "Drone1": {
                "truth": truth,
                "estimate": {
                    "position": [float(step) + 0.1, 2.0 * float(step) - 0.2, -1.0],
                    "velocity": [1.0, 2.0, 0.0],
                    # This is intentionally on the opposite side of the
                    # +/-pi wrap from the truth at step zero.
                    "yaw": -np.pi + 0.1 * step,
                    "covariance": [
                        [0.04, 0.0, 0.0],
                        [0.0, 0.09, 0.0],
                        [0.0, 0.0, 0.04],
                    ],
                },
            }
        }
    return record


def test_extract_localization_series_and_metrics_use_position_sigma_and_wrapped_yaw():
    records = [_record(step) for step in range(3)]
    series = extract_localization_series(records)
    value = series["Drone1"]

    assert np.allclose(value["position_sigma"], [[0.2, 0.3]] * 3)
    assert np.all(value["truth_available"] & value["estimate_available"])
    # At step zero the raw yaw difference is approximately -2*pi, but the
    # shortest angular error is zero.
    metrics = compute_localization_metrics(records)
    item = metrics["agents"]["Drone1"]
    assert np.isclose(item["x_rmse_m"], 0.1)
    assert np.isclose(item["y_rmse_m"], 0.2)
    assert item["yaw_max_abs_error_rad"] < 0.5
    assert item["x_one_sigma_coverage"] == 1.0
    assert item["y_one_sigma_coverage"] == 1.0
    assert np.isclose(item["yaw_one_sigma_coverage"], 2.0 / 3.0)


def test_localization_plot_writes_for_current_and_legacy_logs(tmp_path):
    current_path = tmp_path / "localization.png"
    assert has_localization_estimates([_record(0)])
    plot_localization_truth_vs_estimate([_record(0), _record(1)], str(current_path))
    assert current_path.is_file() and current_path.stat().st_size > 0

    legacy = [_record(0, estimate=False), _record(1, estimate=False)]
    assert not has_localization_estimates(legacy)
    legacy_metrics = compute_localization_metrics(legacy)
    assert legacy_metrics["available"] is False
    assert legacy_metrics["legacy_schema"] is True
    legacy_path = tmp_path / "legacy_localization.png"
    plot_localization_truth_vs_estimate(legacy, str(legacy_path))
    assert legacy_path.is_file() and legacy_path.stat().st_size > 0


def test_localization_log_entry_is_json_friendly_and_keeps_covariance_alias():
    entry = localization_log_entry(
        {"position": [1.0, 2.0, 3.0], "velocity": [0.0, 0.0, 0.0], "yaw": 0.2},
        {
            "estimated_position": np.array([1.1, 1.9, 3.0]),
            "estimated_velocity": np.zeros(3),
            "yaw": 0.25,
            "uncertainty_covariance": np.eye(3) * 0.25,
        },
        vehicle_type="drone",
    )
    json.dumps(entry, allow_nan=False)
    assert entry["vehicle_type"] == "drone"
    assert np.allclose(entry["estimate"]["covariance"], [[0.25, 0.0], [0.0, 0.25]])
    assert entry["estimate"]["uncertainty_covariance"] == entry["estimate"]["covariance"]


def test_canonical_localization_object_supports_algorithm_flags_diagnostics_and_per_agent_names(tmp_path):
    records = []
    for step in range(2):
        records.append({
            "step": step,
            "dt": 0.1,
            "states": {"Drone1": {"position": [float(step), 0.0, -5.0], "yaw": 0.0}},
            "localization": {
                "algorithm": "gs_ci",
                "agents": {
                    "Drone1": {
                        "pose": [float(step) + 0.5, 0.0, 0.1],
                        "velocity": [0.0, 0.0],
                        "covariance": [1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.25],
                        "valid": step == 0,
                        "stale": step == 1,
                        "initialized": True,
                        "diagnostics": {
                            "accepted_relative_updates": 2,
                            "accepted_communication_updates": 3,
                            "rejected_updates": 1,
                        },
                    }
                },
            },
        })
    series = extract_localization_series(records)["Drone1"]
    assert series["algorithm_name"] == "gs_ci"
    assert np.allclose(series["position_sigma"][0], [1.0, 2.0])
    assert np.isclose(series["yaw_sigma"][0], 0.5)
    assert not series["estimate_available"][1]
    metrics = compute_localization_metrics(records)["agents"]["Drone1"]
    assert metrics["algorithm"] == "gs_ci"
    assert metrics["stale_samples"] == 1
    assert metrics["accepted_relative_updates"] == 4
    assert metrics["accepted_communication_updates"] == 6
    paths = render_localization_plots(records, str(tmp_path), stem="run")
    assert len(paths) == 1
    assert "gs_ci" in paths[0]
    assert paths[0].endswith("Drone1.png")


def test_outer_flags_create_gaps_for_nested_python_entries_and_filter_target_truth(tmp_path):
    truth = {"position": [0.0, 0.0, 0.0], "yaw": 0.0}
    records = [
        {
            "step": 0,
            "states": {
                "Drone1": truth,
                "Target1": {"position": [10.0, 10.0, 0.0], "yaw": 0.0},
            },
            "localization": {
                "algorithm": "recursive",
                "Drone1": {
                    "truth": truth,
                    "estimate": {"position": [0.1, 0.0, 0.0], "yaw": 0.0},
                    "valid": False,
                    "initialized": True,
                },
            },
        },
    ]
    series = extract_localization_series(records)
    assert "Drone1" in series
    assert not series["Drone1"]["estimate_available"][0]
    assert np.isnan(series["Drone1"]["estimate_position"][0, 0])
    paths = render_localization_plots(records, str(tmp_path), stem="run")
    assert len(paths) == 1
    assert paths[0].endswith("Drone1.png")


def test_both_algorithms_render_all_eight_controlled_agents_with_distinct_names(tmp_path):
    controlled = [
        "Drone1", "Drone2", "SimpleFlight", "Drone4", "Drone5",
        "Husky1", "Husky2", "Husky3",
    ]
    rendered = {}
    for algorithm in ("recursive_decentralized", "gs_ci"):
        states = {}
        agents = {}
        for index, agent in enumerate(controlled):
            x = float(index)
            truth = {"position": [x, -x, 0.0], "yaw": 0.1 * x}
            states[agent] = truth
            agents[agent] = {
                "pose": [x + 0.1, -x - 0.1, 0.1 * x + 0.01],
                "velocity": [0.0, 0.0],
                "covariance": [1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.25],
                "initialized": True,
                "valid": True,
                "stale": False,
            }
        records = [{
            "step": 0,
            "dt": 0.1,
            "states": states,
            "localization": {"algorithm": algorithm, "agents": agents},
        }]
        output_directory = tmp_path / algorithm
        paths = render_localization_plots(records, str(output_directory), stem="comparison")
        assert len(paths) == len(controlled)
        assert {path.rsplit("_", 1)[-1].removesuffix(".png") for path in paths} == set(controlled)
        assert all(algorithm in path for path in paths)
        rendered[algorithm] = set(paths)

    assert rendered["recursive_decentralized"].isdisjoint(rendered["gs_ci"])
