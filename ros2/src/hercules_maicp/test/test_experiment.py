"""Focused protocol and score-contract tests for the MAICP runner."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import sys

import pytest

PACKAGE_ROOT = Path(__file__).parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from hercules_maicp.dynamics import AffineDynamicsModel, fit_affine_models, load_dynamics_models
from hercules_maicp.experiment import DEFAULT_AGENTS, ExperimentRunner, MissionRequest, RunnerConfig
from hercules_maicp.scores import MissionFailedError, MissionScoreError, collision_scores, score_records


def _vehicle_types() -> dict[str, str]:
    return {
        agent: ("drone" if index < 3 else "ugv")
        for index, agent in enumerate(DEFAULT_AGENTS)
    }


def _collision_records(count: int = 4) -> list[dict]:
    records = []
    for index in range(count):
        states = {}
        controls = {}
        collisions = {}
        for agent in DEFAULT_AGENTS:
            states[agent] = {
                "position": [float(index), 0.0, 0.0],
                "velocity": [0.1 * index, 0.0, 0.0],
                "source_timestamp": 0.1 * index,
            }
            controls[agent] = [1.0, 0.0]
            collisions[agent] = {"available": True, "relevant": False}
        records.append({
            "step": index,
            "dt": 0.1,
            "vehicle_types": _vehicle_types(),
            "states": states,
            "safe_controls": controls,
            "collisions": collisions,
        })
    return records


def test_live_command_and_reset_are_argv_templates() -> None:
    config = RunnerConfig(
        cases=("tracking",),
        methods=("nominal",),
        mode="in_process",
        steps=100,
    )
    runner = ExperimentRunner(config, mission_executor=lambda request: None)
    request = MissionRequest(
        "tracking", "nominal", 0, 0, "deployment", 0, 12, 0.0, 0.0, 100,
        Path("out.jsonl"), Path("reset.json"), None, "localhost", 10.0,
    )
    command = runner.build_command(request)
    reset = runner.build_reset_command(request)
    assert command[0:4] == ["ros2", "launch", "hercules_maicp", "case_study.launch.py"]
    assert "case:=tracking" in command
    assert "seed:=12" in command
    assert "duration:=10.0" in command
    assert "dynamics_file:=" not in command
    assert reset[:4] == ["ros2", "run", "hercules_maicp", "reset_scene"]
    assert "--seed" in reset and "12" in reset


def test_collision_score_uses_actual_source_time_and_reports_collision_separately() -> None:
    models = {
        "uav": AffineDynamicsModel(
            "uav", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            provenance={"fit_source": "separate_training_logs", "training_logs": ["train.jsonl"]},
        ),
        "ugv": AffineDynamicsModel(
            "ugv", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            provenance={"fit_source": "separate_training_logs", "training_logs": ["train.jsonl"]},
        ),
    }
    records = _collision_records()
    records[2]["states"]["Drone1"]["source_timestamp"] = 0.25
    records[2]["states"]["Drone1"]["velocity"][0] = 0.25
    records[2]["collisions"]["Drone1"]["relevant"] = True
    scores, diagnostics = collision_scores(records, models)
    assert scores["Drone1"] == pytest.approx(0.0)
    result = score_records(records, "collision", models=models, expected_per_class={"uav": 3, "ugv": 3})
    assert result.collision_free is False
    assert result.diagnostics["collision_free"]["events"] == 1
    assert result.diagnostics["fail_safe"] is False


def test_collision_score_accepts_logged_frozen_prediction() -> None:
    records = _collision_records()
    for row in records:
        row["maicp"] = {
            "learned_dynamics": {agent: [0.0, 0.0] for agent in DEFAULT_AGENTS}
        }
    scores, diagnostics = collision_scores(records, None)
    assert scores["Drone1"] == pytest.approx(0.0)
    assert diagnostics["agents"]["Drone1"]["dynamics_sources"] == ["inline_prediction"]


def test_collision_duplicate_packet_is_legal_but_changed_state_is_rejected() -> None:
    records = _collision_records()
    for row in records:
        row["maicp"] = {
            "learned_dynamics": {agent: [0.0, 0.0] for agent in DEFAULT_AGENTS}
        }

    # Row 2 is a repeated packet for the state at row 1.  Different commands
    # are held across the repeated packets, so the interval to row 3 must use
    # the largest residual rather than fabricate a zero-time acceleration.
    records[2]["states"]["Drone1"] = deepcopy(records[1]["states"]["Drone1"])
    records[1]["safe_controls"]["Drone1"] = [2.0, 0.0]
    records[2]["safe_controls"]["Drone1"] = [3.0, 0.0]
    scores, diagnostics = collision_scores(records, None)
    assert scores["Drone1"] == pytest.approx(2.0)
    assert diagnostics["duplicate_source_samples"] == 1
    assert diagnostics["agents"]["Drone1"]["duplicate_source_samples"] == 1
    assert diagnostics["agents"]["Drone1"]["unique_intervals"] == 2
    assert diagnostics["agents"]["Drone1"]["command_observations"] == 3

    changed = _collision_records()
    changed[2]["states"]["Drone1"]["source_timestamp"] = changed[1]["states"]["Drone1"]["source_timestamp"]
    changed[2]["states"]["Drone1"]["position"][0] += 0.01
    with pytest.raises(MissionScoreError, match="inconsistent duplicate"):
        collision_scores(changed, {
            "uav": AffineDynamicsModel(
                "uav", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                provenance={"fit_source": "separate_training_logs", "training_logs": ["train.jsonl"]},
            ),
            "ugv": AffineDynamicsModel(
                "ugv", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                provenance={"fit_source": "separate_training_logs", "training_logs": ["train.jsonl"]},
            ),
        })


def test_failed_and_truncated_jsonl_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mission.jsonl"
    path.write_text(json.dumps({"step": 0}) + "\n", encoding="utf-8")
    (tmp_path / "mission.jsonl.failed").write_text("controller failed\n", encoding="utf-8")
    with pytest.raises(MissionFailedError):
        from hercules_maicp.scores import load_mission_jsonl

        load_mission_jsonl(path, expected_steps=2)
    (tmp_path / "mission.jsonl.failed").unlink()
    with pytest.raises(MissionFailedError, match="truncated"):
        from hercules_maicp.scores import load_mission_jsonl

        load_mission_jsonl(path, expected_steps=2)


def test_frozen_affine_fit_has_explicit_training_provenance(tmp_path: Path) -> None:
    path = tmp_path / "training.jsonl"
    rows = []
    positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 0.0)]
    for index, (x, y) in enumerate(positions):
        states = {}
        controls = {}
        for agent in DEFAULT_AGENTS:
            if index == 0:
                velocity = [0.0, 0.0]
            else:
                px, py = positions[index - 1]
                velocity = [0.1 * (1.0 + 0.1 * px + 0.2 * py), 0.1 * (0.3 + 0.4 * px + 0.5 * py)]
            states[agent] = {"position": [x, y, 0.0], "velocity": velocity, "source_timestamp": 0.1 * index}
            controls[agent] = [1.0, 0.3]
        rows.append({"vehicle_types": _vehicle_types(), "states": states, "safe_controls": controls})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "dynamics.json"
    fit_affine_models([path], output)
    models = load_dynamics_models(output)
    assert set(models) == {"uav", "ugv"}
    assert models["uav"].provenance["fit_source"] == "separate_training_logs"


def test_frozen_fit_collapses_duplicate_intervals_and_averages_commands(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-training.jsonl"
    positions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 0.0)]
    velocities = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0)]
    rows = []
    for index, ((x, y), (vx, vy)) in enumerate(zip(positions, velocities)):
        states = {}
        controls = {}
        for agent in DEFAULT_AGENTS:
            states[agent] = {
                "position": [x, y, 0.0],
                "velocity": [vx, vy, 0.0],
                "source_timestamp": 0.1 * index,
            }
            controls[agent] = [1.0, 0.0]
        rows.append({"vehicle_types": _vehicle_types(), "states": states, "safe_controls": controls})
    duplicate = deepcopy(rows[1])
    duplicate["safe_controls"] = {agent: [4.0, 0.0] for agent in DEFAULT_AGENTS}
    rows.insert(2, duplicate)
    rows[1]["safe_controls"] = {agent: [2.0, 0.0] for agent in DEFAULT_AGENTS}
    rows[4]["safe_controls"] = {agent: [3.0, 0.0] for agent in DEFAULT_AGENTS}
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    models = fit_affine_models([path])
    # Four unique transitions per agent, including the averaged command on
    # the duplicated interval; six agents contribute to each class.
    assert models["uav"].sample_count == 12
    assert models["uav"].provenance["duplicate_source_samples"] == 6
    assert models["uav"].evaluate((1.0, 0.0))[0] == pytest.approx(-2.0)
