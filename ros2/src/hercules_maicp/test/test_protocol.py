"""Protocol-level coverage for the in-process MA-ICP experiment runner."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import pytest

PACKAGE_ROOT = Path(__file__).parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from hercules_maicp.experiment import IN_PROCESS_MODE, DEFAULT_AGENTS, ExperimentRunner, RunnerConfig
from hercules_maicp.scores import MissionScore


METHODS = ("nominal", "fixed_margin", "centralized", "unshifted", "maicp")


def _score(request: object) -> MissionScore:
    """Return a small deterministic tracking score for an injected mission."""

    phase = getattr(request, "phase")
    value = 0.01 if phase == "calibration" else 0.02
    robot_scores = {agent: value for agent in DEFAULT_AGENTS}
    return MissionScore(
        case="tracking",
        robot_scores=robot_scores,
        class_scores={"uav": value, "ugv": value},
        collision_free=True,
        diagnostics={"all_returned_home": True},
        record_count=2,
    )


def test_runner_protocol_all_methods_and_paired_deployment_seeds(tmp_path: Path) -> None:
    calls = []

    def mission_executor(request: object) -> MissionScore:
        calls.append(request)
        return _score(request)

    def order_statistic(stores: object, rank: int, iterations: int, config: object) -> float:
        # The injected mission scores are tied, so this is a deterministic
        # finite upper bound for both distributed methods.
        return 0.01

    config = RunnerConfig(
        cases=("tracking",),
        methods=METHODS,
        repeats=1,
        rounds=2,
        calibration_missions=10,
        deployment_missions=2,
        steps=2,
        mode=IN_PROCESS_MODE,
        truth_seed=31415,
        output_dir=tmp_path / "runs",
    )
    report = ExperimentRunner(
        config,
        mission_executor=mission_executor,
        order_statistic=order_statistic,
    ).run()

    rounds = {
        method: report["cases"]["tracking"][method][0]["rounds"]
        for method in METHODS
    }
    assert len(calls) == 90  # 4 nominal + 14 fixed + 24 for each other method.

    # Nominal has no calibration batch and remains at zero margins.
    for entry in rounds["nominal"]:
        assert entry["calibration"]["requested"] == 0
        assert entry["calibration"]["result"] is None
        assert entry["margins"] == {"uav": 0.0, "ugv": 0.0}

    # Fixed margin calibrates once, then reuses the pilot threshold in round 2.
    assert rounds["fixed_margin"][0]["calibration"]["requested"] == 10
    assert rounds["fixed_margin"][1]["calibration"]["requested"] == 0
    assert rounds["fixed_margin"][0]["margins"] == rounds["fixed_margin"][1]["margins"]
    assert rounds["fixed_margin"][0]["margins"]["uav"] == pytest.approx(0.01)
    assert all((request.margin_uav, request.margin_ugv) == (0.0, 0.0)
               for request in calls
               if request.method == "fixed_margin" and request.phase == "calibration")

    # Centralized and MA-ICP retain the configured sensitivity; unshifted does
    # not. The core intentionally reports all configured variants uncertified.
    for method in ("centralized", "maicp"):
        for entry in rounds[method]:
            assert entry["calibration"]["requested"] == 10
            result = entry["calibration"]["result"]
            assert result["kappa"] == {"uav": 0.6, "ugv": 0.6}
            assert result["certified"] is False
    for entry in rounds["unshifted"]:
        assert entry["calibration"]["requested"] == 10
        assert entry["calibration"]["result"]["kappa"] == {"uav": 0.0, "ugv": 0.0}

    grouped = defaultdict(list)
    for request in calls:
        grouped[(request.method, request.phase, request.round_index)].append(request)

    # A runner batch uses one margin pair throughout that batch. Deployment
    # requests must also agree with the margin recorded for that round.
    for method in METHODS:
        for round_index in range(2):
            for phase in ("calibration", "deployment"):
                batch = grouped[(method, phase, round_index)]
                if not batch:
                    continue
                margin_pairs = {(item.margin_uav, item.margin_ugv) for item in batch}
                assert len(margin_pairs) == 1
                if phase == "deployment":
                    expected = rounds[method][round_index]["margins"]
                    assert margin_pairs == {(expected["uav"], expected["ugv"])}

    calibration_seeds = {
        request.seed
        for request in calls
        if request.phase == "calibration"
    }
    deployment_seeds = {
        request.seed
        for request in calls
        if request.phase == "deployment"
    }
    assert calibration_seeds.isdisjoint(deployment_seeds)

    # Calibration is method-specific, while deployment uses paired scenes.
    calibration_by_method = {
        method: {
            request.seed
            for request in calls
            if request.method == method and request.phase == "calibration"
        }
        for method in METHODS
    }
    assert all(
        calibration_by_method[left].isdisjoint(calibration_by_method[right])
        for index, left in enumerate(METHODS)
        for right in METHODS[index + 1 :]
    )
    for round_index in range(2):
        deployment_by_method = {
            method: [
                request.seed
                for request in calls
                if request.method == method
                and request.phase == "deployment"
                and request.round_index == round_index
            ]
            for method in METHODS
        }
        assert len({tuple(seeds) for seeds in deployment_by_method.values()}) == 1
