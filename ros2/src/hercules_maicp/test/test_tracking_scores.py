"""Completed DRWT outputs must be compared with truth at their own epoch."""
from copy import deepcopy

import pytest

from hercules_maicp.scores import MissionScoreError, tracking_scores


def records():
    return [dict(step=i, dt=1.0, states={"Drone1": {}},
                 target_truth=dict(position=[float(i), 0.0], velocity=[1.0, 0.0]),
                 target_tracking={"agents": {"Drone1": dict(
                     epoch_timed_out=False,
                     estimate=dict(active=True, iterations=50, timestamp=0.0,
                                   position=[0.1, 0.0], velocity=[1.0, 0.0]))}})
            for i in range(2)]


def test_tracking_compares_a_held_estimate_with_its_epoch_truth():
    scores, diagnostics = tracking_scores(records(), start_time=1.0)
    assert scores["Drone1"] == pytest.approx(0.1)
    assert diagnostics["agents"]["Drone1"]["max_current_time_position_error"] == pytest.approx(0.9)
    assert diagnostics["agents"]["Drone1"]["max_estimate_age_sec"] == 1.0


@pytest.mark.parametrize("iterations,timed_out", [(49, False), (50, True), (None, False)])
def test_incomplete_or_timed_out_estimates_cannot_be_calibration_samples(iterations, timed_out):
    rows = deepcopy(records())
    entry = rows[1]["target_tracking"]["agents"]["Drone1"]
    entry["epoch_timed_out"] = timed_out
    entry["estimate"]["iterations"] = iterations
    with pytest.raises(MissionScoreError, match="incomplete or timed-out"):
        tracking_scores(rows, start_time=1.0)
