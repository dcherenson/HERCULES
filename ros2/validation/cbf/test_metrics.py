from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rural_nominal"))
from render_validation import cbf_metrics


def test_old_log_without_cbf_fields_is_accepted():
    assert cbf_metrics([{"step": 0}]) == {"available": False, "steps": 0, "entries": 0}


def test_cbf_metrics_counts_interventions_and_fallbacks():
    records = [{"cbf": {"entries": [
        {"enabled": True, "fallback": False, "intervention_norm": 0.0,
         "minimum_barrier": 1.0, "final_feasible": True, "effective_method": "mestres"},
        {"enabled": True, "fallback": True, "intervention_norm": 0.5,
         "minimum_barrier": None, "final_feasible": False, "effective_method": "mestres"},
    ]}}]
    result = cbf_metrics(records)
    assert result["enabled_entries"] == 2
    assert result["fallback_entries"] == 1
    assert result["fraction_intervened"] == 0.5
    assert result["minimum_reported_barrier"] == 1.0
