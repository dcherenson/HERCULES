#!/usr/bin/env python3
"""Compare the Python-only and ROS mission log formats.

The comparator is deliberately host-side and dependency-light.  The core
normalization, interpolation, CSV/JSON output, and Markdown report use only
the Python standard library.  Matplotlib is optional and is used only for a
diagnostic plot.

Typical usage::

    python3 ros2/validation/python_ros/compare_python_ros.py \
        runs/python_nominal runs/ros_nominal \
        --output-dir runs/comparison

Each input may be a run directory containing ``mission.jsonl`` or a direct
JSON-lines file.  The two logs are aligned at their mission starts and
resampled onto a common grid (0.1 seconds by default) over their overlapping
time interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class ComparisonError(RuntimeError):
    """An input or comparison error that should be shown without a traceback."""


CONTROLLED_AGENT_DEFAULTS = (
    "Drone1",
    "Drone2",
    "SimpleFlight",
    "Drone4",
    "Drone5",
    "Husky1",
    "Husky2",
    "Husky3",
)

VECTOR_FIELDS = (
    "position",
    "velocity",
    "desired_slot",
    "command",
    "nominal_control",
    "safe_control",
    "tracking_estimate_position",
    "tracking_estimate_velocity",
)

SCALAR_NUMERIC_FIELDS = (
    "yaw",
    "yaw_rate",
    "slot_error",
    "cbf_minimum_barrier",
    "cbf_solver_time_ms",
    "cbf_filter_time_ms",
    "cbf_intervention_norm",
    "cbf_constraint_count",
    "cbf_active_constraints",
    "cbf_distributed_rounds",
    "cbf_primal_residual",
    "cbf_dual_residual",
    "cbf_maximum_row_violation",
    "tracking_position_error_m",
    "tracking_velocity_error_mps",
    "tracking_iterations",
    "tracking_residual",
    "measurement_age_s",
    "obstacle_age_s",
    "obstacle_proxy_count",
)

DISCRETE_FIELDS = (
    "cbf_enabled",
    "cbf_success",
    "cbf_fallback",
    "cbf_deadline_miss",
    "tracking_active",
    "tracking_direct_observation",
    "measurement_valid",
    "measurement_visible",
    "collision_relevant",
    "collision_has_collided",
    "cbf_status",
    "observation_source",
    "obstacle_valid",
    "obstacle_stale",
    "command_rejected",
)

ROW_NUMERIC_FIELDS = (
    "tracking_active_count",
    "tracking_direct_observation_count",
    "tracking_epochs_timed_out",
    "consensus_messages_published",
    "consensus_messages_rejected",
    "handoffs_sent",
    "handoffs_accepted",
    "camera_captures",
    "camera_visible_detections",
    "camera_invalid_detections",
    "camera_rpc_errors",
    "mean_capture_rate_hz",
    "capture_interval_p95_sec",
    "capture_interval_max_sec",
    "mean_image_rpc_duration_sec",
    "max_image_rpc_duration_sec",
    "cpu_percent",
    "memory_mb",
)

TARGET_VECTOR_FIELDS = ("position", "velocity")
TARGET_NUMERIC_FIELDS = ("yaw", "index", "sample_count", "progress")
TARGET_DISCRETE_FIELDS = ("phase", "pattern")

CSV_VECTOR_FIELDS = {
    "position": "position",
    "velocity": "velocity",
    "desired_slot": "desired_slot",
    "command": "command",
    "nominal_control": "nominal_control",
    "safe_control": "safe_control",
    "tracking_estimate_position": "tracking_estimate_position",
    "tracking_estimate_velocity": "tracking_estimate_velocity",
}


def _is_finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if _is_finite(value):
        return float(value)
    return None


def _as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "on", "enabled", "enable", "1"}:
            return True
        if lowered in {"false", "no", "off", "disabled", "disable", "0"}:
            return False
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _as_vector(value: Any, length: int = 3, pad: Optional[float] = None) -> List[Optional[float]]:
    """Return a fixed-width numeric vector, preserving missing components."""

    if isinstance(value, Mapping):
        # Accept either x/y/z fields or a numbered mapping.
        values = []
        for index, axis in enumerate(("x", "y", "z", "w")):
            if axis in value:
                values.append(_as_float(value[axis]))
            elif str(index) in value:
                values.append(_as_float(value[str(index)]))
        if not values:
            value = None
        else:
            value = values

    if isinstance(value, (list, tuple)):
        result = [_as_float(component) for component in value[:length]]
        while len(result) < length:
            result.append(pad)
        return result

    scalar = _as_float(value)
    if scalar is not None:
        return [scalar] + [pad] * (length - 1)
    return [pad] * length


def _as_control(value: Any) -> List[Optional[float]]:
    """Normalize velocity or unicycle commands to three components.

    UGV commands are commonly ``[speed, yaw]``.  The third component is a
    semantically meaningful zero for comparison, unlike a missing position
    component, so controls use zero-padding.
    """

    if value is None:
        return [None, None, None]
    return _as_vector(value, length=3, pad=0.0)


def _vector_norm(value: Sequence[Optional[float]]) -> Optional[float]:
    components = [float(component) for component in value if _is_finite(component)]
    if not components:
        return None
    return math.sqrt(sum(component * component for component in components))


def _vector_distance(
    left: Sequence[Optional[float]], right: Sequence[Optional[float]]
) -> Optional[float]:
    common = []
    for a, b in zip(left, right):
        if _is_finite(a) and _is_finite(b):
            common.append(float(a) - float(b))
    if not common:
        return None
    return math.sqrt(sum(component * component for component in common))


def _wrapped_angle_difference(left: Any, right: Any) -> Optional[float]:
    a = _as_float(left)
    b = _as_float(right)
    if a is None or b is None:
        return None
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _is_target_name(name: str) -> bool:
    return str(name).strip().lower().startswith("target")


def _agent_map(value: Any) -> Mapping[str, Any]:
    return _mapping(value)


def _extract_slot_error(value: Any) -> Optional[float]:
    if isinstance(value, Mapping):
        value = _first(
            value,
            "norm",
            "magnitude",
            "position_error",
            "error",
            "distance",
        )
    if isinstance(value, (list, tuple)):
        return _vector_norm(_as_vector(value, length=3, pad=None))
    return _as_float(value)


def _target_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract target truth while accepting both logger generations."""

    candidate = _first(row, "target_truth", "target")
    if not isinstance(candidate, Mapping) or "position" not in candidate:
        targets = _mapping(row.get("targets"))
        candidate = None
        for name, value in targets.items():
            if _is_target_name(str(name)) and isinstance(value, Mapping):
                candidate = value
                break
    if not isinstance(candidate, Mapping) or "position" not in candidate:
        states = _mapping(row.get("states"))
        candidate = None
        for name, value in states.items():
            if _is_target_name(str(name)) and isinstance(value, Mapping):
                candidate = value
                break
    candidate = _mapping(candidate)
    pattern = _mapping(candidate.get("pattern"))
    index = _first(candidate, "index", "current_index", "target_index")
    if index is None:
        index = _first(pattern, "index", "current_index", "target_index")
    sample_count = _first(candidate, "sample_count", "path_length", "count")
    if sample_count is None:
        sample_count = _first(pattern, "sample_count", "path_length", "count")
    progress = _first(candidate, "progress", "path_progress", "progress_fraction")
    if progress is None:
        progress = _first(pattern, "progress", "path_progress", "progress_fraction")
    if progress is None:
        progress = _first(row, "target_progress", "target_path_progress")
    return {
        "position": _as_vector(_first(candidate, "position", "pos"), 3, None),
        "velocity": _as_vector(_first(candidate, "velocity", "vel"), 3, None),
        "yaw": _as_float(_first(candidate, "yaw", "heading")),
        "index": _as_float(index),
        "sample_count": _as_float(sample_count),
        "progress": _as_float(progress),
        "phase": _first(candidate, "phase"),
        "pattern": _first(candidate, "pattern"),
    }


def _tracking_record(row: Mapping[str, Any], agent: str, target: Mapping[str, Any]) -> Dict[str, Any]:
    tracking = _mapping(row.get("target_tracking"))
    agents = _mapping(_first(tracking, "agents", "estimates", "tracks"))
    info = _mapping(agents.get(agent))
    estimate = _mapping(_first(info, "estimate", "target_estimate"))
    if not estimate and any(key in info for key in ("position", "velocity")):
        estimate = info
    measurement = _mapping(_first(info, "measurement", "observation"))

    estimate_position = _as_vector(_first(estimate, "position", "pos"), 3, None)
    estimate_velocity = _as_vector(_first(estimate, "velocity", "vel"), 3, None)
    position_error = _as_float(_first(info, "position_error_m", "position_error"))
    if position_error is None:
        position_error = _vector_distance(estimate_position, target["position"])
    velocity_error = _as_float(_first(info, "velocity_error_mps", "velocity_error"))
    if velocity_error is None:
        velocity_error = _vector_distance(estimate_velocity, target["velocity"])

    active = _as_bool(_first(info, "active", "enabled", "valid"))
    if active is None:
        active = _as_bool(_first(estimate, "active", "enabled", "valid"))
    direct = _as_bool(_first(info, "direct_observation", "direct", "is_direct"))
    if direct is None:
        source = _first(info, "observation_source", "source")
        if isinstance(source, str):
            direct = source.strip().lower() in {"direct", "direct_observation", "sensor"}
    valid = _as_bool(_first(measurement, "valid", "is_valid"))
    visible = _as_bool(_first(measurement, "visible", "is_visible", "valid"))
    if valid is None:
        valid = _as_bool(_first(info, "measurement_valid", "valid"))
    if visible is None:
        visible = _as_bool(_first(info, "measurement_visible", "visible"))

    return {
        "tracking_estimate_position": estimate_position,
        "tracking_estimate_velocity": estimate_velocity,
        "tracking_position_error_m": position_error,
        "tracking_velocity_error_mps": velocity_error,
        "tracking_iterations": _as_float(
            _first(estimate, "iterations", "consensus_iterations")
            if _first(estimate, "iterations", "consensus_iterations") is not None
            else _first(info, "iterations", "consensus_iterations")
        ),
        "tracking_residual": _as_float(
            _first(estimate, "consensus_residual", "residual")
            if _first(estimate, "consensus_residual", "residual") is not None
            else _first(info, "consensus_residual", "residual")
        ),
        "tracking_active": active,
        "tracking_direct_observation": direct,
        "measurement_valid": valid,
        "measurement_visible": visible,
        "measurement_age_s": _as_float(_first(measurement, "age_s", "age", "timestamp_age")),
        "observation_source": _first(info, "observation_source", "source")
        or _first(measurement, "source", "observation_source")
        or _first(tracking, "observation_source", "source"),
    }


def _cbf_entry(row: Mapping[str, Any], agent: str) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return (entry, top-level CBF object) for Python or ROS schemas."""

    top = _mapping(row.get("cbf"))
    entries = top.get("entries")
    if isinstance(entries, list):
        for candidate in entries:
            if isinstance(candidate, Mapping) and str(candidate.get("agent_id")) == agent:
                return candidate, top
        return {}, top
    candidate = top.get(agent)
    return (_mapping(candidate), top)


def _normalize_cbf(row: Mapping[str, Any], agent: str) -> Dict[str, Any]:
    entry, top = _cbf_entry(row, agent)
    top_is_ros_entries = isinstance(top.get("entries"), list)
    nominal_controls = _agent_map(row.get("nominal_controls"))
    safe_controls = _agent_map(row.get("safe_controls"))

    nominal_value = _first(entry, "nominal_control", "nominal", "u_nominal")
    if nominal_value is None:
        nominal_value = nominal_controls.get(agent)
    safe_value = _first(entry, "safe_control", "safe", "u_safe")
    if safe_value is None:
        safe_value = safe_controls.get(agent)

    has_entry = bool(entry)
    enabled = _as_bool(entry.get("enabled"))
    if enabled is None and top_is_ros_entries:
        enabled = _as_bool(top.get("enabled"))
    if enabled is None and has_entry:
        # Python's per-agent CBF result has no explicit enabled field.
        enabled = True

    success = _as_bool(_first(entry, "success", "final_feasible"))
    fallback = _as_bool(_first(entry, "fallback", "used_fallback"))
    status = _first(entry, "solver_status", "status")
    intervention = _as_float(_first(entry, "intervention_norm", "control_intervention_norm"))
    nominal = _as_control(nominal_value)
    safe = _as_control(safe_value)
    if intervention is None and any(_is_finite(x) for x in nominal) and any(_is_finite(x) for x in safe):
        intervention = _vector_distance(nominal, safe)

    active_constraints = _first(entry, "active_constraints", "active_constraint_count")
    if isinstance(active_constraints, (list, tuple, set, Mapping)):
        active_constraints = len(active_constraints)

    return {
        "nominal_control": nominal,
        "safe_control": safe,
        "cbf_enabled": enabled,
        "cbf_success": success,
        "cbf_fallback": fallback,
        "cbf_status": status,
        "cbf_minimum_barrier": _as_float(_first(entry, "minimum_barrier", "min_barrier")),
        "cbf_solver_time_ms": _as_float(_first(entry, "solver_time_ms", "solve_time_ms")),
        "cbf_filter_time_ms": _as_float(_first(entry, "filter_time_ms", "filter_compute_time_ms")),
        "cbf_intervention_norm": intervention,
        "cbf_constraint_count": _as_float(_first(entry, "constraint_count", "constraints")),
        "cbf_active_constraints": _as_float(active_constraints),
        "cbf_distributed_rounds": _as_float(_first(entry, "distributed_rounds", "rounds")),
        "cbf_primal_residual": _as_float(_first(entry, "primal_residual", "primal_residual_norm")),
        "cbf_dual_residual": _as_float(_first(entry, "dual_residual", "dual_residual_norm")),
        "cbf_maximum_row_violation": _as_float(
            _first(entry, "maximum_row_violation", "max_row_violation")
        ),
        "cbf_deadline_miss": _as_bool(_first(entry, "deadline_miss", "deadline_missed")),
    }


def _row_agents(row: Mapping[str, Any]) -> Iterable[str]:
    keys: set[str] = set()
    for field in (
        "states",
        "vehicle_types",
        "desired_slots",
        "slot_errors",
        "commands",
        "nominal_controls",
        "safe_controls",
        "target_tracking",
        "collisions",
    ):
        value = _mapping(row.get(field))
        if field == "target_tracking":
            value = _mapping(_first(value, "agents", "estimates", "tracks"))
        keys.update(str(key) for key in value.keys())
    cbf = _mapping(row.get("cbf"))
    if isinstance(cbf.get("entries"), list):
        for entry in cbf["entries"]:
            if isinstance(entry, Mapping) and entry.get("agent_id") is not None:
                keys.add(str(entry["agent_id"]))
    else:
        keys.update(str(key) for key in cbf.keys() if key != "schema_version")
    return sorted(key for key in keys if not _is_target_name(key))


def _vehicle_type(row: Mapping[str, Any], agent: str) -> str:
    value = _mapping(row.get("vehicle_types")).get(agent)
    if value is not None:
        return str(value)
    lowered = agent.lower()
    if lowered.startswith("drone") or lowered.startswith("uav"):
        return "drone"
    if lowered.startswith("ugv") or lowered.startswith("rover"):
        return "ugv"
    return "unknown"


def _normalize_agent(row: Mapping[str, Any], agent: str, target: Mapping[str, Any]) -> Dict[str, Any]:
    states = _mapping(row.get("states"))
    state = _mapping(states.get(agent))
    desired = _mapping(row.get("desired_slots")).get(agent)
    slot_error = _mapping(row.get("slot_errors")).get(agent)
    commands = _mapping(row.get("commands")).get(agent)
    obstacle = _mapping(_mapping(row.get("obstacles")).get(agent))
    command_diagnostics = _mapping(_mapping(row.get("command_diagnostics")).get(agent))
    tracking = _tracking_record(row, agent, target)
    cbf = _normalize_cbf(row, agent)
    collision = _mapping(row.get("collisions")).get(agent)
    collision = _mapping(collision)
    timing = _mapping(row.get("timing"))

    obstacle_valid = _as_bool(_first(obstacle, "valid", "sensor_valid"))
    obstacle_age = _as_float(_first(obstacle, "age", "age_s"))
    obstacle_stale = _as_bool(_first(obstacle, "stale", "sensor_stale"))
    if obstacle_stale is None and obstacle_valid is not None:
        obstacle_stale = not obstacle_valid
    result: Dict[str, Any] = {
        "vehicle_type": _vehicle_type(row, agent),
        "position": _as_vector(_first(state, "position", "pos"), 3, None),
        "velocity": _as_vector(_first(state, "velocity", "vel"), 3, None),
        "yaw": _as_float(_first(state, "yaw", "heading")),
        "yaw_rate": _as_float(_first(state, "yaw_rate", "angular_velocity")),
        "desired_slot": _as_vector(desired, 3, None),
        "slot_error": _extract_slot_error(slot_error),
        "command": _as_control(commands),
        "command_rejected": _as_bool(_first(command_diagnostics, "rejected", "command_rejected")),
        "obstacle_valid": obstacle_valid,
        "obstacle_stale": obstacle_stale,
        "obstacle_age_s": obstacle_age,
        "obstacle_proxy_count": _as_float(
            _first(obstacle, "count", "proxy_count")
            if _first(obstacle, "count", "proxy_count") is not None
            else (len(obstacle.get("proxies", [])) if isinstance(obstacle.get("proxies"), list) else None)
        ),
        "collision_relevant": _as_bool(_first(collision, "relevant", "is_relevant")),
        "collision_has_collided": _as_bool(
            _first(collision, "has_collided", "collision", "collided")
        ),
        "timing_cycle_ms": _as_float(_first(timing, "cycle_ms", "total_cycle_ms")),
        "timing_deadline_miss": _as_bool(_first(timing, "deadline_miss", "deadline_missed")),
    }
    result.update(tracking)
    result.update(cbf)
    if result["cbf_deadline_miss"] is None:
        result["cbf_deadline_miss"] = result["timing_deadline_miss"]
    return result


def discover_log(run_or_log: os.PathLike[str] | str) -> Path:
    """Resolve a direct JSONL file or a run directory.

    ``mission.jsonl`` is preferred.  A directory with exactly one JSONL file
    is also accepted, while ambiguous directories fail with an actionable
    message instead of silently selecting the wrong run.
    """

    path = Path(run_or_log).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise ComparisonError(f"Run path does not exist: {path}")
    preferred = path / "mission.jsonl"
    if preferred.is_file():
        return preferred
    direct = sorted(path.glob("*.jsonl"))
    if len(direct) == 1:
        return direct[0]
    nested = sorted(path.rglob("mission.jsonl"))
    if len(nested) == 1:
        return nested[0]
    if not direct and not nested:
        raise ComparisonError(
            f"No mission.jsonl (or unique *.jsonl) found under run directory: {path}"
        )
    candidates = direct + [candidate for candidate in nested if candidate not in direct]
    names = ", ".join(str(candidate) for candidate in candidates[:8])
    suffix = " ..." if len(candidates) > 8 else ""
    raise ComparisonError(f"Ambiguous run directory {path}; choose a log explicitly. Candidates: {names}{suffix}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSON-lines, also accepting a JSON array for convenience."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ComparisonError(f"Unable to read {path}: {exc}") from exc
    stripped = text.strip()
    if not stripped:
        raise ComparisonError(f"Log is empty: {path}")
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ComparisonError(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(payload, list) or not all(isinstance(item, Mapping) for item in payload):
            raise ComparisonError(f"JSON array in {path} must contain objects")
        return [dict(item) for item in payload]

    records: List[Dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ComparisonError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ComparisonError(f"Expected an object at {path}:{line_number}")
        records.append(dict(value))
    if not records:
        raise ComparisonError(f"Log contains no JSON records: {path}")
    return records


def _raw_time(row: Mapping[str, Any], index: int, default_dt: float) -> float:
    timestamp = _as_float(row.get("timestamp"))
    if timestamp is not None:
        return timestamp
    step = _as_float(row.get("step"))
    if step is not None:
        row_dt = _as_float(row.get("dt")) or default_dt
        return step * row_dt
    return index * default_dt


def _row_diagnostics(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize run-wide counters/health fields shared by both loggers."""
    tracking = _mapping(row.get("target_tracking"))
    observation = _mapping(row.get("target_observation_diagnostics"))
    performance = _mapping(_first(row, "performance", "resources", "resource_usage"))
    handoffs = tracking.get("handoffs")
    handoff_count = len(handoffs) if isinstance(handoffs, list) else None
    active_count = _first(tracking, "active_track_count")
    direct_count = _first(tracking, "direct_observation_count")
    if active_count is None:
        active_count = sum(
            1 for value in _mapping(tracking.get("agents")).values()
            if _as_bool(_first(_mapping(value), "active")) is True
        )
    if direct_count is None:
        direct_count = sum(
            1 for value in _mapping(tracking.get("agents")).values()
            if _as_bool(_first(_mapping(value), "direct_observation", "direct")) is True
        )
    return {
        "tracking_active_count": _as_float(active_count),
        "tracking_direct_observation_count": _as_float(direct_count),
        "tracking_epochs_timed_out": _as_float(_first(tracking, "epochs_timed_out", "epoch_timeouts")),
        "consensus_messages_published": _as_float(_first(tracking, "consensus_messages_published")),
        "consensus_messages_rejected": _as_float(_first(tracking, "consensus_messages_rejected")),
        "handoffs_sent": _as_float(_first(tracking, "handoffs_sent")) if _first(tracking, "handoffs_sent") is not None else _as_float(handoff_count),
        "handoffs_accepted": _as_float(_first(tracking, "handoffs_accepted")),
        "camera_captures": _as_float(_first(observation, "camera_captures", "captures")),
        "camera_visible_detections": _as_float(_first(observation, "camera_visible_detections", "visible")),
        "camera_invalid_detections": _as_float(_first(observation, "camera_invalid_detections", "invalid")),
        "camera_rpc_errors": _as_float(_first(observation, "camera_rpc_errors", "rpc_errors", "errors")),
        "mean_capture_rate_hz": _as_float(_first(observation, "mean_capture_rate_hz")),
        "capture_interval_p95_sec": _as_float(_first(observation, "capture_interval_p95_sec")),
        "capture_interval_max_sec": _as_float(_first(observation, "capture_interval_max_sec")),
        "mean_image_rpc_duration_sec": _as_float(_first(observation, "mean_image_rpc_duration_sec")),
        "max_image_rpc_duration_sec": _as_float(_first(observation, "max_image_rpc_duration_sec")),
        "cpu_percent": _as_float(_first(performance, "cpu_percent", "cpu")),
        "memory_mb": _as_float(_first(performance, "memory_mb", "memory", "rss_mb")),
    }


def normalize_raw(path: Path, label: str, raw_records: Sequence[Mapping[str, Any]], default_dt: float = 0.1) -> Dict[str, Any]:
    """Normalize one logger's raw rows to a common semantic schema."""

    if not raw_records:
        raise ComparisonError(f"No records to normalize for {label}: {path}")
    timed = [(_raw_time(row, index, default_dt), index, row) for index, row in enumerate(raw_records)]
    timed.sort(key=lambda item: (item[0], item[1]))
    origin = timed[0][0]
    records: List[Dict[str, Any]] = []
    for source_time, _, row in timed:
        relative_time = source_time - origin
        target = _target_record(row)
        agents = {}
        for agent in _row_agents(row):
            agents[agent] = _normalize_agent(row, agent, target)
        timing = _mapping(row.get("timing"))
        diagnostics = _row_diagnostics(row)
        step = _as_float(row.get("step"))
        normalized = {
            "time": relative_time,
            "source_time": source_time,
            "step": int(step) if step is not None and float(step).is_integer() else step,
            "target": target,
            "agents": agents,
            "timing_cycle_ms": _as_float(_first(timing, "cycle_ms", "total_cycle_ms")),
            "timing_deadline_miss": _as_bool(_first(timing, "deadline_miss", "deadline_missed")),
            **diagnostics,
        }
        # Keep the latest row when duplicate timestamps occur.  The original
        # order is retained for distinct timestamps, so this is deterministic.
        if records and abs(float(records[-1]["time"]) - relative_time) < 1e-12:
            records[-1] = normalized
        else:
            records.append(normalized)

    agents = sorted({agent for record in records for agent in record["agents"]})
    vehicle_types: Dict[str, str] = {}
    for agent in agents:
        values = [record["agents"].get(agent, {}).get("vehicle_type") for record in records]
        values = [str(value) for value in values if value]
        vehicle_types[agent] = values[0] if values else "unknown"
    times = [float(record["time"]) for record in records]
    return {
        "implementation": label,
        "source_log": str(path),
        "dt": default_dt,
        "raw_record_count": len(raw_records),
        "record_count": len(records),
        "raw_times_s": times,
        "raw_duration_s": max(times) if times else 0.0,
        "agents": agents,
        "vehicle_types": vehicle_types,
        "records": records,
    }


def _series(records: Sequence[Mapping[str, Any]], selector: Any) -> Tuple[List[float], List[Any]]:
    times: List[float] = []
    values: List[Any] = []
    for record in records:
        value = selector(record)
        if value is not None:
            times.append(float(record["time"]))
            values.append(value)
    return times, values


def _linear_scalar(times: Sequence[float], values: Sequence[Any], target: float) -> Optional[float]:
    usable = [(float(time), _as_float(value)) for time, value in zip(times, values) if _as_float(value) is not None]
    if not usable:
        return None
    if len(usable) == 1:
        return usable[0][1]
    if target <= usable[0][0]:
        return usable[0][1]
    if target >= usable[-1][0]:
        return usable[-1][1]
    for (left_time, left_value), (right_time, right_value) in zip(usable, usable[1:]):
        if target <= right_time:
            span = right_time - left_time
            if span <= 0:
                return right_value
            fraction = (target - left_time) / span
            return float(left_value) + fraction * (float(right_value) - float(left_value))
    return usable[-1][1]


def _hold_value(times: Sequence[float], values: Sequence[Any], target: float) -> Any:
    usable = [(float(time), value) for time, value in zip(times, values) if value is not None]
    if not usable:
        return None
    selected = usable[0][1]
    for time, value in usable:
        if time > target:
            break
        selected = value
    return selected


def _interpolate_field(
    records: Sequence[Mapping[str, Any]], selector: Any, target: float, discrete: bool = False
) -> Any:
    times, values = _series(records, selector)
    if discrete:
        return _hold_value(times, values, target)
    return _linear_scalar(times, values, target)


def _interpolate_vector(
    records: Sequence[Mapping[str, Any]], selector: Any, target: float, pad: Optional[float] = None
) -> List[Optional[float]]:
    values = []
    for component in range(3):
        values.append(
            _interpolate_field(
                records,
                lambda record, component=component: (
                    selector(record)[component]
                    if isinstance(selector(record), (list, tuple)) and len(selector(record)) > component
                    else pad
                ),
                target,
            )
        )
    return values


def _agent_selector(agent: str, field: str) -> Any:
    return lambda record: _mapping(_mapping(record.get("agents")).get(agent)).get(field)


def _target_selector(field: str) -> Any:
    return lambda record: _mapping(record.get("target")).get(field)


def _coverage(records: Sequence[Mapping[str, Any]], target: float, agent: Optional[str] = None) -> bool:
    if not records:
        return False
    first = float(records[0]["time"])
    last = float(records[-1]["time"])
    if target < first - 1e-9 or target > last + 1e-9:
        return False
    if agent is None:
        return True
    position = _interpolate_vector(records, _agent_selector(agent, "position"), target)
    return _is_finite(position[0]) and _is_finite(position[1])


def resample_dataset(dataset: Mapping[str, Any], grid: Sequence[float]) -> Dict[str, Any]:
    """Resample a normalized dataset onto the supplied canonical time grid."""

    raw_records = list(dataset["records"])
    resampled_records: List[Dict[str, Any]] = []
    for target_time in grid:
        target = {}
        for field in TARGET_VECTOR_FIELDS:
            target[field] = _interpolate_vector(raw_records, _target_selector(field), target_time)
        for field in TARGET_NUMERIC_FIELDS:
            target[field] = _interpolate_field(raw_records, _target_selector(field), target_time)
        for field in TARGET_DISCRETE_FIELDS:
            target[field] = _interpolate_field(raw_records, _target_selector(field), target_time, discrete=True)

        agents: Dict[str, Dict[str, Any]] = {}
        for agent in dataset["agents"]:
            agent_record: Dict[str, Any] = {
                "vehicle_type": dataset["vehicle_types"].get(agent, "unknown"),
                "coverage": _coverage(raw_records, target_time, agent),
            }
            for field in VECTOR_FIELDS:
                agent_record[field] = _interpolate_vector(
                    raw_records, _agent_selector(agent, field), target_time, pad=0.0 if field in {"command", "nominal_control", "safe_control"} else None
                )
            for field in SCALAR_NUMERIC_FIELDS:
                agent_record[field] = _interpolate_field(
                    raw_records, _agent_selector(agent, field), target_time
                )
            for field in DISCRETE_FIELDS:
                agent_record[field] = _interpolate_field(
                    raw_records, _agent_selector(agent, field), target_time, discrete=True
                )
            agent_record["timing_cycle_ms"] = _interpolate_field(
                raw_records,
                lambda record: _mapping(_mapping(record.get("agents")).get(agent)).get("timing_cycle_ms"),
                target_time,
            )
            agent_record["timing_deadline_miss"] = _interpolate_field(
                raw_records,
                lambda record: _mapping(_mapping(record.get("agents")).get(agent)).get("timing_deadline_miss"),
                target_time,
                discrete=True,
            )
            agents[agent] = agent_record

        resampled_records.append(
            {
                "time": float(target_time),
                "target": target,
                "agents": agents,
                "timing_cycle_ms": _interpolate_field(
                    raw_records, lambda record: record.get("timing_cycle_ms"), target_time
                ),
                "timing_deadline_miss": _interpolate_field(
                    raw_records,
                    lambda record: record.get("timing_deadline_miss"),
                    target_time,
                    discrete=True,
                ),
                **{
                    field: _interpolate_field(
                        raw_records,
                        lambda record, field=field: record.get(field),
                        target_time,
                    )
                    for field in ROW_NUMERIC_FIELDS
                },
            }
        )

    return {
        "implementation": dataset["implementation"],
        "source_log": dataset["source_log"],
        "dt": dataset["dt"],
        "raw_record_count": dataset["raw_record_count"],
        "record_count": dataset["record_count"],
        "raw_times_s": list(dataset["raw_times_s"]),
        "raw_duration_s": dataset["raw_duration_s"],
        "duration_s": float(grid[-1]) if grid else 0.0,
        "agents": list(dataset["agents"]),
        "vehicle_types": dict(dataset["vehicle_types"]),
        "time": [float(value) for value in grid],
        "records": resampled_records,
    }


def _values_for_agent(dataset: Mapping[str, Any], agent: str, field: str) -> List[Any]:
    return [
        _mapping(_mapping(record.get("agents")).get(agent)).get(field)
        for record in dataset.get("records", [])
    ]


def _finite_values(values: Iterable[Any]) -> List[float]:
    return [float(value) for value in values if _is_finite(value)]


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _stats(values: Iterable[Any]) -> Dict[str, Any]:
    finite = _finite_values(values)
    if not finite:
        return {"count": 0, "mean": None, "rms": None, "max": None, "p95": None}
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "rms": math.sqrt(sum(value * value for value in finite) / len(finite)),
        "max": max(finite),
        "p95": _percentile(finite, 0.95),
    }


def _bool_fraction(values: Iterable[Any]) -> Optional[float]:
    present = [value for value in values if isinstance(value, bool)]
    if not present:
        return None
    return sum(1 for value in present if value) / len(present)


def _count_true(values: Iterable[Any]) -> int:
    return sum(1 for value in values if value is True)


def _target_progress(target: Mapping[str, Any]) -> Optional[float]:
    explicit = _as_float(target.get("progress"))
    if explicit is not None:
        return explicit
    index = _as_float(target.get("index"))
    sample_count = _as_float(target.get("sample_count"))
    if index is not None and sample_count is not None and sample_count > 0:
        return index / sample_count
    return None


def _pairwise_min_distance(dataset: Mapping[str, Any]) -> Optional[float]:
    minimum: Optional[float] = None
    vehicle_types = _mapping(dataset.get("vehicle_types"))
    for record in dataset.get("records", []):
        positions = {}
        for agent, value in _mapping(record.get("agents")).items():
            position = _mapping(value).get("position")
            if isinstance(position, (list, tuple)) and _is_finite(position[0]) and _is_finite(position[1]):
                positions[agent] = position
        names = sorted(positions)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                left_type = str(vehicle_types.get(left_name, "")).lower()
                right_type = str(vehicle_types.get(right_name, "")).lower()
                if left_type and right_type and left_type != right_type:
                    continue
                distance = _vector_distance(positions[left_name], positions[right_name])
                if distance is not None and (minimum is None or distance < minimum):
                    minimum = distance
    return minimum


def summarize_dataset(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    """Produce implementation-local metrics from the canonical grid."""

    agent_summary: Dict[str, Any] = {}
    for agent in dataset.get("agents", []):
        position = _values_for_agent(dataset, agent, "position")
        desired = _values_for_agent(dataset, agent, "desired_slot")
        slot_errors = _values_for_agent(dataset, agent, "slot_error")
        target_positions = [_mapping(record.get("target")).get("position") for record in dataset.get("records", [])]
        target_velocities = [_mapping(record.get("target")).get("velocity") for record in dataset.get("records", [])]
        position_target_errors = [
            _vector_distance(value, target)
            for value, target in zip(position, target_positions)
            if isinstance(value, (list, tuple)) and isinstance(target, (list, tuple))
        ]
        velocity_target_errors = [
            _vector_distance(value, target)
            for value, target in zip(_values_for_agent(dataset, agent, "velocity"), target_velocities)
            if isinstance(value, (list, tuple)) and isinstance(target, (list, tuple))
        ]
        z_errors = []
        for value, target in zip(position, desired):
            if isinstance(value, (list, tuple)) and isinstance(target, (list, tuple)):
                if len(value) > 2 and len(target) > 2 and _is_finite(value[2]) and _is_finite(target[2]):
                    z_errors.append(abs(float(value[2]) - float(target[2])))
        target_radii = []
        for value, target in zip(position, target_positions):
            if isinstance(value, (list, tuple)) and isinstance(target, (list, tuple)):
                common = []
                for component in range(min(2, len(value), len(target))):
                    if _is_finite(value[component]) and _is_finite(target[component]):
                        common.append(float(value[component]) - float(target[component]))
                if common:
                    target_radii.append(math.sqrt(sum(component * component for component in common)))

        agent_summary[agent] = {
            "vehicle_type": dataset.get("vehicle_types", {}).get(agent, "unknown"),
            "coverage_fraction": _bool_fraction(_values_for_agent(dataset, agent, "coverage")),
            "slot_error_m": _stats(slot_errors),
            "altitude_error_m": _stats(z_errors),
            "target_position_error_m": _stats(position_target_errors),
            "target_velocity_error_mps": _stats(velocity_target_errors),
            "target_radius_xy_m": _stats(target_radii),
            "tracking_active_fraction": _bool_fraction(
                _values_for_agent(dataset, agent, "tracking_active")
            ),
            "direct_observation_fraction": _bool_fraction(
                _values_for_agent(dataset, agent, "tracking_direct_observation")
            ),
            "tracking_position_error_m": _stats(
                _values_for_agent(dataset, agent, "tracking_position_error_m")
            ),
            "tracking_velocity_error_mps": _stats(
                _values_for_agent(dataset, agent, "tracking_velocity_error_mps")
            ),
            "tracking_iterations": _stats(_values_for_agent(dataset, agent, "tracking_iterations")),
            "tracking_residual": _stats(_values_for_agent(dataset, agent, "tracking_residual")),
            "cbf_enabled_fraction": _bool_fraction(_values_for_agent(dataset, agent, "cbf_enabled")),
            "cbf_success_fraction": _bool_fraction(_values_for_agent(dataset, agent, "cbf_success")),
            "cbf_fallback_count": _count_true(_values_for_agent(dataset, agent, "cbf_fallback")),
            "cbf_deadline_miss_count": _count_true(
                _values_for_agent(dataset, agent, "cbf_deadline_miss")
            ),
            "cbf_minimum_barrier": _stats(
                _values_for_agent(dataset, agent, "cbf_minimum_barrier")
            ),
            "cbf_intervention_norm": _stats(
                _values_for_agent(dataset, agent, "cbf_intervention_norm")
            ),
            "cbf_solver_time_ms": _stats(_values_for_agent(dataset, agent, "cbf_solver_time_ms")),
            "cbf_filter_time_ms": _stats(_values_for_agent(dataset, agent, "cbf_filter_time_ms")),
            "collision_relevant_count": _count_true(
                _values_for_agent(dataset, agent, "collision_relevant")
            ),
            "collision_count": _count_true(
                _values_for_agent(dataset, agent, "collision_has_collided")
            ),
        }

    cycle_times = [record.get("timing_cycle_ms") for record in dataset.get("records", [])]
    gaps = [
        right - left
        for left, right in zip(dataset.get("raw_times_s", []), dataset.get("raw_times_s", [])[1:])
        if right > left
    ]
    progress = [_target_progress(_mapping(record.get("target"))) for record in dataset.get("records", [])]
    return {
        "implementation": dataset.get("implementation"),
        "source_log": dataset.get("source_log"),
        "raw_record_count": dataset.get("raw_record_count"),
        "resampled_record_count": len(dataset.get("records", [])),
        "raw_duration_s": dataset.get("raw_duration_s"),
        "duration_s": dataset.get("duration_s", 0.0),
        "agents": list(dataset.get("agents", [])),
        "loop_timing": {
            "cycle_ms": _stats(cycle_times),
            "raw_gap_s": _stats(gaps),
            "raw_rate_hz": (1.0 / _percentile(gaps, 0.5)) if _percentile(gaps, 0.5) else None,
            "deadline_miss_count": _count_true(
                record.get("timing_deadline_miss") for record in dataset.get("records", [])
            ),
        },
        "tracking_counters": {
            field: _stats(record.get(field) for record in dataset.get("records", []))
            for field in ROW_NUMERIC_FIELDS[:7]
        },
        "observation_health": {
            field: _stats(record.get(field) for record in dataset.get("records", []))
            for field in ROW_NUMERIC_FIELDS[7:15]
        },
        "resources": {
            field: _stats(record.get(field) for record in dataset.get("records", []))
            for field in ROW_NUMERIC_FIELDS[15:]
        },
        "pairwise_min_distance_m": _pairwise_min_distance(dataset),
        "target_progress": _stats(progress),
        "agents_summary": agent_summary,
    }


def _diff_values(left: Any, right: Any) -> Optional[float]:
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return _vector_distance(left, right)
    if _is_finite(left) and _is_finite(right):
        return abs(float(left) - float(right))
    return None


def _compare_bool(left: Any, right: Any) -> Optional[bool]:
    if isinstance(left, bool) and isinstance(right, bool):
        return left != right
    return None


def compare_datasets(python: Mapping[str, Any], ros: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare two already aligned, resampled canonical datasets."""

    python_agents = set(python.get("agents", []))
    ros_agents = set(ros.get("agents", []))
    common_agents = sorted(python_agents & ros_agents)
    grid = list(python.get("time", []))
    if len(grid) != len(ros.get("time", [])):
        raise ComparisonError("Aligned datasets have different grid lengths")
    comparison_rows: List[Dict[str, Any]] = []
    per_agent_values: Dict[str, Dict[str, List[Any]]] = {
        agent: {
            "position_diff_m": [],
            "velocity_diff_mps": [],
            "yaw_diff_rad": [],
            "desired_slot_diff_m": [],
            "slot_error_diff_m": [],
            "command_diff": [],
            "nominal_control_diff": [],
            "safe_control_diff": [],
            "target_truth_diff_m": [],
            "target_truth_velocity_diff_mps": [],
            "cbf_minimum_barrier_diff": [],
            "cbf_solver_time_diff_ms": [],
            "cbf_intervention_diff": [],
        }
        for agent in common_agents
    }
    mismatch_counts = {agent: {} for agent in common_agents}
    global_values: Dict[str, List[Any]] = {key: [] for key in per_agent_values[common_agents[0]]} if common_agents else {}

    for index, time in enumerate(grid):
        python_record = _mapping(python["records"][index])
        ros_record = _mapping(ros["records"][index])
        python_target = _mapping(python_record.get("target"))
        ros_target = _mapping(ros_record.get("target"))
        target_diff = _diff_values(python_target.get("position"), ros_target.get("position"))
        target_velocity_diff = _diff_values(python_target.get("velocity"), ros_target.get("velocity"))
        for agent in common_agents:
            left = _mapping(_mapping(python_record.get("agents")).get(agent))
            right = _mapping(_mapping(ros_record.get("agents")).get(agent))
            fields = {
                "position_diff_m": _diff_values(left.get("position"), right.get("position")),
                "velocity_diff_mps": _diff_values(left.get("velocity"), right.get("velocity")),
                "yaw_diff_rad": _wrapped_angle_difference(left.get("yaw"), right.get("yaw")),
                "desired_slot_diff_m": _diff_values(left.get("desired_slot"), right.get("desired_slot")),
                "slot_error_diff_m": _diff_values(left.get("slot_error"), right.get("slot_error")),
                "command_diff": _diff_values(left.get("command"), right.get("command")),
                "nominal_control_diff": _diff_values(
                    left.get("nominal_control"), right.get("nominal_control")
                ),
                "safe_control_diff": _diff_values(left.get("safe_control"), right.get("safe_control")),
                "target_truth_diff_m": target_diff,
                "target_truth_velocity_diff_mps": target_velocity_diff,
                "cbf_minimum_barrier_diff": _diff_values(
                    left.get("cbf_minimum_barrier"), right.get("cbf_minimum_barrier")
                ),
                "cbf_solver_time_diff_ms": _diff_values(
                    left.get("cbf_solver_time_ms"), right.get("cbf_solver_time_ms")
                ),
                "cbf_intervention_diff": _diff_values(
                    left.get("cbf_intervention_norm"), right.get("cbf_intervention_norm")
                ),
            }
            for key, value in fields.items():
                per_agent_values[agent][key].append(value)
                global_values.setdefault(key, []).append(value)
            mismatch_fields = {
                "cbf_enabled": _compare_bool(left.get("cbf_enabled"), right.get("cbf_enabled")),
                "cbf_success": _compare_bool(left.get("cbf_success"), right.get("cbf_success")),
                "cbf_fallback": _compare_bool(left.get("cbf_fallback"), right.get("cbf_fallback")),
                "tracking_active": _compare_bool(left.get("tracking_active"), right.get("tracking_active")),
                "tracking_direct_observation": _compare_bool(
                    left.get("tracking_direct_observation"), right.get("tracking_direct_observation")
                ),
            }
            for key, mismatch in mismatch_fields.items():
                if mismatch is not None:
                    mismatch_counts[agent][key] = mismatch_counts[agent].get(key, 0) + int(mismatch)

            comparison_rows.append(
                {
                    "time": time,
                    "agent": agent,
                    "python_coverage": left.get("coverage"),
                    "ros_coverage": right.get("coverage"),
                    "coverage": bool(left.get("coverage")) and bool(right.get("coverage")),
                    **fields,
                    **{f"{key}_mismatch": value for key, value in mismatch_fields.items()},
                }
            )

    per_agent = {}
    for agent in common_agents:
        per_agent[agent] = {
            key: _stats(values) for key, values in per_agent_values[agent].items()
        }
        per_agent[agent]["discrete_mismatches"] = mismatch_counts[agent]
        per_agent[agent]["sample_count"] = len(grid)

    warnings = []
    if not common_agents:
        warnings.append("No common controlled agents were present in both logs.")
    if python_agents - ros_agents:
        warnings.append("Agents present only in Python: " + ", ".join(sorted(python_agents - ros_agents)))
    if ros_agents - python_agents:
        warnings.append("Agents present only in ROS: " + ", ".join(sorted(ros_agents - python_agents)))
    if any(
        value.get("cbf_enabled") is False
        for record in ros.get("records", [])
        for value in _mapping(record.get("agents")).values()
    ):
        warnings.append("ROS CBF diagnostics include disabled samples; compare enabled and disabled runs separately.")

    return {
        "python_source_log": python.get("source_log"),
        "ros_source_log": ros.get("source_log"),
        "dt": python.get("dt"),
        "duration_s": python.get("duration_s", 0.0),
        "grid_count": len(grid),
        "common_agents": common_agents,
        "python_only_agents": sorted(python_agents - ros_agents),
        "ros_only_agents": sorted(ros_agents - python_agents),
        "per_agent": per_agent,
        "global": {key: _stats(values) for key, values in global_values.items()},
        "warnings": warnings,
        "rows": comparison_rows,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _flatten_agent_row(
    implementation: str, time: float, target: Mapping[str, Any], agent: str, value: Mapping[str, Any]
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "implementation": implementation,
        "time": time,
        "agent": agent,
        "vehicle_type": value.get("vehicle_type"),
        "coverage": value.get("coverage"),
    }
    for field, prefix in CSV_VECTOR_FIELDS.items():
        vector = value.get(field)
        for index, axis in enumerate(("x", "y", "z")):
            row[f"{prefix}_{axis}"] = vector[index] if isinstance(vector, (list, tuple)) and len(vector) > index else None
    for field in SCALAR_NUMERIC_FIELDS + DISCRETE_FIELDS:
        row[field] = value.get(field)
    row["timing_cycle_ms"] = value.get("timing_cycle_ms")
    row["timing_deadline_miss"] = value.get("timing_deadline_miss")
    for field in TARGET_VECTOR_FIELDS:
        vector = target.get(field)
        for index, axis in enumerate(("x", "y", "z")):
            row[f"target_{field}_{axis}"] = vector[index] if isinstance(vector, (list, tuple)) and len(vector) > index else None
    for field in TARGET_NUMERIC_FIELDS + TARGET_DISCRETE_FIELDS:
        row[f"target_{field}"] = target.get(field)
    return row


def write_normalized_csv(path: Path, dataset: Mapping[str, Any]) -> None:
    rows = []
    for record in dataset.get("records", []):
        for agent, value in _mapping(record.get("agents")).items():
            rows.append(
                _flatten_agent_row(
                    str(dataset.get("implementation")),
                    float(record.get("time", 0.0)),
                    _mapping(record.get("target")),
                    str(agent),
                    _mapping(value),
                )
            )
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def write_comparison_csv(path: Path, comparison: Mapping[str, Any]) -> None:
    rows = list(comparison.get("rows", []))
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _metric(summary: Mapping[str, Any], key: str, statistic: str = "max") -> Any:
    value = summary.get(key)
    return _mapping(value).get(statistic) if isinstance(value, Mapping) else value


def render_markdown(
    output_path: Path,
    python: Mapping[str, Any],
    ros: Mapping[str, Any],
    python_summary: Mapping[str, Any],
    ros_summary: Mapping[str, Any],
    comparison: Mapping[str, Any],
    plot_status: str,
) -> None:
    lines = [
        "# Python-only vs ROS mission comparison",
        "",
        "This report compares canonicalized mission telemetry. Each log is aligned to its own mission start and resampled at **" 
        + _fmt(comparison.get("dt"))
        + " s** over the overlapping run interval.",
        "",
        "## Inputs",
        "",
        "| implementation | source log | raw records | raw duration (s) | resampled records |",
        "|---|---|---:|---:|---:|",
        f"| Python | `{python.get('source_log')}` | {_fmt(python.get('raw_record_count'), 0)} | {_fmt(python.get('raw_duration_s'))} | {_fmt(python.get('record_count'), 0)} |",
        f"| ROS | `{ros.get('source_log')}` | {_fmt(ros.get('raw_record_count'), 0)} | {_fmt(ros.get('raw_duration_s'))} | {_fmt(ros.get('record_count'), 0)} |",
        "",
        "## Aligned comparison",
        "",
        f"Common interval: **{_fmt(comparison.get('duration_s'))} s**, **{_fmt(comparison.get('grid_count'), 0)}** samples; common agents: **{', '.join(comparison.get('common_agents', [])) or 'none'}**.",
        "",
        "| agent | position RMS / max (m) | velocity RMS / max (m/s) | slot-error RMS / max (m) | command RMS / max | target truth RMS / max (m) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for agent in comparison.get("common_agents", []):
        metrics = _mapping(comparison.get("per_agent", {}).get(agent))
        lines.append(
            "| "
            + agent
            + " | "
            + _fmt(_metric(metrics, "position_diff_m", "rms"))
            + " / "
            + _fmt(_metric(metrics, "position_diff_m", "max"))
            + " | "
            + _fmt(_metric(metrics, "velocity_diff_mps", "rms"))
            + " / "
            + _fmt(_metric(metrics, "velocity_diff_mps", "max"))
            + " | "
            + _fmt(_metric(metrics, "slot_error_diff_m", "rms"))
            + " / "
            + _fmt(_metric(metrics, "slot_error_diff_m", "max"))
            + " | "
            + _fmt(_metric(metrics, "command_diff", "rms"))
            + " / "
            + _fmt(_metric(metrics, "command_diff", "max"))
            + " | "
            + _fmt(_metric(metrics, "target_truth_diff_m", "rms"))
            + " / "
            + _fmt(_metric(metrics, "target_truth_diff_m", "max"))
            + " |"
        )
    if not comparison.get("common_agents"):
        lines.append("| _none_ | n/a | n/a | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Implementation-local metrics",
            "",
            "| implementation | min pairwise distance (m) | max slot error (m) | target progress end | CBF enabled fraction | CBF fallback count | deadline misses |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, summary in (("Python", python_summary), ("ROS", ros_summary)):
        agents_summary = _mapping(summary.get("agents_summary"))
        slot_values = [_metric(value, "slot_error_m", "max") for value in agents_summary.values()]
        cbf_values = [_metric(value, "cbf_enabled_fraction", "max") for value in agents_summary.values()]
        fallback_count = sum(int(_metric(value, "cbf_fallback_count") or 0) for value in agents_summary.values())
        deadline_count = sum(int(_metric(value, "cbf_deadline_miss_count") or 0) for value in agents_summary.values())
        target_progress = _mapping(summary.get("target_progress"))
        lines.append(
            f"| {label} | {_fmt(summary.get('pairwise_min_distance_m'))} | {_fmt(max((v for v in slot_values if v is not None), default=None))} | {_fmt(target_progress.get('max'))} | {_fmt(max((v for v in cbf_values if v is not None), default=None))} | {fallback_count} | {deadline_count} |"
        )

    lines.extend(
        [
            "",
            "## CBF and tracking notes",
            "",
            "The Python log's per-agent CBF dictionary and the ROS log's `cbf.entries` array are mapped to the same fields. Missing fields remain `n/a`; they are not treated as zero or as a failed solve.",
            "",
            "| agent | CBF enabled mismatch samples | CBF success mismatch samples | CBF fallback mismatch samples | tracking-active mismatch samples |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for agent in comparison.get("common_agents", []):
        mismatches = _mapping(comparison.get("per_agent", {}).get(agent, {}).get("discrete_mismatches"))
        lines.append(
            f"| {agent} | {mismatches.get('cbf_enabled', 0)} | {mismatches.get('cbf_success', 0)} | {mismatches.get('cbf_fallback', 0)} | {mismatches.get('tracking_active', 0)} |"
        )
    lines.extend(["", f"Plot: **{plot_status}**.", ""])

    warnings = list(comparison.get("warnings", []))
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.extend([f"- {warning}", ""])

    lines.extend(
        [
            "## Interpretation caveats",
            "",
            "- Python timestamps are normally mission-step time; ROS timestamps are elapsed mission wall time. Start alignment and interpolation reduce this difference but do not establish deterministic replay.",
            "- Compare CBF-enabled Python against ROS CBF-enabled runs for safety-filter behavior. ROS CBF-disabled runs are still useful as a baseline, but should not be presented as an apples-to-apples CBF comparison.",
            "- Matching target truth does not prove matching Unreal/AirSim physics, sensor timing, or route configuration. Use the same recorded inputs or a fixed simulator seed where possible.",
            "- Position and control differences are descriptive metrics, not pass/fail thresholds. Add scenario-specific tolerances after the baseline runs are known.",
            "",
            "## Artifacts",
            "",
            "- `normalized_python.csv` / `normalized_ros.csv`: long-form per-agent canonical telemetry.",
            "- `normalized_python.json` / `normalized_ros.json`: canonical resampled records and source metadata.",
            "- `comparison.csv`: one row per common agent and grid time with absolute differences.",
            "- `comparison.json`: machine-readable metrics and comparison rows.",
            "- `comparison_plot.png`: optional diagnostic plot when Matplotlib is installed.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _max_series(dataset: Mapping[str, Any], field: str, agents: Sequence[str]) -> List[Optional[float]]:
    result = []
    for record in dataset.get("records", []):
        values = [
            _as_float(_mapping(_mapping(record.get("agents")).get(agent)).get(field))
            for agent in agents
        ]
        finite = [value for value in values if value is not None]
        result.append(max(finite) if finite else None)
    return result


def make_plot(path: Path, python: Mapping[str, Any], ros: Mapping[str, Any], comparison: Mapping[str, Any]) -> str:
    """Make a small diagnostic plot, returning a human-readable status."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on host environment
        return f"not generated (Matplotlib unavailable: {exc})"

    try:
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        labels = ((python, "Python", "tab:blue"), (ros, "ROS", "tab:orange"))
        for dataset, label, color in labels:
            times = dataset.get("time", [])
            slot = _max_series(dataset, "slot_error", dataset.get("agents", []))
            axes[0][0].plot(times, [value if value is not None else math.nan for value in slot], label=label, color=color)
            target_x = []
            target_y = []
            for record in dataset.get("records", []):
                position = _mapping(record.get("target")).get("position")
                if isinstance(position, (list, tuple)) and _is_finite(position[0]) and _is_finite(position[1]):
                    target_x.append(float(position[0]))
                    target_y.append(float(position[1]))
            if target_x:
                axes[0][1].plot(target_x, target_y, label=label, color=color)
            command = _max_series(dataset, "cbf_intervention_norm", dataset.get("agents", []))
            axes[1][1].plot(times, [value if value is not None else math.nan for value in command], label=label, color=color)

        diff_by_time: List[Optional[float]] = []
        for time in python.get("time", []):
            values = [
                row.get("position_diff_m")
                for row in comparison.get("rows", [])
                if abs(float(row.get("time", 0.0)) - float(time)) < 1e-9
            ]
            finite = [float(value) for value in values if _is_finite(value)]
            diff_by_time.append(max(finite) if finite else None)
        axes[1][0].plot(
            python.get("time", []),
            [value if value is not None else math.nan for value in diff_by_time],
            label="ROS − Python",
            color="tab:green",
        )

        axes[0][0].set_title("Maximum slot error")
        axes[0][0].set_xlabel("mission time (s)")
        axes[0][0].set_ylabel("m")
        axes[0][1].set_title("Target truth XY path")
        axes[0][1].set_xlabel("x")
        axes[0][1].set_ylabel("y")
        axes[1][0].set_title("Maximum position difference")
        axes[1][0].set_xlabel("mission time (s)")
        axes[1][0].set_ylabel("m")
        axes[1][1].set_title("Maximum CBF intervention norm")
        axes[1][1].set_xlabel("mission time (s)")
        axes[1][1].set_ylabel("control units")
        for axis in axes.flat:
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best")
        figure.savefig(path, dpi=150)
        plt.close(figure)
    except Exception as exc:  # pragma: no cover - depends on plotting backend
        return f"not generated (plotting failed: {exc})"
    return str(path)


def _canonical_grid(python: Mapping[str, Any], ros: Mapping[str, Any], dt: float) -> List[float]:
    python_duration = float(python.get("raw_duration_s", 0.0))
    ros_duration = float(ros.get("raw_duration_s", 0.0))
    duration = min(python_duration, ros_duration)
    if duration < 0:
        raise ComparisonError("Run duration cannot be negative")
    if duration == 0:
        return [0.0]
    count = int(math.floor((duration + 1e-9) / dt))
    return [round(index * dt, 10) for index in range(count + 1)]


def run_comparison(args: argparse.Namespace) -> Dict[str, Any]:
    python_path = discover_log(args.python_run)
    ros_path = discover_log(args.ros_run)
    python_raw = load_jsonl(python_path)
    ros_raw = load_jsonl(ros_path)
    python_normalized = normalize_raw(python_path, "python", python_raw, args.dt)
    ros_normalized = normalize_raw(ros_path, "ros", ros_raw, args.dt)
    grid = _canonical_grid(python_normalized, ros_normalized, args.dt)
    python_resampled = resample_dataset(python_normalized, grid)
    ros_resampled = resample_dataset(ros_normalized, grid)
    python_summary = summarize_dataset(python_resampled)
    ros_summary = summarize_dataset(ros_resampled)
    comparison = compare_datasets(python_resampled, ros_resampled)
    comparison["python_summary"] = python_summary
    comparison["ros_summary"] = ros_summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "normalized_python.json", python_resampled)
    write_json(args.output_dir / "normalized_ros.json", ros_resampled)
    write_json(args.output_dir / "comparison.json", comparison)
    write_normalized_csv(args.output_dir / "normalized_python.csv", python_resampled)
    write_normalized_csv(args.output_dir / "normalized_ros.csv", ros_resampled)
    write_comparison_csv(args.output_dir / "comparison.csv", comparison)

    plot_path = args.output_dir / "comparison_plot.png"
    if args.no_plot:
        plot_status = "disabled by --no-plot"
    else:
        plot_status = make_plot(plot_path, python_resampled, ros_resampled, comparison)
    comparison["plot_status"] = plot_status
    write_json(args.output_dir / "comparison.json", comparison)
    render_markdown(
        args.output_dir / "comparison.md",
        python_resampled,
        ros_resampled,
        python_summary,
        ros_summary,
        comparison,
        plot_status,
    )
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("python_run", help="Python run directory or mission.jsonl path")
    parser.add_argument("ros_run", help="ROS run directory or mission.jsonl path")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for normalized CSV/JSON, comparison artifacts, and report",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Common resampling interval in seconds (default: 0.1)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip optional Matplotlib plot generation")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return failure if there is no overlapping duration or no common agents",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dt <= 0 or not math.isfinite(args.dt):
        parser.error("--dt must be a finite positive number")
    try:
        comparison = run_comparison(args)
        if args.strict and (comparison.get("duration_s", 0.0) <= 0 or not comparison.get("common_agents")):
            raise ComparisonError("Strict comparison requires positive overlap and at least one common agent")
    except ComparisonError as exc:
        parser.error(str(exc))
    print(f"Compared {comparison['grid_count']} samples across {len(comparison['common_agents'])} common agents.")
    print(f"Report: {args.output_dir / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
