"""Mission-log parsing and MAICP robot/mission scores.

The AirSim mission node writes one JSON object per control cycle.  This
module deliberately treats that JSONL as an audit boundary: malformed,
incomplete, failed, or missing observations raise :class:`MissionScoreError`
instead of becoming a zero score.  The runner can therefore fail a mission
and keep the failure separate from calibration data.

The three application scores follow the simulation section of the MAICP
paper:

* collision avoidance uses the norm of the acceleration-model residual;
* target tracking uses target-position error at the estimate timestamp; and
* localization uses planar position error at the localization timestamp.

The functions accept the current ROS log schema and a few older aliases so
that archived validation logs remain useful.  No ROS imports are required.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

try:  # NumPy is available in the ROS image, but keep imports lightweight.
    import numpy as np
except ImportError:  # pragma: no cover - useful error is raised at call time.
    np = None  # type: ignore[assignment]


class MissionScoreError(ValueError):
    """A mission cannot provide a valid finite score."""


class MissionFailedError(MissionScoreError):
    """A mission has an explicit failure sidecar or a failed observation."""


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MissionScoreError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise MissionScoreError(f"{name} must be a finite number")
    return result


def _vector(value: object, name: str, *, dimension: int = 2) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise MissionScoreError(f"{name} must be a numeric vector")
    try:
        values = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise MissionScoreError(f"{name} must be a numeric vector") from exc
    if len(values) < dimension or any(not math.isfinite(item) for item in values[:dimension]):
        raise MissionScoreError(f"{name} must contain {dimension} finite coordinates")
    return values


def _planar(value: object, name: str) -> tuple[float, float]:
    values = _vector(value, name, dimension=2)
    return float(values[0]), float(values[1])


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in vector))


def normalize_class(value: object) -> str:
    """Normalize vehicle-type spellings to the two paper classes."""

    text = str(getattr(value, "value", value)).strip().lower()
    if text in {"uav", "drone", "multirotor", "quadrotor", "air"}:
        return "uav"
    if text in {"ugv", "ground", "car", "husky", "vehicle"}:
        return "ugv"
    raise MissionScoreError(f"unknown vehicle class {value!r}")


def infer_class(agent_id: str, record: Mapping[str, Any] | None = None) -> str:
    """Infer a class from ``vehicle_types`` or the conventional AirSim ID."""

    if record is not None:
        vehicle_types = record.get("vehicle_types")
        if isinstance(vehicle_types, Mapping) and agent_id in vehicle_types:
            return normalize_class(vehicle_types[agent_id])
    text = str(agent_id).lower()
    if text.startswith(("drone", "uav", "simpleflight", "multirotor")):
        return "uav"
    if text.startswith(("husky", "ugv", "car", "ground")):
        return "ugv"
    raise MissionScoreError(f"cannot infer vehicle class for agent {agent_id!r}")


def load_mission_jsonl(
    path: str | Path,
    *,
    expected_steps: int | None = None,
    require_done: bool = False,
) -> list[dict[str, Any]]:
    """Load a complete mission log and reject failed or truncated artifacts.

    ``require_done`` is enabled by :class:`~experiment.ExperimentRunner` for
    live runs.  It is intentionally optional for offline scoring of archived
    JSONL files that predate the ``.done`` sidecar contract.  If
    ``expected_steps`` is supplied, the row count must match exactly in both
    modes.
    """

    log_path = Path(path)
    if not log_path.is_file():
        raise MissionScoreError(f"mission log is missing: {log_path}")
    failed_path = Path(str(log_path) + ".failed")
    done_path = Path(str(log_path) + ".done")
    if failed_path.exists():
        reason = failed_path.read_text(encoding="utf-8", errors="replace").strip()
        raise MissionFailedError(
            f"mission reported failure in {failed_path}: {reason or 'unspecified failure'}"
        )
    if done_path.exists() and failed_path.exists():
        raise MissionFailedError(f"mission has both completion and failure sidecars: {log_path}")
    if require_done and not done_path.is_file():
        raise MissionFailedError(f"mission completion sidecar is missing: {done_path}")

    rows: list[dict[str, Any]] = []
    try:
        with log_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MissionScoreError(
                        f"invalid JSON in {log_path} at line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise MissionScoreError(
                        f"mission row {line_number} in {log_path} is not an object"
                    )
                rows.append(value)
    except OSError as exc:
        raise MissionScoreError(f"cannot read mission log {log_path}: {exc}") from exc
    if not rows:
        raise MissionScoreError(f"mission log is empty: {log_path}")
    if expected_steps is not None:
        if not isinstance(expected_steps, int) or expected_steps < 1:
            raise ValueError("expected_steps must be a positive integer")
        if len(rows) != expected_steps:
            raise MissionFailedError(
                f"mission log is truncated: expected {expected_steps} rows, found {len(rows)}"
            )
        # A same-length log with a dropped/duplicated control step is also
        # incomplete.  Archived logs without a step field remain supported,
        # but once the field is present it must cover the fixed mission
        # horizon exactly.
        step_values = [row.get("step") for row in rows]
        if any(value is not None for value in step_values):
            if step_values != list(range(expected_steps)):
                raise MissionFailedError(
                    f"mission log has noncontiguous steps; expected 0..{expected_steps - 1}"
                )
    return rows


# Short aliases used by validation scripts and downstream callers.
load_jsonl = load_mission_jsonl
load_mission_records = load_mission_jsonl


def _row_time(row: Mapping[str, Any], index: int) -> float:
    """Return the logical mission time used by tracking truth interpolation."""

    # Tracking timestamps are mission logical seconds (step * dt).  The ROS
    # logger's wall/receipt timestamp can be on a different clock, so prefer
    # the deterministic step clock whenever both fields are available.
    if "step" in row and "dt" in row:
        try:
            step = _finite(row["step"], f"row {index} step")
            dt = _finite(row["dt"], f"row {index} dt")
            if dt > 0.0:
                return step * dt
        except MissionScoreError:
            pass
    for key in ("timestamp", "source_timestamp", "time"):
        if key in row and row[key] is not None:
            return _finite(row[key], f"row {index} {key}")
    return float(index)


def _agent_ids(records: Sequence[Mapping[str, Any]], explicit: Iterable[str] | None = None) -> tuple[str, ...]:
    if explicit is not None:
        result = tuple(str(item) for item in explicit)
        if not result:
            raise MissionScoreError("at least one agent is required")
        return result
    for row in records:
        states = row.get("states")
        if isinstance(states, Mapping) and states:
            # Target1 is logged alongside the controlled fleet for truth and
            # tracking, but it has no safe-control record and is not one of
            # the paper's three-UAV/three-UGV score members.
            controlled = tuple(
                str(item)
                for item in states
                if not str(item).strip().lower().startswith("target")
            )
            return controlled or tuple(str(item) for item in states)
    raise MissionScoreError("mission records contain no states")


def _class_map(records: Sequence[Mapping[str, Any]], agents: Sequence[str]) -> dict[str, str]:
    first = records[0]
    result = {agent: infer_class(agent, first) for agent in agents}
    return result


def _state_sample(row: Mapping[str, Any], agent: str, row_index: int) -> tuple[tuple[float, float], tuple[float, float], float]:
    states = row.get("states")
    if not isinstance(states, Mapping) or agent not in states or not isinstance(states[agent], Mapping):
        raise MissionScoreError(f"row {row_index} is missing states[{agent!r}]")
    state = states[agent]
    position = _planar(state.get("position"), f"states[{agent}].position at row {row_index}")
    velocity = _planar(state.get("velocity"), f"states[{agent}].velocity at row {row_index}")
    if "source_timestamp" not in state or state.get("source_timestamp") is None:
        raise MissionScoreError(f"states[{agent}] is missing source_timestamp at row {row_index}")
    timestamp = _finite(state["source_timestamp"], f"states[{agent}].source_timestamp at row {row_index}")
    return position, velocity, timestamp


def _safe_control(row: Mapping[str, Any], agent: str, row_index: int) -> tuple[float, float]:
    candidates = ("safe_controls", "maicp_safe_controls", "controls")
    controls: object | None = None
    for key in candidates:
        if key in row:
            controls = row[key]
            break
    if not isinstance(controls, Mapping) or agent not in controls:
        raise MissionScoreError(f"row {row_index} is missing safe_controls[{agent!r}]")
    return _planar(controls[agent], f"safe_controls[{agent}] at row {row_index}")


def _model_for(models: Mapping[str, Any], class_name: str, agent: str) -> Any:
    if class_name in models:
        return models[class_name]
    if agent in models:
        return models[agent]
    for key, value in models.items():
        try:
            if normalize_class(key) == class_name:
                return value
        except MissionScoreError:
            continue
    raise MissionScoreError(f"dynamics model for class {class_name!r} is missing")


def _prediction_from_row(
    row: Mapping[str, Any], agent: str, row_index: int,
) -> tuple[float, float] | None:
    """Read an exact per-record learned-dynamics prediction when logged.

    The mission logger may include ``maicp.learned_dynamics`` so an archived
    scorer can reproduce the controller's frozen model without importing its
    launch configuration.  Several descriptive aliases are accepted for
    forward/backward compatibility; a present but malformed value is an
    error rather than a fallback to zero.
    """

    containers: list[Mapping[str, Any]] = [row]
    maicp = row.get("maicp")
    if isinstance(maicp, Mapping):
        containers.append(maicp)
    keys = (
        "learned_dynamics",
        "learned_acceleration",
        "model_prediction",
        "dynamics_prediction",
        "predicted_acceleration",
    )
    for container in containers:
        for key in keys:
            if key not in container:
                continue
            values = container[key]
            if not isinstance(values, Mapping) or agent not in values:
                continue
            raw = values[agent]
            if isinstance(raw, Mapping):
                for nested_key in ("acceleration", "prediction", "value", "dynamics"):
                    if nested_key in raw:
                        raw = raw[nested_key]
                        break
            return _planar(raw, f"{key}[{agent}] at row {row_index}")
    return None


def _inline_affine_prediction(
    row: Mapping[str, Any], class_name: str, position: Sequence[float], row_index: int,
) -> tuple[float, float] | None:
    """Evaluate coefficients embedded in a mission record, if present."""

    containers: list[Mapping[str, Any]] = [row]
    maicp = row.get("maicp")
    if isinstance(maicp, Mapping):
        containers.append(maicp)
    aliases = {
        "uav": ("model_uav", "uav_model"),
        "ugv": ("model_ugv", "ugv_model"),
    }
    raw_model: object | None = None
    for container in containers:
        for key in ("models", "dynamics_models", "affine_models", "model"):
            value = container.get(key)
            if isinstance(value, Mapping) and class_name in value:
                raw_model = value[class_name]
                break
        if raw_model is None:
            for key in aliases[class_name]:
                if key in container:
                    raw_model = container[key]
                    break
        if raw_model is not None:
            break
    if raw_model is None:
        return None
    if isinstance(raw_model, Mapping):
        raw_model = raw_model.get("coefficients", raw_model.get("coeffs"))
    if isinstance(raw_model, str):
        raw_model = raw_model.split()
    try:
        coefficients = tuple(float(value) for value in raw_model) if raw_model is not None else ()
    except (TypeError, ValueError) as exc:
        raise MissionScoreError(
            f"embedded {class_name} dynamics at row {row_index} must have six finite coefficients"
        ) from exc
    if len(coefficients) != 6 or any(not math.isfinite(value) for value in coefficients):
        raise MissionScoreError(
            f"embedded {class_name} dynamics at row {row_index} must have six finite coefficients"
        )
    x, y = _planar(position, f"embedded {class_name} model position at row {row_index}")
    b_x, xx, xy, b_y, yx, yy = coefficients
    return b_x + xx * x + xy * y, b_y + yx * x + yy * y


@dataclass
class _StateGroup:
    """Consecutive packets for one physical state sample.

    AirSim can publish the same state more than once while the mission loop
    is waiting for the next simulator sample.  ``first_index`` is retained so
    every command held during that interval can be scored; ``last_index`` is
    retained as an audit pointer to the last identical packet.
    """

    position: tuple[float, float]
    velocity: tuple[float, float]
    timestamp: float
    first_index: int
    last_index: int
    full_position: tuple[float, ...]
    full_velocity: tuple[float, ...]


def _state_vectors(
    row: Mapping[str, Any], agent: str, row_index: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return finite full state vectors for duplicate-state comparisons."""

    states = row.get("states")
    if not isinstance(states, Mapping) or agent not in states or not isinstance(states[agent], Mapping):
        raise MissionScoreError(f"row {row_index} is missing states[{agent!r}]")
    state = states[agent]
    position = _vector(state.get("position"), f"states[{agent}].position at row {row_index}")
    velocity = _vector(state.get("velocity"), f"states[{agent}].velocity at row {row_index}")
    if any(not math.isfinite(value) for value in position + velocity):
        raise MissionScoreError(f"states[{agent}] contains a non-finite state at row {row_index}")
    return position, velocity


def _collapse_state_samples(
    records: Sequence[Mapping[str, Any]], agent: str, end: int,
) -> tuple[list[_StateGroup], int]:
    """Collapse consecutive identical source samples and count duplicates.

    A repeated timestamp is valid only when the complete position and velocity
    packet is identical.  Backward timestamps, nonfinite values, and changed
    state packets remain hard scoring errors.
    """

    groups: list[_StateGroup] = []
    duplicate_count = 0
    for row_index in range(end):
        position, velocity, timestamp = _state_sample(records[row_index], agent, row_index)
        full_position, full_velocity = _state_vectors(records[row_index], agent, row_index)
        if not groups:
            groups.append(
                _StateGroup(
                    position=position,
                    velocity=velocity,
                    timestamp=timestamp,
                    first_index=row_index,
                    last_index=row_index,
                    full_position=full_position,
                    full_velocity=full_velocity,
                )
            )
            continue
        previous = groups[-1]
        if timestamp < previous.timestamp:
            raise MissionScoreError(
                f"states[{agent}] source_timestamp moves backwards at row {row_index}"
            )
        if timestamp == previous.timestamp:
            if full_position != previous.full_position or full_velocity != previous.full_velocity:
                raise MissionScoreError(
                    f"states[{agent}] has an inconsistent duplicate source timestamp at row {row_index}"
                )
            previous.last_index = row_index
            duplicate_count += 1
            continue
        dt = timestamp - previous.timestamp
        # Preserve the old strict guard against effectively zero distinct
        # intervals; otherwise one bad packet would create an unbounded score.
        if not math.isfinite(dt) or dt <= 1e-12:
            raise MissionScoreError(
                f"states[{agent}] source_timestamp is not strictly increasing at row {row_index}"
            )
        groups.append(
            _StateGroup(
                position=position,
                velocity=velocity,
                timestamp=timestamp,
                first_index=row_index,
                last_index=row_index,
                full_position=full_position,
                full_velocity=full_velocity,
            )
        )
    return groups, duplicate_count


def collision_scores(
    records: Sequence[Mapping[str, Any]],
    models: Mapping[str, Any] | None,
    *,
    agents: Iterable[str] | None = None,
    horizon: int | None = None,
    start_time: float = 0.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute frozen-affine acceleration residual scores.

    The residual at row ``i`` is
    ``(v_i-v_{i-1})/(t_i-t_{i-1}) - u_{i-1} - d_hat(p_{i-1})``.
    ``source_timestamp`` is used for the time difference; the JSONL row
    timestamp is never substituted silently.
    """

    if models is None:
        models = {}
    if not isinstance(models, Mapping):
        raise MissionScoreError("collision scoring dynamics models must be a mapping")
    selected = _agent_ids(records, agents)
    class_map = _class_map(records, selected)
    end = len(records) if horizon is None else min(len(records), int(horizon))
    if end < 2:
        raise MissionScoreError("collision score horizon must contain at least two rows")
    scores: dict[str, float] = {}
    diagnostics: dict[str, Any] = {
        "score": "acceleration_residual_l2",
        "agents": {},
        "fail_safe_events": [],
        "missing_fail_safe_rows": [],
        "duplicate_source_samples": 0,
    }
    for row_index, row in enumerate(records[:end]):
        cbf = row.get("cbf")
        entries = cbf.get("entries") if isinstance(cbf, Mapping) else None
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            diagnostics["missing_fail_safe_rows"].append(row_index)
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            if bool(entry.get("fallback", False)) or bool(entry.get("deadline_miss", False)) or not bool(entry.get("final_feasible", True)):
                diagnostics["fail_safe_events"].append({
                    "row": row_index,
                    "agent": entry.get("agent_id"),
                    "solver_status": entry.get("solver_status"),
                    "fallback": bool(entry.get("fallback", False)),
                    "deadline_miss": bool(entry.get("deadline_miss", False)),
                    "final_feasible": entry.get("final_feasible"),
                })
    diagnostics["fail_safe_diagnostics_available"] = not bool(
        diagnostics["missing_fail_safe_rows"]
    )
    for agent in selected:
        groups, duplicate_count = _collapse_state_samples(records, agent, end)
        if len(groups) < 2:
            raise MissionScoreError(f"no distinct collision state samples for {agent}")
        diagnostics["duplicate_source_samples"] += duplicate_count
        maximum = -math.inf
        count = 0
        residuals: list[float] = []
        model = None
        if models:
            try:
                model = _model_for(models, class_map[agent], agent)
            except MissionScoreError:
                model = None
        dynamics_sources: set[str] = set()
        unique_intervals = 0
        for previous, current in zip(groups, groups[1:]):
            if _row_time(records[current.first_index], current.first_index) < start_time:
                continue
            dt = current.timestamp - previous.timestamp
            if not math.isfinite(dt) or dt <= 1e-12:
                raise MissionScoreError(
                    f"states[{agent}] source_timestamp is not strictly increasing at row {current.first_index}"
                )
            # Commands are held from the first packet of one physical state
            # until the first packet of the next distinct state.  With no
            # duplicate packets this range has one command, exactly matching
            # the original row-to-row convention.  With duplicates, every
            # intervening command is evaluated and the largest residual is
            # retained conservatively.
            command_indices = range(previous.first_index, current.first_index)
            interval_observations = 0
            for command_index in command_indices:
                command = _safe_control(records[command_index], agent, command_index)
                predicted_xy = _prediction_from_row(records[command_index], agent, command_index)
                if predicted_xy is not None:
                    dynamics_sources.add("inline_prediction")
                if predicted_xy is None:
                    predicted_xy = _inline_affine_prediction(
                        records[command_index], class_map[agent], previous.position, command_index
                    )
                    if predicted_xy is not None:
                        dynamics_sources.add("inline_affine")
                if predicted_xy is None:
                    if model is None:
                        raise MissionScoreError(
                            f"dynamics model/prediction for {agent} is missing at row {command_index}"
                        )
                    predicted = model.evaluate(previous.position) if hasattr(model, "evaluate") else model(previous.position)
                    predicted_xy = _planar(
                        predicted, f"dynamics model {class_map[agent]} at row {command_index}"
                    )
                    dynamics_sources.add("frozen_model")
                residual = tuple(
                    (current.velocity[coordinate] - previous.velocity[coordinate]) / dt
                    - command[coordinate]
                    - predicted_xy[coordinate]
                    for coordinate in range(2)
                )
                value = _norm(residual)
                maximum = max(maximum, value)
                residuals.append(value)
                count += 1
                interval_observations += 1
            if interval_observations == 0:
                raise MissionScoreError(f"no command observation for {agent} before row {current.first_index}")
            unique_intervals += 1
        if count == 0 or not math.isfinite(maximum):
            raise MissionScoreError(f"no valid collision observations for {agent}")
        scores[agent] = float(maximum)
        diagnostics["agents"][agent] = {
            "class": class_map[agent],
            "observations": count,
            "command_observations": count,
            "unique_intervals": unique_intervals,
            "duplicate_source_samples": duplicate_count,
            "max_residual": float(maximum),
            "mean_residual": float(sum(residuals) / len(residuals)),
            "dynamics_sources": sorted(dynamics_sources),
        }
    return scores, diagnostics


def _target_truth_sample(
    row: Mapping[str, Any], row_index: int, *, fallback_time: float,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    truth = row.get("target_truth")
    if not isinstance(truth, Mapping):
        truth = row.get("target")
    if not isinstance(truth, Mapping):
        raise MissionScoreError(f"row {row_index} is missing target_truth")
    position = _planar(truth.get("position"), f"target_truth.position at row {row_index}")
    velocity_value = truth.get("velocity", (0.0, 0.0))
    velocity = _planar(velocity_value, f"target_truth.velocity at row {row_index}")
    # ``source_timestamp`` is often an AirSim/ROS wall-clock stamp while the
    # tracker reports its estimate epoch in mission time.  The tracking
    # scorer below considers both domains explicitly.  This helper's legacy
    # single timestamp is therefore the explicit logical timestamp, falling
    # back to the deterministic step clock.
    timestamp_value = truth.get("timestamp", fallback_time)
    timestamp = _finite(timestamp_value, f"target_truth timestamp at row {row_index}")
    return position, velocity, timestamp


def _target_truth_values(
    row: Mapping[str, Any], row_index: int, *, logical_time: float,
) -> tuple[tuple[float, float], tuple[float, float], float, float | None, float | None]:
    """Read target truth and retain every timestamp domain it provides.

    AirSim truth can carry a simulator/source stamp while a tracker estimate
    may carry either that stamp or the mission's deterministic step clock.
    Keeping both lets the scorer match epochs without silently comparing
    unrelated clocks.
    """

    truth = row.get("target_truth")
    if not isinstance(truth, Mapping):
        truth = row.get("target")
    if not isinstance(truth, Mapping):
        raise MissionScoreError(f"row {row_index} is missing target_truth")
    position = _planar(truth.get("position"), f"target_truth.position at row {row_index}")
    velocity = _planar(
        truth.get("velocity", (0.0, 0.0)),
        f"target_truth.velocity at row {row_index}",
    )
    explicit_timestamp = None
    source_timestamp = None
    if truth.get("timestamp") is not None:
        explicit_timestamp = _finite(
            truth["timestamp"], f"target_truth timestamp at row {row_index}"
        )
    if truth.get("source_timestamp") is not None:
        source_timestamp = _finite(
            truth["source_timestamp"],
            f"target_truth source_timestamp at row {row_index}",
        )
    return position, velocity, float(logical_time), explicit_timestamp, source_timestamp


def _ordered_series(
    samples: Sequence[tuple[float, tuple[float, ...]]], name: str,
) -> list[tuple[float, tuple[float, ...]]]:
    if not samples:
        return []
    normalized: list[tuple[float, tuple[float, ...]]] = []
    for index, (raw_timestamp, raw_value) in enumerate(samples):
        timestamp = _finite(raw_timestamp, f"{name} timestamp at sample {index}")
        try:
            value = tuple(float(item) for item in raw_value)
        except (TypeError, ValueError) as exc:
            raise MissionScoreError(f"truth value is invalid for {name} at sample {index}") from exc
        if not value or any(not math.isfinite(item) for item in value):
            raise MissionScoreError(f"truth value is not finite for {name} at sample {index}")
        normalized.append((timestamp, value))
    ordered = sorted(normalized, key=lambda item: item[0])
    deduplicated: list[tuple[float, tuple[float, ...]]] = []
    for timestamp, value in ordered:
        if deduplicated and timestamp == deduplicated[-1][0]:
            if value != deduplicated[-1][1]:
                raise MissionScoreError(
                    f"truth has an inconsistent duplicate timestamp for {name}"
                )
            # Repeated source packets with the same truth value are valid;
            # retain one sample for interpolation.
            continue
        if deduplicated and timestamp < deduplicated[-1][0]:
            raise MissionScoreError(f"truth timestamps move backwards for {name}")
        deduplicated.append((timestamp, value))
    return deduplicated


def _choose_truth_domain(
    domains: Mapping[str, Sequence[tuple[float, tuple[float, ...]]]],
    timestamp: float,
    name: str,
) -> tuple[str, list[tuple[float, tuple[float, ...]]]]:
    """Choose the truth clock whose horizon and samples match an estimate.

    If several domains overlap, the nearest truth sample wins.  A source
    timestamp therefore wins for source-epoch estimates, while a logical
    step timestamp wins when the source clock is offset by simulator delay.
    Outside every complete domain is an explicit mission-score failure.
    """

    candidates: list[tuple[float, int, str, list[tuple[float, tuple[float, ...]]]]] = []
    # Prefer the deterministic logical clock when the estimate is in that
    # horizon.  Current MAICP tracker epochs are step*dt; source-clock
    # domains remain available when a ROS/AirSim epoch is clearly outside it.
    priority = {
        "logical": 0,
        "explicit": 1,
        "source": 2,
        "state_source": 3,
    }
    horizons: list[str] = []
    for label, raw_samples in domains.items():
        if not raw_samples:
            continue
        raw_times = [float(sample_time) for sample_time, _ in raw_samples]
        lower, upper = min(raw_times), max(raw_times)
        # Do not validate an unrelated clock.  Logs can legitimately contain
        # repeated source stamps while the scored estimate is in logical
        # time; only the selected domain must be strictly increasing.
        if not (lower - 1e-9 <= timestamp <= upper + 1e-9):
            continue
        ordered = _ordered_series(raw_samples, f"{name} ({label})")
        horizons.append(f"{label}=[{ordered[0][0]}, {ordered[-1][0]}]")
        if ordered[0][0] - 1e-9 <= timestamp <= ordered[-1][0] + 1e-9:
            nearest = min(abs(timestamp - sample_time) for sample_time, _ in ordered)
            candidates.append((nearest, priority.get(label, 3), label, ordered))
    if not candidates:
        joined = ", ".join(horizons) or "no truth samples"
        raise MissionScoreError(
            f"{name} observation timestamp {timestamp} lies outside truth horizons ({joined})"
        )
    _, _, label, ordered = min(candidates, key=lambda item: (item[1], item[0]))
    return label, ordered


def _truth_domains(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[tuple[float, tuple[float, float]]]]:
    """Build logical, explicit, and source target-truth time series."""

    domains: dict[str, list[tuple[float, tuple[float, float]]]] = {
        "logical": [],
        "explicit": [],
        "source": [],
        "state_source": [],
    }
    for index, row in enumerate(records):
        logical_time = _row_time(row, index)
        position, _, logical, explicit, source = _target_truth_values(
            row, index, logical_time=logical_time
        )
        domains["logical"].append((logical, position))
        if explicit is not None:
            domains["explicit"].append((explicit, position))
        if source is not None:
            domains["source"].append((source, position))
        states = row.get("states")
        if isinstance(states, Mapping):
            for state in states.values():
                if isinstance(state, Mapping) and state.get("source_timestamp") is not None:
                    domains["state_source"].append(
                        (
                            _finite(
                                state["source_timestamp"],
                                f"state source_timestamp at row {index}",
                            ),
                            position,
                        )
                    )
                    break
    return domains


def _truth_value_domains(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[tuple[float, tuple[float, float]]]]:
    """Build velocity domains matching :func:`_truth_domains`."""

    domains: dict[str, list[tuple[float, tuple[float, float]]]] = {
        "logical": [],
        "explicit": [],
        "source": [],
        "state_source": [],
    }
    for index, row in enumerate(records):
        logical_time = _row_time(row, index)
        _, velocity, logical, explicit, source = _target_truth_values(
            row, index, logical_time=logical_time
        )
        domains["logical"].append((logical, velocity))
        if explicit is not None:
            domains["explicit"].append((explicit, velocity))
        if source is not None:
            domains["source"].append((source, velocity))
        states = row.get("states")
        if isinstance(states, Mapping):
            for state in states.values():
                if isinstance(state, Mapping) and state.get("source_timestamp") is not None:
                    domains["state_source"].append(
                        (
                            _finite(
                                state["source_timestamp"],
                                f"state source_timestamp at row {index}",
                            ),
                            velocity,
                        )
                    )
                    break
    return domains


def _interpolate(samples: Sequence[tuple[float, tuple[float, ...]]], timestamp: float, name: str) -> tuple[float, ...]:
    if not samples:
        raise MissionScoreError(f"no truth samples are available for {name}")
    ordered = _ordered_series(samples, name)
    if timestamp < ordered[0][0] - 1e-9 or timestamp > ordered[-1][0] + 1e-9:
        raise MissionScoreError(
            f"{name} observation timestamp {timestamp} lies outside truth horizon "
            f"[{ordered[0][0]}, {ordered[-1][0]}]"
        )
    if len(ordered) == 1:
        return tuple(ordered[0][1])
    for index, (left_time, left_value) in enumerate(ordered):
        if timestamp <= left_time + 1e-12:
            return tuple(left_value)
        if index + 1 >= len(ordered):
            return tuple(left_value)
        right_time, right_value = ordered[index + 1]
        if timestamp <= right_time + 1e-12:
            fraction = (timestamp - left_time) / (right_time - left_time)
            return tuple(
                float(left_value[coordinate])
                + fraction * (float(right_value[coordinate]) - float(left_value[coordinate]))
                for coordinate in range(len(left_value))
            )
    return tuple(ordered[-1][1])


def tracking_scores(
    records: Sequence[Mapping[str, Any]],
    *,
    agents: Iterable[str] | None = None,
    horizon: int | None = None,
    start_time: float = 0.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Score active target estimates against truth at matched epochs."""

    selected = _agent_ids(records, agents)
    class_map = _class_map(records, selected)
    end = len(records) if horizon is None else min(len(records), int(horizon))
    score_indices = [index for index in range(end) if _row_time(records[index], index) >= start_time]
    if not score_indices:
        raise MissionScoreError("target-tracking score horizon has no samples after score_start")
    # Build truth over the full mission history.  ``horizon`` selects the
    # scored estimates; it must not truncate interpolation support when an
    # estimate near the score window edge refers to a later truth epoch.
    truth_position_domains = _truth_domains(records)
    truth_velocity_domains = _truth_value_domains(records)
    scores: dict[str, float] = {}
    diagnostics: dict[str, Any] = {"score": "target_position_l2", "agents": {}}
    tracking_root = "target_tracking"
    for agent in selected:
        maximum = -math.inf
        count = 0
        velocity_errors: list[float] = []
        current_position_errors: list[float] = []
        estimate_ages: list[float] = []
        for index in score_indices:
            row = records[index]
            tracking = row.get(tracking_root)
            if not isinstance(tracking, Mapping):
                raise MissionScoreError(f"row {index} is missing target_tracking")
            by_agent = tracking.get("agents")
            if not isinstance(by_agent, Mapping) or agent not in by_agent:
                raise MissionScoreError(f"row {index} is missing target_tracking.agents[{agent!r}]")
            entry = by_agent[agent]
            estimate = entry.get("estimate") if isinstance(entry, Mapping) else None
            if not isinstance(estimate, Mapping) or not bool(estimate.get("active", False)):
                raise MissionScoreError(
                    f"row {index} has no active target estimate for {agent}; "
                    "missing observations are a failed mission"
                )
            if estimate.get("iterations") != 50 or bool(entry.get("epoch_timed_out", False)):
                raise MissionScoreError(
                    f"row {index} has an incomplete or timed-out 50-round target estimate for {agent}"
                )
            estimate_position = _planar(
                estimate.get("position"),
                f"target_tracking.agents[{agent}].estimate.position at row {index}",
            )
            estimate_velocity = _planar(
                estimate.get("velocity"),
                f"target_tracking.agents[{agent}].estimate.velocity at row {index}",
            )
            estimate_timestamp = _finite(
                estimate.get("timestamp"),
                f"target_tracking.agents[{agent}].estimate.timestamp at row {index}",
            )
            truth_domain, truth_positions = _choose_truth_domain(
                truth_position_domains, estimate_timestamp, "target_truth"
            )
            truth_position = _interpolate(truth_positions, estimate_timestamp, f"target_truth ({truth_domain})")
            value = _norm((estimate_position[0] - truth_position[0], estimate_position[1] - truth_position[1]))
            current_truth = _target_truth_values(row, index, logical_time=_row_time(row, index))[0]
            current_position_errors.append(_norm((estimate_position[0] - current_truth[0],
                                                  estimate_position[1] - current_truth[1])))
            if truth_domain == "logical":
                estimate_ages.append(_row_time(row, index) - estimate_timestamp)
            velocity_domain, truth_velocities = _choose_truth_domain(
                truth_velocity_domains, estimate_timestamp, "target_truth velocity"
            )
            # A position and velocity domain should normally agree.  Keep the
            # selected labels visible in diagnostics if a legacy log supplied
            # only one of the optional timestamp fields.
            _ = velocity_domain
            truth_velocity = _interpolate(
                truth_velocities, estimate_timestamp, f"target_truth velocity ({velocity_domain})"
            )
            velocity_errors.append(_norm((estimate_velocity[0] - truth_velocity[0], estimate_velocity[1] - truth_velocity[1])))
            maximum = max(maximum, value)
            count += 1
        if count == 0 or not math.isfinite(maximum):
            raise MissionScoreError(f"no valid target estimates for {agent}")
        scores[agent] = float(maximum)
        diagnostics["agents"][agent] = {
            "class": class_map[agent],
            "observations": count,
            "max_position_error": float(maximum),
            "max_velocity_error": float(max(velocity_errors)),
            "max_current_time_position_error": float(max(current_position_errors)),
            "max_estimate_age_sec": max(estimate_ages) if estimate_ages else None,
        }
    return scores, diagnostics


def _localization_sample(
    row: Mapping[str, Any], agent: str, row_index: int,
) -> tuple[tuple[float, float], float]:
    localization = row.get("localization")
    if not isinstance(localization, Mapping):
        raise MissionScoreError(f"row {row_index} is missing localization")
    by_agent = localization.get("agents")
    if not isinstance(by_agent, Mapping) or agent not in by_agent or not isinstance(by_agent[agent], Mapping):
        raise MissionScoreError(f"row {row_index} is missing localization.agents[{agent!r}]")
    entry = by_agent[agent]
    if not bool(entry.get("valid", False)) or bool(entry.get("stale", False)):
        raise MissionScoreError(
            f"row {row_index} has invalid or stale localization for {agent}; "
            "missing observations are a failed mission"
        )
    pose_value = entry.get("pose")
    pose = _planar(pose_value, f"localization.agents[{agent}].pose at row {row_index}")
    state = row.get("states", {}).get(agent) if isinstance(row.get("states"), Mapping) else None
    state_timestamp = state.get("source_timestamp") if isinstance(state, Mapping) else None
    timestamp_value = entry.get("timestamp", state_timestamp)
    if timestamp_value is None:
        timestamp_value = _row_time(row, row_index)
    timestamp = _finite(timestamp_value, f"localization.agents[{agent}].timestamp at row {row_index}")
    return pose, timestamp


def localization_scores(
    records: Sequence[Mapping[str, Any]],
    *,
    agents: Iterable[str] | None = None,
    horizon: int | None = None,
    start_time: float = 0.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Score valid, fresh planar localization estimates against state truth."""

    selected = _agent_ids(records, agents)
    class_map = _class_map(records, selected)
    end = len(records) if horizon is None else min(len(records), int(horizon))
    score_indices = [index for index in range(end) if _row_time(records[index], index) >= start_time]
    if not score_indices:
        raise MissionScoreError("localization score horizon has no samples after score_start")
    # As with tracking, keep full truth support while selecting only the
    # fixed scored horizon.  Localization estimates may carry either source
    # or logical mission timestamps, so retain both domains.
    truth_by_agent: dict[str, dict[str, list[tuple[float, tuple[float, float]]]]] = {
        agent: {"source": [], "logical": []} for agent in selected
    }
    for index, row in enumerate(records):
        for agent in selected:
            position, _, timestamp = _state_sample(row, agent, index)
            truth_by_agent[agent]["source"].append((timestamp, position))
            truth_by_agent[agent]["logical"].append((_row_time(row, index), position))
    scores: dict[str, float] = {}
    diagnostics: dict[str, Any] = {"score": "localization_position_l2", "agents": {}}
    for agent in selected:
        maximum = -math.inf
        count = 0
        for index in score_indices:
            row = records[index]
            pose, timestamp = _localization_sample(row, agent, index)
            truth_domain, truth_samples = _choose_truth_domain(
                truth_by_agent[agent], timestamp, f"states[{agent}]"
            )
            truth = _interpolate(
                truth_samples, timestamp, f"states[{agent}] ({truth_domain})"
            )
            maximum = max(maximum, _norm((pose[0] - truth[0], pose[1] - truth[1])))
            count += 1
        if count == 0 or not math.isfinite(maximum):
            raise MissionScoreError(f"no valid localization observations for {agent}")
        scores[agent] = float(maximum)
        diagnostics["agents"][agent] = {
            "class": class_map[agent],
            "observations": count,
            "max_position_error": float(maximum),
        }
    return scores, diagnostics


def collision_free_status(records: Sequence[Mapping[str, Any]]) -> tuple[bool | None, dict[str, Any]]:
    """Read simulator collision diagnostics independently of model scores.

    The ROS mission node currently calls the field ``collisions``; the
    ``mission_collisions`` alias is accepted for the research runner and old
    fixtures.  Missing collision telemetry is reported as unavailable rather
    than being interpreted as collision-free.
    """

    observed = False
    available = False
    events = 0
    details: list[dict[str, Any]] = []
    for row_index, row in enumerate(records):
        value = row.get("mission_collisions", row.get("collisions"))
        if value is None:
            continue
        observed = True
        entries: Iterable[tuple[str, Any]]
        if isinstance(value, Mapping):
            entries = value.items()
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            entries = ((str(index), item) for index, item in enumerate(value))
        else:
            raise MissionScoreError(f"collision telemetry at row {row_index} is not an object/list")
        for agent, entry in entries:
            if not isinstance(entry, Mapping):
                raise MissionScoreError(f"collision telemetry for {agent} at row {row_index} is invalid")
            if bool(entry.get("available", True)):
                available = True
            collided = bool(entry.get("relevant", False)) or bool(entry.get("has_collided", False))
            if collided:
                events += 1
                details.append({"row": row_index, "agent": str(agent), "entry": dict(entry)})
    if not observed or not available:
        return None, {"available": False, "events": events, "details": details}
    return events == 0, {"available": True, "events": events, "details": details}


def order_statistic(values: Iterable[float], rank: int) -> float:
    """Return a finite one-indexed order statistic."""

    try:
        samples = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise MissionScoreError("order-statistic values must be numeric") from exc
    if not samples or any(not math.isfinite(value) for value in samples):
        raise MissionScoreError("order-statistic values must be nonempty and finite")
    if not isinstance(rank, int) or not 1 <= rank <= len(samples):
        raise MissionScoreError(f"order-statistic rank {rank!r} is outside 1..{len(samples)}")
    return float(sorted(samples)[rank - 1])


def aggregate_mission_scores(
    robot_scores: Mapping[str, float],
    class_by_agent: Mapping[str, str],
    *,
    permitted_violations: int = 1,
    expected_per_class: Mapping[str, int] | None = None,
) -> dict[str, float]:
    """Aggregate robot scores using the paper's class order statistic."""

    if permitted_violations < 0:
        raise ValueError("permitted_violations must be nonnegative")
    by_class: dict[str, list[float]] = {}
    for agent, raw_score in robot_scores.items():
        if agent not in class_by_agent:
            raise MissionScoreError(f"missing class for scored agent {agent!r}")
        value = _finite(raw_score, f"score for {agent}")
        by_class.setdefault(normalize_class(class_by_agent[agent]), []).append(value)
    result: dict[str, float] = {}
    for class_name, values in by_class.items():
        expected = (expected_per_class or {}).get(class_name, len(values))
        if expected != len(values):
            raise MissionScoreError(
                f"class {class_name} has {len(values)} robot scores; expected {expected}"
            )
        rank = len(values) - permitted_violations
        if rank < 1:
            raise MissionScoreError(f"class {class_name} has too many permitted violations")
        result[class_name] = order_statistic(values, rank)
    return result


@dataclass(frozen=True)
class MissionScore:
    """Scores and independent mission diagnostics."""

    case: str
    robot_scores: dict[str, float]
    class_scores: dict[str, float]
    collision_free: bool | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    record_count: int = 0

    @property
    def mission_scores(self) -> dict[str, float]:
        return dict(self.class_scores)

    @property
    def failed(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "robot_scores": dict(self.robot_scores),
            "class_scores": dict(self.class_scores),
            "collision_free": self.collision_free,
            "diagnostics": self.diagnostics,
            "record_count": self.record_count,
        }


def score_records(
    records: Sequence[Mapping[str, Any]],
    case: str,
    *,
    models: Mapping[str, Any] | None = None,
    agents: Iterable[str] | None = None,
    horizon: int | None = None,
    permitted_violations: int = 1,
    expected_per_class: Mapping[str, int] | None = None,
    start_time: float = 0.0,
) -> MissionScore:
    """Score already-loaded mission records for one MAICP application."""

    canonical = str(case).lower().replace("-", "_")
    if canonical in {"collision", "collision_avoidance", "cbf"}:
        robot_scores, diagnostics = collision_scores(
            records, models, agents=agents, horizon=horizon, start_time=start_time
        )
    elif canonical in {"tracking", "target_tracking", "drwt"}:
        robot_scores, diagnostics = tracking_scores(
            records, agents=agents, horizon=horizon, start_time=start_time
        )
    elif canonical in {"localization", "cooperative_localization", "coop_localization"}:
        robot_scores, diagnostics = localization_scores(
            records, agents=agents, horizon=horizon, start_time=start_time
        )
    else:
        raise ValueError(f"unknown MAICP case {case!r}")
    class_by_agent = {
        agent: diagnostics["agents"][agent]["class"] for agent in robot_scores
    }
    class_scores = aggregate_mission_scores(
        robot_scores,
        class_by_agent,
        permitted_violations=permitted_violations,
        expected_per_class=expected_per_class,
    )
    collision_free, collision_diagnostics = collision_free_status(records)
    diagnostics["collision_free"] = collision_diagnostics
    diagnostics["fail_safe"] = bool(diagnostics.get("fail_safe_events"))
    diagnostics["fail_safe_unknown"] = bool(diagnostics.get("missing_fail_safe_rows"))
    returned = records[-1].get("maicp", {}).get("returned_home", {})
    # The mission computes this from simulator truth and requires actual
    # travel as well as return, so a parked robot is not a completed patrol.
    diagnostics["returned_home"] = {agent: returned.get(agent) for agent in robot_scores}
    known_returns = list(diagnostics["returned_home"].values())
    diagnostics["all_returned_home"] = (all(known_returns)
        if all(isinstance(value, bool) for value in known_returns) else None)
    return MissionScore(
        case=canonical,
        robot_scores=dict(robot_scores),
        class_scores=dict(class_scores),
        collision_free=collision_free,
        diagnostics=diagnostics,
        record_count=len(records),
    )


def score_mission(
    path: str | Path,
    case: str,
    *,
    models: Mapping[str, Any] | None = None,
    dynamics_file: str | Path | None = None,
    agents: Iterable[str] | None = None,
    expected_steps: int | None = None,
    require_done: bool = False,
    horizon: int | None = None,
    permitted_violations: int = 1,
    expected_per_class: Mapping[str, int] | None = None,
    start_time: float = 0.0,
) -> MissionScore:
    """Load and score one mission JSONL artifact."""

    records = load_mission_jsonl(path, expected_steps=expected_steps, require_done=require_done)
    if dynamics_file is not None:
        from .dynamics import load_dynamics_models

        file_models = load_dynamics_models(dynamics_file)
        if models is not None:
            merged = dict(file_models)
            merged.update(models)
            models = merged
        else:
            models = file_models
    return score_records(
        records,
        case,
        models=models,
        agents=agents,
        horizon=horizon,
        permitted_violations=permitted_violations,
        expected_per_class=expected_per_class,
        start_time=start_time,
    )


compute_scores = score_records
score_mission_log = score_mission
mission_score = aggregate_mission_scores


__all__ = [
    "MissionFailedError",
    "MissionScore",
    "MissionScoreError",
    "aggregate_mission_scores",
    "collision_free_status",
    "collision_scores",
    "compute_scores",
    "infer_class",
    "load_jsonl",
    "load_mission_jsonl",
    "load_mission_records",
    "localization_scores",
    "mission_score",
    "normalize_class",
    "order_statistic",
    "score_mission",
    "score_mission_log",
    "score_records",
    "tracking_scores",
]
