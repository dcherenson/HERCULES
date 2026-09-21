"""Localization diagnostics for mission JSONL logs.

The Python and ROS mission runners have not always used the same JSONL
schema.  This module deliberately keeps the schema adapter at the edge of
the plotting code: current logs may contain an explicit ``localization``
mapping with ``truth`` and ``estimate`` entries, while older logs generally
put truth poses in ``states`` and may omit localization estimates entirely.

The public helpers return ordinary NumPy arrays and JSON-serialisable metric
dictionaries.  They do not import matplotlib until a plot is requested, so
the parser and metrics remain usable in headless validation and unit tests.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


_LOCALIZATION_CONTAINER_KEYS = (
    "localization",
    "localization_states",
    "localization_estimates",
    "local_state_estimates",
    "state_estimates",
    "estimates",
)
_TRUTH_CONTAINER_KEYS = ("truth_states", "ground_truth", "ground_truth_states")
_TRUTH_KEYS = ("truth", "ground_truth", "actual", "true", "state")
_ESTIMATE_KEYS = (
    "estimate",
    "estimated",
    "local_state_estimate",
    "state_estimate",
    "localization_estimate",
)
_POSITION_KEYS = ("position", "estimated_position", "pos", "location")
_VELOCITY_KEYS = ("velocity", "planar_velocity", "estimated_velocity", "vel")
_YAW_KEYS = ("yaw", "estimated_yaw", "heading", "course")
_YAW_RATE_KEYS = ("yaw_rate", "estimated_yaw_rate", "heading_rate")
_COVARIANCE_KEYS = (
    "pose_covariance",
    "position_covariance",
    "uncertainty_covariance",
    "covariance",
    "covariance_matrix",
    "state_covariance",
)
_DIAGNOSTIC_KEYS = (
    "diagnostics", "diagnostic", "stats", "statistics",
    "localization_diagnostics", "communication_diagnostics",
)
_LOCALIZATION_METADATA_KEYS = {
    "algorithm",
    "algorithm_name",
    "agents",
    "timestamp",
    "sequence",
    "valid",
    "stale",
    "initialized",
    "status",
    "diagnostics",
    "diagnostic",
    "stats",
    "statistics",
    "localization_diagnostics",
    "communication_diagnostics",
}


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _lookup_mapping(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _localization_agents(value: Any) -> Mapping[str, Any] | None:
    """Return the agent map from either a per-agent or wrapped container.

    ROS validation logs wrap entries in ``localization.agents`` and the
    Python logger historically wrote entries directly under ``localization``.
    A small amount of metadata may also live beside ``agents``; it must not be
    treated as an agent when selecting files or computing aggregate metrics.
    """

    if not _is_mapping(value):
        return None
    nested = value.get("agents")
    if _is_mapping(nested):
        return nested
    return {
        str(key): item for key, item in value.items()
        if str(key) not in _LOCALIZATION_METADATA_KEYS
    }


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _vector(value: Any, dimensions: int = 3) -> np.ndarray:
    """Convert a JSON vector or pose mapping to a fixed-size float vector."""

    result = np.full(dimensions, np.nan, dtype=float)
    if _is_mapping(value):
        # Pose messages sometimes wrap the numeric vector in ``xyz`` or
        # ``position``.  Avoid recursively accepting a mapping's unrelated
        # fields as a vector.
        nested = _lookup_mapping(value, ("xyz", "value", "vector", "position"))
        if nested is not None and nested is not value:
            return _vector(nested, dimensions)
        if all(key in value for key in ("x", "y")):
            result[0] = _finite_float(value.get("x"))
            result[1] = _finite_float(value.get("y"))
            if dimensions > 2 and "z" in value:
                result[2] = _finite_float(value.get("z"))
            return result
        return result
    try:
        values = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return result
    if values.size:
        result[: min(dimensions, values.size)] = values[:dimensions]
    return result


def _quaternion_yaw(value: Any) -> float:
    try:
        quaternion = np.asarray(value, dtype=float).reshape(4)
    except (TypeError, ValueError):
        return float("nan")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12 or not np.all(np.isfinite(quaternion)):
        return float("nan")
    w, x, y, z = quaternion / norm
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _yaw(value: Any) -> float:
    if _is_mapping(value):
        direct = _lookup_mapping(value, _YAW_KEYS)
        if direct is not None:
            return _finite_float(direct)
        for key in ("orientation_quaternion", "quaternion", "orientation"):
            if value.get(key) is not None:
                return _quaternion_yaw(value[key])
        return float("nan")
    return _finite_float(value)


def _field(sample: Any, keys: Iterable[str]) -> Any:
    if not _is_mapping(sample):
        return None
    direct = _lookup_mapping(sample, keys)
    if direct is not None:
        return direct
    for nested_key in ("pose", "state", "odometry", "measurement"):
        nested = sample.get(nested_key)
        if _is_mapping(nested):
            direct = _lookup_mapping(nested, keys)
            if direct is not None:
                return direct
    return None


def _pose_fields(sample: Any) -> Dict[str, Any]:
    """Extract position, velocity, yaw, yaw-rate and covariance from a pose."""

    if not _is_mapping(sample):
        return {
            "position": np.full(3, np.nan),
            "velocity": np.full(3, np.nan),
            "yaw": float("nan"),
            "yaw_rate": float("nan"),
            "covariance": np.full((2, 2), np.nan),
            "full_covariance": np.full((3, 3), np.nan),
        }
    planar_pose = sample.get("pose") if _is_mapping(sample) else None
    full_covariance = _full_covariance(_field(sample, _COVARIANCE_KEYS))
    # The canonical localization message stores planar [x, y, yaw] as
    # ``pose``.  A mapping-valued pose is handled by the normal nested-field
    # path below; a numeric pose must not be mistaken for [x, y, z].
    if planar_pose is not None and not _is_mapping(planar_pose):
        planar = _vector(planar_pose, 3)
        position = np.array([planar[0], planar[1], np.nan], dtype=float)
        yaw = planar[2]
    else:
        position_value = _field(sample, _POSITION_KEYS)
        position = _vector(position_value, 3)
        yaw_value = _field(sample, _YAW_KEYS)
        if yaw_value is None:
            yaw_value = _field(sample, ("orientation_quaternion", "quaternion", "orientation"))
        yaw = _yaw(yaw_value) if yaw_value is not None else _yaw(sample)
    velocity_value = _field(sample, _VELOCITY_KEYS)
    covariance = full_covariance[:2, :2]
    return {
        "position": position,
        "velocity": _vector(velocity_value, 3),
        "yaw": yaw,
        "yaw_rate": _finite_float(_field(sample, _YAW_RATE_KEYS)),
        "covariance": covariance,
        "full_covariance": full_covariance,
    }


def _full_covariance(value: Any) -> np.ndarray:
    """Return a 3x3 pose covariance from common covariance encodings."""

    result = np.full((3, 3), np.nan, dtype=float)
    if value is None:
        return result
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return result
    if values.ndim == 2 and values.shape[0] >= 3 and values.shape[1] >= 3:
        result[:, :] = values[:3, :3]
    elif values.ndim == 2 and values.shape[0] >= 2 and values.shape[1] >= 2:
        result[:2, :2] = values[:2, :2]
    else:
        flat = values.reshape(-1)
        if flat.size >= 16:
            matrix = flat[:16].reshape(4, 4)
            result[:, :] = matrix[:3, :3]
        elif flat.size >= 9:
            matrix = flat[:9].reshape(3, 3)
            result[:, :] = matrix
        elif flat.size >= 4:
            result[:2, :2] = flat[:4].reshape(2, 2)
    # Covariance is expected to be symmetric.  Symmetrizing here protects the
    # square-root operation in the plotting/coverage code against tiny
    # serialization or numerical asymmetries.
    finite = np.isfinite(result)
    if finite.all():
        result = 0.5 * (result + result.T)
    elif np.isfinite(result[:2, :2]).all():
        result[:2, :2] = 0.5 * (result[:2, :2] + result[:2, :2].T)
    return result


def _covariance(value: Any) -> np.ndarray:
    """Return the x/y covariance block from common covariance encodings."""

    return _full_covariance(value)[:2, :2]


def _container_entry(record: Mapping[str, Any], name: str) -> Any:
    for key in _LOCALIZATION_CONTAINER_KEYS:
        container = record.get(key)
        if key == "localization":
            container = _localization_agents(container)
        if _is_mapping(container) and name in container:
            return container[name]
    for key in _TRUTH_CONTAINER_KEYS:
        container = record.get(key)
        if _is_mapping(container) and name in container:
            return container[name]
    states = record.get("states")
    if _is_mapping(states) and name in states:
        return states[name]
    return None


def _name_set(records: Sequence[Mapping[str, Any]]) -> List[str]:
    names = set()
    for record in records:
        for key in ("states", *_LOCALIZATION_CONTAINER_KEYS, *_TRUTH_CONTAINER_KEYS):
            container = record.get(key)
            if key == "localization":
                container = _localization_agents(container)
            if _is_mapping(container):
                names.update(str(name) for name in container)
    return sorted(names)


def _explicit_localization_names(records: Sequence[Mapping[str, Any]]) -> List[str]:
    """Return names represented by an explicit localization container."""

    names = set()
    for record in records:
        for key in _LOCALIZATION_CONTAINER_KEYS:
            container = record.get(key)
            if key == "localization":
                container = _localization_agents(container)
            if not _is_mapping(container):
                continue
            if _is_mapping(container):
                names.update(str(name) for name in container)
    return sorted(names)


def _record_time_axis(records: Sequence[Mapping[str, Any]], default_dt: float = 0.1) -> np.ndarray:
    """Return mission-relative times while tolerating old logs."""

    if not records:
        return np.empty(0, dtype=float)
    timestamps: List[float] = []
    has_timestamps = True
    for record in records:
        value = _finite_float(record.get("timestamp"))
        if not np.isfinite(value):
            has_timestamps = False
            break
        timestamps.append(value)
    # Wall-clock timestamps are occasionally stored as ``timestamp`` in old
    # logs.  Prefer step/dt in that case; mission timestamps are normally
    # small (seconds from the run start).
    if has_timestamps and timestamps and max(abs(item) for item in timestamps) < 1e7:
        return np.asarray(timestamps, dtype=float) - timestamps[0]
    try:
        dt = float(records[0].get("dt", default_dt))
    except (TypeError, ValueError):
        dt = default_dt
    steps: List[float] = []
    for index, record in enumerate(records):
        try:
            steps.append(float(record.get("step", index)))
        except (TypeError, ValueError):
            steps.append(float(index))
    return (np.asarray(steps, dtype=float) - steps[0]) * dt


def _split_entry(entry: Any) -> tuple[Any, Any]:
    """Return explicit truth and estimate objects from one log entry."""

    if not _is_mapping(entry):
        return None, None
    truth = _lookup_mapping(entry, _TRUTH_KEYS)
    estimate = _lookup_mapping(entry, _ESTIMATE_KEYS)
    # A field-level localization object is also accepted, e.g.
    # ``{"estimated_position": ..., "uncertainty_covariance": ...}``.
    if estimate is None and (
        any(key in entry for key in (
            "estimated_position", "estimated_velocity", "estimated_yaw",
            "initialized", "valid", "stale", "sequence", "algorithm",
            "algorithm_name", "covariance", "pose_covariance",
            "uncertainty_covariance", "planar_velocity",
        ))
        or ("pose" in entry and "velocity" not in entry)
    ):
        estimate = entry
    # Some loggers flatten the truth and estimate at the same level.  A plain
    # ``position`` object is truth when no explicit estimate is present; an
    # ``estimate`` object always wins above.
    if truth is None and estimate is not entry and any(key in entry for key in ("position", "velocity", "yaw")):
        truth = entry
    if truth is None and _is_mapping(entry.get("truth_state")):
        truth = entry["truth_state"]
    if estimate is None and _is_mapping(entry.get("estimate_state")):
        estimate = entry["estimate_state"]
    return truth, estimate


def _estimate_from_record(record: Mapping[str, Any], name: str, entry: Any) -> Any:
    _, estimate = _split_entry(entry)
    # Python logs place validity/staleness on the outer record while the
    # estimate payload remains nested.  Apply those flags before returning the
    # nested payload so invalid samples become plotting gaps.
    if _is_mapping(entry) and (
        entry.get("valid") is False
        or entry.get("initialized") is False
        or entry.get("stale") is True
    ):
        return None
    if estimate is not None:
        if _is_mapping(estimate) and (
            estimate.get("valid") is False or estimate.get("initialized") is False or
            estimate.get("stale") is True
        ):
            return None
        return estimate
    # Explicit estimate containers are checked separately because an entry in
    # ``states`` may contain only the truth pose.
    for key in ("localization_estimates", "local_state_estimates", "state_estimates", "estimates"):
        container = record.get(key)
        if _is_mapping(container) and name in container:
            candidate = container[name]
            _, nested = _split_entry(candidate)
            estimate = nested if nested is not None else candidate
            if _is_mapping(estimate) and (estimate.get("valid") is False or
                                          estimate.get("initialized") is False or
                                          estimate.get("stale") is True):
                return None
            return estimate
    return None


def _truth_from_record(record: Mapping[str, Any], name: str, entry: Any) -> Any:
    truth, _ = _split_entry(entry)
    if truth is not None:
        return truth
    for key in _TRUTH_CONTAINER_KEYS:
        container = record.get(key)
        if _is_mapping(container) and name in container:
            candidate = container[name]
            nested, _ = _split_entry(candidate)
            return nested if nested is not None else candidate
    states = record.get("states")
    if _is_mapping(states) and name in states:
        candidate = states[name]
        nested, _ = _split_entry(candidate)
        return nested if nested is not None else candidate
    return None


def extract_localization_series(
    records: Sequence[Mapping[str, Any]],
    names: Sequence[str] | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract truth/estimate arrays for each agent from mission records.

    Returned dictionaries contain ``truth_position`` and ``estimate_position``
    arrays of shape ``(N, 3)``, ``truth_yaw``/``estimate_yaw`` arrays of shape
    ``(N,)``, ``estimate_covariance`` of shape ``(N, 2, 2)``, and matching
    ``position_sigma``/``yaw_sigma`` arrays.  Missing values are represented by
    NaN, which lets callers handle asynchronous or legacy records with masks.
    """

    sequence = list(records)
    selected = [str(name) for name in names] if names is not None else _name_set(sequence)
    time = _record_time_axis(sequence)
    output: Dict[str, Dict[str, np.ndarray]] = {}
    for name in selected:
        truth_position = np.full((len(sequence), 3), np.nan, dtype=float)
        estimate_position = np.full((len(sequence), 3), np.nan, dtype=float)
        truth_velocity = np.full((len(sequence), 3), np.nan, dtype=float)
        estimate_velocity = np.full((len(sequence), 3), np.nan, dtype=float)
        truth_yaw = np.full(len(sequence), np.nan, dtype=float)
        estimate_yaw = np.full(len(sequence), np.nan, dtype=float)
        truth_yaw_rate = np.full(len(sequence), np.nan, dtype=float)
        estimate_yaw_rate = np.full(len(sequence), np.nan, dtype=float)
        covariance = np.full((len(sequence), 2, 2), np.nan, dtype=float)
        pose_covariance = np.full((len(sequence), 3, 3), np.nan, dtype=float)
        valid = np.zeros(len(sequence), dtype=bool)
        stale = np.zeros(len(sequence), dtype=bool)
        initialized = np.zeros(len(sequence), dtype=bool)
        sequence_numbers = np.full(len(sequence), np.nan, dtype=float)
        algorithms = np.full(len(sequence), "", dtype=object)
        diagnostics: List[Mapping[str, Any] | None] = [None] * len(sequence)
        for index, record in enumerate(sequence):
            entry = _container_entry(record, name)
            truth = _truth_from_record(record, name, entry)
            _, raw_estimate = _split_entry(entry)
            estimate = _estimate_from_record(record, name, entry)
            truth_fields = _pose_fields(truth)
            estimate_fields = _pose_fields(estimate)
            # Canonical entries keep validity and diagnostics alongside the
            # estimate.  Nested legacy entries may put covariance/flags on
            # the outer object, so use it as a fallback.
            metadata = entry if _is_mapping(entry) else {}
            global_localization = record.get("localization")
            if _is_mapping(global_localization) and _is_mapping(global_localization.get("agents")):
                global_localization = global_localization
            else:
                global_localization = {}
            if estimate is not None and _is_mapping(estimate):
                estimate_metadata = estimate
            elif _is_mapping(raw_estimate):
                estimate_metadata = raw_estimate
            else:
                estimate_metadata = metadata
            # Validity, sequencing and algorithm metadata may be attached to
            # the outer ``localization`` entry (the Python logger does this)
            # while the numerical estimate is nested.  Let explicit outer
            # fields override the nested defaults without copying unrelated
            # payload fields.
            if metadata is not estimate_metadata:
                merged_metadata = dict(estimate_metadata)
                for metadata_key in (
                    "valid", "stale", "initialized", "sequence", "algorithm",
                    "algorithm_name", *_DIAGNOSTIC_KEYS,
                ):
                    if metadata_key in metadata:
                        merged_metadata[metadata_key] = metadata[metadata_key]
                estimate_metadata = merged_metadata
            has_estimate_fields = estimate is not None or (
                _is_mapping(metadata) and any(
                    key in metadata for key in (
                        "pose", "estimated_position", "estimated_velocity",
                        "estimated_yaw", "estimate", "estimated",
                    )
                )
            )
            if metadata is not estimate:
                metadata_covariance = _full_covariance(_field(metadata, _COVARIANCE_KEYS))
                estimate_covariance = np.asarray(estimate_fields["full_covariance"], dtype=float)
                missing_covariance = ~np.isfinite(estimate_covariance) & np.isfinite(metadata_covariance)
                if np.any(missing_covariance):
                    estimate_fields = dict(estimate_fields)
                    estimate_covariance = estimate_covariance.copy()
                    estimate_covariance[missing_covariance] = metadata_covariance[missing_covariance]
                    estimate_fields["full_covariance"] = estimate_covariance
                    estimate_fields["covariance"] = estimate_covariance[:2, :2]
            truth_position[index] = truth_fields["position"]
            truth_velocity[index] = truth_fields["velocity"]
            truth_yaw[index] = truth_fields["yaw"]
            truth_yaw_rate[index] = truth_fields["yaw_rate"]
            estimate_position[index] = estimate_fields["position"]
            estimate_velocity[index] = estimate_fields["velocity"]
            estimate_yaw[index] = estimate_fields["yaw"]
            estimate_yaw_rate[index] = estimate_fields["yaw_rate"]
            covariance[index] = estimate_fields["covariance"]
            pose_covariance[index] = estimate_fields["full_covariance"]
            if _is_mapping(estimate_metadata):
                valid[index] = bool(estimate_metadata.get("valid", has_estimate_fields))
                stale[index] = bool(estimate_metadata.get("stale", False))
                initialized[index] = bool(estimate_metadata.get("initialized", has_estimate_fields))
                sequence_numbers[index] = _finite_float(estimate_metadata.get("sequence"))
                algorithms[index] = str(
                    estimate_metadata.get(
                        "algorithm",
                        estimate_metadata.get(
                            "algorithm_name",
                            metadata.get(
                                "algorithm",
                                metadata.get(
                                    "algorithm_name", global_localization.get("algorithm", "")
                                ),
                            ),
                        ),
                    ) or ""
                )
                diagnostic_value = _lookup_mapping(estimate_metadata, _DIAGNOSTIC_KEYS)
                if diagnostic_value is None and metadata is not estimate_metadata:
                    diagnostic_value = _lookup_mapping(metadata, _DIAGNOSTIC_KEYS)
                if diagnostic_value is None:
                    diagnostic_value = _lookup_mapping(global_localization, _DIAGNOSTIC_KEYS)
                if diagnostic_value is None and any(
                    "accept" in str(key).lower()
                    or "reject" in str(key).lower()
                    or str(key).lower() in {"updates", "peer_estimates_received"}
                    for key in estimate_metadata
                ):
                    diagnostic_value = estimate_metadata
                diagnostics[index] = diagnostic_value if _is_mapping(diagnostic_value) else None
            # Invalid, stale, or uninitialized estimates must render as gaps;
            # retain the flags separately so metrics can report their counts.
            if not valid[index] or stale[index] or not initialized[index]:
                estimate_position[index] = np.nan
                estimate_velocity[index] = np.nan
                estimate_yaw[index] = np.nan
                estimate_yaw_rate[index] = np.nan
                covariance[index] = np.nan
                pose_covariance[index] = np.nan
        diagonal = np.diagonal(covariance, axis1=1, axis2=2)
        sigma = np.full((len(sequence), 2), np.nan, dtype=float)
        finite_diagonal = np.isfinite(diagonal)
        sigma[finite_diagonal] = np.sqrt(np.maximum(diagonal[finite_diagonal], 0.0))
        # Keep yaw's third diagonal separately; x/y bands remain exactly the
        # reported sqrt(Pxx)/sqrt(Pyy) values.
        yaw_variance = np.full(len(sequence), np.nan, dtype=float)
        for index, record in enumerate(sequence):
            entry = _container_entry(record, name)
            estimate = _estimate_from_record(record, name, entry)
            full = _pose_fields(estimate)["full_covariance"]
            if np.isfinite(full[2, 2]):
                yaw_variance[index] = full[2, 2]
        yaw_sigma = np.sqrt(np.maximum(yaw_variance, 0.0))
        yaw_diagonal = pose_covariance[:, 2, 2]
        finite_yaw_diagonal = np.isfinite(yaw_diagonal)
        yaw_sigma[finite_yaw_diagonal] = np.sqrt(np.maximum(yaw_diagonal[finite_yaw_diagonal], 0.0))
        algorithm_values = [value for value in algorithms if value]
        algorithm_name = Counter(algorithm_values).most_common(1)[0][0] if algorithm_values else "legacy"
        output[name] = {
            "time": time.copy(),
            "truth_position": truth_position,
            "estimate_position": estimate_position,
            "truth_velocity": truth_velocity,
            "estimate_velocity": estimate_velocity,
            "truth_yaw": truth_yaw,
            "estimate_yaw": estimate_yaw,
            "truth_yaw_rate": truth_yaw_rate,
            "estimate_yaw_rate": estimate_yaw_rate,
            "estimate_covariance": covariance,
            "pose_covariance": pose_covariance,
            "position_sigma": sigma,
            "yaw_sigma": yaw_sigma,
            "truth_available": np.isfinite(truth_position[:, :2]).all(axis=1),
            "estimate_available": (
                np.isfinite(estimate_position[:, :2]).all(axis=1)
                & valid & ~stale & initialized
            ),
            "valid": valid,
            "stale": stale,
            "initialized": initialized,
            "sequence": sequence_numbers,
            "algorithm": algorithms,
            "algorithm_name": algorithm_name,
            "diagnostics": diagnostics,
        }
    return output


def has_localization_estimates(
    records: Sequence[Mapping[str, Any]],
    names: Sequence[str] | None = None,
) -> bool:
    """Return whether at least one truth/estimate position pair is logged."""

    for series in extract_localization_series(records, names).values():
        if np.any(series["truth_available"] & series["estimate_available"]):
            return True
    return False


def has_localization_data(
    records: Sequence[Mapping[str, Any]],
    names: Sequence[str] | None = None,
) -> bool:
    """Return whether records carry an explicit localization estimate field.

    This is intentionally weaker than :func:`has_localization_estimates`:
    runs that start before initialization, or contain only stale samples, can
    still produce useful per-agent plots with visible gaps and diagnostics.
    """

    selected = set(str(name) for name in names) if names is not None else None
    for record in records:
        for key in _LOCALIZATION_CONTAINER_KEYS:
            container = record.get(key)
            if key == "localization":
                container = _localization_agents(container)
            if not _is_mapping(container):
                continue
            if selected is None:
                return bool(container)
            if any(str(name) in container for name in selected):
                return True
    return has_localization_estimates(records, names)


def wrap_angle(angle: Any) -> np.ndarray:
    """Wrap radians to ``[-pi, pi)`` while preserving NaNs."""

    values = np.asarray(angle, dtype=float)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _metric_value(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _agent_metrics(series: Mapping[str, np.ndarray], sigma_level: float) -> Dict[str, Any]:
    truth = np.asarray(series["truth_position"], dtype=float)
    estimate = np.asarray(series["estimate_position"], dtype=float)
    position_mask = np.isfinite(truth[:, :2]).all(axis=1) & np.isfinite(estimate[:, :2]).all(axis=1)
    errors = estimate[:, :2] - truth[:, :2]
    errors = errors[position_mask]
    truth_yaw = np.asarray(series["truth_yaw"], dtype=float)
    estimate_yaw = np.asarray(series["estimate_yaw"], dtype=float)
    yaw_mask = np.isfinite(truth_yaw) & np.isfinite(estimate_yaw)
    yaw_errors = wrap_angle(estimate_yaw[yaw_mask] - truth_yaw[yaw_mask])
    sigma = np.asarray(series["position_sigma"], dtype=float)
    sigma_samples = sigma[position_mask] if len(sigma) else np.empty((0, 2))
    sigma_mask = np.isfinite(sigma_samples) & (sigma_samples >= 0.0)
    coverage = []
    for dimension in range(2):
        valid = sigma_mask[:, dimension] if len(sigma_samples) else np.empty(0, dtype=bool)
        coverage.append(
            _metric_value(np.mean(
                np.abs(errors[valid, dimension])
                <= sigma_samples[valid, dimension] * sigma_level + 1e-12
            ))
            if np.any(valid) else None
        )
    if len(errors):
        euclidean = np.linalg.norm(errors, axis=1)
        x_rmse = np.sqrt(np.mean(errors[:, 0] ** 2))
        y_rmse = np.sqrt(np.mean(errors[:, 1] ** 2))
        position_rmse = np.sqrt(np.mean(euclidean ** 2))
        position_max = np.max(euclidean)
        position_mean = np.mean(euclidean)
    else:
        x_rmse = y_rmse = position_rmse = position_max = position_mean = None
    if len(yaw_errors):
        yaw_rmse = np.sqrt(np.mean(yaw_errors ** 2))
        yaw_mean = np.mean(np.abs(yaw_errors))
        yaw_max = np.max(np.abs(yaw_errors))
    else:
        yaw_rmse = yaw_mean = yaw_max = None
    yaw_sigma = np.asarray(series.get("yaw_sigma", np.full(len(truth_yaw), np.nan)), dtype=float)
    mean_yaw_sigma = np.nanmean(yaw_sigma) if np.any(np.isfinite(yaw_sigma)) else None
    yaw_sigma_samples = yaw_sigma[yaw_mask] if len(yaw_sigma) else np.empty(0)
    yaw_sigma_valid = (
        np.isfinite(yaw_sigma_samples) & (yaw_sigma_samples >= 0.0)
        if len(yaw_sigma_samples) else np.empty(0, dtype=bool)
    )
    yaw_coverage = (
        _metric_value(
            np.mean(np.abs(yaw_errors[yaw_sigma_valid])
                   <= yaw_sigma_samples[yaw_sigma_valid] * sigma_level + 1e-12)
        )
        if np.any(yaw_sigma_valid) else None
    )
    # NEES uses the reported planar [x, y, yaw] covariance when available and
    # falls back to the x/y block for older logs that predate yaw covariance.
    nees_values = []
    paired_indices = np.flatnonzero(position_mask)
    full_covariance = np.asarray(
        series.get("pose_covariance", np.full((len(truth), 3, 3), np.nan)), dtype=float
    )
    for error_index, index in enumerate(paired_indices):
        covariance = full_covariance[index] if len(full_covariance) > index else np.full((3, 3), np.nan)
        if (
            np.any(yaw_mask & (np.arange(len(yaw_mask)) == index))
            and covariance.shape == (3, 3) and np.isfinite(covariance).all()
        ):
            covariance = 0.5 * (covariance + covariance.T)
            error = np.asarray([errors[error_index, 0], errors[error_index, 1], yaw_errors[np.flatnonzero(yaw_mask).tolist().index(index)]])
        else:
            covariance = np.asarray(series["estimate_covariance"][index], dtype=float)
            error = errors[error_index]
        if covariance.ndim != 2 or covariance.shape[0] not in (2, 3) or covariance.shape[1] != covariance.shape[0] or not np.isfinite(covariance).all():
            continue
        try:
            covariance = 0.5 * (covariance + covariance.T)
            nees_values.append(float(error @ np.linalg.pinv(covariance) @ error))
        except (ValueError, np.linalg.LinAlgError):
            pass

    diagnostics = series.get("diagnostics", [])

    def numeric(mapping: Mapping[str, Any], key: str) -> float | None:
        value = mapping.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            return None
        value = float(value)
        return value if np.isfinite(value) else None

    # Diagnostics are often serialized with both a descriptive alias and a
    # canonical counter (for example ``accepted_relative_updates`` and
    # ``updates``).  Pick the first available alias per sample so the same
    # counter is not double-counted when producing totals.
    relative_total = 0.0
    communication_total = 0.0
    accepted_total: float | None = None
    rejected_total = 0.0
    rejected_measurement_total = 0.0
    rejected_communication_total = 0.0
    if isinstance(diagnostics, Sequence):
        for diagnostic in diagnostics:
            if not _is_mapping(diagnostic):
                continue

            def first_numeric(*keys: str) -> float | None:
                for key in keys:
                    value = numeric(diagnostic, key)
                    if value is not None:
                        return value
                return None

            relative = first_numeric(
                "accepted_relative_updates", "relative_updates_accepted",
                "accepted_measurements", "updates",
            )
            communication = first_numeric(
                "accepted_communication_updates", "communication_updates_accepted",
                "peer_estimates_received",
            )
            if relative is not None:
                relative_total += relative
            if communication is not None:
                communication_total += communication
            explicit_accepted = numeric(diagnostic, "accepted_updates")
            if explicit_accepted is not None:
                accepted_total = (accepted_total or 0.0) + explicit_accepted
            elif relative is not None or communication is not None:
                # There is no explicit total in this sample, so derive it
                # from the disjoint relative/communication counters.
                accepted_total = (accepted_total or 0.0) + (relative or 0.0) + (communication or 0.0)
            rejected = first_numeric(
                "rejected_updates", "rejected_communications",
                "rejected_relative_updates", "rejected_communication_updates",
                "peer_estimates_rejected",
            )
            rejected_measurements = first_numeric(
                "rejected_measurements", "measurements_rejected",
            )
            if rejected is not None:
                rejected_total += rejected
                rejected_communication_total += rejected
            if rejected_measurements is not None:
                rejected_measurement_total += rejected_measurements
            # If only one broad rejection counter is present, expose it in
            # both aliases; if both categories are present, their sum is the
            # broad total.
            if rejected is None and rejected_measurements is not None:
                rejected_total += rejected_measurements
            if rejected is not None and rejected_measurements is None:
                rejected_measurement_total += rejected

    valid_samples = np.asarray(series.get("valid", []), dtype=bool)
    stale_samples = np.asarray(series.get("stale", []), dtype=bool)
    initialized_samples = np.asarray(series.get("initialized", []), dtype=bool)
    return {
        "samples": int(len(series["time"])),
        "paired_position_samples": int(np.sum(position_mask)),
        "paired_yaw_samples": int(np.sum(yaw_mask)),
        "position_rmse_m": _metric_value(position_rmse),
        "position_rms_m": _metric_value(position_rmse),
        "position_mean_error_m": _metric_value(position_mean),
        "position_max_error_m": _metric_value(position_max),
        "x_rmse_m": _metric_value(x_rmse),
        "y_rmse_m": _metric_value(y_rmse),
        "x_rmse": _metric_value(x_rmse),
        "y_rmse": _metric_value(y_rmse),
        "rmse_x_m": _metric_value(x_rmse),
        "rmse_y_m": _metric_value(y_rmse),
        "yaw_rmse_rad": _metric_value(yaw_rmse),
        "yaw_rmse": _metric_value(yaw_rmse),
        "rmse_yaw_rad": _metric_value(yaw_rmse),
        "yaw_mean_abs_error_rad": _metric_value(yaw_mean),
        "yaw_max_abs_error_rad": _metric_value(yaw_max),
        "mean_x_sigma_m": _metric_value(np.nanmean(sigma[:, 0])) if np.any(np.isfinite(sigma[:, 0])) else None,
        "mean_y_sigma_m": _metric_value(np.nanmean(sigma[:, 1])) if np.any(np.isfinite(sigma[:, 1])) else None,
        "mean_sigma_x": _metric_value(np.nanmean(sigma[:, 0])) if np.any(np.isfinite(sigma[:, 0])) else None,
        "mean_sigma_y": _metric_value(np.nanmean(sigma[:, 1])) if np.any(np.isfinite(sigma[:, 1])) else None,
        "mean_yaw_sigma_rad": _metric_value(mean_yaw_sigma),
        "mean_reported_sigma_x_m": _metric_value(np.nanmean(sigma[:, 0])) if np.any(np.isfinite(sigma[:, 0])) else None,
        "mean_reported_sigma_y_m": _metric_value(np.nanmean(sigma[:, 1])) if np.any(np.isfinite(sigma[:, 1])) else None,
        "mean_reported_sigma_m": _metric_value(np.nanmean(sigma)) if np.any(np.isfinite(sigma)) else None,
        "nees": _metric_value(np.mean(nees_values)) if nees_values else None,
        "nees_mean": _metric_value(np.mean(nees_values)) if nees_values else None,
        "nees_max": _metric_value(np.max(nees_values)) if nees_values else None,
        "nees_samples": int(len(nees_values)),
        "x_one_sigma_coverage": coverage[0],
        "y_one_sigma_coverage": coverage[1],
        "yaw_one_sigma_coverage": yaw_coverage,
        "coverage_x": coverage[0],
        "coverage_y": coverage[1],
        "coverage_yaw": yaw_coverage,
        "empirical_x_one_sigma_coverage": coverage[0],
        "empirical_y_one_sigma_coverage": coverage[1],
        "empirical_yaw_one_sigma_coverage": yaw_coverage,
        "one_sigma_coverage": {"x": coverage[0], "y": coverage[1], "yaw": yaw_coverage},
        "valid_samples": int(np.sum(valid_samples)) if len(valid_samples) else 0,
        "invalid_samples": int(np.sum(~valid_samples)) if len(valid_samples) else 0,
        "stale_samples": int(np.sum(stale_samples)) if len(stale_samples) else 0,
        "uninitialized_samples": int(np.sum(~initialized_samples)) if len(initialized_samples) else 0,
        "accepted_relative_updates": int(round(relative_total)),
        "accepted_communication_updates": int(round(communication_total)),
        "rejected_updates": int(round(rejected_total)),
        "rejected_measurements": int(round(rejected_measurement_total)),
        "rejected_relative_updates": int(round(rejected_measurement_total)),
        "rejected_communication_updates": int(round(rejected_communication_total)),
        "accepted_updates": int(round(accepted_total)) if accepted_total is not None else 0,
        "algorithm": str(series.get("algorithm_name", "legacy")),
        "available": bool(len(errors) or len(yaw_errors)),
    }


def compute_localization_metrics(
    records_or_series: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, np.ndarray]],
    names: Sequence[str] | None = None,
    sigma_level: float = 1.0,
) -> Dict[str, Any]:
    """Compute position/yaw estimation errors and 1-sigma coverage.

    The ``records_or_series`` argument may be raw JSONL records or the result
    of :func:`extract_localization_series`.  Missing estimate fields produce
    an ``available: false`` result rather than making old logs fail.
    """

    if sigma_level <= 0.0:
        raise ValueError("sigma_level must be positive")
    if _looks_like_series(records_or_series):
        series = records_or_series  # type: ignore[assignment]
        if names is not None:
            series = {str(name): series[str(name)] for name in names if str(name) in series}
    else:
        selected_names = names
        if selected_names is None:
            explicit_names = _explicit_localization_names(records_or_series)
            selected_names = explicit_names or None
        series = extract_localization_series(records_or_series, selected_names)  # type: ignore[arg-type]
    per_agent = {
        str(name): _agent_metrics(value, float(sigma_level))
        for name, value in sorted(series.items())
    }
    all_position_errors: List[np.ndarray] = []
    all_yaw_errors: List[np.ndarray] = []
    for value in series.values():
        truth = np.asarray(value["truth_position"], dtype=float)
        estimate = np.asarray(value["estimate_position"], dtype=float)
        mask = np.isfinite(truth[:, :2]).all(axis=1) & np.isfinite(estimate[:, :2]).all(axis=1)
        if np.any(mask):
            all_position_errors.append(estimate[mask, :2] - truth[mask, :2])
        truth_yaw = np.asarray(value["truth_yaw"], dtype=float)
        estimate_yaw = np.asarray(value["estimate_yaw"], dtype=float)
        mask = np.isfinite(truth_yaw) & np.isfinite(estimate_yaw)
        if np.any(mask):
            all_yaw_errors.append(wrap_angle(estimate_yaw[mask] - truth_yaw[mask]))
    position = np.vstack(all_position_errors) if all_position_errors else np.empty((0, 2))
    yaw = np.concatenate(all_yaw_errors) if all_yaw_errors else np.empty(0)
    if len(position):
        position_norm = np.linalg.norm(position, axis=1)
        global_position_rmse = np.sqrt(np.mean(position_norm ** 2))
        global_x_rmse = np.sqrt(np.mean(position[:, 0] ** 2))
        global_y_rmse = np.sqrt(np.mean(position[:, 1] ** 2))
    else:
        global_position_rmse = global_x_rmse = global_y_rmse = None
    global_yaw_rmse = np.sqrt(np.mean(yaw ** 2)) if len(yaw) else None
    algorithms = sorted({str(value.get("algorithm", "legacy")) for value in per_agent.values()})
    global_valid_samples = sum(int(value.get("valid_samples", 0)) for value in per_agent.values())
    global_invalid_samples = sum(int(value.get("invalid_samples", 0)) for value in per_agent.values())
    global_stale_samples = sum(int(value.get("stale_samples", 0)) for value in per_agent.values())
    global_uninitialized_samples = sum(int(value.get("uninitialized_samples", 0)) for value in per_agent.values())
    global_nees_values = [
        float(value["nees"]) for value in per_agent.values()
        if value.get("nees") is not None
    ]
    global_metrics = {
        "agents": int(len(per_agent)),
        "paired_position_samples": int(len(position)),
        "paired_yaw_samples": int(len(yaw)),
        "position_rmse_m": _metric_value(global_position_rmse),
        "position_rms_m": _metric_value(global_position_rmse),
        "x_rmse_m": _metric_value(global_x_rmse),
        "y_rmse_m": _metric_value(global_y_rmse),
        "x_rmse": _metric_value(global_x_rmse),
        "y_rmse": _metric_value(global_y_rmse),
        "yaw_rmse_rad": _metric_value(global_yaw_rmse),
        "yaw_rmse": _metric_value(global_yaw_rmse),
        "valid_samples": int(global_valid_samples),
        "invalid_samples": int(global_invalid_samples),
        "stale_samples": int(global_stale_samples),
        "uninitialized_samples": int(global_uninitialized_samples),
        "nees": _metric_value(np.mean(global_nees_values)) if global_nees_values else None,
        "algorithms": algorithms,
    }
    available = bool(len(position) or len(yaw))
    return {
        "available": available,
        "legacy_schema": not any(
            _is_mapping(record.get("localization")) for record in records_or_series
        ) if not _looks_like_series(records_or_series) else False,
        "agents": per_agent,
        "per_agent": per_agent,
        "global": global_metrics,
    }


def _looks_like_series(value: Any) -> bool:
    if not _is_mapping(value) or not value:
        return False
    first = next(iter(value.values()))
    return _is_mapping(first) and "time" in first and "truth_position" in first


def _unwrap_finite(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(result)
    if not np.any(finite):
        return result
    indices = np.flatnonzero(finite)
    for start, stop in zip(indices[:-1], indices[1:]):
        if stop > start + 1:
            result[start + 1:stop] = np.nan
    # There can be multiple finite runs; unwrap each independently so a
    # missing estimate cannot produce an artificial 2*pi jump in the plot.
    runs = []
    start = indices[0]
    previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            runs.append((start, previous + 1))
            start = index
        previous = index
    runs.append((start, previous + 1))
    for start, stop in runs:
        result[start:stop] = np.unwrap(result[start:stop])
    return result


def _wrapped_yaw_traces(
    truth_values: np.ndarray,
    estimate_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unwrap yaw traces while keeping paired estimates on truth's branch."""

    truth = _unwrap_finite(truth_values)
    estimate = _unwrap_finite(estimate_values)
    paired = np.isfinite(truth_values) & np.isfinite(estimate_values)
    if np.any(paired):
        # The raw yaw values can straddle +/-pi even when the physical error
        # is small.  Align paired estimates with the truth branch using the
        # shortest angular residual before drawing the uncertainty band.
        estimate[paired] = truth[paired] + wrap_angle(
            np.asarray(estimate_values, dtype=float)[paired]
            - np.asarray(truth_values, dtype=float)[paired]
        )
    return truth, estimate


def plot_localization_truth_vs_estimate(
    records_or_series: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, np.ndarray]],
    output_path: str,
    agent_names: Sequence[str] | None = None,
    *,
    agent_name: str | None = None,
    title: str | None = None,
    dpi: int = 150,
) -> str:
    """Save x/y/yaw truth-vs-estimate plots with covariance bands.

    Each selected agent gets matching colored truth and estimate traces.  For
    x and y, finite covariance diagonals are rendered as a translucent
    estimate +/- one-standard-deviation band.  Yaw is plotted on its own
    subplot and unwrapped for readability; metric error computation still
    uses a wrapped shortest-angle difference.
    """

    if _looks_like_series(records_or_series):
        series = records_or_series  # type: ignore[assignment]
    else:
        selected_names = agent_names
        if selected_names is None:
            # Explicit localization records should not pull unrelated
            # truth-only targets from the historical ``states`` container
            # into the comparison figure.  Legacy logs still fall back to
            # all state names for a useful truth-only artifact.
            selected_names = _explicit_localization_names(records_or_series)
            if not selected_names:
                selected_names = None
        series = extract_localization_series(records_or_series, selected_names)  # type: ignore[arg-type]
    selected = [str(agent_name)] if agent_name is not None else (
        [str(name) for name in agent_names] if agent_names is not None else sorted(series)
    )
    selected = [name for name in selected if name in series]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    axis_labels = ("X position (m)", "Y position (m)", "Yaw (rad)")
    colors = plt.get_cmap("tab10")
    plotted = False
    for color_index, name in enumerate(selected):
        value = series[name]
        time = np.asarray(value["time"], dtype=float)
        color = colors(color_index % 10)
        truth_position = np.asarray(value["truth_position"], dtype=float)
        estimate_position = np.asarray(value["estimate_position"], dtype=float)
        sigma = np.asarray(value["position_sigma"], dtype=float)
        truth_yaw, estimate_yaw = _wrapped_yaw_traces(
            np.asarray(value["truth_yaw"], dtype=float),
            np.asarray(value["estimate_yaw"], dtype=float),
        )
        yaw_sigma = np.asarray(value.get("yaw_sigma", np.full(len(time), np.nan)), dtype=float)
        truth_valid = np.isfinite(truth_position[:, :2]).all(axis=1)
        estimate_valid = np.isfinite(estimate_position[:, :2]).all(axis=1)
        if np.any(truth_valid):
            axes[0].plot(time, np.where(np.isfinite(truth_position[:, 0]), truth_position[:, 0], np.nan),
                         color=color, linewidth=1.7, label=f"{name} truth")
            axes[1].plot(time, np.where(np.isfinite(truth_position[:, 1]), truth_position[:, 1], np.nan),
                         color=color, linewidth=1.7)
            plotted = True
        if np.any(estimate_valid):
            axes[0].plot(time, np.where(np.isfinite(estimate_position[:, 0]), estimate_position[:, 0], np.nan),
                         color=color, linestyle="--", linewidth=1.4, label=f"{name} estimate")
            axes[1].plot(time, np.where(np.isfinite(estimate_position[:, 1]), estimate_position[:, 1], np.nan),
                         color=color, linestyle="--", linewidth=1.4)
            for axis_index in range(2):
                finite_band = estimate_valid & np.isfinite(sigma[:, axis_index])
                if np.any(finite_band):
                    center = estimate_position[:, axis_index]
                    width = sigma[:, axis_index]
                    axes[axis_index].fill_between(
                        time,
                        np.where(finite_band, center - width, np.nan),
                        np.where(finite_band, center + width, np.nan),
                        color=color, alpha=0.14, linewidth=0.0,
                        label=f"{name} +/- 1 sigma" if axis_index == 0 else None,
                    )
            plotted = True
        yaw_truth_valid = np.isfinite(truth_yaw)
        yaw_estimate_valid = np.isfinite(estimate_yaw)
        if np.any(yaw_truth_valid):
            axes[2].plot(time, np.where(yaw_truth_valid, truth_yaw, np.nan),
                         color=color, linewidth=1.7)
        if np.any(yaw_estimate_valid):
            axes[2].plot(time, np.where(yaw_estimate_valid, estimate_yaw, np.nan),
                         color=color, linestyle="--", linewidth=1.4)
            finite_yaw_band = yaw_estimate_valid & np.isfinite(yaw_sigma)
            if np.any(finite_yaw_band):
                axes[2].fill_between(
                    time,
                    np.where(finite_yaw_band, estimate_yaw - yaw_sigma, np.nan),
                    np.where(finite_yaw_band, estimate_yaw + yaw_sigma, np.nan),
                    color=color, alpha=0.10, linewidth=0.0,
                    label=f"{name} yaw +/- 1 sigma",
                )
    if not plotted:
        for axis in axes:
            axis.text(0.5, 0.5, "No localization estimate data", ha="center", va="center",
                      transform=axis.transAxes, color="0.4")
    for axis, label in zip(axes, axis_labels):
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.28)
    axes[-1].set_xlabel("Mission time (s)")
    axes[0].legend(loc="best", fontsize="small", ncol=2)
    axes[0].set_title(title or "Localization truth versus estimate")
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


def render_localization_plots(
    records_or_series: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, np.ndarray]],
    output_dir: str,
    stem: str = "localization",
    names: Sequence[str] | None = None,
    *,
    dpi: int = 150,
) -> List[str]:
    """Render one truth-vs-estimate figure per agent.

    Filenames include the selected estimator algorithm and agent name, for
    example ``mission_localization_gs_ci_Drone1.png``.  This keeps outputs
    from recursive and GS-CI runs comparable without overwriting one another.
    A legacy log uses ``legacy`` as its algorithm label and still renders a
    truth-only/empty estimate figure when explicitly requested.
    """

    if _looks_like_series(records_or_series):
        series = records_or_series  # type: ignore[assignment]
        if names is not None:
            series = {str(name): series[str(name)] for name in names if str(name) in series}
    else:
        selected_names = names
        if selected_names is None:
            explicit_names = _explicit_localization_names(records_or_series)
            selected_names = explicit_names or None
        series = extract_localization_series(records_or_series, selected_names)  # type: ignore[arg-type]
    os.makedirs(output_dir, exist_ok=True)

    def safe_name(value: Any) -> str:
        text = str(value or "legacy")
        return "".join(character if character.isalnum() or character in "-_" else "_" for character in text)

    paths: List[str] = []
    for name in sorted(series):
        algorithm = safe_name(series[name].get("algorithm_name", "legacy"))
        path = os.path.join(output_dir, f"{safe_name(stem)}_{algorithm}_{safe_name(name)}.png")
        plot_localization_truth_vs_estimate(
            series, path, agent_name=name,
            title=f"{name} localization ({algorithm})",
            dpi=dpi,
        )
        paths.append(path)
    return paths


def plot_localization_per_agent(*args: Any, **kwargs: Any) -> List[str]:
    """Compatibility alias for :func:`render_localization_plots`."""

    return render_localization_plots(*args, **kwargs)


# Short aliases are intentionally kept in this module and re-exported by
# mission_plots.  They make the helper convenient for validation scripts while
# preserving the descriptive public name above.
plot_localization = plot_localization_truth_vs_estimate
localization_metrics = compute_localization_metrics
compute_localization_error_metrics = compute_localization_metrics
plot_localization_comparison = plot_localization_truth_vs_estimate


def localization_log_entry(
    truth_state: Mapping[str, Any] | None,
    estimate_state: Mapping[str, Any] | None,
    *,
    vehicle_type: str | None = None,
    algorithm: str = "python_placeholder",
    valid: bool | None = None,
    stale: bool = False,
    initialized: bool | None = None,
    timestamp: float | None = None,
    sequence: int | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the explicit, JSON-friendly localization log entry.

    This small adapter is used by the Python mission logger and is also useful
    to tests/synthetic log producers.  Unknown estimate fields are preserved
    only when they are already JSON-compatible; the canonical pose and
    covariance keys are normalized to lists/scalars.
    """

    def serializable_pose(value: Mapping[str, Any] | None) -> Dict[str, Any]:
        fields = _pose_fields(value)
        def serializable_vector(array: np.ndarray) -> List[float | None]:
            return [float(item) if np.isfinite(item) else None for item in np.asarray(array).reshape(-1)]

        output: Dict[str, Any] = {
            "position": serializable_vector(fields["position"]),
            "velocity": serializable_vector(fields["velocity"]),
            "yaw": _metric_value(fields["yaw"]),
            "yaw_rate": _metric_value(fields["yaw_rate"]),
        }
        covariance = fields["covariance"]
        if np.isfinite(covariance).all():
            output["covariance"] = covariance.tolist()
            output["uncertainty_covariance"] = covariance.tolist()
        full_covariance = fields["full_covariance"]
        if np.isfinite(full_covariance).all():
            output["pose_covariance"] = full_covariance.reshape(-1).tolist()
        return output

    estimate_fields = _pose_fields(estimate_state)
    estimate_position = estimate_fields["position"]
    estimate_pose = [
        float(estimate_position[0]) if np.isfinite(estimate_position[0]) else None,
        float(estimate_position[1]) if np.isfinite(estimate_position[1]) else None,
        _metric_value(estimate_fields["yaw"]),
    ]
    estimate_velocity = estimate_fields["velocity"]
    canonical_velocity = [
        float(item) if np.isfinite(item) else None for item in estimate_velocity[:2]
    ]
    canonical_covariance = estimate_fields["full_covariance"]
    result: Dict[str, Any] = {
        "truth": serializable_pose(truth_state),
        "estimate": serializable_pose(estimate_state),
        "algorithm": str(algorithm),
        "valid": bool(valid) if valid is not None else bool(estimate_state is not None),
        "stale": bool(stale),
        "initialized": bool(initialized) if initialized is not None else bool(estimate_state is not None),
        "pose": estimate_pose,
        "velocity": canonical_velocity,
    }
    if np.isfinite(canonical_covariance).all():
        result["covariance"] = canonical_covariance.reshape(-1).tolist()
        result["pose_covariance"] = canonical_covariance.reshape(-1).tolist()
    if timestamp is not None:
        result["timestamp"] = float(timestamp)
    if sequence is not None:
        result["sequence"] = int(sequence)
    if diagnostics is not None:
        def serializable_value(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {str(key): serializable_value(item) for key, item in value.items()}
            if isinstance(value, np.ndarray):
                return [serializable_value(item) for item in value.tolist()]
            if isinstance(value, (list, tuple)):
                return [serializable_value(item) for item in value]
            if isinstance(value, np.generic):
                return serializable_value(value.item())
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        result["diagnostics"] = serializable_value(diagnostics)
    if vehicle_type is not None:
        result["vehicle_type"] = str(vehicle_type)
    return result


__all__ = [
    "compute_localization_metrics",
    "compute_localization_error_metrics",
    "extract_localization_series",
    "has_localization_data",
    "has_localization_estimates",
    "localization_log_entry",
    "localization_metrics",
    "plot_localization",
    "plot_localization_comparison",
    "plot_localization_per_agent",
    "plot_localization_truth_vs_estimate",
    "render_localization_plots",
    "wrap_angle",
]
