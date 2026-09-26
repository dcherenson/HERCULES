"""Frozen affine acceleration-residual models for collision scoring.

The collision score in :mod:`scores` evaluates

``||dv/dt - u_previous - d_hat(position_previous)||``.

``d_hat`` is fit once from independent training logs and then serialized. A
runner must load and validate the serialized file before it starts calibration;
this module intentionally contains no online fitting path.  The command line
entry point is useful for creating the file explicitly::

    python -m hercules_maicp.dynamics fit --output dynamics.json train/*.jsonl
    python -m hercules_maicp.dynamics verify dynamics.json

The coefficient order is fixed and shared with the ROS launch parameters:
``[bias_x, xx, xy, bias_y, yx, yy]``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - the ROS image includes NumPy.
    np = None  # type: ignore[assignment]


COEFFICIENT_ORDER = ("bias_x", "xx", "xy", "bias_y", "yx", "yy")
SCHEMA_VERSION = 1
SCHEMA = "hercules_maicp_affine_acceleration_residual"


class DynamicsFileError(ValueError):
    """A frozen dynamics file is absent, malformed, or not auditable."""


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DynamicsFileError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise DynamicsFileError(f"{name} must be finite")
    return result


def _planar(value: object, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise DynamicsFileError(f"{name} must be a planar vector")
    try:
        values = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise DynamicsFileError(f"{name} must be a planar vector") from exc
    if len(values) < 2 or any(not math.isfinite(item) for item in values[:2]):
        raise DynamicsFileError(f"{name} must contain finite x/y values")
    return float(values[0]), float(values[1])


def _full_vector(value: object, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise DynamicsFileError(f"{name} must be a finite state vector")
    try:
        values = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise DynamicsFileError(f"{name} must be a finite state vector") from exc
    if len(values) < 2 or any(not math.isfinite(item) for item in values):
        raise DynamicsFileError(f"{name} must be a finite state vector")
    return values


def normalize_class(value: object) -> str:
    text = str(getattr(value, "value", value)).strip().lower()
    if text in {"uav", "drone", "multirotor", "quadrotor", "air"}:
        return "uav"
    if text in {"ugv", "ground", "car", "husky", "vehicle"}:
        return "ugv"
    raise DynamicsFileError(f"unknown dynamics class {value!r}")


@dataclass(frozen=True)
class AffineDynamicsModel:
    """One frozen class-specific affine residual model."""

    platform: str
    coefficients: tuple[float, ...]
    sample_count: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    fit_rank: int | None = None
    train_mse: float | None = None

    def __post_init__(self) -> None:
        platform = normalize_class(self.platform)
        coefficients = tuple(float(value) for value in self.coefficients)
        if len(coefficients) != 6 or any(not math.isfinite(value) for value in coefficients):
            raise DynamicsFileError(
                f"{platform} coefficients must be six finite values in {COEFFICIENT_ORDER} order"
            )
        if self.sample_count is not None and (not isinstance(self.sample_count, int) or self.sample_count < 1):
            raise DynamicsFileError("sample_count must be a positive integer when present")
        if self.fit_rank is not None and (not isinstance(self.fit_rank, int) or self.fit_rank < 1):
            raise DynamicsFileError("fit_rank must be a positive integer when present")
        if self.train_mse is not None and (not math.isfinite(float(self.train_mse)) or float(self.train_mse) < 0.0):
            raise DynamicsFileError("train_mse must be finite and nonnegative when present")
        object.__setattr__(self, "platform", platform)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def coeffs(self) -> tuple[float, ...]:
        return self.coefficients

    def evaluate(self, position: Sequence[float]) -> tuple[float, float]:
        x, y = _planar(position, f"{self.platform} model position")
        b_x, xx, xy, b_y, yx, yy = self.coefficients
        return (b_x + xx * x + xy * y, b_y + yx * x + yy * y)

    __call__ = evaluate

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "platform": self.platform,
            "coefficients": list(self.coefficients),
            "coefficient_order": list(COEFFICIENT_ORDER),
            "provenance": dict(self.provenance),
        }
        if self.sample_count is not None:
            value["sample_count"] = self.sample_count
        if self.fit_rank is not None:
            value["fit_rank"] = self.fit_rank
        if self.train_mse is not None:
            value["train_mse"] = self.train_mse
        return value


FrozenAffineDynamics = AffineDynamicsModel
AffineModel = AffineDynamicsModel


def _agent_class(agent: str, row: Mapping[str, Any]) -> str:
    vehicle_types = row.get("vehicle_types")
    if isinstance(vehicle_types, Mapping) and agent in vehicle_types:
        return normalize_class(vehicle_types[agent])
    text = str(agent).lower()
    if text.startswith(("drone", "uav", "simpleflight", "multirotor")):
        return "uav"
    if text.startswith(("husky", "ugv", "car", "ground")):
        return "ugv"
    raise DynamicsFileError(f"cannot infer class for agent {agent!r}")


def _row_timestamp(row: Mapping[str, Any], state: Mapping[str, Any], row_index: int) -> float:
    if state.get("source_timestamp") is not None:
        return _finite(state["source_timestamp"], f"state timestamp at row {row_index}")
    if row.get("timestamp") is not None:
        return _finite(row["timestamp"], f"row timestamp at row {row_index}")
    raise DynamicsFileError(f"state at row {row_index} has no source_timestamp")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DynamicsFileError(f"training log is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DynamicsFileError(f"invalid training JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise DynamicsFileError(f"training row at {path}:{line_number} is not an object")
            rows.append(row)
    if len(rows) < 2:
        raise DynamicsFileError(f"training log has fewer than two rows: {path}")
    return rows


def _collect_training_samples(paths: Sequence[Path]) -> tuple[dict[str, list[tuple[tuple[float, float, float], tuple[float, float]]]], dict[str, int]]:
    """Collect one residual sample per distinct source-time interval.

    Repeated simulator packets describe one physical state, rather than a
    zero-length acceleration interval.  The command is held over each JSONL
    row, so a transition spanning repeated packets subtracts the average of
    all intervening commands.  This preserves the trace while avoiding a
    fabricated infinite acceleration at the duplicate timestamp.
    """

    samples: dict[str, list[tuple[tuple[float, float, float], tuple[float, float]]]] = {"ugv": [], "uav": []}
    skipped: dict[str, int] = {"missing_or_invalid": 0, "duplicate_source_samples": 0}
    for path in paths:
        rows = _load_rows(path)
        first_states = rows[0].get("states")
        first_controls = rows[0].get("safe_controls")
        if not isinstance(first_states, Mapping) or not first_states:
            raise DynamicsFileError(f"training log has no states: {path}")
        if not isinstance(first_controls, Mapping) or not first_controls:
            raise DynamicsFileError(f"training log has no safe_controls: {path}")
        # Target truth is commonly logged in ``states`` but has no control;
        # only controlled agents contribute acceleration residuals.
        agents = tuple(
            str(agent) for agent in first_states
            if str(agent) in first_controls
        )
        if not agents:
            raise DynamicsFileError(f"training log has no controlled agents: {path}")

        for agent in agents:
            groups: list[dict[str, Any]] = []
            controls_by_row: dict[int, tuple[float, float]] = {}
            for row_index, row in enumerate(rows):
                states = row.get("states")
                controls = row.get("safe_controls")
                if not isinstance(states, Mapping) or not isinstance(controls, Mapping):
                    raise DynamicsFileError(
                        f"training row {row_index} in {path} lacks states/safe_controls"
                    )
                state = states.get(agent)
                if not isinstance(state, Mapping) or agent not in controls:
                    raise DynamicsFileError(
                        f"training row {row_index} in {path} is missing {agent} state/control"
                    )
                position_values = _full_vector(
                    state.get("position"), f"training {path} row {row_index} {agent} position"
                )
                velocity_values = _full_vector(
                    state.get("velocity"), f"training {path} row {row_index} {agent} velocity"
                )
                position = (position_values[0], position_values[1])
                velocity = (velocity_values[0], velocity_values[1])
                timestamp = _row_timestamp(row, state, row_index)
                command = _planar(
                    controls[agent], f"training {path} row {row_index} {agent} safe_control"
                )
                controls_by_row[row_index] = command
                if not groups:
                    groups.append({
                        "position": position,
                        "velocity": velocity,
                        "full_position": position_values,
                        "full_velocity": velocity_values,
                        "timestamp": timestamp,
                        "first_index": row_index,
                        "last_index": row_index,
                    })
                    continue
                previous = groups[-1]
                previous_timestamp = float(previous["timestamp"])
                if timestamp < previous_timestamp:
                    raise DynamicsFileError(
                        f"training timestamps for {agent} move backwards at row {row_index}"
                    )
                if timestamp == previous_timestamp:
                    if (
                        position_values != previous["full_position"]
                        or velocity_values != previous["full_velocity"]
                    ):
                        raise DynamicsFileError(
                            f"training {path} has an inconsistent duplicate source timestamp "
                            f"for {agent} at row {row_index}"
                        )
                    previous["last_index"] = row_index
                    skipped["duplicate_source_samples"] += 1
                    continue
                dt = timestamp - previous_timestamp
                if not math.isfinite(dt) or dt <= 1e-12:
                    raise DynamicsFileError(
                        f"training timestamps for {agent} are not strictly increasing at row {row_index}"
                    )
                groups.append({
                    "position": position,
                    "velocity": velocity,
                    "full_position": position_values,
                    "full_velocity": velocity_values,
                    "timestamp": timestamp,
                    "first_index": row_index,
                    "last_index": row_index,
                })

            if len(groups) < 2:
                continue
            for previous, current in zip(groups, groups[1:]):
                dt = float(current["timestamp"]) - float(previous["timestamp"])
                if not math.isfinite(dt) or dt <= 1e-12:
                    raise DynamicsFileError(
                        f"training timestamps for {agent} are not strictly increasing"
                    )
                # One command per logged row.  A duplicate state group starts
                # at ``first_index`` and the next distinct state starts at
                # ``current.first_index``; averaging this interval preserves
                # every held command in the trace.
                command_indices = range(
                    int(previous["first_index"]), int(current["first_index"])
                )
                commands_for_interval = [controls_by_row[index] for index in command_indices]
                if not commands_for_interval:
                    raise DynamicsFileError(
                        f"training interval for {agent} has no safe control"
                    )
                command = tuple(
                    sum(value[coordinate] for value in commands_for_interval)
                    / len(commands_for_interval)
                    for coordinate in range(2)
                )
                prior_velocity = previous["velocity"]
                velocity = current["velocity"]
                residual = tuple(
                    (velocity[coordinate] - prior_velocity[coordinate]) / dt - command[coordinate]
                    for coordinate in range(2)
                )
                class_name = _agent_class(agent, rows[int(previous["first_index"])])
                prior_position = previous["position"]
                samples[class_name].append(((1.0, prior_position[0], prior_position[1]), residual))
    if not any(samples.values()):
        raise DynamicsFileError("training logs contain no valid acceleration residual samples")
    return samples, skipped


def _fit_one(platform: str, samples: Sequence[tuple[tuple[float, float, float], tuple[float, float]]], provenance: Mapping[str, Any]) -> AffineDynamicsModel:
    if len(samples) < 3:
        raise DynamicsFileError(f"{platform} requires at least three training samples")
    if np is not None:
        design = np.asarray([row[0] for row in samples], dtype=float)
        targets = np.asarray([row[1] for row in samples], dtype=float)
        rank = int(np.linalg.matrix_rank(design))
        if rank < 3:
            raise DynamicsFileError(f"{platform} training positions do not identify an affine model (rank {rank})")
        solution, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
        # Columns are x/y output; serialize x coefficients followed by y.
        coefficients = (
            float(solution[0, 0]), float(solution[1, 0]), float(solution[2, 0]),
            float(solution[0, 1]), float(solution[1, 1]), float(solution[2, 1]),
        )
        predictions = design @ solution
        mse = float(np.mean((predictions - targets) ** 2))
    else:  # pragma: no cover - exercised in minimal Python environments.
        # Solve the two independent normal equations with a tiny pure-Python
        # Gaussian eliminator.  Keeping fitting NumPy-free makes the offline
        # CLI usable before the ROS image is installed.
        normal = [[0.0 for _ in range(4)] for _ in range(3)]
        for features, target in samples:
            for row in range(3):
                for col in range(3):
                    normal[row][col] += features[row] * features[col]
                normal[row][3] += features[row] * target[0]
        normal_y = [[normal[row][col] for col in range(4)] for row in range(3)]
        for row in range(3):
            normal_y[row][3] = 0.0
        for features, target in samples:
            for row in range(3):
                normal_y[row][3] += features[row] * target[1]

        def solve(matrix: list[list[float]]) -> tuple[float, float, float]:
            matrix = [list(row) for row in matrix]
            for pivot in range(3):
                selected = max(range(pivot, 3), key=lambda index: abs(matrix[index][pivot]))
                if abs(matrix[selected][pivot]) <= 1e-12:
                    raise DynamicsFileError(f"{platform} training positions do not identify an affine model")
                matrix[pivot], matrix[selected] = matrix[selected], matrix[pivot]
                scale = matrix[pivot][pivot]
                matrix[pivot] = [value / scale for value in matrix[pivot]]
                for row in range(3):
                    if row == pivot:
                        continue
                    factor = matrix[row][pivot]
                    matrix[row] = [
                        matrix[row][col] - factor * matrix[pivot][col]
                        for col in range(4)
                    ]
            return tuple(matrix[row][3] for row in range(3))  # type: ignore[return-value]

        x_solution = solve(normal)
        y_solution = solve(normal_y)
        coefficients = (
            x_solution[0], x_solution[1], x_solution[2],
            y_solution[0], y_solution[1], y_solution[2],
        )
        errors = []
        for features, target in samples:
            predicted = (
                sum(features[col] * x_solution[col] for col in range(3)),
                sum(features[col] * y_solution[col] for col in range(3)),
            )
            errors.extend((predicted[0] - target[0], predicted[1] - target[1]))
        mse = sum(error * error for error in errors) / len(errors)
        rank = 3
    return AffineDynamicsModel(
        platform=platform,
        coefficients=coefficients,
        sample_count=len(samples),
        fit_rank=rank,
        train_mse=mse,
        provenance=provenance,
    )


def fit_affine_models(
    training_logs: Sequence[str | Path] | Mapping[str, Sequence[str | Path]],
    output: str | Path | None = None,
    *,
    provenance: Mapping[str, Any] | None = None,
    require_both_classes: bool = True,
) -> dict[str, AffineDynamicsModel]:
    """Fit frozen UGV/UAV affine residuals from separate JSONL logs.

    ``training_logs`` is either a sequence of paths (classes are inferred from
    each row's ``vehicle_types``) or a class-to-path mapping.  Paths are
    de-duplicated and stored in provenance, so accidentally passing a
    calibration/deployment log twice is visible in the artifact.  The fit is
    entirely offline; callers should invoke it before creating a runner.
    """

    if isinstance(training_logs, Mapping):
        flattened: list[Path] = []
        for values in training_logs.values():
            flattened.extend(Path(value).expanduser().resolve() for value in values)
        paths = flattened
    else:
        paths = [Path(value).expanduser().resolve() for value in training_logs]
    unique_paths = list(dict.fromkeys(paths))
    if not unique_paths:
        raise DynamicsFileError("at least one separate training log is required")
    sample_sets, skipped = _collect_training_samples(unique_paths)
    metadata: dict[str, Any] = {
        "fit_source": "separate_training_logs",
        "training_logs": [str(path) for path in unique_paths],
        "coefficient_order": list(COEFFICIENT_ORDER),
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "skipped_samples": skipped,
        "duplicate_source_samples": int(skipped.get("duplicate_source_samples", 0)),
    }
    if provenance:
        metadata.update(dict(provenance))
    models: dict[str, AffineDynamicsModel] = {}
    for platform in ("ugv", "uav"):
        if not sample_sets[platform]:
            if require_both_classes:
                raise DynamicsFileError(f"training logs contain no {platform} residual samples")
            continue
        models[platform] = _fit_one(platform, sample_sets[platform], metadata)
    if output is not None:
        save_dynamics_models(output, models, provenance=metadata)
    return models


fit_affine_dynamics = fit_affine_models
fit_dynamics = fit_affine_models


def _model_entry(value: object, platform: str) -> AffineDynamicsModel:
    if isinstance(value, AffineDynamicsModel):
        if value.platform != platform:
            raise DynamicsFileError(f"model platform mismatch: expected {platform}, got {value.platform}")
        return value
    if isinstance(value, Mapping):
        coefficients = value.get("coefficients", value.get("coeffs"))
        if coefficients is None:
            raise DynamicsFileError(f"{platform} model has no coefficients")
        return AffineDynamicsModel(
            platform=str(value.get("platform", platform)),
            coefficients=tuple(coefficients),
            sample_count=value.get("sample_count"),
            fit_rank=value.get("fit_rank"),
            train_mse=value.get("train_mse"),
            provenance=value.get("provenance", {}),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return AffineDynamicsModel(platform=platform, coefficients=tuple(value))
    raise DynamicsFileError(f"invalid {platform} dynamics model")


def save_dynamics_models(
    path: str | Path,
    models: Mapping[str, AffineDynamicsModel | Mapping[str, Any] | Sequence[float]],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write the canonical frozen model JSON artifact."""

    normalized: dict[str, AffineDynamicsModel] = {}
    for raw_platform, value in models.items():
        platform = normalize_class(raw_platform)
        normalized[platform] = _model_entry(value, platform)
    if not normalized:
        raise DynamicsFileError("at least one dynamics model is required")
    metadata = dict(provenance or {})
    metadata.setdefault("fit_source", "separate_training_logs")
    metadata.setdefault("coefficient_order", list(COEFFICIENT_ORDER))
    payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "coefficient_order": list(COEFFICIENT_ORDER),
        "provenance": metadata,
        "training_logs": list(metadata.get("training_logs", [])),
        "classes": {platform: model.as_dict() for platform, model in sorted(normalized.items())},
        # ``models`` is a readable compatibility alias for integrations that
        # use that term; both entries intentionally contain the same values.
        "models": {platform: model.as_dict() for platform, model in sorted(normalized.items())},
    }
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def load_dynamics_models(
    path: str | Path,
    *,
    required_classes: Iterable[str] = ("ugv", "uav"),
    require_provenance: bool = True,
) -> dict[str, AffineDynamicsModel]:
    """Load and fail-closed validate an already-fitted dynamics artifact."""

    target = Path(path).expanduser()
    if not target.is_file():
        raise DynamicsFileError(f"frozen dynamics file is missing: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DynamicsFileError(f"cannot read dynamics file {target}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DynamicsFileError("dynamics artifact must be a JSON object")
    try:
        schema_version = int(payload.get("schema_version", -1))
    except (TypeError, ValueError) as exc:
        raise DynamicsFileError("dynamics artifact has an invalid schema_version") from exc
    if payload.get("schema") != SCHEMA or schema_version != SCHEMA_VERSION:
        raise DynamicsFileError("unsupported frozen dynamics schema")
    try:
        order = tuple(payload.get("coefficient_order", ()))
    except TypeError as exc:
        raise DynamicsFileError("dynamics artifact has an invalid coefficient_order") from exc
    if order != COEFFICIENT_ORDER:
        raise DynamicsFileError(f"coefficient_order must be {list(COEFFICIENT_ORDER)}")
    provenance = payload.get("provenance")
    if require_provenance:
        if not isinstance(provenance, Mapping) or provenance.get("fit_source") != "separate_training_logs":
            raise DynamicsFileError("frozen dynamics provenance must identify separate training logs")
        training_logs = provenance.get("training_logs", payload.get("training_logs"))
        if (
            isinstance(training_logs, (str, bytes))
            or not isinstance(training_logs, Sequence)
            or not training_logs
        ):
            raise DynamicsFileError("frozen dynamics provenance has no training_logs")
    entries = payload.get("classes", payload.get("models"))
    if not isinstance(entries, Mapping):
        raise DynamicsFileError("dynamics artifact has no classes/models map")
    models: dict[str, AffineDynamicsModel] = {}
    for raw_platform, value in entries.items():
        try:
            platform = normalize_class(raw_platform)
        except DynamicsFileError:
            continue
        models[platform] = _model_entry(value, platform)
    required = tuple(normalize_class(value) for value in required_classes)
    missing = [platform for platform in required if platform not in models]
    if missing:
        raise DynamicsFileError(f"frozen dynamics artifact is missing classes: {missing}")
    return models


load_dynamics = load_dynamics_models
verify_dynamics_file = load_dynamics_models


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Fit or verify frozen MAICP affine dynamics")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit", help="fit from independent training JSONL logs")
    fit.add_argument("--output", required=True, type=Path)
    fit.add_argument("logs", nargs="+", type=Path)
    fit.add_argument("--provenance", type=Path, help="optional JSON object merged into provenance")
    verify = subparsers.add_parser("verify", help="validate an existing frozen artifact")
    verify.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "fit":
        provenance = None
        if args.provenance is not None:
            value = json.loads(args.provenance.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                raise SystemExit("--provenance must contain a JSON object")
            provenance = value
        models = fit_affine_models(args.logs, args.output, provenance=provenance)
        print(json.dumps({platform: model.as_dict() for platform, model in models.items()}, indent=2))
        return 0
    models = load_dynamics_models(args.path)
    print(json.dumps({platform: model.as_dict() for platform, model in models.items()}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point used by ``ros2 run ... fit_dynamics``."""

    # argparse reads sys.argv when ``argv`` is omitted.  Keeping this tiny
    # wrapper public makes the ament entry point and ``python -m`` identical.
    if argv is None:
        return _cli()
    import sys

    previous = sys.argv
    try:
        sys.argv = [previous[0], *argv]
        return _cli()
    finally:
        sys.argv = previous


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests.
    raise SystemExit(main())


__all__ = [
    "AffineDynamicsModel",
    "AffineModel",
    "COEFFICIENT_ORDER",
    "DynamicsFileError",
    "FrozenAffineDynamics",
    "SCHEMA",
    "SCHEMA_VERSION",
    "fit_affine_dynamics",
    "fit_affine_models",
    "fit_dynamics",
    "load_dynamics",
    "load_dynamics_models",
    "main",
    "save_dynamics_models",
    "verify_dynamics_file",
]
