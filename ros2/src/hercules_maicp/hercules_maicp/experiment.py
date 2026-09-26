"""Small batch runner for the three ROS 2 AirSim MAICP case studies.

The runner owns the experiment protocol, while :mod:`core` owns the numerical
calibration update and :mod:`scores` owns the mission artifact contract.  A
live run uses a reset command before *every* mission and launches each mission
in its own process group.  Unit tests may opt into ``mode="in_process"`` and
provide a mission callback; the default remains the real ROS launch path.

Calibration and deployment are intentionally separate batches.  A calibration
batch is held at the current margin, the margin is updated once, and only then
is a fresh deployment batch launched.  Deployment seeds are deterministic and
shared by all methods for paired comparisons.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Literal

from .dynamics import DynamicsFileError, load_dynamics_models
from .scores import (
    MissionFailedError,
    MissionScore,
    MissionScoreError,
    infer_class,
    score_mission,
)


DEFAULT_AGENTS = ("Drone1", "Drone2", "SimpleFlight", "Husky1", "Husky2", "Husky3")
_DEFAULT_Y = {
    "Drone1": -9.0,
    "Drone2": 0.0,
    "SimpleFlight": 9.0,
    "Husky1": -9.0,
    "Husky2": 0.0,
    "Husky3": 9.0,
}
# The reset fixture has 9 m lane spacing.  A 10 m communication radius joins
# adjacent lanes and same-lane UAV/UGV pairs while leaving the outer lanes
# directly disconnected.
DEFAULT_GRAPH = {
    agent: tuple(
        peer for peer in DEFAULT_AGENTS
        if peer != agent and abs(_DEFAULT_Y[agent] - _DEFAULT_Y[peer]) <= 9.0 + 1e-9
    )
    for agent in DEFAULT_AGENTS
}
DEFAULT_METHODS = ("nominal", "fixed_margin", "centralized", "unshifted", "maicp")
LIVE_MODE = "ros"
IN_PROCESS_MODE = "in_process"


def _float(value: object, name: str, *, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{name} must be finite{ ' and nonnegative' if nonnegative else ''}")
    return result


def _canonical_method(value: str) -> str:
    name = str(value).strip().lower().replace("-", "_")
    aliases = {
        "fixed": "fixed_margin",
        "fixedmargin": "fixed_margin",
        "distributed": "maicp",
        "adaptive": "maicp",
        "oracle": "centralized",
        "centralised": "centralized",
    }
    return aliases.get(name, name)


def _canonical_case(value: str) -> str:
    name = str(value).strip().lower().replace("-", "_")
    aliases = {
        "cbf": "collision",
        "collision_avoidance": "collision",
        "target_tracking": "tracking",
        "drwt": "tracking",
        "cooperative_localization": "localization",
        "coop_localization": "localization",
    }
    return aliases.get(name, name)


def _format_argv(template: Sequence[str], values: Mapping[str, object]) -> list[str]:
    """Format an argv template without invoking a shell."""

    class StrictValues(dict[str, object]):
        def __missing__(self, key: str) -> object:
            raise ValueError(f"command template contains unknown placeholder {{{key}}}")

    try:
        return [str(argument).format_map(StrictValues(values)) for argument in template]
    except KeyError as exc:  # pragma: no cover - StrictValues normally handles this.
        raise ValueError(f"command template contains unknown placeholder {{{exc.args[0]}}}") from exc


@dataclass(frozen=True)
class MissionRequest:
    """One reset/launch/score operation."""

    case: str
    method: str
    repeat: int
    round_index: int
    phase: Literal["calibration", "deployment"]
    mission_index: int
    seed: int
    margin_ugv: float
    margin_uav: float
    steps: int
    log_path: Path
    reset_path: Path
    dynamics_file: Path | None
    host: str
    duration_sec: float = 0.0

    @property
    def round(self) -> int:
        return self.round_index

    def placeholders(self) -> dict[str, object]:
        dynamics = "" if self.dynamics_file is None else str(self.dynamics_file)
        return {
            "case": self.case,
            "method": self.method,
            "repeat": self.repeat,
            "round": self.round_index,
            "phase": self.phase,
            "mission_index": self.mission_index,
            "seed": self.seed,
            "truth_seed": self.seed,
            "margin_ugv": self.margin_ugv,
            "margin_uav": self.margin_uav,
            "steps": self.steps,
            "log_path": str(self.log_path),
            "reset_path": str(self.reset_path),
            "dynamics_file": dynamics,
            "host": self.host,
            "duration_sec": self.duration_sec,
        }


@dataclass(frozen=True)
class RunnerConfig:
    """Protocol and process settings.

    Defaults preserve the paper's calibration/deployment counts and the
    six-agent graph, while using a 100-sample (10 second) live horizon.
    """

    cases: tuple[str, ...] = ("tracking",)
    methods: tuple[str, ...] = DEFAULT_METHODS
    repeats: int = 3
    rounds: int = 5
    calibration_missions: int = 200
    deployment_missions: int = 200
    steps: int = 100
    control_dt: float = 0.1
    alpha: float = 0.10
    delta_cal: float = 0.99
    kappa: float = 0.6
    permitted_violations: int = 1
    pinball_iterations: int = 100
    threshold_iterations: int = 100
    truth_seed: int = 20260926
    output_dir: Path = Path("maicp_runs")
    host: str = "localhost"
    mode: str = LIVE_MODE
    mission_command: tuple[str, ...] = (
        "ros2",
        "launch",
        "hercules_maicp",
        "case_study.launch.py",
        "case:={case}",
        "steps:={steps}",
        "margin_ugv:={margin_ugv}",
        "margin_uav:={margin_uav}",
        "seed:={seed}",
        "dynamics_file:={dynamics_file}",
        "host:={host}",
        # case_study.launch.py exposes the ROS launch argument as ``duration``
        # (the wrapper forwards it to the mission's duration_sec parameter).
        "duration:={duration_sec}",
        "log_path:={log_path}",
    )
    reset_command: tuple[str, ...] | None = (
        "ros2",
        "run",
        "hercules_maicp",
        "reset_scene",
        "--seed",
        "{seed}",
        "--case",
        "{case}",
        "--host",
        "{host}",
        "--output",
        "{reset_path}",
    )
    update_graph: Mapping[str, Sequence[str]] = field(default_factory=lambda: dict(DEFAULT_GRAPH))
    dynamics_file: Path | None = None
    startup_timeout: float = 30.0
    mission_timeout: float = 30.0
    reset_timeout: float = 60.0
    process_terminate_timeout: float = 5.0
    poll_interval: float = 0.05
    require_done_sidecar: bool = True
    expected_steps: int | None = None
    score_horizon: int | None = None
    score_start_sec: float = 2.0
    # ``replicas`` is the terminology used in the paper; keep ``repeats`` as
    # the readable API while accepting either spelling at construction time.
    replicas: int | None = None

    def __post_init__(self) -> None:
        if self.replicas is not None:
            if not isinstance(self.replicas, int) or self.replicas < 1:
                raise ValueError("replicas must be a positive integer")
            object.__setattr__(self, "repeats", self.replicas)
        object.__setattr__(self, "replicas", int(self.repeats))
        cases = tuple(_canonical_case(value) for value in self.cases)
        methods = tuple(_canonical_method(value) for value in self.methods)
        if not cases or any(value not in {"collision", "tracking", "localization"} for value in cases):
            raise ValueError("cases must contain collision, tracking, or localization")
        allowed = set(DEFAULT_METHODS)
        if not methods or any(value not in allowed for value in methods):
            raise ValueError(f"methods must be drawn from {sorted(allowed)}")
        if len(set(cases)) != len(cases) or len(set(methods)) != len(methods):
            raise ValueError("cases and methods must not contain duplicates")
        for name in ("repeats", "rounds", "calibration_missions", "deployment_missions", "steps", "pinball_iterations", "threshold_iterations", "permitted_violations"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.repeats < 1 or self.rounds < 1 or self.calibration_missions < 1 or self.deployment_missions < 1:
            raise ValueError("repeats, rounds, and calibration/deployment counts must be positive")
        if self.steps < 2:
            raise ValueError("steps must be at least two")
        if self.permitted_violations < 0:
            raise ValueError("permitted_violations must be nonnegative")
        if not 0.0 < self.alpha < 1.0 or not 0.0 < self.delta_cal < 1.0:
            raise ValueError("alpha and delta_cal must lie in (0, 1)")
        _float(self.control_dt, "control_dt")
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if self.mode == LIVE_MODE and self.steps * self.control_dt > 10.0 + 1e-9:
            raise ValueError("live missions must be no longer than 10 seconds")
        _float(self.kappa, "kappa", nonnegative=True)
        for name in ("startup_timeout", "mission_timeout", "reset_timeout", "process_terminate_timeout", "poll_interval"):
            value = _float(getattr(self, name), name, nonnegative=True)
            if name != "poll_interval" and value <= 0.0:
                raise ValueError(f"{name} must be positive")
            if name == "poll_interval" and value <= 0.0:
                raise ValueError("poll_interval must be positive")
        if self.expected_steps is not None and (not isinstance(self.expected_steps, int) or self.expected_steps < 2):
            raise ValueError("expected_steps must be at least two")
        if self.score_horizon is not None and (not isinstance(self.score_horizon, int) or self.score_horizon < 2):
            raise ValueError("score_horizon must be at least two")
        if not math.isfinite(float(self.score_start_sec)) or self.score_start_sec < 0.0:
            raise ValueError("score_start_sec must be finite and nonnegative")
        if self.mode not in {LIVE_MODE, IN_PROCESS_MODE}:
            raise ValueError("mode must be 'ros' or 'in_process'")
        if self.mode == LIVE_MODE and self.reset_command is None:
            raise ValueError("a reset_command is required for live ROS runs")
        graph = {str(node): tuple(str(peer) for peer in peers) for node, peers in self.update_graph.items()}
        if not graph:
            raise ValueError("update_graph must not be empty")
        if set(graph) != {peer for peers in graph.values() for peer in peers} | set(graph):
            # This catches a common typo but permits isolated nodes to be
            # diagnosed by the core/ROS transport rather than here.
            unknown = {peer for peers in graph.values() for peer in peers} - set(graph)
            if unknown:
                raise ValueError(f"update_graph contains unknown peers: {sorted(unknown)}")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "dynamics_file", None if self.dynamics_file is None else Path(self.dynamics_file))
        object.__setattr__(self, "update_graph", graph)


ExperimentConfig = RunnerConfig


@dataclass(frozen=True)
class MissionOutcome:
    request: MissionRequest
    score: MissionScore | None
    failed: bool
    failure_reason: str | None = None
    mode: str = LIVE_MODE

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.request.case,
            "method": self.request.method,
            "repeat": self.request.repeat,
            "round": self.request.round_index,
            "phase": self.request.phase,
            "mission_index": self.request.mission_index,
            "seed": self.request.seed,
            "margin_ugv": self.request.margin_ugv,
            "margin_uav": self.request.margin_uav,
            "log_path": str(self.request.log_path),
            "score_path": str(self.request.log_path.with_suffix(".score.json")),
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "mode": self.mode,
            "score": None if self.score is None else self.score.as_dict(),
        }


def _load_core_config(settings: RunnerConfig) -> Any:
    """Build the numerical config without importing ROS."""

    from .config import paper_config

    kwargs = {
        "outer_replicates": settings.repeats,
        "rounds": settings.rounds,
        "calibration_missions": settings.calibration_missions,
        "deployment_missions": settings.deployment_missions,
        "simulation_steps": settings.steps,
        "alpha": settings.alpha,
        "delta_cal": settings.delta_cal,
        "K_S": settings.pinball_iterations,
        "K_q": settings.threshold_iterations,
    }
    return paper_config(**kwargs)


def _stable_seed(base: int, case: str, repeat: int, round_index: int, phase: str, index: int, method: str | None = None) -> int:
    """Derive deterministic independent 32-bit seeds without Python hash()."""

    method_text = "shared" if method is None else method
    token = f"{int(base)}|{case}|{repeat}|{round_index}|{phase}|{index}|{method_text}".encode()
    digest = hashlib.blake2b(token, digest_size=8, person=b"hercules-maicp").digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def _core_graph(settings: RunnerConfig) -> dict[str, set[str]]:
    return {str(node): set(str(peer) for peer in peers) for node, peers in settings.update_graph.items()}


class ExperimentRunner:
    """Execute and summarize the nested MAICP experiment protocol."""

    def __init__(
        self,
        config: RunnerConfig | None = None,
        *,
        order_statistic: Callable[..., object] | None = None,
        mission_executor: Callable[[MissionRequest], object] | None = None,
        reset_executor: Callable[[MissionRequest], object] | None = None,
        core_config: Any | None = None,
    ) -> None:
        self.config = RunnerConfig() if config is None else config
        self.order_statistic = order_statistic
        self.mission_executor = mission_executor
        self.reset_executor = reset_executor
        self.core_config = _load_core_config(self.config) if core_config is None else core_config
        self._ros_order_statistic: Callable[..., object] | None = None
        self._models: Mapping[str, Any] | None = None
        if "collision" in self.config.cases:
            if self.config.dynamics_file is None:
                raise DynamicsFileError("collision experiments require a frozen dynamics_file")
            # Verification happens at construction, before any calibration
            # mission can run.  No fitting or model mutation is performed.
            self._models = load_dynamics_models(self.config.dynamics_file)

    def _request(
        self,
        case: str,
        method: str,
        repeat: int,
        round_index: int,
        phase: Literal["calibration", "deployment"],
        mission_index: int,
        margins: Mapping[str, float],
    ) -> MissionRequest:
        # Deployment seeds intentionally omit method so every method receives
        # the same independent scene realization.  Calibration batches remain
        # separate per method and round.
        seed = _stable_seed(
            self.config.truth_seed,
            case,
            repeat,
            round_index,
            phase,
            mission_index,
            None if phase == "deployment" else method,
        )
        stem = f"{case}/{method}/repeat_{repeat:03d}/round_{round_index:03d}"
        filename = f"{phase}_{mission_index:04d}"
        log_path = self.config.output_dir / stem / f"{filename}.jsonl"
        reset_path = self.config.output_dir / stem / f"{filename}.reset.json"
        return MissionRequest(
            case=case,
            method=method,
            repeat=repeat,
            round_index=round_index,
            phase=phase,
            mission_index=mission_index,
            seed=seed,
            margin_ugv=_float(margins.get("ugv", 0.0), "margin_ugv", nonnegative=True),
            margin_uav=_float(margins.get("uav", 0.0), "margin_uav", nonnegative=True),
            steps=self.config.steps,
            log_path=log_path,
            reset_path=reset_path,
            dynamics_file=self.config.dynamics_file,
            host=self.config.host,
            duration_sec=self.config.steps * self.config.control_dt,
        )

    def build_command(self, request: MissionRequest) -> list[str]:
        """Return the launch argv for a request, never a shell string."""

        command = _format_argv(self.config.mission_command, request.placeholders())
        if request.dynamics_file is None:
            # ROS 2 launch treats an explicitly empty ``dynamics_file:=`` as
            # a malformed parameter in non-collision cases.  The default
            # template includes the token for collision runs, so remove only
            # the empty assignment after formatting; custom nonempty paths
            # remain untouched.
            command = [
                argument for argument in command
                if argument not in {"dynamics_file:=", "dynamics_file=", "--dynamics-file="}
            ]
        return command

    def build_reset_command(self, request: MissionRequest) -> list[str]:
        if self.config.reset_command is None:
            raise ValueError("reset command is required for live ROS runs")
        return _format_argv(self.config.reset_command, request.placeholders())

    def _cleanup_sidecars(self, path: Path) -> None:
        # Each request owns its output path.  Remove a prior artifact before
        # launch so a newly-created ``.done`` marker can never certify stale
        # JSONL from an interrupted retry.
        if path.exists():
            path.unlink()
        for suffix in (".done", ".failed"):
            candidate = Path(str(path) + suffix)
            if candidate.exists():
                candidate.unlink()

    def _call_reset(self, request: MissionRequest) -> None:
        request.reset_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.mode == IN_PROCESS_MODE:
            if self.reset_executor is not None:
                self.reset_executor(request)
            return
        if self.reset_executor is not None:
            self.reset_executor(request)
            return
        command = self.build_reset_command(request)
        try:
            process = subprocess.Popen(command, start_new_session=True)
        except OSError as exc:
            raise MissionFailedError(f"could not start reset command {command!r}: {exc}") from exc
        try:
            return_code = process.wait(timeout=self.config.reset_timeout)
        except subprocess.TimeoutExpired as exc:
            self._terminate_process(process)
            raise MissionFailedError(f"reset command timed out after {self.config.reset_timeout}s") from exc
        if return_code != 0:
            raise MissionFailedError(f"reset command exited with status {return_code}: {command!r}")

    def _terminate_process(self, process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        try:
            # ROS launch forwards SIGINT to its children itself.
            process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=self.config.process_terminate_timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=self.config.process_terminate_timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=self.config.process_terminate_timeout)

    def _run_live(self, request: MissionRequest) -> Path:
        request.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._cleanup_sidecars(request.log_path)
        command = self.build_command(request)
        stdout_path = Path(str(request.log_path) + ".stdout")
        try:
            stdout = stdout_path.open("w", encoding="utf-8")
        except OSError as exc:
            raise MissionFailedError(f"could not create launch output {stdout_path}: {exc}") from exc
        try:
            process = subprocess.Popen(
                command,
                cwd=str(Path.cwd()),
                stdout=stdout,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            stdout.close()
            raise MissionFailedError(f"could not start mission command {command!r}: {exc}") from exc
        started_at = time.monotonic()
        log_seen = False
        mission_deadline: float | None = None
        try:
            while True:
                failed_sidecar = Path(str(request.log_path) + ".failed")
                done_sidecar = Path(str(request.log_path) + ".done")
                if failed_sidecar.exists():
                    reason = failed_sidecar.read_text(encoding="utf-8", errors="replace").strip()
                    raise MissionFailedError(
                        f"mission failed ({request.case}/{request.method}/{request.phase}/{request.mission_index}): "
                        f"{reason or 'unspecified failure'}"
                    )
                if request.log_path.exists():
                    if not log_seen:
                        log_seen = True
                        mission_deadline = time.monotonic() + self.config.mission_timeout
                now = time.monotonic()
                if done_sidecar.exists():
                    if not request.log_path.exists():
                        raise MissionFailedError(f"mission marked done without log: {request.log_path}")
                    return request.log_path
                if process.poll() is not None:
                    raise MissionFailedError(
                        f"mission process exited {process.returncode} without completion sidecar: {request.log_path}"
                    )
                if not log_seen and now - started_at >= self.config.startup_timeout:
                    raise MissionFailedError(
                        f"mission did not create a log within {self.config.startup_timeout}s: {request.log_path}"
                    )
                if mission_deadline is not None and now >= mission_deadline:
                    raise MissionFailedError(
                        f"mission exceeded {self.config.mission_timeout}s after log startup: {request.log_path}"
                    )
                time.sleep(self.config.poll_interval)
        finally:
            # The runner owns this process group.  It is safe to stop launch
            # after the mission sidecar has been written and prevents orphaned
            # ROS nodes from contaminating the next reset.
            try:
                self._terminate_process(process)
            finally:
                stdout.close()

    def _run_in_process(self, request: MissionRequest) -> object:
        if self.mission_executor is None:
            raise MissionFailedError("mode='in_process' requires mission_executor")
        return self.mission_executor(request)

    def _score_executor_result(self, request: MissionRequest, result: object) -> MissionScore:
        if isinstance(result, MissionScore):
            return result
        if isinstance(result, (str, Path)):
            log_path = Path(result)
        elif isinstance(result, Mapping) and "score" in result and isinstance(result["score"], MissionScore):
            return result["score"]
        elif isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
            # Test callbacks may return already-loaded records.  Keep the
            # scoring contract identical to the live path.
            from .scores import score_records

            return score_records(
                list(result),
                request.case,
                models=self._models,
                expected_per_class={"uav": 3, "ugv": 3},
                horizon=self.config.score_horizon,
                permitted_violations=self.config.permitted_violations,
                start_time=(self.config.score_start_sec if request.case in {"tracking", "localization"} else 0.0),
            )
        else:
            log_path = request.log_path
        return score_mission(
            log_path,
            request.case,
            dynamics_file=request.dynamics_file,
            expected_steps=self.config.expected_steps or request.steps,
            require_done=self.config.require_done_sidecar,
            horizon=self.config.score_horizon,
            permitted_violations=self.config.permitted_violations,
            expected_per_class={"uav": 3, "ugv": 3},
            start_time=(self.config.score_start_sec if request.case in {"tracking", "localization"} else 0.0),
        )

    def _persist_outcome(self, outcome: MissionOutcome) -> None:
        """Write one auditable result immediately after its mission ends."""

        target = outcome.request.log_path.with_suffix(".score.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(
            json.dumps(_jsonable(outcome.as_dict()), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    def run_mission(self, request: MissionRequest) -> MissionOutcome:
        """Reset, execute, and score one mission; failures remain explicit."""

        try:
            self._call_reset(request)
            result = self._run_live(request) if self.config.mode == LIVE_MODE else self._run_in_process(request)
            score = self._score_executor_result(request, result)
            outcome = MissionOutcome(request=request, score=score, failed=False, mode=self.config.mode)
        except (MissionScoreError, DynamicsFileError, OSError, RuntimeError, ValueError) as exc:
            outcome = MissionOutcome(
                request=request,
                score=None,
                failed=True,
                failure_reason=str(exc),
                mode=self.config.mode,
            )
        try:
            self._persist_outcome(outcome)
        except OSError as exc:
            # Persistence is part of the mission contract.  A successful
            # score whose audit artifact cannot be written is reported as a
            # failure instead of being returned as an unrecorded result.
            outcome = MissionOutcome(
                request=request,
                score=None,
                failed=True,
                failure_reason=f"could not persist mission score: {exc}",
                mode=self.config.mode,
            )
        return outcome

    @staticmethod
    def _data_for_core(scores: Sequence[MissionScore]) -> list[dict[str, dict[str, float]]]:
        data: list[dict[str, dict[str, float]]] = []
        for score in scores:
            classes: dict[str, dict[str, float]] = {"uav": {}, "ugv": {}}
            for agent, value in score.robot_scores.items():
                class_name = score.diagnostics.get("agents", {}).get(agent, {}).get("class")
                if class_name is None:
                    class_name = infer_class(agent)
                classes[str(class_name)][agent] = float(value)
            if any(not values for values in classes.values()):
                raise MissionScoreError("calibration mission did not score both classes")
            data.append(classes)
        return data

    def _calibrate(
        self,
        method: str,
        current: Mapping[str, float],
        scores: Sequence[MissionScore],
    ) -> Any:
        if not scores:
            raise MissionScoreError("calibration requires at least one mission")
        from .core import calibrate_round

        canonical = _canonical_method(method)
        kwargs: dict[str, Any] = {
            "graph": _core_graph(self.config),
        }
        if canonical in {"maicp", "unshifted"}:
            callback = self._order_statistic_callback()
            kwargs["order_statistic"] = callback
            if self.config.mode == LIVE_MODE and self.order_statistic is None:
                kwargs["margin_update"] = callback.solve_update
        if canonical in {"maicp", "centralized"}:
            kwargs["kappa"] = {"uav": self.config.kappa, "ugv": self.config.kappa}
            # The configured kappa is an experiment setting, not an analytic
            # certificate.  Keep the core result explicitly non-certified.
            kwargs["sensitivity_certified"] = False
        elif canonical == "unshifted":
            kwargs["kappa"] = {"uav": 0.0, "ugv": 0.0}
            kwargs["sensitivity_certified"] = False
        data = self._data_for_core(scores)
        result = calibrate_round(canonical, dict(current), data, self.core_config, **kwargs)
        return result

    def _order_statistic_callback(self) -> Callable[..., object] | None:
        """Return the configured callback, starting ROS workers when needed."""

        if self.order_statistic is not None:
            return self.order_statistic
        if self.config.mode != LIVE_MODE:
            return None
        if self._ros_order_statistic is None:
            try:
                from .ros_transport import RosOrderStatistic

                self._ros_order_statistic = RosOrderStatistic(_core_graph(self.config))
            except Exception as exc:
                raise MissionFailedError(
                    f"could not start ROS distributed calibration workers: {exc}"
                ) from exc
        return self._ros_order_statistic

    def close(self) -> None:
        """Stop owned ROS calibration workers and release their node."""

        callback = self._ros_order_statistic
        self._ros_order_statistic = None
        if callback is not None:
            close = getattr(callback, "close", None)
            if callable(close):
                close()

    def _run_batch(
        self,
        case: str,
        method: str,
        repeat: int,
        round_index: int,
        phase: Literal["calibration", "deployment"],
        margins: Mapping[str, float],
        count: int,
        *,
        fail_fast: bool = False,
    ) -> tuple[list[MissionOutcome], list[MissionScore]]:
        outcomes: list[MissionOutcome] = []
        scores: list[MissionScore] = []
        for mission_index in range(count):
            request = self._request(case, method, repeat, round_index, phase, mission_index, margins)
            outcome = self.run_mission(request)
            outcomes.append(outcome)
            if outcome.failed or outcome.score is None:
                # A failed/truncated mission cannot become a finite calibration
                # observation or a zero deployment score.
                if fail_fast:
                    raise MissionFailedError(
                        f"{case}/{method}/repeat {repeat}/round {round_index}/"
                        f"{phase}/{mission_index} failed: {outcome.failure_reason or 'unknown failure'}"
                    )
                continue
            scores.append(outcome.score)
        return outcomes, scores

    @staticmethod
    def _deployment_metrics(
        outcomes: Sequence[MissionOutcome],
        scores: Sequence[MissionScore],
        margins: Mapping[str, float],
    ) -> dict[str, Any]:
        """Summarize fixed-horizon coverage and collision telemetry."""

        class_coverage: dict[str, dict[str, Any]] = {}
        for class_name in ("uav", "ugv"):
            values = [
                float(score.class_scores[class_name])
                for score in scores
                if class_name in score.class_scores
            ]
            successes = sum(value <= float(margins.get(class_name, 0.0)) for value in values)
            class_coverage[class_name] = {
                "margin": float(margins.get(class_name, 0.0)),
                "successes": successes,
                "total": len(values),
                "rate": None if not values else successes / len(values),
            }
        all_successes = 0
        all_total = 0
        for score in scores:
            if not all(class_name in score.class_scores for class_name in ("uav", "ugv")):
                continue
            all_total += 1
            if all(
                float(score.class_scores[class_name]) <= float(margins.get(class_name, 0.0))
                for class_name in ("uav", "ugv")
            ):
                all_successes += 1

        completed = [score.diagnostics.get("all_returned_home") for score in scores]
        known_completions = [value for value in completed if isinstance(value, bool)]
        controller_failures = [bool(score.diagnostics["fail_safe"])
                               for score in scores if score.case == "collision"
                               and not score.diagnostics.get("fail_safe_unknown", True)]
        collision_values = [score.collision_free for score in scores]
        known = [value for value in collision_values if value is not None]
        collision_successes = sum(bool(value) for value in known)
        collision_failures = len(known) - collision_successes
        collision_unknown = len(collision_values) - len(known)
        return {
            "completion_rate": (sum(known_completions) / len(known_completions)
                                if known_completions else None),
            "completion_known": len(known_completions),
            "controller_failure_rate": (sum(controller_failures) / len(controller_failures)
                                        if controller_failures else None),
            "class_coverage": class_coverage,
            "all_class_coverage": {
                "successes": all_successes,
                "total": all_total,
                "rate": None if all_total == 0 else all_successes / all_total,
            },
            "collision_free_rate": (
                None if not known else collision_successes / len(known)
            ),
            "collision_free_known": len(known),
            "collision_free_successes": collision_successes,
            "collision_free_failures": collision_failures,
            "collision_free_unknown": collision_unknown,
            "mission_failures": sum(outcome.failed for outcome in outcomes),
        }

    def run(self) -> dict[str, Any]:
        """Run every requested case/method/repeat/round and return a report."""

        report: dict[str, Any] = {
            "schema_version": 1,
            "mode": self.config.mode,
            "defaults": {
                "control_dt": self.config.control_dt,
                "steps": self.config.steps,
                "calibration_missions": self.config.calibration_missions,
                "deployment_missions": self.config.deployment_missions,
                "repeats": self.config.repeats,
                "replicas": self.config.repeats,
                "rounds": self.config.rounds,
                "alpha": self.config.alpha,
                "delta_cal": self.config.delta_cal,
                "kappa": self.config.kappa,
                "margin_bounds": {
                    name: [cls.margin_min, cls.margin_max]
                    for name, cls in self.core_config.classes.items()
                },
                "permitted_violations": self.config.permitted_violations,
                "score_start_sec": self.config.score_start_sec,
                "duration_sec": self.config.steps * self.config.control_dt,
                "update_graph": {node: list(peers) for node, peers in self.config.update_graph.items()},
                "methods": list(self.config.methods),
            },
            "limitations": [
                "scores are finite sampled-horizon diagnostics, not continuous-time certificates",
                "configured kappa is used as supplied and is not estimated by the runner",
                "failed or incomplete missions are retained and excluded from calibration",
                "collision-free telemetry is reported separately from conformal score coverage",
                "tracking and localization score from the fixed 2 second warm-up boundary",
            ],
            "cases": {},
        }
        try:
            for case in self.config.cases:
                case_report: dict[str, Any] = {}
                for method in self.config.methods:
                    method_report: list[dict[str, Any]] = []
                    for repeat in range(self.config.repeats):
                        if method in {"nominal", "fixed_margin"}:
                            margins = {"uav": 0.0, "ugv": 0.0}
                        else:
                            configured = getattr(self.core_config, "initial_margins", None)
                            if isinstance(configured, Mapping):
                                margins = {
                                    "uav": float(configured.get("uav", 0.2)),
                                    "ugv": float(configured.get("ugv", 0.2)),
                                }
                            else:
                                margins = {"uav": 0.2, "ugv": 0.2}
                        fixed_margins: dict[str, float] | None = None
                        rounds: list[dict[str, Any]] = []
                        for round_index in range(self.config.rounds):
                            calibration_outcomes: list[MissionOutcome] = []
                            calibration_scores: list[MissionScore] = []
                            calibration_result: Any | None = None
                            needs_calibration = method not in {"nominal"} and (
                                method != "fixed_margin" or fixed_margins is None
                            )
                            if needs_calibration:
                                calibration_outcomes, calibration_scores = self._run_batch(
                                    case,
                                    method,
                                    repeat,
                                    round_index,
                                    "calibration",
                                    margins,
                                    self.config.calibration_missions,
                                    fail_fast=True,
                                )
                                if len(calibration_scores) != self.config.calibration_missions:
                                    raise MissionFailedError(
                                        f"{case}/{method}/repeat {repeat}/round {round_index}: "
                                        f"only {len(calibration_scores)}/{self.config.calibration_missions} "
                                        "calibration missions were valid"
                                    )
                                calibration_result = self._calibrate(method, margins, calibration_scores)
                                candidate = getattr(calibration_result, "margins", None)
                                if not isinstance(candidate, Mapping):
                                    raise MissionScoreError("core calibration returned no margin mapping")
                                margins = {str(key): float(value) for key, value in candidate.items()}
                                if method == "fixed_margin":
                                    fixed_margins = dict(margins)
                            elif method == "fixed_margin" and fixed_margins is not None:
                                margins = dict(fixed_margins)
                            deployment_outcomes, deployment_scores = self._run_batch(
                                case,
                                method,
                                repeat,
                                round_index,
                                "deployment",
                                margins,
                                self.config.deployment_missions,
                            )
                            deployment_valid = len(deployment_scores)
                            deployment_metrics = self._deployment_metrics(
                                deployment_outcomes, deployment_scores, margins
                            )
                            rounds.append(
                                {
                                    "round": round_index,
                                    "margins": dict(margins),
                                    "calibration": {
                                        "requested": self.config.calibration_missions if needs_calibration else 0,
                                        "valid": len(calibration_scores),
                                        "failed": sum(outcome.failed for outcome in calibration_outcomes),
                                        "result": None if calibration_result is None else _jsonable(calibration_result),
                                        "missions": [outcome.as_dict() for outcome in calibration_outcomes],
                                    },
                                    "deployment": {
                                        "requested": self.config.deployment_missions,
                                        "valid": deployment_valid,
                                        "failed": sum(outcome.failed for outcome in deployment_outcomes),
                                        "completion_rate": deployment_metrics["completion_rate"],
                                        "completion_known": deployment_metrics["completion_known"],
                                        "controller_failure_rate": deployment_metrics["controller_failure_rate"],
                                        "class_coverage": deployment_metrics["class_coverage"],
                                        "all_class_coverage": deployment_metrics["all_class_coverage"],
                                        "collision_free_rate": deployment_metrics["collision_free_rate"],
                                        "collision_free_known": deployment_metrics["collision_free_known"],
                                        "collision_free_failures": deployment_metrics["collision_free_failures"],
                                        "collision_free_unknown": deployment_metrics["collision_free_unknown"],
                                        "collision_free": [
                                            outcome.score.collision_free
                                            for outcome in deployment_outcomes
                                            if outcome.score is not None
                                        ],
                                        "missions": [outcome.as_dict() for outcome in deployment_outcomes],
                                    },
                                }
                            )
                        method_report.append({"repeat": repeat, "rounds": rounds})
                    case_report[method] = method_report
                report["cases"][case] = case_report
            return report
        finally:
            self.close()


def _jsonable(value: object) -> object:
    """Convert dataclasses/numpy-like values into JSON-safe diagnostics."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return str(value)


def run_experiment(config: RunnerConfig | None = None, **kwargs: Any) -> dict[str, Any]:
    return ExperimentRunner(config, **kwargs).run()


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ROS 2 AirSim MAICP case studies")
    parser.add_argument("--cases", default=",".join(RunnerConfig().cases))
    parser.add_argument("--methods", default=",".join(RunnerConfig().methods))
    parser.add_argument("--repeats", "--replicas", dest="repeats", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--calibration-missions", type=int, default=200)
    parser.add_argument("--deployment-missions", type=int, default=200)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--kappa", type=float, default=0.6)
    parser.add_argument("--delta-cal", type=float, default=0.99)
    parser.add_argument("--truth-seed", type=int, default=RunnerConfig().truth_seed)
    parser.add_argument("--output-dir", type=Path, default=RunnerConfig().output_dir)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--mode", choices=(LIVE_MODE, IN_PROCESS_MODE), default=LIVE_MODE)
    parser.add_argument("--dynamics-file", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    config = RunnerConfig(
        cases=_parse_csv(args.cases),
        methods=_parse_csv(args.methods),
        repeats=args.repeats,
        rounds=args.rounds,
        calibration_missions=args.calibration_missions,
        deployment_missions=args.deployment_missions,
        steps=args.steps,
        kappa=args.kappa,
        delta_cal=args.delta_cal,
        truth_seed=args.truth_seed,
        output_dir=args.output_dir,
        host=args.host,
        mode=args.mode,
        dynamics_file=args.dynamics_file,
    )
    report = ExperimentRunner(config).run()
    if args.report is None:
        print(json.dumps(_jsonable(report), indent=2))
    else:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(_jsonable(report), indent=2) + "\n", encoding="utf-8")
    return 0


__all__ = [
    "DEFAULT_AGENTS",
    "DEFAULT_GRAPH",
    "DEFAULT_METHODS",
    "ExperimentConfig",
    "ExperimentRunner",
    "IN_PROCESS_MODE",
    "LIVE_MODE",
    "MissionOutcome",
    "MissionRequest",
    "RunnerConfig",
    "run_experiment",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised through ros2 run.
    raise SystemExit(main())
