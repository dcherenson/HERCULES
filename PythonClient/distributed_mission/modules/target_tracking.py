"""Distributed rolling-window target tracking.

This module implements the linear-Gaussian Distributed Rolling Window
Tracking (DRWT) estimator from the target-tracking reference paper. The
orchestrator supplies the communication graph; each tracker keeps its own
measurements, information matrix, dual variable, and estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


# This remains deliberately separate from the conformal robustness placeholder.
# Edit this constant while tuning the target obstacle footprint.
TARGET_CBF_SIGMA_MULTIPLIER = 2.0

STATE_DIM = 4


def constant_velocity_transition(dt: float) -> np.ndarray:
    """Return the planar constant-velocity transition matrix."""

    delta = max(0.0, float(dt))
    return np.array([
        [1.0, 0.0, delta, 0.0],
        [0.0, 1.0, 0.0, delta],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def constant_acceleration_process_noise(dt: float, spectral_density: float = 1.0) -> np.ndarray:
    """Return the standard white-acceleration process covariance."""

    delta = max(0.0, float(dt))
    q = max(0.0, float(spectral_density))
    return q * np.array([
        [delta ** 3 / 3.0, 0.0, delta ** 2 / 2.0, 0.0],
        [0.0, delta ** 3 / 3.0, 0.0, delta ** 2 / 2.0],
        [delta ** 2 / 2.0, 0.0, delta, 0.0],
        [0.0, delta ** 2 / 2.0, 0.0, delta],
    ], dtype=float)


def position_measurement_matrix() -> np.ndarray:
    """Return the linear position measurement matrix ``C``."""

    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=float)


def _positive_definite(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    values = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    eigenvalues = np.maximum(eigenvalues, float(floor))
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
    values = _positive_definite(matrix)
    try:
        return np.linalg.inv(values)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(values)


def solve_block_tridiagonal(system: np.ndarray, vector: np.ndarray, block_dim: int = STATE_DIM) -> np.ndarray:
    """Solve an SPD block-tridiagonal system with Cholesky passes.

    This is the forward/backward primal update from Algorithm 2. The dense
    matrix boundary keeps the helper straightforward to validate, while the
    solver only accesses diagonal and first off-diagonal blocks.
    """

    matrix = np.asarray(system, dtype=float)
    rhs = np.asarray(vector, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != rhs.size:
        raise ValueError("system and vector dimensions do not agree")
    if matrix.shape[0] % block_dim:
        raise ValueError("system dimension must be a multiple of block_dim")
    count = matrix.shape[0] // block_dim
    lower = []
    factors = []
    for index in range(count):
        start = index * block_dim
        stop = start + block_dim
        diagonal = matrix[start:stop, start:stop].copy()
        if index:
            diagonal -= lower[index - 1] @ lower[index - 1].T
        factor = np.linalg.cholesky(_positive_definite(diagonal))
        factors.append(factor)
        if index < count - 1:
            next_start = (index + 1) * block_dim
            off_diagonal = matrix[next_start:next_start + block_dim, start:stop]
            lower.append(off_diagonal @ np.linalg.inv(factor.T))

    forward = np.zeros_like(rhs)
    for index, factor in enumerate(factors):
        start = index * block_dim
        stop = start + block_dim
        value = rhs[start:stop].copy()
        if index:
            previous = (index - 1) * block_dim
            value -= lower[index - 1] @ forward[previous:previous + block_dim]
        forward[start:stop] = np.linalg.solve(factor, value)

    solution = np.zeros_like(rhs)
    for index in range(count - 1, -1, -1):
        start = index * block_dim
        stop = start + block_dim
        value = forward[start:stop].copy()
        if index < count - 1:
            following = (index + 1) * block_dim
            value -= lower[index].T @ solution[following:following + block_dim]
        solution[start:stop] = np.linalg.solve(factors[index].T, value)
    return solution


def assemble_window_information(
    times: Sequence[float],
    measurements: Sequence[Optional["TargetMeasurement"]],
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    process_noise_spectral_density: float = 1.0,
    active_tracker_count: int = 1,
    handoff_information: Optional[np.ndarray] = None,
    handoff_information_vector: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the local DRWT information matrix and vector."""

    if len(times) == 0 or len(times) != len(measurements):
        raise ValueError("times and measurements must have equal nonzero length")
    count = len(times)
    dimension = STATE_DIM * count
    information = np.zeros((dimension, dimension), dtype=float)
    information_vector = np.zeros(dimension, dtype=float)
    scale = max(1, int(active_tracker_count))
    prior_mean = np.asarray(prior_mean, dtype=float).reshape(STATE_DIM)

    def add_factor(factors: Mapping[int, np.ndarray], value: np.ndarray, covariance: np.ndarray) -> None:
        inverse = _safe_inverse(np.asarray(covariance, dtype=float))
        residual_value = np.asarray(value, dtype=float).reshape(-1)
        for first_index, first_matrix in factors.items():
            first_slice = slice(first_index * STATE_DIM, (first_index + 1) * STATE_DIM)
            information_vector[first_slice] += first_matrix.T @ inverse @ residual_value
            for second_index, second_matrix in factors.items():
                second_slice = slice(second_index * STATE_DIM, (second_index + 1) * STATE_DIM)
                information[first_slice, second_slice] += first_matrix.T @ inverse @ second_matrix

    add_factor({0: np.eye(STATE_DIM)}, prior_mean, np.asarray(prior_covariance, dtype=float) * scale)
    if handoff_information is not None:
        information[:STATE_DIM, :STATE_DIM] += _positive_definite(np.asarray(handoff_information, dtype=float))
        if handoff_information_vector is not None:
            information_vector[:STATE_DIM] += np.asarray(handoff_information_vector, dtype=float).reshape(STATE_DIM)

    for index in range(1, count):
        delta = max(1e-6, float(times[index]) - float(times[index - 1]))
        transition = constant_velocity_transition(delta)
        process = constant_acceleration_process_noise(delta, process_noise_spectral_density) * scale
        add_factor({index - 1: -transition, index: np.eye(STATE_DIM)}, np.zeros(STATE_DIM), process)

    measurement_matrix = position_measurement_matrix()
    for index, measurement in enumerate(measurements):
        if measurement is None or not measurement.valid:
            continue
        add_factor(
            {index: measurement_matrix},
            np.asarray(measurement.position, dtype=float)[:2],
            _positive_definite(np.asarray(measurement.covariance, dtype=float).reshape((2, 2))),
        )
    return _positive_definite(information), information_vector


def dense_information_solution(information: np.ndarray, information_vector: np.ndarray) -> np.ndarray:
    """Solve the dense Equation 15 system, mainly for regression tests."""

    return np.linalg.solve(_positive_definite(information), np.asarray(information_vector, dtype=float))


@dataclass
class TargetMeasurement:
    target_id: str
    position: np.ndarray
    covariance: np.ndarray
    timestamp: float
    valid: bool = True
    source: str = "unknown"
    capture_id: Optional[str] = None
    sensor: Optional[str] = None
    visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(-1)
        if self.position.size < 2:
            raise ValueError("target measurement position must contain x and y")
        self.position = self.position[:2]
        self.covariance = _positive_definite(np.asarray(self.covariance, dtype=float).reshape((2, 2)))
        self.timestamp = float(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "position": self.position.tolist(),
            "covariance": self.covariance.tolist(),
            "timestamp": self.timestamp,
            "valid": bool(self.valid),
            "source": self.source,
            "capture_id": self.capture_id,
            "sensor": self.sensor,
            "visible": bool(self.visible),
            "metadata": dict(self.metadata),
        }


@dataclass
class _Track:
    target_id: str
    window_seconds: float
    process_noise_spectral_density: float
    measurement_std: float
    rho: float
    max_iterations: int
    tolerance: float
    prior_mean: Optional[np.ndarray] = None
    prior_covariance: np.ndarray = field(default_factory=lambda: np.eye(STATE_DIM) * 25.0)
    times: list = field(default_factory=list)
    measurements: list = field(default_factory=list)
    x: Optional[np.ndarray] = None
    dual: Optional[np.ndarray] = None
    information: Optional[np.ndarray] = None
    information_vector: Optional[np.ndarray] = None
    covariance: np.ndarray = field(default_factory=lambda: np.eye(STATE_DIM) * 25.0)
    latest_time: Optional[float] = None
    last_direct_observation: Optional[float] = None
    active: bool = False
    admm_iterations: int = 0
    consensus_residual: float = float("inf")
    handoffs: list = field(default_factory=list)
    pending_handoff_information: Optional[np.ndarray] = None
    pending_handoff_vector: Optional[np.ndarray] = None
    last_handoff_time: Optional[float] = None
    last_handoff_sent: Optional[float] = None

    def ensure_initialized(self, measurement: Optional[TargetMeasurement], timestamp: float) -> bool:
        if self.prior_mean is not None:
            return True
        if measurement is None or not measurement.valid:
            return False
        self.prior_mean = np.array([measurement.position[0], measurement.position[1], 0.0, 0.0], dtype=float)
        velocity_variance = max(1.0, 4.0 * self.measurement_std ** 2)
        self.prior_covariance = np.diag([
            max(float(measurement.covariance[0, 0]), 1e-4),
            max(float(measurement.covariance[1, 1]), 1e-4),
            velocity_variance,
            velocity_variance,
        ])
        self.latest_time = float(timestamp)
        return True

    def append_sample(self, timestamp: float, measurement: Optional[TargetMeasurement]) -> None:
        self.times.append(float(timestamp))
        self.measurements.append(measurement)
        self.latest_time = float(timestamp)
        if measurement is not None and measurement.valid:
            self.last_direct_observation = float(measurement.timestamp)

        cutoff = float(timestamp) - max(0.0, self.window_seconds)
        while len(self.times) > 1 and self.times[1] < cutoff:
            self.times.pop(0)
            self.measurements.pop(0)
            if self.x is not None and self.x.size >= 2 * STATE_DIM:
                self.prior_mean = self.x[STATE_DIM:2 * STATE_DIM].copy()
                self.prior_covariance = self.covariance[STATE_DIM:2 * STATE_DIM, STATE_DIM:2 * STATE_DIM].copy()

    def begin(self, timestamp: float, measurement: Optional[TargetMeasurement], active_count: int) -> bool:
        timestamp = float(timestamp)
        support_times = [value for value in (self.last_direct_observation, self.last_handoff_time)
                         if value is not None]
        if self.active and support_times and timestamp - max(support_times) > self.window_seconds:
            self.active = False
        if not self.active and self.prior_mean is not None:
            if measurement is None or not measurement.valid:
                return False
            # A new direct observation after expiry starts a fresh rolling
            # window rather than silently reviving stale information.
            self.prior_mean = None
            self.times.clear()
            self.measurements.clear()
            self.x = None
            self.dual = None
            self.information = None
            self.information_vector = None
        if not self.ensure_initialized(measurement, timestamp):
            return False
        self.append_sample(timestamp, measurement)
        if self.prior_mean is None:
            return False
        self.information, self.information_vector = assemble_window_information(
            self.times,
            self.measurements,
            self.prior_mean,
            self.prior_covariance,
            self.process_noise_spectral_density,
            active_count,
            self.pending_handoff_information,
            self.pending_handoff_vector,
        )
        self.pending_handoff_information = None
        self.pending_handoff_vector = None
        self.x = solve_block_tridiagonal(self.information, self.information_vector)
        self.dual = np.zeros_like(self.x)
        self.covariance = np.linalg.pinv(self.information)
        self.active = True
        self.admm_iterations = 0
        self.consensus_residual = float("inf")
        return True

    def latest_state(self) -> np.ndarray:
        if self.x is None:
            return np.zeros(STATE_DIM)
        return np.asarray(self.x[-STATE_DIM:], dtype=float).copy()

    def message(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "trajectory": self.x.tolist() if self.x is not None else None,
            "times": list(self.times),
            "state": self.latest_state().tolist() if self.x is not None else None,
            "active": bool(self.active and self.x is not None),
            "last_direct_observation": self.last_direct_observation,
        }

    def consensus_step(self, neighbor_messages: Iterable[Mapping[str, Any]]) -> float:
        if self.x is None or self.information is None or self.information_vector is None:
            return 0.0
        neighbors = [message for message in neighbor_messages
                     if message.get("target_id") == self.target_id
                     and message.get("trajectory") is not None
                     and bool(message.get("active", True))]
        previous = self.x.copy()
        if self.dual is None:
            self.dual = np.zeros_like(previous)
        trajectories = []
        for message in neighbors:
            values = np.asarray(message["trajectory"], dtype=float).reshape(-1, STATE_DIM)
            message_times = np.asarray(message.get("times") or [], dtype=float).reshape(-1)
            if message_times.size == values.shape[0] and len(self.times):
                aligned = np.vstack([
                    np.interp(np.asarray(self.times, dtype=float), message_times, values[:, component])
                    for component in range(STATE_DIM)
                ]).T.reshape(-1)
            elif values.shape[0] == previous.size // STATE_DIM:
                aligned = values.reshape(-1)
            else:
                # A disconnected component can have a shorter history when
                # it first receives a handoff. Use its latest state on the
                # local epoch grid until the next synchronized append.
                aligned = np.tile(values[-1], previous.size // STATE_DIM)
            trajectories.append(aligned)
        if trajectories:
            degree = len(trajectories)
            system = self.information + 2.0 * self.rho * degree * np.eye(previous.size)
            vector = self.information_vector - self.dual + self.rho * sum((previous + value) for value in trajectories)
            self.x = solve_block_tridiagonal(system, vector)
            # Equation 14 updates the local dual with the new primal value
            # against the synchronized neighbor trajectory from this round.
            self.dual += self.rho * sum((self.x - value) for value in trajectories)
        else:
            self.x = solve_block_tridiagonal(self.information, self.information_vector)
        self.admm_iterations += 1
        self.consensus_residual = float(np.linalg.norm(self.x - previous))
        return self.consensus_residual

    def finalize(self) -> Dict[str, Any]:
        state = self.latest_state()
        covariance = self.covariance[-STATE_DIM:, -STATE_DIM:] if self.covariance is not None else np.eye(STATE_DIM)
        residual = None
        latest_measurement = self.measurements[-1] if self.measurements else None
        if latest_measurement is not None and latest_measurement.valid:
            residual = (state[:2] - latest_measurement.position[:2]).tolist()
        return {
            "target_id": self.target_id,
            "position": state[:2].tolist(),
            "velocity": state[2:].tolist(),
            "covariance": covariance[:2, :2].tolist(),
            "state_covariance": covariance.tolist(),
            "timestamp": float(self.latest_time or 0.0),
            "active": bool(self.active),
            "admm_iterations": int(self.admm_iterations),
            "consensus_residual": float(self.consensus_residual),
            "measurement_residual": residual,
            "last_direct_observation": self.last_direct_observation,
            "handoffs": list(self.handoffs),
        }

    def predicted(self, timestamp: float) -> Dict[str, Any]:
        estimate = self.finalize()
        if not estimate["active"]:
            return estimate
        delta = max(0.0, float(timestamp) - float(estimate["timestamp"]))
        state = constant_velocity_transition(delta) @ np.asarray(
            [*estimate["position"], *estimate["velocity"]], dtype=float
        )
        covariance = np.asarray(estimate["covariance"], dtype=float)
        covariance += constant_acceleration_process_noise(delta, self.process_noise_spectral_density)[:2, :2]
        estimate["position"] = state[:2].tolist()
        estimate["velocity"] = state[2:].tolist()
        estimate["covariance"] = covariance.tolist()
        estimate["timestamp"] = float(timestamp)
        return estimate

    def handoff_message(self, timestamp: float) -> Optional[Dict[str, Any]]:
        if not self.active:
            return None
        support_times = [value for value in (self.last_direct_observation, self.last_handoff_time)
                         if value is not None]
        if not support_times or float(timestamp) - max(support_times) < self.window_seconds:
            return None
        if self.last_handoff_sent is not None and float(timestamp) - self.last_handoff_sent < self.window_seconds:
            return None
        estimate = self.finalize()
        covariance = _positive_definite(np.asarray(estimate["state_covariance"], dtype=float))
        information = _safe_inverse(covariance)
        state = np.asarray([*estimate["position"], *estimate["velocity"]], dtype=float)
        self.last_handoff_sent = float(timestamp)
        return {
            "type": "handoff",
            "target_id": self.target_id,
            "source_id": "",
            "information_matrix": information.tolist(),
            "information_vector": (information @ state).tolist(),
            "timestamp": float(timestamp),
        }

    def accept_handoff(self, message: Mapping[str, Any]) -> None:
        if str(message.get("target_id", self.target_id)) != self.target_id:
            return
        matrix = np.asarray(message.get("information_matrix"), dtype=float)
        vector = np.asarray(message.get("information_vector"), dtype=float)
        if matrix.shape != (STATE_DIM, STATE_DIM) or vector.shape != (STATE_DIM,):
            return
        covariance = _safe_inverse(matrix)
        if self.prior_mean is None:
            self.prior_mean = covariance @ vector
            self.prior_covariance = covariance
            self.times = [float(message.get("timestamp", 0.0))]
            self.measurements = [None]
            self.latest_time = self.times[0]
            self.x = self.prior_mean.copy()
            self.dual = np.zeros_like(self.x)
            self.information = matrix.copy()
            self.information_vector = vector.copy()
        self.active = True
        self.last_handoff_time = float(message.get("timestamp", 0.0))
        if self.pending_handoff_information is None:
            self.pending_handoff_information = matrix.copy()
            self.pending_handoff_vector = vector.copy()
        else:
            self.pending_handoff_information += matrix
            self.pending_handoff_vector += vector
        self.handoffs.append({"source_id": message.get("source_id"), "timestamp": message.get("timestamp")})


class TargetTrackingModule:
    """One decentralized DRWT estimator owned by one physical agent."""

    def __init__(
        self,
        agent_id: str,
        window_seconds: float = 5.0,
        process_noise_spectral_density: float = 1.0,
        measurement_std: float = 0.5,
        rho: float = 1.0,
        max_iterations: int = 20,
        tolerance: float = 1e-3,
    ) -> None:
        self.agent_id = str(agent_id)
        self.window_seconds = float(window_seconds)
        self.process_noise_spectral_density = float(process_noise_spectral_density)
        self.measurement_std = float(measurement_std)
        self.rho = float(rho)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.tracks: Dict[str, _Track] = {}
        self.last_measurements: Dict[str, Optional[TargetMeasurement]] = {}

    def _track(self, target_id: str) -> _Track:
        target_id = str(target_id)
        if target_id not in self.tracks:
            self.tracks[target_id] = _Track(
                target_id=target_id,
                window_seconds=self.window_seconds,
                process_noise_spectral_density=self.process_noise_spectral_density,
                measurement_std=self.measurement_std,
                rho=self.rho,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
            )
        return self.tracks[target_id]

    def begin_epoch(
        self,
        timestamp: float,
        measurements: Mapping[str, Optional[TargetMeasurement]],
        active_tracker_count: int = 1,
    ) -> None:
        self.last_measurements = dict(measurements)
        target_ids = set(self.tracks) | {str(target_id) for target_id in measurements}
        for target_id in sorted(target_ids):
            self._track(target_id).begin(
                timestamp,
                measurements.get(target_id),
                active_tracker_count,
            )

    def seed_from_messages(self, messages: Iterable[Mapping[str, Any]], timestamp: float) -> None:
        """Initialize a silent agent from a neighbor's estimate announcement."""

        for message in messages:
            target_id = message.get("target_id")
            state = message.get("state")
            if target_id is None or state is None or str(target_id) in self.tracks:
                continue
            values = np.asarray(state, dtype=float).reshape(-1)
            if values.size != STATE_DIM:
                continue
            track = self._track(str(target_id))
            track.prior_mean = values.copy()
            track.prior_covariance = np.eye(STATE_DIM) * 16.0
            track.latest_time = float(timestamp)
            track.last_handoff_time = float(timestamp)
            track.active = True
            track.times = [float(timestamp)]
            track.measurements = [None]
            track.information, track.information_vector = assemble_window_information(
                track.times, track.measurements, track.prior_mean, track.prior_covariance,
                track.process_noise_spectral_density, 1,
            )
            track.x = track.prior_mean.copy()
            track.dual = np.zeros(STATE_DIM)

    def messages(self) -> Dict[str, Dict[str, Any]]:
        return {target_id: track.message() for target_id, track in self.tracks.items() if track.active}

    def consensus_round(self, inbound_messages: Mapping[str, Sequence[Mapping[str, Any]]]) -> float:
        maximum = 0.0
        for target_id, track in self.tracks.items():
            maximum = max(maximum, track.consensus_step(inbound_messages.get(target_id, [])))
        return maximum

    def finish_epoch(self) -> Dict[str, Dict[str, Any]]:
        return {target_id: track.finalize() for target_id, track in self.tracks.items() if track.active}

    def predicted_estimates(self, timestamp: float) -> Dict[str, Dict[str, Any]]:
        return {target_id: track.predicted(timestamp) for target_id, track in self.tracks.items() if track.active}

    def handoff_messages(self, timestamp: float) -> Dict[str, Dict[str, Any]]:
        messages = {}
        for target_id, track in self.tracks.items():
            message = track.handoff_message(timestamp)
            if message is not None:
                message["source_id"] = self.agent_id
                messages[target_id] = message
        return messages

    def accept_handoff(self, message: Mapping[str, Any]) -> None:
        target_id = message.get("target_id")
        if target_id is not None:
            self._track(str(target_id)).accept_handoff(message)

    def update_and_track(
        self,
        sensor_data: Dict[str, Any],
        inbound_msgs: Dict[str, Any],
        robustness_margin: float,
        local_state_estimate: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
        """Backward-compatible one-agent wrapper for older callers."""

        timestamp = float(sensor_data.get("timestamp", 0.0))
        measurements = sensor_data.get("target_measurements") or {}
        normalized = {
            target_id: value if isinstance(value, TargetMeasurement) else None
            for target_id, value in measurements.items()
        }
        self.begin_epoch(timestamp, normalized, 1)
        estimates = self.finish_epoch()
        first = next(iter(estimates.values()), None)
        if first is None:
            estimate = {
                "target_position": np.zeros(3),
                "target_velocity": np.zeros(3),
                "uncertainty_covariance": np.eye(3) * max(0.0, float(robustness_margin)),
                "active": False,
            }
            return estimate, {"agent_id": self.agent_id, "targets": {}}, np.zeros(3)
        position = np.array([first["position"][0], first["position"][1], 0.0])
        velocity = np.array([first["velocity"][0], first["velocity"][1], 0.0])
        covariance = np.asarray(first["covariance"], dtype=float)
        estimate = {
            "target_id": first["target_id"],
            "target_position": position,
            "target_velocity": velocity,
            "uncertainty_covariance": np.pad(covariance, ((0, 1), (0, 1))),
            "active": bool(first["active"]),
        }
        self_position = np.asarray(local_state_estimate.get("estimated_position", np.zeros(3)), dtype=float)
        direction = position - self_position
        distance = float(np.linalg.norm(direction))
        nominal_velocity = direction / distance * min(2.0, distance) if distance > 1e-9 else np.zeros(3)
        return estimate, {"agent_id": self.agent_id, "targets": estimates}, nominal_velocity


class DistributedTargetTracking:
    """Synchronous network harness for independent DRWT agent modules.

    This class only routes messages and clocked epochs. It never combines
    measurements centrally; every primal and dual update is performed by the
    receiving agent's :class:`TargetTrackingModule`.
    """

    def __init__(self, modules: Mapping[str, TargetTrackingModule], max_iterations: int = 20,
                 tolerance: float = 1e-3) -> None:
        self.modules = dict(modules)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.last_result: Dict[str, Any] = {}

    def update(
        self,
        timestamp: float,
        measurements: Mapping[str, Mapping[str, Optional[TargetMeasurement]]],
        adjacency: Mapping[str, Sequence[str]],
    ) -> Dict[str, Any]:
        """Run one clocked DRWT estimation epoch over the current graph."""

        target_ids = sorted({str(target_id) for values in measurements.values() for target_id in values})
        for values in self.modules.values():
            target_ids.extend(str(target_id) for target_id in values.tracks)
        target_ids = sorted(set(target_ids))
        active_count = max(1, len(self.modules))
        for agent_id, module in self.modules.items():
            module.begin_epoch(timestamp, measurements.get(agent_id, {}), active_count)

        # A new target is announced by a directly observing agent and seeded
        # into silent neighbors before ADMM rounds begin. This is still a
        # normal estimate message, not a central measurement fusion step.
        for _ in range(max(1, len(self.modules))):
            changed = False
            announcements = {agent_id: module.messages() for agent_id, module in self.modules.items()}
            for agent_id, module in self.modules.items():
                if all(target_id in module.tracks for target_id in target_ids):
                    continue
                neighbors = []
                for neighbor in adjacency.get(agent_id, []):
                    neighbors.extend(announcements.get(neighbor, {}).values())
                before = set(module.tracks)
                module.seed_from_messages(neighbors, timestamp)
                changed |= set(module.tracks) != before
            if not changed:
                break

        residual = float("inf")
        rounds = 0
        for rounds in range(1, self.max_iterations + 1):
            messages = {agent_id: module.messages() for agent_id, module in self.modules.items()}
            round_residual = 0.0
            for agent_id, module in self.modules.items():
                inbound: Dict[str, list] = {}
                for neighbor in adjacency.get(agent_id, []):
                    for target_id, message in messages.get(neighbor, {}).items():
                        inbound.setdefault(target_id, []).append(message)
                module_residual = module.consensus_round(inbound)
                round_residual = max(round_residual, module_residual)
            residual = round_residual
            if residual <= self.tolerance:
                break

        estimates = {
            agent_id: module.finish_epoch()
            for agent_id, module in self.modules.items()
        }
        final_messages = {agent_id: module.messages() for agent_id, module in self.modules.items()}
        self.last_result = {
            "estimates": estimates,
            "messages": final_messages,
            "iterations": rounds,
            "residual": float(residual),
            "active_targets": target_ids,
            "handoffs": [],
        }
        return self.last_result

    def predicted_estimates(self, timestamp: float) -> Dict[str, Dict[str, Dict[str, Any]]]:
        return {
            agent_id: module.predicted_estimates(timestamp)
            for agent_id, module in self.modules.items()
        }

    def perform_handoffs(self, timestamp: float, adjacency: Mapping[str, Sequence[str]]) -> list:
        """Transfer stale tracker information to one continuing neighbor."""

        handoffs = []
        for source_id, module in self.modules.items():
            for target_id, message in module.handoff_messages(timestamp).items():
                candidates = [neighbor for neighbor in adjacency.get(source_id, [])
                              if target_id in self.modules[neighbor].tracks]
                if not candidates:
                    continue
                receiver_id = sorted(candidates)[0]
                message["source_id"] = source_id
                message["receiver_id"] = receiver_id
                self.modules[receiver_id].accept_handoff(message)
                handoffs.append(dict(message))
        self.last_result.setdefault("handoffs", []).extend(handoffs)
        return handoffs
