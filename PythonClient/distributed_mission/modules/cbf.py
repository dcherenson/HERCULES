"""Distributed control-barrier-function safety filters.

The module separates model-level controls from AirSim commands: double
integrators return acceleration, while the Mestres unicycle controller returns
linear speed and yaw rate. The simulator adapter performs final conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class AgentState:
    agent_id: str
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    acceleration: Optional[np.ndarray] = None
    vehicle_type: str = "drone"
    timestamp: float = 0.0
    yaw_rate: float = 0.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(-1)
        self.velocity = np.asarray(self.velocity, dtype=float).reshape(-1)
        if self.acceleration is not None:
            self.acceleration = np.asarray(self.acceleration, dtype=float).reshape(-1)


@dataclass
class ObstacleProxy:
    obstacle_id: str
    center: np.ndarray
    radius: float
    source: str = "unknown"
    timestamp: float = 0.0
    point_count: int = 0
    is_planar: bool = False
    velocity: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(-1)
        self.radius = float(max(0.0, self.radius))
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=float).reshape(-1)


@dataclass
class CBFConfig:
    method: str = "mestres"
    k1: float = 2.0
    k2: float = 2.0
    alpha: float = 2.0
    # Placeholder robustness radius; set to zero until the estimator supplies
    # a calibrated uncertainty model.
    uncertainty_radius: float = 0.0
    uav_radius: float = 1.0
    ugv_radius: float = 1.25
    obstacle_margin: float = 0.0
    uav_acceleration_limit: float = 6.0
    ugv_acceleration_limit: float = 3.0
    uav_velocity_limit: float = 3.0
    ugv_speed_limit: float = 3.0
    ugv_yaw_rate_limit: float = 1.5
    lookahead_distance: float = 1.0
    # Maximum permitted NED Z for UAVs. The mission supplies a map-specific
    # value corresponding to one metre above the calibrated ground reference.
    uav_altitude_floor: float = -1.0
    distributed_rounds: int = 20
    distributed_tolerance: float = 1e-3
    solver_eps_abs: float = 1e-5
    solver_eps_rel: float = 1e-5
    solver_max_iter: int = 4000


@dataclass
class CBFRequest:
    ego: AgentState
    nominal_control: np.ndarray
    neighbors: Sequence[AgentState] = field(default_factory=list)
    obstacles: Sequence[ObstacleProxy] = field(default_factory=list)
    uncertainty_radius: Optional[float] = None
    control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    sensor_valid: bool = True

    def __post_init__(self) -> None:
        self.nominal_control = np.asarray(self.nominal_control, dtype=float).reshape(-1)


@dataclass
class CBFResult:
    safe_control: np.ndarray
    success: bool
    status: str
    message: str = ""
    minimum_barrier: float = float("inf")
    robust_terms: List[float] = field(default_factory=list)
    active_constraints: int = 0
    constraint_count: int = 0
    distributed_rounds: int = 0
    solve_time_ms: float = 0.0
    fallback: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _as3(value: Optional[np.ndarray]) -> np.ndarray:
    result = np.zeros(3)
    if value is not None:
        vector = np.asarray(value, dtype=float).reshape(-1)
        result[: min(3, len(vector))] = vector[:3]
    return result


def _solve_qp(
    nominal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rows: Sequence[np.ndarray],
    rhs: Sequence[float],
    config: CBFConfig,
) -> Tuple[np.ndarray, bool, str, Dict[str, Any]]:
    """Solve min 1/2||u-u_nom||² subject to A u >= b and box bounds."""
    n = len(nominal)
    if not rows:
        return np.clip(nominal, lower, upper), True, "solved_no_constraints", {}

    matrix = np.asarray(rows, dtype=float).reshape((-1, n))
    vector = np.asarray(rhs, dtype=float)

    try:
        import osqp
        from scipy import sparse

        problem = osqp.OSQP()
        problem.setup(
            P=sparse.csc_matrix(np.eye(n)),
            q=-np.asarray(nominal, dtype=float),
            A=sparse.csc_matrix(matrix),
            l=vector,
            u=np.full(len(vector), np.inf),
            eps_abs=config.solver_eps_abs,
            eps_rel=config.solver_eps_rel,
            max_iter=config.solver_max_iter,
            polishing=False,
            verbose=False,
        )
        result = problem.solve()
        status = str(result.info.status).lower()
        success = result.x is not None and "solved" in status
        control = np.clip(np.asarray(result.x, dtype=float), lower, upper) if success else np.zeros(n)
        info = {
            "solver": "osqp",
            "status": status,
            "iterations": int(getattr(result.info, "iter", 0)),
            "primal_residual": float(getattr(result.info, "prim_res", np.nan)),
            "dual_residual": float(getattr(result.info, "dual_res", np.nan)),
        }
        return control, success, status, info
    except ImportError:
        # Keeps local unit tests usable before installing the documented OSQP
        # dependency. Production runs use OSQP.
        try:
            from scipy.optimize import minimize

            constraints = [
                {"type": "ineq", "fun": lambda u, row=row, b=b: np.dot(row, u) - b}
                for row, b in zip(matrix, vector)
            ]
            result = minimize(
                lambda u: 0.5 * float(np.sum((u - nominal) ** 2)),
                np.clip(nominal, lower, upper),
                method="SLSQP",
                bounds=list(zip(lower, upper)),
                constraints=constraints,
                options={"maxiter": config.solver_max_iter, "ftol": config.solver_eps_abs},
            )
            return np.asarray(result.x), bool(result.success), str(result.message), {"solver": "scipy-slsqp"}
        except ImportError as exc:
            return np.zeros(n), False, "solver_unavailable", {"error": str(exc)}


class DistributedCBFModule:
    """Selectable Mestres or Wang safety filter."""

    def __init__(self, agent_id: str, vehicle_type: str = "drone", config: Optional[CBFConfig] = None):
        self.agent_id = agent_id
        self.vehicle_type = vehicle_type
        self.config = config or CBFConfig()
        if self.config.method not in {"mestres", "wang"}:
            raise ValueError("method must be 'mestres' or 'wang'")
        self.last_result: Optional[CBFResult] = None

    def compute_safe_control(
        self,
        nominal_velocity: Optional[np.ndarray] = None,
        local_state_estimate: Optional[Dict[str, Any]] = None,
        neighbor_states: Optional[Sequence[Union[Dict[str, Any], AgentState]]] = None,
        robustness_margin: Optional[float] = None,
        obstacles: Optional[Sequence[ObstacleProxy]] = None,
        nominal_control: Optional[np.ndarray] = None,
        sensor_valid: bool = True,
    ) -> np.ndarray:
        """Backward-compatible wrapper returning only the safe command."""
        state = local_state_estimate or {}
        ego = AgentState(
            self.agent_id,
            state.get("estimated_position", np.zeros(3)),
            state.get("estimated_velocity", np.zeros(3)),
            float(state.get("yaw", 0.0)),
            vehicle_type=self.vehicle_type,
        )
        neighbors = [self._state_from_any(item) for item in (neighbor_states or [])]
        control = nominal_control if nominal_control is not None else nominal_velocity
        if control is None:
            control = np.zeros(2 if self._is_unicycle() else 3)
        result = self.filter(CBFRequest(ego, np.asarray(control), neighbors, list(obstacles or []), robustness_margin, sensor_valid=sensor_valid))
        self.last_result = result
        return result.safe_control

    def filter(self, request: CBFRequest) -> CBFResult:
        import time

        start = time.perf_counter()
        ego = request.ego
        dimension = 2 if self._is_unicycle() else 3
        if not request.sensor_valid:
            return self._failure(dimension, "invalid_or_stale_sensor", start)
        nominal = np.asarray(request.nominal_control, dtype=float).reshape(-1)[:dimension]
        if len(nominal) != dimension or not np.all(np.isfinite(nominal)):
            return self._failure(dimension, "invalid_nominal_control", start)

        lower, upper = self._bounds(request, dimension)
        rows: List[np.ndarray] = []
        rhs: List[float] = []
        barriers: List[float] = []
        robust_terms: List[float] = []

        if self._is_unicycle():
            self._build_unicycle_constraints(request, rows, rhs, barriers, robust_terms)
        else:
            self._build_double_integrator_constraints(request, rows, rhs, barriers, robust_terms)

        control, success, status, solver_info = _solve_qp(nominal, lower, upper, rows, rhs, self.config)
        distributed_rounds = 0
        if success and self.config.method == "mestres" and rows:
            # The paper's distributed projected saddle dynamics are represented
            # here by local projected correction rounds. Each round uses only
            # this agent's constraint rows and preserves the control box.
            control, distributed_rounds = self._project_constraints(control, lower, upper, rows, rhs)
        minimum = min(barriers) if barriers else float("inf")
        active = sum(abs(float(np.dot(row, control) - bound)) < 1e-3 for row, bound in zip(rows, rhs))
        result = CBFResult(
            safe_control=control if success else self._fail_safe_control(ego, dimension),
            success=success,
            status=status,
            message="" if success else "CBF solver failed; fail-safe control applied",
            minimum_barrier=minimum,
            robust_terms=robust_terms,
            active_constraints=int(active),
            constraint_count=len(rows),
            distributed_rounds=distributed_rounds,
            solve_time_ms=(time.perf_counter() - start) * 1000.0,
            fallback=not success,
            diagnostics=solver_info,
        )
        self.last_result = result
        return result

    def _project_constraints(
        self,
        control: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        rows: Sequence[np.ndarray],
        rhs: Sequence[float],
    ) -> Tuple[np.ndarray, int]:
        value = np.asarray(control, dtype=float).copy()
        for round_index in range(self.config.distributed_rounds):
            previous = value.copy()
            for row, bound in zip(rows, rhs):
                violation = float(bound - np.dot(row, value))
                if violation > 0.0:
                    denominator = float(np.dot(row, row)) + 1e-9
                    value = value + (violation / denominator) * np.asarray(row)
                    value = np.clip(value, lower, upper)
            if np.linalg.norm(value - previous) <= self.config.distributed_tolerance:
                return value, round_index + 1
        return value, self.config.distributed_rounds

    def _build_double_integrator_constraints(
        self,
        request: CBFRequest,
        rows: List[np.ndarray],
        rhs: List[float],
        barriers: List[float],
        robust: List[float],
    ) -> None:
        ego = request.ego
        p_i, v_i = _as3(ego.position), _as3(ego.velocity)
        uncertainty = max(0.0, float(self.config.uncertainty_radius if request.uncertainty_radius is None else request.uncertainty_radius))
        ego_radius = self._vehicle_radius(ego.vehicle_type)

        def add_pair(
            p_j: np.ndarray,
            v_j: np.ndarray,
            radius_j: float,
            acceleration_j: Optional[np.ndarray],
            split: bool,
        ) -> None:
            dp = p_i - _as3(p_j)
            dv = v_i - _as3(v_j)
            clearance = ego_radius + radius_j + self.config.obstacle_margin
            h = float(np.dot(dp, dp) - clearance * clearance)
            h_dot = float(2.0 * np.dot(dp, dv))
            psi = h_dot + self.config.k1 * h
            neighbor_acc = _as3(acceleration_j)
            base = 2.0 * float(np.dot(dv, dv)) + self.config.k1 * h_dot + self.config.k2 * psi
            required = -base + 2.0 * float(np.dot(dp, neighbor_acc))

            # Literal Euclidean norm of the full [p,v] gradient, as selected
            # for v1 despite the mixed position/velocity units.
            grad_p = 2.0 * dv + 2.0 * self.config.k1 * dp
            grad_v = 2.0 * dp
            robust_term = float(np.linalg.norm(np.concatenate((grad_p, grad_v))) * uncertainty)
            required += robust_term
            if split:
                required *= 0.5

            rows.append(2.0 * dp)
            rhs.append(required)
            barriers.append(h)
            robust.append(robust_term)

        for neighbor in request.neighbors:
            if neighbor.vehicle_type != ego.vehicle_type:
                continue
            add_pair(
                neighbor.position,
                neighbor.velocity,
                self._vehicle_radius(neighbor.vehicle_type),
                neighbor.acceleration,
                self.config.method == "wang",
            )

        for obstacle in request.obstacles:
            add_pair(
                obstacle.center,
                obstacle.velocity if obstacle.velocity is not None else np.zeros(3),
                float(obstacle.radius),
                np.zeros(3),
                False,
            )

        # AirSim uses NED coordinates. This altitude guard maintains the
        # initial UAV/UGV separation while cross-type pair CBFs remain off.
        if ego.vehicle_type == "drone":
            altitude_floor = float(self.config.uav_altitude_floor)
            h = altitude_floor - p_i[2]
            rows.append(np.array([0.0, 0.0, -1.0]))
            rhs.append(-self.config.alpha * h + uncertainty)
            barriers.append(float(h))
            robust.append(uncertainty)

    def _build_unicycle_constraints(
        self,
        request: CBFRequest,
        rows: List[np.ndarray],
        rhs: List[float],
        barriers: List[float],
        robust: List[float],
    ) -> None:
        ego = request.ego
        position = _as3(ego.position)
        theta = float(ego.yaw)
        lookahead = self.config.lookahead_distance
        q = position[:2] + lookahead * np.array([np.cos(theta), np.sin(theta)])
        control_map = np.array([
            [np.cos(theta), -lookahead * np.sin(theta)],
            [np.sin(theta), lookahead * np.cos(theta)],
        ])
        uncertainty = max(0.0, float(self.config.uncertainty_radius if request.uncertainty_radius is None else request.uncertainty_radius))
        own_radius = self._vehicle_radius(ego.vehicle_type)

        def add_circle(center: np.ndarray, radius: float, velocity: np.ndarray) -> None:
            delta = q - _as3(center)[:2]
            clearance = own_radius + radius + self.config.obstacle_margin
            h = float(np.dot(delta, delta) - clearance * clearance)
            other_velocity = _as3(velocity)[:2]
            row = 2.0 * delta @ control_map
            b = -self.config.alpha * h + 2.0 * float(np.dot(delta, other_velocity))
            robust_term = float(np.linalg.norm(2.0 * delta) * uncertainty)
            rows.append(row)
            rhs.append(b + robust_term)
            barriers.append(h)
            robust.append(robust_term)

        for neighbor in request.neighbors:
            if neighbor.vehicle_type == ego.vehicle_type:
                add_circle(neighbor.position, self._vehicle_radius(neighbor.vehicle_type), neighbor.velocity)
        for obstacle in request.obstacles:
            add_circle(
                obstacle.center,
                obstacle.radius,
                obstacle.velocity if obstacle.velocity is not None else np.zeros(3),
            )

    def _bounds(self, request: CBFRequest, dimension: int) -> Tuple[np.ndarray, np.ndarray]:
        if request.control_bounds is not None:
            lower, upper = request.control_bounds
            return np.asarray(lower, dtype=float)[:dimension], np.asarray(upper, dtype=float)[:dimension]
        if self._is_unicycle():
            return np.array([0.0, -self.config.ugv_yaw_rate_limit]), np.array([self.config.ugv_speed_limit, self.config.ugv_yaw_rate_limit])
        limit = self.config.uav_acceleration_limit if self.vehicle_type == "drone" else self.config.ugv_acceleration_limit
        return np.full(dimension, -limit), np.full(dimension, limit)

    def _fail_safe_control(self, ego: AgentState, dimension: int) -> np.ndarray:
        if self._is_unicycle():
            return np.zeros(2)
        return np.clip(-_as3(ego.velocity), -self.config.uav_acceleration_limit, self.config.uav_acceleration_limit)[:dimension]

    def _failure(self, dimension: int, status: str, start: float) -> CBFResult:
        import time
        return CBFResult(
            np.zeros(dimension), False, status,
            message="fail-safe control applied", fallback=True,
            solve_time_ms=(time.perf_counter() - start) * 1000.0,
        )

    def _vehicle_radius(self, vehicle_type: str) -> float:
        return self.config.uav_radius if vehicle_type == "drone" else self.config.ugv_radius

    def _is_unicycle(self) -> bool:
        return self.vehicle_type == "ugv" and self.config.method == "mestres"

    @staticmethod
    def _state_from_any(value: Union[Dict[str, Any], AgentState]) -> AgentState:
        if isinstance(value, AgentState):
            return value
        return AgentState(
            value.get("agent_id", "neighbor"),
            value.get("estimated_position", value.get("position", np.zeros(3))),
            value.get("estimated_velocity", value.get("velocity", np.zeros(3))),
            value.get("yaw", 0.0),
            value.get("acceleration"),
            value.get("vehicle_type", "drone"),
            value.get("timestamp", 0.0),
            value.get("yaw_rate", 0.0),
        )
