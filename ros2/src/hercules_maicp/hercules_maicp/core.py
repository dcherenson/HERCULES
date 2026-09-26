"""ROS-independent numerical primitives for multi-agent interactive CP.

The module implements the numerical objects in the paper directly:

* strict-subgradient pinball iterations with synchronous Metropolis mixing;
* tagged finite flooding and finite upper recovery of an order statistic;
* centralized order statistics used only by the oracle baseline;
* class-wise mission aggregation and split conformal ranks; and
* the coupled second-order-cone margin update.

No ROS message, executor, or global experiment state is imported here.  A ROS
adapter can exchange the same local iterates and tagged summaries over a real
graph, or can pass ``order_statistic`` and ``margin_update`` callbacks to
:func:`calibrate_round`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from inspect import signature
from math import ceil

import numpy as np

try:  # cvxpy is an optional dependency for users of the order-statistic core.
    import cvxpy as cp
except ImportError:  # pragma: no cover - exercised only in minimal ROS images.
    cp = None  # type: ignore[assignment]

from .config import ClassConfig, MAICPConfig, paper_config


Node = Hashable
Graph = Mapping[Node, Iterable[Node]]


@dataclass(frozen=True)
class OrderStatisticResult:
    """Result of finite distributed order-statistic computation.

    ``value`` is the exact order statistic for the centralized helper and is
    a finite-time upper bound for the distributed helper.  The distributed
    path deliberately leaves ``exact_value`` as ``None`` so it cannot silently
    claim access to all raw samples.
    """

    value: float
    rank: int
    sample_count: int
    exact_value: float | None = None
    pinball_history: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    pinball_rounds: int = 0
    flooding_rounds: int = 0
    recovery_steps: int = 0
    owners: dict[Node, tuple[Hashable, ...]] = field(default_factory=dict)
    certified: bool = True

    @property
    def upper_bound(self) -> float:
        return self.value

    @property
    def exact(self) -> float | None:
        return self.exact_value

    @property
    def total_communication_rounds(self) -> int:
        return int(self.pinball_rounds + self.flooding_rounds)

    @property
    def communication_rounds(self) -> int:
        return self.total_communication_rounds


@dataclass(frozen=True)
class MarginUpdateResult:
    """Accepted solution of the coupled margin update SOCP."""

    margins: dict[Hashable, float]
    previous_margins: dict[Hashable, float]
    thresholds: dict[Hashable, float]
    kappa: dict[Hashable, float]
    normalized_change: float
    objective: float
    constraint_slacks: dict[Hashable, float]
    status: str = "optimal"
    certified: bool = False
    # A local solve is one replica.  A ROS coordinator replaces this with the
    # number of workers that independently returned agreeing solutions.
    replica_count: int = 1

    @property
    def delta(self) -> float:
        return self.normalized_change


@dataclass(frozen=True)
class CalibrationResult:
    """One numerical calibration round and its margin candidate."""

    method: str
    margins: dict[Hashable, float]
    previous_margins: dict[Hashable, float]
    mission_scores: dict[Hashable, tuple[float, ...]]
    certified_mission_scores: dict[Hashable, tuple[float, ...]]
    exact_thresholds: dict[Hashable, float]
    thresholds: dict[Hashable, float]
    conformal_ranks: dict[Hashable, int]
    kappa: dict[Hashable, float]
    status: str
    certified: bool
    margin_update: MarginUpdateResult | None = None

    @property
    def q(self) -> dict[Hashable, float]:
        return dict(self.thresholds)


def _sort_key(value: Node) -> tuple[str, str]:
    """Stable ordering for graph nodes without requiring cross-type ordering."""

    return (type(value).__name__, repr(value))


def _equal_values(left: object, right: object) -> bool:
    if isinstance(left, (float, np.floating)) and isinstance(
        right, (float, np.floating)
    ):
        if np.isnan(left) and np.isnan(right):
            return True
    try:
        result = left == right
        return bool(result)
    except Exception:
        return left is right


def validate_graph(graph: Graph) -> dict[Node, set[Node]]:
    """Validate and copy a nonempty, undirected, connected graph.

    Self loops, unknown neighbors, asymmetric edges, and disconnected graphs
    are rejected before any numerical operation starts.  Empty owner stores
    are allowed; the graph itself must still contain the participating nodes.
    """

    if not isinstance(graph, Mapping) or not graph:
        raise ValueError("communication graph must be a nonempty mapping")
    nodes = tuple(sorted(graph, key=_sort_key))
    node_set = set(nodes)
    canonical: dict[Node, set[Node]] = {}
    for node in nodes:
        raw_neighbors = graph[node]
        if isinstance(raw_neighbors, (str, bytes)):
            raise TypeError(f"neighbors for node {node!r} must be an iterable of nodes")
        try:
            neighbors = list(raw_neighbors)
        except TypeError as exc:
            raise TypeError(f"neighbors for node {node!r} must be iterable") from exc
        if len(set(neighbors)) != len(neighbors):
            raise ValueError(f"graph has duplicate neighbors for node {node!r}")
        if node in neighbors:
            raise ValueError(f"graph must not contain a self edge at {node!r}")
        unknown = set(neighbors) - node_set
        if unknown:
            raise ValueError(f"graph has unknown neighbors for {node!r}: {unknown!r}")
        canonical[node] = set(neighbors)
    for node in nodes:
        for neighbor in canonical[node]:
            if node not in canonical[neighbor]:
                raise ValueError("communication graph must be undirected")

    seen: set[Node] = {nodes[0]}
    queue: deque[Node] = deque([nodes[0]])
    while queue:
        node = queue.popleft()
        for neighbor in canonical[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    if seen != node_set:
        raise ValueError("distributed computation requires a connected graph")
    return canonical


def reachable_nodes(graph: Graph, start: Node) -> set[Node]:
    """Return nodes reachable from ``start`` after graph validation."""

    canonical = validate_graph(graph)
    if start not in canonical:
        raise KeyError(f"unknown graph start node {start!r}")
    seen = {start}
    queue: deque[Node] = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in canonical[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def graph_diameter(graph: Graph) -> int:
    """Compute the exact unweighted diameter used by finite flooding."""

    canonical = validate_graph(graph)
    diameter = 0
    for start in canonical:
        distances: dict[Node, int] = {start: 0}
        queue: deque[Node] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in canonical[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        diameter = max(diameter, max(distances.values(), default=0))
    return diameter


def metropolis_weights(graph: Graph) -> tuple[tuple[Node, ...], np.ndarray]:
    """Return symmetric doubly-stochastic Metropolis weights for ``graph``."""

    canonical = validate_graph(graph)
    nodes = tuple(sorted(canonical, key=_sort_key))
    index = {node: i for i, node in enumerate(nodes)}
    degrees = {node: len(canonical[node]) for node in nodes}
    matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
    for node in nodes:
        row = index[node]
        for neighbor in canonical[node]:
            col = index[neighbor]
            matrix[row, col] = 1.0 / (1.0 + max(degrees[node], degrees[neighbor]))
        matrix[row, row] = 1.0 - float(matrix[row].sum())
    if not np.allclose(matrix, matrix.T) or not np.allclose(matrix.sum(axis=0), 1.0):
        raise AssertionError("Metropolis weights are not symmetric and doubly stochastic")
    return nodes, matrix


metropolis_matrix = metropolis_weights


def _as_tagged_store(store: object, owner: Node) -> dict[Hashable, float]:
    """Convert a node's local values to a tagged scalar store."""

    if isinstance(store, Mapping):
        result: dict[Hashable, float] = {}
        for tag, value in store.items():
            try:
                result[tag] = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"local value {tag!r} at node {owner!r} is not scalar") from exc
        return result
    if isinstance(store, (str, bytes)):
        raise TypeError(f"local values at node {owner!r} must be a mapping or sequence")
    try:
        values = list(store)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"local values at node {owner!r} must be a mapping or sequence") from exc
    return {(owner, index): float(value) for index, value in enumerate(values)}


def _normalise_local_values(
    local_values: Mapping[Node, object] | Sequence[float] | np.ndarray,
    graph: Mapping[Node, set[Node]],
) -> dict[Node, dict[Hashable, float]]:
    """Normalize owner stores while preserving empty graph owners."""

    nodes = tuple(sorted(graph, key=_sort_key))
    if isinstance(local_values, Mapping):
        unknown = set(local_values) - set(nodes)
        if unknown:
            raise ValueError(f"local stores contain owners outside the graph: {unknown!r}")
        return {
            node: _as_tagged_store(local_values.get(node, {}), node)
            for node in nodes
        }
    if isinstance(local_values, (str, bytes)):
        raise TypeError("local_values must be owner stores or a numeric sequence")
    values = list(local_values)
    return {nodes[0]: {(nodes[0], i): float(value) for i, value in enumerate(values)}}


def _validate_owned_samples(
    stores: Mapping[Node, Mapping[Hashable, float]],
) -> tuple[int, set[Hashable]]:
    seen: set[Hashable] = set()
    count = 0
    for owner, store in stores.items():
        if not isinstance(store, Mapping):
            raise TypeError(f"local store for owner {owner!r} must be a mapping")
        for tag, raw_value in store.items():
            if tag in seen:
                raise ValueError(f"every scalar value must have exactly one owner: {tag!r}")
            seen.add(tag)
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError("finite order-statistic recovery requires finite values")
            count += 1
    if count == 0:
        raise ValueError("at least one scalar value is required")
    return count, seen


def flood_tagged(
    graph: Graph,
    local_values: Mapping[Node, Mapping[Hashable, object]],
) -> tuple[dict[Node, dict[Hashable, object]], int]:
    """Flood uniquely tagged values for exactly the graph diameter.

    The local packets contain one entry per owner/tag.  A repeated tag is an
    ownership error even when the repeated values happen to be equal; this is
    what prevents a duplicated robot score from silently changing an order
    statistic.  Every returned node has the same complete tagged map.
    """

    canonical = validate_graph(graph)
    nodes = tuple(sorted(canonical, key=_sort_key))
    if set(local_values) - set(nodes):
        raise ValueError("local flood values contain an owner outside the graph")
    stores = {node: dict(local_values.get(node, {})) for node in nodes}
    owners: dict[Hashable, Node] = {}
    for owner in nodes:
        for tag in stores[owner]:
            if tag in owners:
                raise ValueError(f"tag {tag!r} has multiple owners")
            owners[tag] = owner
    diameter = graph_diameter(canonical)
    for _ in range(diameter):
        previous = {node: dict(stores[node]) for node in nodes}
        for node in nodes:
            for neighbor in canonical[node]:
                for tag, value in previous[neighbor].items():
                    if tag in stores[node]:
                        if not _equal_values(stores[node][tag], value):
                            raise ValueError(f"conflicting values for tag {tag!r}")
                    else:
                        stores[node][tag] = value
    expected = set(owners)
    if any(set(stores[node]) != expected for node in nodes):
        raise AssertionError("tagged flooding did not deliver every unique tag")
    return stores, diameter


tagged_flood = flood_tagged


def pinball_gradient(data: Iterable[float] | Mapping[Hashable, float], q: float, tau: float) -> float:
    """Return the strict-subgradient used by the paper's pinball iteration."""

    if not np.isfinite(q) or not np.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise ValueError("q and tau must be finite with tau in [0, 1]")
    values = data.values() if isinstance(data, Mapping) else data
    gradient = 0.0
    for raw_value in values:
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError("pinball values must be finite")
        gradient += float(value < q) - tau
    return float(gradient)


def local_pinball_step(
    data: Iterable[float] | Mapping[Hashable, float],
    q: float,
    mixed: float,
    tau: float,
    gamma: float,
) -> float:
    """Perform one local pinball update after neighbor mixing.

    ``data`` is only the caller's local store.  The function does not inspect
    or sort any other node's values, making it suitable for a ROS worker.
    """

    if not np.isfinite(mixed) or not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("mixed iterate must be finite and gamma must be positive")
    value = float(mixed - gamma * pinball_gradient(data, q, tau))
    if not np.isfinite(value):
        raise ValueError("pinball update produced a nonfinite iterate")
    return value


pinball_step = local_pinball_step


def _flood_summary(
    graph: Mapping[Node, set[Node]],
    summaries: Mapping[Node, object],
) -> tuple[dict[Node, object], int]:
    """Flood one tagged summary per graph owner and return one common view."""

    local = {node: {("summary", node): summaries[node]} for node in graph}
    flooded, rounds = flood_tagged(graph, local)
    reference = next(iter(flooded))
    return {
        tag[1]: value
        for tag, value in flooded[reference].items()
        if isinstance(tag, tuple) and len(tag) == 2 and tag[0] == "summary"
    }, rounds


def _recover_finite_upper_bound(
    graph: Mapping[Node, set[Node]],
    stores: Mapping[Node, Mapping[Hashable, float]],
    rank: int,
    q: Mapping[Node, float],
) -> tuple[float, int, int]:
    """Apply the finite tagged recovery procedure from the paper."""

    local_max = {
        node: max(stores[node].values(), default=float("-inf"))
        for node in graph
    }
    max_summaries, rounds = _flood_summary(
        graph,
        {node: (local_max[node], q[node]) for node in graph},
    )
    # ``_flood_summary`` returns a node -> summary mapping.  Aggregate the
    # tuple values here rather than asking any node to inspect raw remote data.
    maxima = list(max_summaries.values())
    a_max = max(float(summary[0]) for summary in maxima)
    q_max = max(float(summary[1]) for summary in maxima)
    y = min(a_max, q_max)
    recovery_steps = 0

    def flooded_counts(candidate: float) -> tuple[int, float, int]:
        local_summary = {}
        for node in graph:
            values = stores[node].values()
            count = sum(float(value) <= candidate for value in values)
            successor = min(
                (float(value) for value in values if float(value) > candidate),
                default=float("inf"),
            )
            local_summary[node] = (int(count), float(successor))
        summaries, flood_rounds = _flood_summary(graph, local_summary)
        count = sum(int(summary[0]) for summary in summaries.values())
        successor = min(float(summary[1]) for summary in summaries.values())
        return count, successor, flood_rounds

    count, successor, flood_rounds = flooded_counts(y)
    rounds += flood_rounds
    while count < rank:
        if not np.isfinite(successor):
            raise RuntimeError("finite recovery could not find the requested rank")
        y = successor
        recovery_steps += 1
        count, successor, flood_rounds = flooded_counts(y)
        rounds += flood_rounds
    return float(y), rounds, recovery_steps


def pinball_order_statistic(
    local_values: Mapping[Node, object] | Sequence[float] | np.ndarray,
    rank: int,
    iterations: int,
    q_init: float = 0.0,
    *,
    graph: Graph | None = None,
    gamma_0: float = 0.05,
    exponent: float = 0.75,
) -> OrderStatisticResult:
    """Compute a finite upper bound on a distributed order statistic.

    ``local_values`` maps graph owners to uniquely tagged scalar stores.  Empty
    stores are valid.  If no graph is supplied, the owners form a complete
    graph (or a one-node graph for a sequence), which is convenient for pure
    numerical tests.  The returned ``value`` is obtained by finite recovery,
    so it is at least the exact rank even when the pinball iterations have not
    converged.
    """

    if not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if not np.isfinite(q_init):
        raise ValueError("q_init must be finite")
    if not np.isfinite(gamma_0) or gamma_0 <= 0.0:
        raise ValueError("gamma_0 must be finite and positive")
    if not np.isfinite(exponent) or not 0.5 < exponent <= 1.0:
        raise ValueError("exponent must lie in (0.5, 1]")

    if graph is None:
        if isinstance(local_values, Mapping):
            raw_nodes = tuple(sorted(local_values, key=_sort_key))
            if not raw_nodes:
                raise ValueError("at least one graph owner is required")
            graph = {
                node: set(raw_nodes) - {node}
                for node in raw_nodes
            }
        else:
            graph = {"local": set()}
    canonical = validate_graph(graph)
    stores = _normalise_local_values(local_values, canonical)
    sample_count, tags = _validate_owned_samples(stores)
    if not isinstance(rank, int) or not 1 <= rank <= sample_count:
        raise ValueError("rank is outside the owned value set")

    nodes, mixing = metropolis_weights(canonical)
    index = {node: i for i, node in enumerate(nodes)}
    q = {node: float(q_init) for node in nodes}
    history = [np.asarray([q[node] for node in nodes], dtype=float)]
    tau = (rank - 0.5) / sample_count
    for iteration in range(iterations):
        next_q: dict[Node, float] = {}
        for node in nodes:
            row = index[node]
            mixed = float(mixing[row, row] * q[node])
            for neighbor in canonical[node]:
                mixed += float(mixing[row, index[neighbor]] * q[neighbor])
            step = gamma_0 / ((iteration + 1.0) ** exponent)
            next_q[node] = local_pinball_step(stores[node], q[node], mixed, tau, step)
        q = next_q
        history.append(np.asarray([q[node] for node in nodes], dtype=float))

    value, flooding_rounds, recovery_steps = _recover_finite_upper_bound(
        canonical, stores, rank, q
    )
    owners = {
        node: tuple(sorted(stores[node].keys(), key=_sort_key))
        for node in nodes
    }
    return OrderStatisticResult(
        value=value,
        rank=rank,
        sample_count=sample_count,
        exact_value=None,
        pinball_history=np.asarray(history, dtype=float),
        pinball_rounds=iterations,
        flooding_rounds=flooding_rounds,
        recovery_steps=recovery_steps,
        owners=owners,
        certified=True,
    )


def _flatten_values(values: object) -> tuple[float, ...]:
    """Flatten common centralized or owner-store representations."""

    if isinstance(values, Mapping):
        if not values:
            return ()
        raw_values = list(values.values())
        if all(np.isscalar(item) for item in raw_values):
            result = tuple(float(item) for item in raw_values)
        else:
            seen: set[Hashable] = set()
            flattened: list[float] = []
            for owner, local in values.items():
                if isinstance(local, Mapping):
                    items = local.items()
                else:
                    try:
                        items = (((owner, i), item) for i, item in enumerate(local))
                    except TypeError as exc:
                        raise TypeError("nested score stores must be mappings or sequences") from exc
                for tag, value in items:
                    if tag in seen:
                        raise ValueError(f"every scalar value must have exactly one owner: {tag!r}")
                    seen.add(tag)
                    flattened.append(float(value))
            result = tuple(flattened)
    elif isinstance(values, (str, bytes)):
        raise TypeError("scores must be numeric sequences or mappings")
    else:
        try:
            result = tuple(float(value) for value in values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("scores must be numeric sequences or mappings") from exc
    if any(not np.isfinite(value) for value in result):
        raise ValueError("order-statistic scores must be finite")
    return result


def exact_order_statistic(values: object, rank: int) -> float:
    """Return the exact centralized order statistic, including ties."""

    samples = _flatten_values(values)
    if not samples:
        raise ValueError("at least one value is required")
    if not isinstance(rank, int) or not 1 <= rank <= len(samples):
        raise ValueError("rank is outside the value set")
    return float(sorted(samples)[rank - 1])


centralized_order_statistic = exact_order_statistic
centralized_exact_order_statistic = exact_order_statistic


def certified_order_statistic(
    graph: Graph,
    local_values: Mapping[Node, object] | Sequence[float] | np.ndarray,
    rank: int,
    iterations: int,
    q_init: float = 0.0,
    gamma_0: float = 0.05,
    exponent: float = 0.75,
) -> OrderStatisticResult:
    """Reference-compatible graph-first spelling of the distributed helper."""

    return pinball_order_statistic(
        local_values,
        rank,
        iterations,
        q_init=q_init,
        graph=graph,
        gamma_0=gamma_0,
        exponent=exponent,
    )


distributed_order_statistic = certified_order_statistic


def mission_score(scores: object, rank: int) -> float:
    """Aggregate robot-level scores into the class mission score."""

    return exact_order_statistic(scores, rank)


centralized_mission_score = mission_score


def standard_split_cp(scores: object, alpha: float) -> tuple[float, int]:
    """One-batch split conformal threshold and its rank.

    The ``M+1`` rank is intentional.  When it exceeds ``M`` the finite-sample
    split-CP threshold is ``+inf`` rather than a clipped sample value.
    """

    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    values = _flatten_values(scores)
    if not values:
        raise ValueError("at least one calibration score is required")
    rank = int(ceil((len(values) + 1) * (1.0 - alpha)))
    if rank > len(values):
        return float("inf"), rank
    return exact_order_statistic(values, rank), rank


def split_conformal_threshold(scores: object, alpha: float) -> float:
    """Return only the one-batch split-CP threshold."""

    return standard_split_cp(scores, alpha)[0]


def normalized_margin_distance(
    candidate: Mapping[Hashable, float],
    current: Mapping[Hashable, float],
    classes: Mapping[Hashable, ClassConfig] | MAICPConfig,
) -> float:
    """Compute the paper's scale-normalized Euclidean margin distance."""

    if isinstance(classes, MAICPConfig):
        classes = classes.classes
    keys = tuple(classes)
    return float(
        np.linalg.norm(
            [
                (float(candidate[key]) - float(current[key])) / classes[key].scale
                for key in keys
            ]
        )
    )


def _resolve_class_mapping(
    config_or_classes: MAICPConfig | Mapping[Hashable, ClassConfig],
) -> Mapping[Hashable, ClassConfig]:
    if isinstance(config_or_classes, MAICPConfig):
        return config_or_classes.classes
    if not isinstance(config_or_classes, Mapping) or not config_or_classes:
        raise ValueError("classes must be a nonempty mapping")
    return config_or_classes


def _mapping_for_keys(
    values: Mapping[object, float] | None,
    keys: Sequence[Hashable],
    classes: Mapping[Hashable, ClassConfig],
    *,
    default: Callable[[Hashable, ClassConfig], float] | None = None,
) -> dict[Hashable, float]:
    result: dict[Hashable, float] = {}
    values = values or {}
    for key in keys:
        matched: object | None = key if key in values else None
        if matched is None:
            cls_name = classes[key].name
            for candidate in values:
                if str(getattr(candidate, "value", candidate)).lower() == cls_name.lower():
                    matched = candidate
                    break
        if matched is None:
            if default is None:
                raise ValueError(f"missing value for class {key!r}")
            result[key] = float(default(key, classes[key]))
        else:
            result[key] = float(values[matched])
    return result


def solve_margin_update(
    current: Mapping[Hashable, float],
    thresholds: Mapping[Hashable, float],
    kappa: Mapping[Hashable, float],
    config_or_classes: MAICPConfig | Mapping[Hashable, ClassConfig] | None = None,
    proximal_weight: float | None = None,
    sensitivity_certified: bool = False,
) -> MarginUpdateResult:
    """Solve the coupled SOCP from the paper.

    The constraints are ``q_c + kappa_c * ||(r-r_prev)/scale|| <= r_c`` and
    the objective is ``sum(w_c r_c) + lambda/2 * ||...||^2``.  Invalid inputs,
    missing cvxpy, solver errors, infeasibility, and residual violations raise
    explicitly; no candidate is clipped or silently held.
    """

    if config_or_classes is None:
        config = paper_config()
        classes = config.classes
    elif isinstance(config_or_classes, MAICPConfig):
        config = config_or_classes
        config.validate()
        classes = config.classes
    else:
        config = None
        classes = _resolve_class_mapping(config_or_classes)
    keys = tuple(classes)
    if set(current) != set(keys) or set(thresholds) != set(keys) or set(kappa) != set(keys):
        raise ValueError("current, thresholds, and kappa must contain exactly the configured classes")
    if proximal_weight is None:
        lam = float(config.proximal_weight if config is not None else 0.25)
    else:
        lam = float(proximal_weight)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("proximal_weight must be finite and positive")

    previous = np.asarray([float(current[key]) for key in keys], dtype=float)
    q = np.asarray([float(thresholds[key]) for key in keys], dtype=float)
    kappas = np.asarray([float(kappa[key]) for key in keys], dtype=float)
    scales = np.asarray([float(classes[key].scale) for key in keys], dtype=float)
    weights = np.asarray([float(classes[key].weight) for key in keys], dtype=float)
    lower = np.asarray([float(classes[key].margin_min) for key in keys], dtype=float)
    upper = np.asarray([float(classes[key].margin_max) for key in keys], dtype=float)
    if (
        np.any(~np.isfinite(previous))
        or np.any(~np.isfinite(q))
        or np.any(~np.isfinite(kappas))
        or np.any(kappas < 0.0)
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
        or np.any(~np.isfinite(lower))
        or np.any(~np.isfinite(upper))
        or np.any(lower > upper)
        or np.any(previous < lower)
        or np.any(previous > upper)
    ):
        raise ValueError("margin update contains nonfinite, negative, or out-of-range data")
    if cp is None:
        raise RuntimeError("cvxpy is required for the coupled margin update")

    margin = cp.Variable(len(keys))
    scaled_change = cp.multiply(1.0 / scales, margin - previous)
    delta = cp.norm(scaled_change, 2)
    constraints = [margin >= lower, margin <= upper, q + kappas * delta <= margin]
    objective = cp.Minimize(weights @ margin + 0.5 * lam * cp.sum_squares(scaled_change))
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver="CLARABEL", verbose=False)
    except Exception as exc:  # cvxpy exposes several solver-specific exceptions.
        raise RuntimeError(f"margin update solver failed: {exc}") from exc
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or margin.value is None:
        raise RuntimeError(f"margin update is infeasible or unsolved: {problem.status}")

    candidate_array = np.asarray(margin.value, dtype=float).reshape(-1)
    if candidate_array.shape != previous.shape or np.any(~np.isfinite(candidate_array)):
        raise RuntimeError("margin update solver returned a nonfinite candidate")
    change = float(np.linalg.norm((candidate_array - previous) / scales))
    slacks_array = candidate_array - q - kappas * change
    tolerance = 2e-6
    if (
        np.any(candidate_array < lower - tolerance)
        or np.any(candidate_array > upper + tolerance)
        or np.any(slacks_array < -tolerance)
        or not np.isfinite(change)
    ):
        raise RuntimeError("margin update solver candidate violates its constraints")
    candidate = {key: float(candidate_array[i]) for i, key in enumerate(keys)}
    slacks = {key: float(slacks_array[i]) for i, key in enumerate(keys)}
    return MarginUpdateResult(
        margins=candidate,
        previous_margins={key: float(current[key]) for key in keys},
        thresholds={key: float(thresholds[key]) for key in keys},
        kappa={key: float(kappa[key]) for key in keys},
        normalized_change=change,
        objective=float(problem.value),
        constraint_slacks=slacks,
        status="optimal",
        certified=bool(sensitivity_certified),
    )


solve_margin_update_replica = solve_margin_update


def _class_keys(config: MAICPConfig) -> tuple[Hashable, ...]:
    return tuple(config.classes)


def _class_key_for(value: object, config: MAICPConfig) -> Hashable:
    if value in config.classes:
        return value  # type: ignore[return-value]
    text = str(getattr(value, "value", value)).lower()
    for key, cls in config.classes.items():
        if str(getattr(key, "value", key)).lower() == text or cls.name.lower() == text:
            return key
    raise KeyError(f"unknown class {value!r}")


def _current_margins(
    current: Mapping[object, float] | None,
    config: MAICPConfig,
) -> dict[Hashable, float]:
    if current is None:
        return {key: float(config.classes[key].initial_margin) for key in _class_keys(config)}
    result: dict[Hashable, float] = {}
    for key in _class_keys(config):
        candidates = [key, config.classes[key].name]
        found = next((candidate for candidate in candidates if candidate in current), None)
        if found is None:
            raise ValueError(f"current margins omit class {config.classes[key].name!r}")
        value = float(current[found])
        if not np.isfinite(value):
            raise ValueError("current margins must be finite")
        result[key] = value
    return result


def _normalise_mission_layout(
    scores_by_mission: object,
    config: MAICPConfig,
) -> dict[Hashable, tuple[object, ...]]:
    """Accept class-major or mission-major score data without copying values."""

    keys = _class_keys(config)
    if isinstance(scores_by_mission, Mapping):
        class_entries: dict[Hashable, object] = {}
        all_class_keys = True
        for raw_key, value in scores_by_mission.items():
            try:
                key = _class_key_for(raw_key, config)
            except KeyError:
                all_class_keys = False
                break
            class_entries[key] = value
        if all_class_keys and class_entries:
            if set(class_entries) != set(keys):
                raise ValueError("scores_by_mission must provide every configured class")
            result: dict[Hashable, tuple[object, ...]] = {}
            for key in keys:
                entries = class_entries[key]
                if isinstance(entries, (str, bytes)) or not isinstance(entries, Iterable):
                    raise TypeError(f"scores for class {key!r} must be an iterable of missions")
                result[key] = tuple(entries)
            counts = {len(values) for values in result.values()}
            if len(counts) != 1 or not counts or next(iter(counts)) == 0:
                raise ValueError("all classes must contain the same nonempty mission batch")
            return result
    if isinstance(scores_by_mission, (str, bytes)) or not isinstance(scores_by_mission, Iterable):
        raise TypeError("scores_by_mission must be class-major or mission-major data")
    missions = tuple(scores_by_mission)
    if not missions:
        raise ValueError("calibration data must contain at least one mission")
    result_lists: dict[Hashable, list[object]] = {key: [] for key in keys}
    for mission_index, mission in enumerate(missions):
        if not isinstance(mission, Mapping):
            raise TypeError(f"mission {mission_index} must map class names to robot scores")
        by_class: dict[Hashable, object] = {}
        for raw_key, value in mission.items():
            by_class[_class_key_for(raw_key, config)] = value
        if set(by_class) != set(keys):
            raise ValueError(f"mission {mission_index} does not provide every class")
        for key in keys:
            result_lists[key].append(by_class[key])
    return {key: tuple(values) for key, values in result_lists.items()}


def _mission_store(
    raw: object,
    graph: Mapping[Node, set[Node]],
) -> dict[Node, dict[Hashable, float]]:
    """Build unique owner/tag stores from one mission's raw robot scores."""

    nodes = tuple(sorted(graph, key=_sort_key))
    stores: dict[Node, dict[Hashable, float]] = {node: {} for node in nodes}

    def assign(index: int, tag: Hashable, value: object) -> None:
        node = nodes[index % len(nodes)]
        stores[node][tag] = float(value)

    if np.isscalar(raw):
        assign(0, "score:0", raw)
        return stores
    if isinstance(raw, Mapping):
        if not raw:
            raise ValueError("a mission must contain at least one robot score")
        values = list(raw.values())
        if all(np.isscalar(value) for value in values):
            unknown_robot_ids = set(raw) - set(stores)
            if unknown_robot_ids and len(stores) > 1:
                raise ValueError(
                    "robot score owners lie outside the graph: "
                    f"{unknown_robot_ids!r}"
                )
            for index, (robot_id, value) in enumerate(raw.items()):
                # A robot-id keyed mission already carries the physical
                # owner.  Preserve it when that id is a graph node so a ROS
                # worker receives its own private scores.  Synthetic ids use
                # deterministic round-robin ownership.
                target = robot_id if robot_id in stores else nodes[index % len(nodes)]
                tag = ("robot", robot_id)
                if tag in stores[target]:
                    raise ValueError(f"duplicate robot-score tag {tag!r}")
                stores[target][tag] = float(value)
            return stores
        index = 0
        for owner, local in raw.items():
            if owner not in stores and len(stores) > 1:
                raise ValueError(f"score owner {owner!r} lies outside the update graph")
            if isinstance(local, Mapping):
                items = local.items()
            else:
                try:
                    items = (((owner, i), value) for i, value in enumerate(local))
                except TypeError as exc:
                    raise TypeError("owner stores must be mappings or sequences") from exc
            # If owner names are graph nodes, preserve them.  Otherwise use a
            # deterministic round-robin owner and retain the source tag.
            target = owner if owner in stores else nodes[index % len(nodes)]
            for local_tag, value in items:
                if local_tag in stores[target]:
                    raise ValueError(f"duplicate robot-score tag {local_tag!r}")
                stores[target][local_tag] = float(value)
                index += 1
        return stores
    try:
        values = list(raw)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("mission robot scores must be scalar, sequence, or mapping") from exc
    for index, value in enumerate(values):
        if not np.isscalar(value):
            raise TypeError("mission sequences must contain scalar robot scores")
        assign(index, ("robot", index), value)
    if not values:
        raise ValueError("a mission must contain at least one robot score")
    return stores


def _invoke_order_statistic(
    callback: Callable[..., object],
    stores: Mapping[Node, Mapping[Hashable, float]],
    rank: int,
    iterations: int,
    config: MAICPConfig,
) -> float:
    """Invoke the documented local/ROS order-statistic callback."""

    # Inspect the signature so a TypeError raised *inside* a callback is not
    # mistaken for an old calling convention and silently retried.
    try:
        parameters = tuple(signature(callback).parameters.values())
        positional = tuple(
            parameter
            for parameter in parameters
            if parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        )
        accepts_varargs = any(
            parameter.kind is parameter.VAR_POSITIONAL for parameter in parameters
        )
    except (TypeError, ValueError):
        positional = ()
        accepts_varargs = True
    if not accepts_varargs and len(positional) < 4:
        raise TypeError(
            "order_statistic callback must accept "
            "(stores, rank, iterations, config)"
        )
    result = callback(stores, rank, iterations, config)
    value = getattr(result, "value", result)
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("order-statistic callback returned a nonfinite bound")
    return value


def _invoke_margin_update(
    callback: Callable[..., object],
    current: Mapping[Hashable, float],
    thresholds: Mapping[Hashable, float],
    kappa: Mapping[Hashable, float],
    config: MAICPConfig,
) -> MarginUpdateResult:
    """Invoke and validate the replicated margin-update callback.

    The callback contract is deliberately small so the ROS adapter can send
    only common summaries and configuration: ``callback(current, thresholds,
    kappa, config)``.  This is the only supported callback shape.
    """

    # A TypeError from the callback itself intentionally propagates.
    result = callback(current, thresholds, kappa, config)
    if not isinstance(result, MarginUpdateResult):
        raise TypeError("margin_update callback must return MarginUpdateResult")
    keys = set(config.classes)
    for name, values in (
        ("margins", result.margins),
        ("previous_margins", result.previous_margins),
        ("thresholds", result.thresholds),
        ("kappa", result.kappa),
        ("constraint_slacks", result.constraint_slacks),
    ):
        if set(values) != keys:
            raise ValueError(
                f"margin_update callback result {name} must contain exactly the configured classes"
            )
        if any(not np.isfinite(float(value)) for value in values.values()):
            raise ValueError(f"margin_update callback result {name} contains nonfinite values")
    if not np.isfinite(float(result.normalized_change)) or not np.isfinite(float(result.objective)):
        raise ValueError("margin_update callback returned nonfinite diagnostics")
    if not isinstance(result.replica_count, int) or result.replica_count < 1:
        raise ValueError("margin_update callback replica_count must be a positive integer")
    return result


def _run_order_statistic(
    stores: Mapping[Node, Mapping[Hashable, float]],
    rank: int,
    iterations: int,
    config: MAICPConfig,
    graph: Graph | None,
    callback: Callable[..., object] | None,
) -> float:
    if callback is not None:
        return _invoke_order_statistic(callback, stores, rank, iterations, config)
    if graph is None:
        owners = tuple(sorted(stores, key=_sort_key))
        graph = {owner: set(owners) - {owner} for owner in owners}
    result = pinball_order_statistic(
        stores,
        rank,
        iterations,
        q_init=0.0,
        graph=graph,
        gamma_0=config.pinball_gamma_0,
        exponent=config.pinball_exponent,
    )
    return float(result.value)


def calibrate_round(
    method: str,
    current: Mapping[object, float] | None,
    scores_by_mission: object,
    config: MAICPConfig | None = None,
    *,
    graph: Graph | None = None,
    kappa: Mapping[object, float] | None = None,
    order_statistic: Callable[..., object] | None = None,
    margin_update: Callable[..., object] | None = None,
    sensitivity_certified: bool = False,
) -> CalibrationResult:
    """Run one of the five paper comparison methods.

    ``scores_by_mission`` is independent calibration data in either
    class-major form ``{class: [mission_scores...]}`` or mission-major form
    ``[{class: robot_scores}, ...]``.  A mission entry can be a robot-score
    sequence, a robot-id-to-score mapping, or owner-local tagged stores.

    ``order_statistic`` may replace the in-memory pinball primitive.  It is
    called as ``callback(stores, rank, iterations, config)`` and may return an
    ``OrderStatisticResult`` or a scalar upper bound.  This is the extension
    point used by ROS workers that exchange the same local data over topics.
    ``margin_update`` is called as ``callback(current, thresholds, kappa,
    config)`` for ``maicp`` and ``unshifted`` only.  It must return a
    ``MarginUpdateResult``; centralized calibration always solves its own
    exact local SOCP even when this callback is supplied.
    ``sensitivity_certified`` is accepted for integration symmetry, but this
    core does not compute the paper's kappa certificate, so configured kappa
    values always produce ``certified=False``.
    """

    config = paper_config() if config is None else config
    config.validate()
    canonical_method = str(method).lower().replace("-", "_")
    aliases = {
        "distributed": "maicp",
        "adaptive": "maicp",
        "oracle": "centralized",
        "centralised": "centralized",
        "fixed": "fixed_margin",
        "fixedmargin": "fixed_margin",
    }
    canonical_method = aliases.get(canonical_method, canonical_method)
    allowed = {"maicp", "centralized", "unshifted", "nominal", "fixed_margin"}
    if canonical_method not in allowed:
        raise ValueError(f"unknown calibration method {method!r}; expected one of {sorted(allowed)}")

    classes = config.classes
    keys = _class_keys(config)
    current_margins = _current_margins(current, config)
    layout = _normalise_mission_layout(scores_by_mission, config)
    mission_count = len(next(iter(layout.values())))
    if mission_count != config.calibration_missions:
        raise ValueError(
            f"expected {config.calibration_missions} independent calibration missions, "
            f"received {mission_count}"
        )
    exact_mission: dict[Hashable, tuple[float, ...]] = {}
    distributed_mission: dict[Hashable, tuple[float, ...]] = {}
    for key in keys:
        cls = classes[key]
        raw_missions = layout[key]
        exact_values: list[float] = []
        certified_values: list[float] = []
        for raw in raw_missions:
            raw_count = len(_flatten_values(raw))
            if raw_count != cls.n:
                raise ValueError(
                    f"class {cls.name!r} missions must contain exactly {cls.n} "
                    f"robot scores; received {raw_count}"
                )
            exact_values.append(mission_score(raw, cls.mission_rank))
            if canonical_method in {"maicp", "unshifted"}:
                stores = _mission_store(raw, _graph_for_data(graph, raw))
                certified_values.append(
                    _run_order_statistic(
                        stores,
                        cls.mission_rank,
                        config.K_S,
                        config,
                        graph,
                        order_statistic,
                    )
                )
            else:
                certified_values.append(exact_values[-1])
        exact_mission[key] = tuple(exact_values)
        distributed_mission[key] = tuple(certified_values)

    exact_thresholds: dict[Hashable, float] = {}
    thresholds: dict[Hashable, float] = {}
    ranks: dict[Hashable, int] = {}
    for key in keys:
        cls = classes[key]
        if canonical_method == "fixed_margin":
            threshold, rank = standard_split_cp(exact_mission[key], cls.alpha)
            exact_thresholds[key] = threshold
            thresholds[key] = threshold
            ranks[key] = rank
        else:
            rank = cls.paper_rank(mission_count)
            ranks[key] = rank
            exact_thresholds[key] = exact_order_statistic(exact_mission[key], rank)
            if canonical_method in {"maicp", "unshifted"}:
                # Mission bounds are themselves distributed scalar samples.
                owner_nodes = _graph_for_data(graph, distributed_mission[key])
                bound_stores: dict[Node, dict[Hashable, float]] = {
                    node: {} for node in owner_nodes
                }
                ordered_nodes = tuple(sorted(owner_nodes, key=_sort_key))
                for index, value in enumerate(distributed_mission[key]):
                    owner = ordered_nodes[index % len(ordered_nodes)]
                    bound_stores[owner][("mission", index)] = float(value)
                thresholds[key] = _run_order_statistic(
                    bound_stores,
                    rank,
                    config.K_q,
                    config,
                    graph,
                    order_statistic,
                )
            else:
                thresholds[key] = exact_thresholds[key]

    if canonical_method == "nominal":
        margins = {key: 0.0 for key in keys}
        return CalibrationResult(
            method=canonical_method,
            margins=margins,
            previous_margins=current_margins,
            mission_scores=exact_mission,
            certified_mission_scores=distributed_mission,
            exact_thresholds=exact_thresholds,
            thresholds=thresholds,
            conformal_ranks=ranks,
            kappa={key: 0.0 for key in keys},
            status="nominal",
            certified=False,
            margin_update=None,
        )

    if canonical_method == "fixed_margin":
        # +inf is a valid split-CP result but cannot be silently forced through
        # a finite SOCP.  The fixed baseline exposes it directly to the caller.
        margins = dict(thresholds)
        return CalibrationResult(
            method=canonical_method,
            margins=margins,
            previous_margins=current_margins,
            mission_scores=exact_mission,
            certified_mission_scores=distributed_mission,
            exact_thresholds=exact_thresholds,
            thresholds=thresholds,
            conformal_ranks=ranks,
            kappa={key: 0.0 for key in keys},
            status="fixed",
            certified=False,
            margin_update=None,
        )

    if kappa is None:
        selected_kappa = {
            key: (0.0 if canonical_method == "unshifted" else classes[key].configured_kappa)
            for key in keys
        }
    else:
        selected_kappa = _mapping_for_keys(kappa, keys, classes)
        if canonical_method == "unshifted":
            selected_kappa = {key: 0.0 for key in keys}
    if margin_update is not None and canonical_method in {"maicp", "unshifted"}:
        update = _invoke_margin_update(
            margin_update,
            current_margins,
            thresholds,
            selected_kappa,
            config,
        )
    else:
        update = solve_margin_update(
            current_margins,
            thresholds,
            selected_kappa,
            config,
            # The paper's sensitivity certificate is not computed by this core;
            # configured kappa values therefore never turn into a formal claim,
            # even when an integration caller passes a legacy boolean flag.
            sensitivity_certified=False,
        )
    # A configured kappa is intentionally not a formal sensitivity certificate.
    certified = False
    return CalibrationResult(
        method=canonical_method,
        margins=dict(update.margins),
        previous_margins=current_margins,
        mission_scores=exact_mission,
        certified_mission_scores=distributed_mission,
        exact_thresholds=exact_thresholds,
        thresholds=thresholds,
        conformal_ranks=ranks,
        kappa=selected_kappa,
        status=update.status,
        certified=certified,
        margin_update=update,
    )


def _graph_for_data(graph: Graph | None, data: object) -> dict[Node, set[Node]]:
    if graph is not None:
        return validate_graph(graph)
    if isinstance(data, Mapping) and data and all(
        not np.isscalar(value) for value in data.values()
    ):
        owners = tuple(sorted(data, key=_sort_key))
    else:
        owners = ("local",)
    if not owners:
        owners = ("local",)
    return {owner: set(owners) - {owner} for owner in owners}


__all__ = [
    "CalibrationResult",
    "MarginUpdateResult",
    "OrderStatisticResult",
    "centralized_exact_order_statistic",
    "centralized_mission_score",
    "centralized_order_statistic",
    "calibrate_round",
    "certified_order_statistic",
    "distributed_order_statistic",
    "exact_order_statistic",
    "flood_tagged",
    "graph_diameter",
    "local_pinball_step",
    "mission_score",
    "metropolis_matrix",
    "metropolis_weights",
    "normalized_margin_distance",
    "pinball_gradient",
    "pinball_order_statistic",
    "pinball_step",
    "reachable_nodes",
    "solve_margin_update",
    "solve_margin_update_replica",
    "split_conformal_threshold",
    "standard_split_cp",
    "tagged_flood",
    "validate_graph",
]
