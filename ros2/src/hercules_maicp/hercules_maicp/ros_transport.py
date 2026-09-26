"""Neighbor-only ROS pinball iterations and tagged finite-time flooding.

The experiment driver schedules jobs and delivers each robot's private store.
During a job only iterates and tagged scalar summaries cross robot boundaries.
Raw samples are never published to neighbors. Nodes return the same upper bound.
"""
from __future__ import annotations

import json
import math
import subprocess
import threading
import time
import uuid
from dataclasses import asdict

from .config import ClassConfig
from .core import MarginUpdateResult, graph_diameter, metropolis_weights, solve_margin_update


_MARGIN_TOLERANCE = 2e-6


def _class_payload(config):
    """Serialize only the class fields needed by a replicated SOCP solve."""

    classes = {}
    for key, cls in config.classes.items():
        if not isinstance(cls, ClassConfig):
            raise TypeError(f"class {key!r} is not a ClassConfig")
        name = str(getattr(key, "value", key))
        if name in classes:
            raise ValueError(f"duplicate serialized class key {name!r}")
        classes[name] = asdict(cls)
    if not classes:
        raise ValueError("margin update requires at least one configured class")
    return {"classes": classes, "proximal_weight": float(config.proximal_weight)}


def _classes_from_payload(payload):
    raw = payload.get("classes")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("margin update job has no class configuration")
    classes = {}
    for key, fields in raw.items():
        if not isinstance(fields, dict):
            raise TypeError(f"class payload for {key!r} must be a mapping")
        classes[str(key)] = ClassConfig(**fields)
    return classes


def _margin_result_payload(result):
    return asdict(result)


def _margin_result_from_packet(packet):
    payload = dict(packet)
    for name in ("job", "owner", "kind"):
        payload.pop(name, None)
    return MarginUpdateResult(**payload)


def _margin_result_is_finite(result):
    values = (
        *result.margins.values(),
        *result.previous_margins.values(),
        *result.thresholds.values(),
        *result.kappa.values(),
        *result.constraint_slacks.values(),
        result.normalized_change,
        result.objective,
    )
    return (
        all(math.isfinite(float(value)) for value in values)
        and isinstance(result.replica_count, int)
        and result.replica_count >= 1
    )


def _numeric_maps_agree(left, right, tolerance=_MARGIN_TOLERANCE):
    if set(left) != set(right):
        return False
    return all(abs(float(left[key]) - float(right[key])) <= tolerance for key in left)


def _margin_results_agree(left, right, tolerance=_MARGIN_TOLERANCE):
    return (
        _margin_result_is_finite(left)
        and _margin_result_is_finite(right)
        and
        _numeric_maps_agree(left.margins, right.margins, tolerance)
        and _numeric_maps_agree(left.previous_margins, right.previous_margins, tolerance)
        and _numeric_maps_agree(left.thresholds, right.thresholds, tolerance)
        and _numeric_maps_agree(left.kappa, right.kappa, tolerance)
        and _numeric_maps_agree(left.constraint_slacks, right.constraint_slacks, tolerance)
        and abs(left.normalized_change - right.normalized_change) <= tolerance
        and abs(left.objective - right.objective) <= tolerance
        and left.status == right.status
        and left.certified == right.certified
    )


def _ros():
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    return rclpy, Node, String


def _worker_type():
    rclpy, Node, String = _ros()

    class Worker(Node):
        def __init__(self):
            super().__init__('maicp_worker')
            self.agent = str(self.declare_parameter('agent_id', 'Drone1').value)
            self.condition = threading.Condition()
            self.inbox = {}
            self.busy = False
            self.peer_pub = self.create_publisher(String, '/hercules_maicp/exchange', 1000)
            self.result_pub = self.create_publisher(String, '/hercules_maicp/results', 100)
            self.create_subscription(String, '/hercules_maicp/exchange', self.receive, 1000)
            self.create_subscription(String, '/hercules_maicp/jobs/' + self.agent, self.start, 10)

        def receive(self, msg):
            packet = json.loads(msg.data)
            key = (packet['job'], packet['stage'], packet['round'])
            with self.condition:
                self.inbox.setdefault(key, {})[packet['owner']] = packet['value']
                self.condition.notify_all()

        def start(self, msg):
            job = json.loads(msg.data)
            if self.busy:
                self.result_pub.publish(String(data=json.dumps({
                    'job': job['job'], 'owner': self.agent, 'error': 'worker busy'})))
                return
            self.busy = True
            threading.Thread(target=self.run_job, args=(job,), daemon=True).start()

        def exchange(self, job, stage, step, value):
            packet = dict(job=job['job'], stage=stage, round=step, owner=self.agent, value=value)
            self.peer_pub.publish(String(data=json.dumps(packet)))
            peers = job['graph'][self.agent]
            key = (job['job'], stage, step)
            deadline = time.monotonic() + 15.0
            with self.condition:
                while not all(peer in self.inbox.get(key, {}) for peer in peers):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not rclpy.ok():
                        raise TimeoutError(f'{self.agent}: missing neighbors at {stage}/{step}')
                    self.condition.wait(min(remaining, 0.2))
                return {peer: self.inbox[key][peer] for peer in peers}

        def flood(self, job, stage, local):
            tagged = {self.agent: local}
            for step in range(graph_diameter(job['graph'])):
                incoming = self.exchange(job, stage, step, tagged)
                merged = dict(tagged)
                for values in incoming.values():
                    for owner, value in values.items():
                        if owner in merged and merged[owner] != value:
                            raise ValueError('conflicting tagged contribution')
                        merged[owner] = value
                tagged = merged
            if set(tagged) != set(job['graph']):
                raise ValueError('flood did not reach every owner')
            return list(tagged.values())

        def run_job(self, job):
            kind = str(job.get('kind', 'order_statistic'))
            try:
                if kind == 'margin_update':
                    # Every worker receives the same summaries and class
                    # configuration.  No raw calibration scores cross ROS.
                    result = solve_margin_update(
                        job['current'],
                        job['thresholds'],
                        job['kappa'],
                        _classes_from_payload(job),
                        proximal_weight=float(job['proximal_weight']),
                        sensitivity_certified=False,
                    )
                    result = _margin_result_payload(result)
                elif kind == 'order_statistic':
                    graph = job['graph']
                    # The core validates connectivity/symmetry before forming W.
                    owners, weights = metropolis_weights(graph)
                    i = owners.index(self.agent)
                    data = [float(v) for v in job['values']]
                    if not all(math.isfinite(v) for v in data):
                        raise ValueError('nonfinite local scores')
                    L, rank = job['count'], job['rank']
                    if L < 1 or not 1 <= rank <= L or job['iterations'] < 0:
                        raise ValueError('invalid rank/count/iterations')
                    totals = self.flood(job, 'counts', len(data))
                    if sum(totals) != L:
                        raise ValueError('sample ownership count mismatch')
                    q = 0.0
                    tau = (rank - 0.5) / L
                    for step in range(job['iterations']):
                        received = self.exchange(job, 'pinball', step, q)
                        mixed = weights[i, i] * q
                        for peer, value in received.items():
                            mixed += weights[i, owners.index(peer)] * value
                        gradient = sum(float(v < q) - tau for v in data)
                        q = mixed - job['gamma_0'] / ((step + 1) ** job['exponent']) * gradient
                    maxima = self.flood(job, 'maxima', [max(data) if data else None, q])
                    a_max = max(v[0] for v in maxima if v[0] is not None)
                    candidate = min(a_max, max(v[1] for v in maxima))
                    recovery = 0
                    while True:
                        larger = [v for v in data if v > candidate]
                        local = [sum(v <= candidate for v in data), min(larger) if larger else None]
                        values = self.flood(job, f'recovery{recovery}', local)
                        if sum(v[0] for v in values) >= rank:
                            break
                        candidates = [v[1] for v in values if v[1] is not None]
                        if not candidates:
                            raise ValueError('finite recovery exhausted samples')
                        candidate = min(candidates)
                        recovery += 1
                    result = dict(value=candidate, recovery_steps=recovery)
                else:
                    raise ValueError(f'unknown calibration job kind {kind!r}')
            except Exception as error:
                result = dict(error=str(error))
            with self.condition:
                self.inbox = {k: v for k, v in self.inbox.items() if k[0] != job['job']}
            self.busy = False
            result.update(job=job['job'], owner=self.agent, kind=kind)
            self.result_pub.publish(String(data=json.dumps(result)))
    return Worker


def worker_main():
    rclpy, _, _ = _ros()
    rclpy.init()
    node = _worker_type()()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


class RosOrderStatistic:
    """Callable calibration executor; owns six worker processes by default."""
    def __init__(self, graph, start_workers=True, timeout=120.0):
        rclpy, Node, String = _ros()
        self.rclpy, self.String = rclpy, String
        self.graph = {node: sorted(peers) for node, peers in graph.items()}
        graph_diameter(self.graph)
        self.timeout = timeout
        self.owns_context = not rclpy.ok()
        if self.owns_context:
            rclpy.init()
        self.node = Node('maicp_experiment_' + uuid.uuid4().hex[:8])
        self.results = {}
        self.publishers = {
            node: self.node.create_publisher(String, '/hercules_maicp/jobs/' + node, 10)
            for node in self.graph}
        self.node.create_subscription(String, '/hercules_maicp/results', self._receive, 100)
        self.process = subprocess.Popen(
            ['ros2', 'launch', 'hercules_maicp', 'calibration.launch.py'],
            start_new_session=True) if start_workers else None

    def _receive(self, msg):
        packet = json.loads(msg.data)
        self.results.setdefault(packet['job'], {})[packet['owner']] = packet

    def _wait_ready(self, deadline):
        while not all(pub.get_subscription_count() > 0 for pub in self.publishers.values()):
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if time.monotonic() >= deadline:
                raise TimeoutError('calibration workers not ready')

    def _collect_results(self, job_id, deadline):
        while len(self.results.get(job_id, {})) < len(self.graph):
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if time.monotonic() >= deadline:
                raise TimeoutError('distributed calibration job timed out')
        return self.results.pop(job_id)

    def __call__(self, stores, rank, iterations, config):
        if set(stores) != set(self.graph):
            raise ValueError('every update robot must have exactly one local store')
        deadline = time.monotonic() + self.timeout
        self._wait_ready(deadline)
        job_id = uuid.uuid4().hex
        count = sum(len(v) for v in stores.values())
        for node, values in stores.items():
            if isinstance(values, dict):
                values = list(values.values())
            job = dict(job=job_id, kind='order_statistic', values=list(values), graph=self.graph, count=count,
                       rank=rank, iterations=iterations, gamma_0=config.pinball_gamma_0,
                       exponent=config.pinball_exponent)
            self.publishers[node].publish(self.String(data=json.dumps(job)))
        results = self._collect_results(job_id, deadline)
        errors = {node: r['error'] for node, r in results.items() if 'error' in r}
        if errors:
            raise RuntimeError(str(errors))
        bounds = [r['value'] for r in results.values()]
        if max(bounds) - min(bounds) > 1e-9:
            raise RuntimeError('distributed bound disagreement')
        return max(bounds)

    def solve_update(self, current, thresholds, kappa, config):
        """Replicate the common margin SOCP on every ROS worker.

        The job contains thresholds, current margins, kappa, and class
        configuration only.  Workers return their complete numerical result;
        the coordinator accepts it only when every worker agrees within the
        solver tolerance.
        """

        class_names = {str(getattr(key, "value", key)) for key in config.classes}
        maps = {
            "current": current,
            "thresholds": thresholds,
            "kappa": kappa,
        }
        if any(not isinstance(values, dict) for values in maps.values()):
            raise TypeError('margin update summaries must be mappings')
        encoded = {
            name: {str(key): float(value) for key, value in values.items()}
            for name, values in maps.items()
        }
        if any(set(values) != class_names for values in encoded.values()):
            raise ValueError('margin update summaries must contain exactly the configured classes')
        job_id = uuid.uuid4().hex
        job = dict(
            job=job_id,
            kind='margin_update',
            current=encoded['current'],
            thresholds=encoded['thresholds'],
            kappa=encoded['kappa'],
            **_class_payload(config),
        )
        deadline = time.monotonic() + self.timeout
        self._wait_ready(deadline)
        for node in self.graph:
            self.publishers[node].publish(self.String(data=json.dumps(job)))
        results = self._collect_results(job_id, deadline)
        errors = {node: packet['error'] for node, packet in results.items() if 'error' in packet}
        if errors:
            raise RuntimeError(str(errors))
        if len(results) != len(self.graph):
            raise RuntimeError('replicated margin update returned an incomplete worker set')
        updates = []
        for node, packet in results.items():
            if packet.get('kind') != 'margin_update':
                raise RuntimeError(f'worker {node!r} returned the wrong margin-update kind')
            try:
                updates.append(_margin_result_from_packet(packet))
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f'worker {node!r} returned an invalid margin update: {exc}') from exc
        reference = updates[0]
        if any(not _margin_results_agree(reference, candidate) for candidate in updates[1:]):
            raise RuntimeError('replicated margin update disagreement')
        return MarginUpdateResult(
            margins=dict(reference.margins),
            previous_margins=dict(reference.previous_margins),
            thresholds=dict(reference.thresholds),
            kappa=dict(reference.kappa),
            normalized_change=reference.normalized_change,
            objective=reference.objective,
            constraint_slacks=dict(reference.constraint_slacks),
            status=reference.status,
            certified=False,
            replica_count=len(updates),
        )

    def close(self):
        if self.process is not None:
            import os
            import signal
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=10)
            self.process = None
        self.node.destroy_node()
        if self.owns_context and self.rclpy.ok():
            self.rclpy.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
