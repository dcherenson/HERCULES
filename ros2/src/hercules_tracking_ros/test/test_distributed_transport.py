import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import rclpy
from hercules_interfaces.msg import (
    TargetEstimate, TargetMeasurement, TrackingDiagnostics, TrackingEpoch, TrackingHandoff,
)
from rclpy.node import Node


SOURCE = pathlib.Path(os.environ["HERCULES_SOURCE_DIR"]).resolve()
sys.path.insert(0, str(SOURCE))
from modules.target_tracking import (  # noqa: E402
    DistributedTargetTracking,
    TargetMeasurement as PythonMeasurement,
    TargetTrackingModule,
)


def setup_module():
    rclpy.init()


def teardown_module():
    if rclpy.ok():
        rclpy.shutdown()


def stamp(seconds):
    from builtin_interfaces.msg import Time
    value = Time()
    value.sec = int(seconds)
    value.nanosec = int(round((seconds - value.sec) * 1e9))
    return value


def measurement(agent, x, y, capture):
    value = TargetMeasurement()
    value.target_id = "Target1"
    value.stamp = stamp(0.0)
    value.capture_stamp = stamp(100.0)
    value.receipt_stamp = stamp(101.0)
    value.position = [x, y]
    value.covariance = [0.0625, 0.0, 0.0, 0.0625]
    value.valid = True
    value.source_id = agent
    value.capture_id = capture
    value.sensor_id = "synthetic"
    value.visible = True
    return value


def python_reference(agent_ids, observations, graph, rounds):
    modules = {
        name: TargetTrackingModule(
            name, window_seconds=5.0, process_noise_spectral_density=0.2,
            measurement_std=0.25, rho=1.0,
            max_iterations=rounds, tolerance=0.0,
        )
        for name in agent_ids
    }
    network = DistributedTargetTracking(modules, max_iterations=rounds, tolerance=0.0)
    result = network.update(0.0, observations, graph)
    return result["estimates"]


def cpp_reference():
    return json.loads(subprocess.check_output(
        [os.environ["TRACKING_REFERENCE_DUMP"]], text=True
    ))


def run_network(agent_ids, graph, direct, epoch_id=1, rounds=8):
    node = Node(f"tracking_transport_driver_{epoch_id}")
    processes = []
    for index, agent in enumerate(agent_ids):
        processes.append(subprocess.Popen([
            os.environ["TRACKER_NODE"], "--ros-args",
            "-r", f"__node:=transport_tracker_{index}",
            "-p", f"agent_id:={agent}",
            "-p", "target_id:=Target1",
            "-p", f"tracking_admm_max_iterations:={rounds}",
            "-p", "tracking_admm_tolerance:=0.0",
            "-p", "round_timeout_sec:=0.20",
            "-p", "seed_timeout_sec:=0.20",
            "-p", "measurement_wait_sec:=0.05",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    publishers = {
        agent: node.create_publisher(
            TargetMeasurement, f"/hercules_tracking/{agent}/Target1/measurement", 20
        )
        for agent in agent_ids
    }
    epoch_publisher = node.create_publisher(TrackingEpoch, "/hercules_tracking/epoch", 20)
    estimates = {}
    subscriptions = [
        node.create_subscription(
            TargetEstimate,
            f"/hercules_tracking/{agent}/Target1/estimate",
            lambda value, agent=agent: estimates.__setitem__(agent, value),
            20,
        )
        for agent in agent_ids
    ]
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            if epoch_publisher.get_subscription_count() >= len(agent_ids) and all(
                publisher.get_subscription_count() >= 1 for publisher in publishers.values()
            ):
                break
        # The counts above prove driver-to-tracker discovery. Give the shared
        # consensus/status endpoints time to finish peer-to-peer discovery as
        # well, so the test does not turn DDS startup latency into packet loss.
        settle_deadline = time.monotonic() + 0.5
        while time.monotonic() < settle_deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        for agent, point in direct.items():
            publishers[agent].publish(measurement(agent, point[0], point[1], f"capture-{agent}"))
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.02)
        epoch = TrackingEpoch()
        epoch.epoch_id = epoch_id
        epoch.stamp = stamp(0.0)
        epoch.target_id = "Target1"
        epoch.agent_ids = list(agent_ids)
        epoch.adjacency = [
            int(second in graph.get(first, []))
            for first in agent_ids for second in agent_ids
        ]
        epoch_publisher.publish(epoch)
        deadline = time.monotonic() + 10.0
        while len(estimates) < len(agent_ids) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
        assert set(estimates) == set(agent_ids)
        return estimates
    finally:
        subscriptions.clear()
        node.destroy_node()
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()


def test_connected_chain_matches_synchronous_reference():
    agents = ["Drone1", "Husky1", "Husky2"]
    graph = {"Drone1": ["Husky1"], "Husky1": ["Drone1", "Husky2"], "Husky2": ["Husky1"]}
    direct = {"Drone1": (0.0, 0.0), "Husky1": (1.0, 0.0)}
    actual = run_network(agents, graph, direct, rounds=8)
    observations = {
        agent: ({"Target1": PythonMeasurement(
            "Target1", np.asarray(point), np.eye(2) * 0.0625, 0.0
        )} if agent in direct else {})
        for agent, point in [(name, direct.get(name)) for name in agents]
    }
    expected = python_reference(agents, observations, graph, 8)
    cpp_expected = cpp_reference()
    for agent in agents:
        reference = expected[agent]["Target1"]
        assert np.allclose(actual[agent].position, reference["position"], atol=1e-7)
        assert np.allclose(actual[agent].velocity, reference["velocity"], atol=1e-7)
        assert np.allclose(actual[agent].position, cpp_expected[agent]["position"], atol=1e-7)
        assert np.allclose(actual[agent].velocity, cpp_expected[agent]["velocity"], atol=1e-7)


def test_disconnected_components_do_not_exchange_estimates():
    agents = ["component_a", "component_b"]
    graph = {agent: [] for agent in agents}
    direct = {"component_a": (0.0, 0.0), "component_b": (10.0, -2.0)}
    actual = run_network(agents, graph, direct, epoch_id=2, rounds=4)
    assert abs(actual["component_a"].position[0]) < 1e-9
    assert abs(actual["component_b"].position[0] - 10.0) < 1e-9


def test_missing_frozen_neighbor_times_out_and_publishes_best_local_estimate():
    agent = "timeout_agent"
    node = Node("tracking_timeout_driver")
    process = subprocess.Popen([
        os.environ["TRACKER_NODE"], "--ros-args", "-r", "__node:=timeout_tracker",
        "-p", f"agent_id:={agent}", "-p", "target_id:=Target1",
        "-p", "tracking_admm_max_iterations:=3", "-p", "tracking_admm_tolerance:=0.0",
        "-p", "round_timeout_sec:=0.05", "-p", "seed_timeout_sec:=0.05",
        "-p", "measurement_wait_sec:=0.03",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    measurement_publisher = node.create_publisher(
        TargetMeasurement, f"/hercules_tracking/{agent}/Target1/measurement", 20)
    epoch_publisher = node.create_publisher(TrackingEpoch, "/hercules_tracking/epoch", 20)
    estimates = []
    diagnostics = []
    estimate_subscription = node.create_subscription(
        TargetEstimate, f"/hercules_tracking/{agent}/Target1/estimate",
        estimates.append, 20)
    diagnostic_subscription = node.create_subscription(
        TrackingDiagnostics, f"/hercules_tracking/{agent}/Target1/diagnostics",
        diagnostics.append, 20)
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.03)
            if epoch_publisher.get_subscription_count() and measurement_publisher.get_subscription_count():
                break
        measurement_publisher.publish(measurement(agent, 2.0, -1.0, "timeout-capture"))
        for _ in range(4):
            rclpy.spin_once(node, timeout_sec=0.03)
        value = TrackingEpoch()
        value.epoch_id = 30
        value.stamp = stamp(0.0)
        value.target_id = "Target1"
        value.agent_ids = [agent, "missing_neighbor"]
        value.adjacency = [0, 1, 1, 0]
        epoch_publisher.publish(value)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and (not estimates or not diagnostics):
            rclpy.spin_once(node, timeout_sec=0.03)
        assert estimates and estimates[-1].active
        assert diagnostics and diagnostics[-1].epoch_timed_out
        assert diagnostics[-1].missing_neighbor_messages > 0
    finally:
        node.destroy_subscription(estimate_subscription)
        node.destroy_subscription(diagnostic_subscription)
        node.destroy_node()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)


def test_handoff_topic_activates_receiving_tracker_without_direct_measurement():
    receiver = "handoff_receiver"
    sender = "handoff_sender"
    node = Node("tracking_handoff_driver")
    process = subprocess.Popen([
        os.environ["TRACKER_NODE"], "--ros-args", "-r", "__node:=handoff_tracker",
        "-p", f"agent_id:={receiver}", "-p", "target_id:=Target1",
        "-p", "tracking_admm_max_iterations:=1", "-p", "tracking_admm_tolerance:=0.0",
        "-p", "round_timeout_sec:=0.04", "-p", "seed_timeout_sec:=0.04",
        "-p", "measurement_wait_sec:=0.02",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    epoch_publisher = node.create_publisher(TrackingEpoch, "/hercules_tracking/epoch", 20)
    handoff_publisher = node.create_publisher(TrackingHandoff, "/hercules_tracking/handoff", 20)
    estimates = []
    diagnostics = []
    estimate_subscription = node.create_subscription(
        TargetEstimate, f"/hercules_tracking/{receiver}/Target1/estimate",
        estimates.append, 20)
    diagnostic_subscription = node.create_subscription(
        TrackingDiagnostics, f"/hercules_tracking/{receiver}/Target1/diagnostics",
        diagnostics.append, 20)

    def publish_epoch(identity, timestamp):
        value = TrackingEpoch()
        value.epoch_id = identity
        value.stamp = stamp(timestamp)
        value.target_id = "Target1"
        value.agent_ids = [receiver, sender]
        value.adjacency = [0, 1, 1, 0]
        epoch_publisher.publish(value)

    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.03)
            if epoch_publisher.get_subscription_count() and handoff_publisher.get_subscription_count():
                break
        publish_epoch(40, 0.0)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not estimates:
            rclpy.spin_once(node, timeout_sec=0.03)
        assert estimates and not estimates[-1].active

        handoff = TrackingHandoff()
        handoff.sender_id = sender
        handoff.receiver_id = receiver
        handoff.target_id = "Target1"
        handoff.epoch_id = 40
        handoff.stamp = stamp(0.1)
        handoff.information_matrix = [1.0 if row == column else 0.0
                                      for row in range(4) for column in range(4)]
        handoff.information_vector = [3.0, 4.0, 0.0, 0.0]
        handoff.has_state_covariance = False
        handoff_publisher.publish(handoff)
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.03)
        estimates.clear()
        diagnostics.clear()
        publish_epoch(41, 0.2)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and (not estimates or not diagnostics):
            rclpy.spin_once(node, timeout_sec=0.03)
        assert estimates and estimates[-1].active
        assert diagnostics and diagnostics[-1].handoffs_accepted == 1
    finally:
        node.destroy_subscription(estimate_subscription)
        node.destroy_subscription(diagnostic_subscription)
        node.destroy_node()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
