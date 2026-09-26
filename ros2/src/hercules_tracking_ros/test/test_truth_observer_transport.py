import os
import signal
import subprocess
import time

import numpy as np
import rclpy
from hercules_interfaces.msg import GroundTruthState, TargetMeasurement, TrackingEpoch
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor


AGENTS = ["Drone1", "Drone2", "SimpleFlight", "Husky1", "Husky2", "Husky3"]


def _state(name, position, vehicle_type="drone"):
    value = GroundTruthState()
    value.agent_id = name
    value.header.frame_id = "airsim_world_ned"
    value.position = list(position)
    value.valid = True
    value.vehicle_type = vehicle_type
    return value


def _epoch(identity, timestamp):
    value = TrackingEpoch()
    value.epoch_id = identity
    value.stamp.sec = int(timestamp)
    value.target_id = "Target1"
    value.agent_ids = list(AGENTS)
    value.adjacency = [0] * (len(AGENTS) ** 2)
    return value


def test_truth_observer_is_seeded_range_gated_and_timestamped():
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node("truth_observer_test_driver", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    state_publishers = {
        name: node.create_publisher(GroundTruthState, f"/hercules_mission/ground_truth/{name}", 10)
        for name in AGENTS + ["Target1"]
    }
    epoch_publisher = node.create_publisher(TrackingEpoch, "/hercules_tracking/epoch", 20)
    received = {name: [] for name in AGENTS}
    subscriptions = [
        node.create_subscription(
            TargetMeasurement, f"/hercules_tracking/{name}/Target1/measurement",
            lambda value, name=name: received[name].append(value), 20)
        for name in AGENTS
    ]
    process = subprocess.Popen([
        os.environ["TARGET_OBSERVER_NODE"], "--ros-args",
        "-p", "observation_source:=truth", "-p", "truth_seed:=7",
        "-p", "tracking_measurement_std:=0.25", "-p", "target_sensing_range:=100.0",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.03)
            if (epoch_publisher.get_subscription_count() and all(
                    publisher.get_subscription_count() for publisher in state_publishers.values())):
                break
        for _ in range(4):
            for name in AGENTS:
                state_publishers[name].publish(_state(name, [0.0, 0.0, 0.0]))
            state_publishers["Target1"].publish(
                _state("Target1", [10.0, 0.0, 0.0], "target_ugv"))
            executor.spin_once(timeout_sec=0.03)
        epoch_publisher.publish(_epoch(1, 2.0))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not all(received.values()):
            executor.spin_once(timeout_sec=0.03)
        assert all(received.values())
        for index, name in enumerate(AGENTS):
            message = received[name][-1]
            expected = np.asarray([10.0, 0.0]) + np.random.default_rng(
                7 + 1009 * index).normal(0.0, 0.25, size=2)
            assert message.valid and message.visible
            assert np.allclose(message.position, expected, atol=1e-12)
            assert message.stamp.sec == 2
            assert message.capture_stamp.sec > 1_000_000_000
            assert message.receipt_stamp.sec > 1_000_000_000
            assert message.capture_id == f"target_truth_000001_{name}"

        for name in AGENTS:
            received[name].clear()
        for _ in range(3):
            state_publishers["Target1"].publish(
                _state("Target1", [200.0, 0.0, 0.0], "target_ugv"))
            executor.spin_once(timeout_sec=0.03)
        epoch_publisher.publish(_epoch(2, 3.0))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not all(received.values()):
            executor.spin_once(timeout_sec=0.03)
        assert all(received.values())
        assert all(not values[-1].valid and not values[-1].visible
                   for values in received.values())
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=5)
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown(context=context)
    assert process.returncode == 0
    assert subscriptions
