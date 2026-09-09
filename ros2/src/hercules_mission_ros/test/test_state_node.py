"""Exercise the read-only state node through synthetic ROS transport."""

import os
import signal
import subprocess
import time

from hercules_interfaces.msg import GroundTruthState
from nav_msgs.msg import Odometry
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor


def test_state_node_reconstructs_distinct_world_positions(tmp_path):
    params = tmp_path / "state.yaml"
    params.write_text(
        """/**:
  ros__parameters:
    agent_ids: [Drone1, Husky1]
    vehicle_types: [drone, ugv]
    odom_topics: [/synthetic/drone, /synthetic/husky]
    vehicle_origins_ned: [1.0, -3.0, -0.2, -2.0, 3.0, 0.4]
    state_topic_prefix: /synthetic/canonical
    freshness_timeout_sec: 0.5
""",
        encoding="utf-8",
    )

    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node("synthetic_state_transport", context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    drone_pub = node.create_publisher(Odometry, "/synthetic/drone", 10)
    husky_pub = node.create_publisher(Odometry, "/synthetic/husky", 10)
    received = {"Drone1": [], "Husky1": []}

    def collect(message):
        received[message.agent_id].append(message)

    subscriptions = [
        node.create_subscription(
            GroundTruthState, "/synthetic/canonical/Drone1", collect, 10),
        node.create_subscription(
            GroundTruthState, "/synthetic/canonical/Husky1", collect, 10),
    ]
    process = subprocess.Popen(
        [os.environ["MISSION_STATE_NODE"], "--ros-args", "--params-file", str(params)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        started = time.monotonic()
        sequence = 1
        while time.monotonic() - started < 8.0:
            message = Odometry()
            message.header.stamp.sec = sequence
            message.pose.pose.orientation.w = 1.0
            message.pose.pose.position.x = 4.0
            message.pose.pose.position.y = -5.0
            message.pose.pose.position.z = 6.0
            message.twist.twist.linear.y = -0.25
            drone_pub.publish(message)
            husky_pub.publish(message)
            executor.spin_once(timeout_sec=0.05)
            if received["Drone1"] and received["Husky1"]:
                break
            sequence += 1
        assert process.poll() is None
        assert received["Drone1"] and received["Husky1"]
        drone = received["Drone1"][-1]
        husky = received["Husky1"][-1]
        assert drone.header.frame_id == "airsim_world_ned"
        assert husky.header.frame_id == "airsim_world_ned"
        assert list(drone.position) == [5.0, 2.0, -6.2]
        assert list(husky.position) == [2.0, 8.0, -5.6]
        assert list(drone.velocity) == [0.0, 0.25, -0.0]
        assert list(drone.position) != list(husky.position)

        # The node suppresses a repeated simulator stamp rather than treating
        # repeated receipt as a fresh canonical sample.
        repeated = Odometry()
        repeated.header.stamp.sec = 1000
        repeated.pose.pose.orientation.w = 1.0
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            drone_pub.publish(repeated)
            executor.spin_once(timeout_sec=0.05)
            if (received["Drone1"] and
                    received["Drone1"][-1].header.stamp.sec == 1000):
                break
        assert received["Drone1"][-1].header.stamp.sec == 1000
        for _ in range(6):
            executor.spin_once(timeout_sec=0.05)
        before = len(received["Drone1"])
        for _ in range(4):
            drone_pub.publish(repeated)
            executor.spin_once(timeout_sec=0.05)
        assert len(received["Drone1"]) == before
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=5)
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown(context=context)
    assert process.returncode == 0, process.stdout.read()
    assert subscriptions
