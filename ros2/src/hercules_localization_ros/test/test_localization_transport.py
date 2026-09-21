import os
import subprocess
import time

import rclpy
from hercules_interfaces.msg import (
    GlobalCiBelief,
    LocalizationDiagnostics,
    LocalizationEstimate,
    LocalizationMeasurement,
    LocalizationPeerEstimate,
    PlanarRelativeMeasurement,
)
from hercules_interfaces.srv import (
    RecursiveLocalizationPair,
    ResetLocalization,
    SetLocalizationAlgorithm,
)
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix


NODE = os.environ["LOCALIZATION_NODE"]


def setup_module():
    rclpy.init()


def teardown_module():
    if rclpy.ok():
        rclpy.shutdown()


def wait_until(node, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.03)
        if predicate():
            return True
    return False


def measurement(agent_id="transport_agent", x=1.0, y=2.0):
    value = LocalizationMeasurement()
    value.agent_id = agent_id
    # A GPS observation seeds the estimator; odometry is intentionally held
    # until that first fix by the transport contract.
    value.source_id = "gps"
    value.frame_id = "map"
    value.position = [x, y, 3.0]
    value.velocity = [0.1, 0.2, 0.3]
    value.orientation = [1.0, 0.0, 0.0, 0.0]
    value.covariance = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    value.valid = True
    return value


def test_localization_transport_selects_algorithm_and_publishes_outputs():
    node_name = "localization_transport_node"
    topic_prefix = "/localization_transport"
    process = subprocess.Popen([
        NODE,
        "--ros-args",
        "-r", f"__node:={node_name}",
        "-p", "agent_id:=transport_agent",
        "-p", "algorithm:=gs_ci",
        "-p", f"measurement_topic:={topic_prefix}/measurement",
        "-p", f"odom_local_topic:={topic_prefix}/odom_local",
        "-p", f"global_gps_topic:={topic_prefix}/global_gps",
        "-p", f"relative_topic:={topic_prefix}/relative",
        "-p", f"peer_topic:={topic_prefix}/peer",
        "-p", f"global_ci_topic:={topic_prefix}/gs_ci",
        "-p", f"estimate_topic:={topic_prefix}/estimate",
        "-p", f"diagnostics_topic:={topic_prefix}/diagnostics",
        "-p", f"peer_output_topic:={topic_prefix}/peer_out",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    driver = Node("localization_transport_driver")
    measurement_publisher = driver.create_publisher(
        LocalizationMeasurement, f"{topic_prefix}/measurement", 20)
    odom_publisher = driver.create_publisher(Odometry, f"{topic_prefix}/odom_local", 20)
    gps_publisher = driver.create_publisher(NavSatFix, f"{topic_prefix}/global_gps", 20)
    relative_publisher = driver.create_publisher(
        PlanarRelativeMeasurement, f"{topic_prefix}/relative", 20)
    peer_publisher = driver.create_publisher(
        LocalizationPeerEstimate, f"{topic_prefix}/peer", 20)
    global_ci_publisher = driver.create_publisher(
        GlobalCiBelief, f"{topic_prefix}/gs_ci", 20)
    estimates = []
    diagnostics = []
    peers = []
    beliefs = []
    driver.create_subscription(
        LocalizationEstimate,
        f"{topic_prefix}/estimate", estimates.append, 20)
    driver.create_subscription(LocalizationDiagnostics,
                               f"{topic_prefix}/diagnostics", diagnostics.append, 20)
    driver.create_subscription(LocalizationPeerEstimate,
                               f"{topic_prefix}/peer_out", peers.append, 20)
    driver.create_subscription(GlobalCiBelief,
                               f"{topic_prefix}/gs_ci", beliefs.append, 20)
    select_client = driver.create_client(
        SetLocalizationAlgorithm, f"/{node_name}/set_algorithm")
    reset_client = driver.create_client(ResetLocalization, f"/{node_name}/reset")
    pair_client = driver.create_client(RecursiveLocalizationPair, f"/{node_name}/recursive_pair")
    try:
        assert wait_until(
            driver,
            lambda: measurement_publisher.get_subscription_count() > 0
            and odom_publisher.get_subscription_count() > 0
            and gps_publisher.get_subscription_count() > 0
            and relative_publisher.get_subscription_count() > 0
            and peer_publisher.get_subscription_count() > 0
            and global_ci_publisher.get_subscription_count() > 0,
            timeout=8.0,
        )

        peer = LocalizationPeerEstimate()
        peer.sender_id = "peer"
        peer.frame_id = "map"
        peer.position = [4.0, 5.0, 6.0]
        peer.velocity = [0.0, 0.0, 0.0]
        peer.orientation = [1.0, 0.0, 0.0, 0.0]
        peer.covariance = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        peer.valid = True
        peer_publisher.publish(peer)
        relative = PlanarRelativeMeasurement()
        relative.source_id = "peer"
        relative.target_id = "transport_agent"
        relative.relative_position = [1.0, 2.0]
        relative.relative_velocity = [0.0, 0.0]
        relative.covariance = [1.0, 0.0, 0.0, 1.0]
        relative.valid = True
        relative_publisher.publish(relative)
        for _ in range(5):
            rclpy.spin_once(driver, timeout_sec=0.03)

        measurement_publisher.publish(measurement())
        assert wait_until(driver, lambda: estimates and diagnostics and peers, timeout=5.0)
        assert estimates[-1].algorithm == "gs_ci"
        assert estimates[-1].agent_id == "transport_agent"
        assert list(estimates[-1].position) == [1.0, 2.0, 3.0]
        assert diagnostics[-1].algorithm == "gs_ci"
        assert diagnostics[-1].peer_estimates_received >= 1
        assert peers[-1].sender_id == "transport_agent"
        assert wait_until(driver, lambda: beliefs, timeout=3.0)
        assert beliefs[-1].agent_id == "transport_agent"
        assert beliefs[-1].confidence_weight == 0.8

        assert pair_client.wait_for_service(timeout_sec=3.0)
        pair_request = RecursiveLocalizationPair.Request()
        pair_request.source_id = "transport_agent"
        pair_request.target_id = "peer"
        pair_request.event_id = "pair-event-1"
        pair_request.measurement.source_id = "transport_agent"
        pair_request.measurement.target_id = "peer"
        pair_request.measurement.relative_position = [3.0, 0.0]
        pair_request.measurement.range = 3.0
        pair_request.measurement.bearing = 0.0
        pair_request.measurement.covariance = [1.0, 0.0, 0.0, 0.01]
        pair_request.measurement.valid = True
        pair_response = pair_client.call_async(pair_request)
        assert wait_until(driver, lambda: pair_response.done(), timeout=3.0)
        assert not pair_response.result().accepted
        assert not pair_response.result().duplicate

        assert select_client.wait_for_service(timeout_sec=3.0)
        request = SetLocalizationAlgorithm.Request()
        # Unknown names are rejected without disturbing the active GS-CI
        # selection.
        request.algorithm = "not-an-estimator"
        response = select_client.call_async(request)
        assert wait_until(driver, lambda: response.done(), timeout=3.0)
        assert not response.result().accepted
        assert response.result().active_algorithm == "gs_ci"

        assert reset_client.wait_for_service(timeout_sec=3.0)
        reset_response = reset_client.call_async(ResetLocalization.Request())
        assert wait_until(driver, lambda: reset_response.done(), timeout=3.0)
        assert reset_response.result().success

        # Reset restores the first-GPS initialization gate; odometry alone
        # must not seed a new estimator state.
        odom = Odometry()
        odom.header.frame_id = "map"
        odom.pose.pose.position.x = 7.0
        odom.pose.pose.orientation.w = 1.0
        odom_publisher.publish(odom)
        assert wait_until(driver, lambda: len(estimates) >= 2, timeout=5.0)
        assert not estimates[-1].initialized
        assert not estimates[-1].valid
    finally:
        driver.destroy_node()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)


def test_recursive_pair_transport_is_sequence_idempotent():
    observer_name = "localization_transport_agent"
    observed_name = "localization_peer"
    observer_prefix = "/recursive_localization_transport/observer"
    observed_prefix = "/recursive_localization_transport/observed"
    relative_topic = "/recursive_localization_transport/relative"
    observer_process = subprocess.Popen([
        NODE,
        "--ros-args",
        "-r", f"__node:={observer_name}",
        "-p", "agent_id:=transport_agent",
        "-p", "algorithm:=recursive_decentralized",
        "-p", f"measurement_topic:={observer_prefix}/measurement",
        "-p", f"relative_topic:={relative_topic}",
        "-p", f"peer_topic:={observer_prefix}/peer",
        "-p", f"global_ci_topic:={observer_prefix}/gs_ci",
        "-p", f"estimate_topic:={observer_prefix}/estimate",
        "-p", f"diagnostics_topic:={observer_prefix}/diagnostics",
        "-p", f"peer_output_topic:={observer_prefix}/peer_out",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    observed_process = subprocess.Popen([
        NODE,
        "--ros-args",
        "-r", f"__node:={observed_name}",
        "-p", "agent_id:=peer",
        "-p", "algorithm:=recursive_decentralized",
        "-p", f"measurement_topic:={observed_prefix}/measurement",
        "-p", f"relative_topic:={relative_topic}",
        "-p", f"peer_topic:={observed_prefix}/peer",
        "-p", f"global_ci_topic:={observed_prefix}/gs_ci",
        "-p", f"estimate_topic:={observed_prefix}/estimate",
        "-p", f"diagnostics_topic:={observed_prefix}/diagnostics",
        "-p", f"peer_output_topic:={observed_prefix}/peer_out",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    driver = Node("recursive_localization_transport_driver")
    observer_measurements = driver.create_publisher(
        LocalizationMeasurement, f"{observer_prefix}/measurement", 20)
    observed_measurements = driver.create_publisher(
        LocalizationMeasurement, f"{observed_prefix}/measurement", 20)
    relative_publisher = driver.create_publisher(
        PlanarRelativeMeasurement, relative_topic, 20)
    observer_estimates = []
    observed_estimates = []
    driver.create_subscription(
        LocalizationEstimate, f"{observer_prefix}/estimate", observer_estimates.append, 20)
    driver.create_subscription(
        LocalizationEstimate, f"{observed_prefix}/estimate", observed_estimates.append, 20)
    pair_client = driver.create_client(
        RecursiveLocalizationPair, f"/{observed_name}/recursive_pair")
    try:
        assert wait_until(
            driver,
            lambda: pair_client.service_is_ready()
            and observer_measurements.get_subscription_count() > 0
            and observed_measurements.get_subscription_count() > 0
            and relative_publisher.get_subscription_count() >= 2,
            timeout=8.0,
        )
        observer_measurements.publish(measurement("transport_agent", 0.0, 0.0))
        observed_measurements.publish(measurement("peer", 5.0, 0.0))
        assert wait_until(driver, lambda: observer_estimates and observed_estimates, timeout=5.0)

        relative = PlanarRelativeMeasurement()
        relative.source_id = "transport_agent"
        relative.target_id = "peer"
        relative.observer_id = "transport_agent"
        relative.observed_id = "peer"
        relative.capture_id = "pair-event-live"
        relative.sequence = 7
        relative.relative_position = [4.0, 0.0]
        relative.range = 4.0
        relative.bearing = 0.0
        relative.covariance = [0.25, 0.0, 0.0, 0.01]
        relative.valid = True
        relative_publisher.publish(relative)
        for _ in range(10):
            rclpy.spin_once(driver, timeout_sec=0.03)

        # A subsequent private input publishes the state installed by the
        # asynchronous two-agent transaction without changing non-anchor GPS.
        observer_count = len(observer_estimates)
        observed_count = len(observed_estimates)
        observer_measurements.publish(measurement("transport_agent", 0.0, 0.0))
        observed_measurements.publish(measurement("peer", 5.0, 0.0))
        assert wait_until(
            driver,
            lambda: len(observer_estimates) > observer_count
            and len(observed_estimates) > observed_count
            and observer_estimates[-1].valid and observed_estimates[-1].valid,
            timeout=5.0,
        )
        assert observer_estimates[-1].position[0] > 0.0
        assert observed_estimates[-1].position[0] < 5.0

        # A direct retry of the same target-side transaction returns the
        # original numerical response, including posterior blocks/factors.
        request = RecursiveLocalizationPair.Request()
        request.source_id = "transport_agent"
        request.target_id = "peer"
        request.event_id = "pair-event-cache"
        request.pair_sequence = 7
        request.measurement.source_id = "transport_agent"
        request.measurement.target_id = "peer"
        request.measurement.relative_position = [4.0, 0.0]
        request.measurement.range = 4.0
        request.measurement.bearing = 0.0
        request.measurement.covariance = [0.25, 0.0, 0.0, 0.01]
        request.measurement.sequence = 7
        request.measurement.valid = True
        request.observer_pose = [0.0, 0.0, 0.0]
        request.observer_covariance = [1.0, 0.0, 0.0,
                                       0.0, 1.0, 0.0,
                                       0.0, 0.0, 1.0]

        first = pair_client.call_async(request)
        assert wait_until(driver, lambda: first.done(), timeout=3.0)
        assert first.result().accepted
        assert not first.result().duplicate
        first_pose = list(first.result().observer_posterior_pose)
        first_factor = list(first.result().correlation_factor)

        retry = pair_client.call_async(request)
        assert wait_until(driver, lambda: retry.done(), timeout=3.0)
        assert retry.result().accepted
        assert retry.result().duplicate
        assert list(retry.result().observer_posterior_pose) == first_pose
        assert list(retry.result().correlation_factor) == first_factor

        request.event_id = "pair-event-conflict"
        request.measurement.sequence = 8
        conflict = pair_client.call_async(request)
        assert wait_until(driver, lambda: conflict.done(), timeout=3.0)
        assert not conflict.result().accepted
        assert not conflict.result().duplicate

        request.pair_sequence = 9
        request.measurement.sequence = 9
        request.event_id = "wrong-target"
        request.target_id = "another_agent"
        request.measurement.target_id = "another_agent"
        wrong_target = pair_client.call_async(request)
        assert wait_until(driver, lambda: wrong_target.done(), timeout=3.0)
        assert not wrong_target.result().accepted
        assert not wrong_target.result().duplicate
    finally:
        driver.destroy_node()
        for process in (observer_process, observed_process):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
