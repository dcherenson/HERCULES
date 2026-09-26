"""Offline regression checks for the AirSim capture worker ownership."""

import importlib.util
import inspect
import sys
from pathlib import Path
import threading
import types

import numpy as np


def _module(monkeypatch):
    """Load the observer with tiny ROS message stubs, without ROS installed."""
    class Message:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Time:
        sec = 0
        nanosec = 0

    class Header(Message):
        pass

    class Node:
        pass

    class DurabilityPolicy:
        TRANSIENT_LOCAL = object()

    class ReliabilityPolicy:
        RELIABLE = object()

    class QoSProfile:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    rclpy = types.ModuleType("rclpy")
    rclpy.executors = types.ModuleType("rclpy.executors")
    rclpy.executors.ExternalShutdownException = type(
        "ExternalShutdownException", (Exception,), {})
    rclpy.node = types.ModuleType("rclpy.node")
    rclpy.node.Node = Node
    rclpy.qos = types.ModuleType("rclpy.qos")
    rclpy.qos.DurabilityPolicy = DurabilityPolicy
    rclpy.qos.QoSProfile = QoSProfile
    rclpy.qos.ReliabilityPolicy = ReliabilityPolicy

    builtin_interfaces = types.ModuleType("builtin_interfaces")
    builtin_interfaces.msg = types.ModuleType("builtin_interfaces.msg")
    builtin_interfaces.msg.Time = Time
    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs.msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs.msg.PointStamped = Message
    hercules_interfaces = types.ModuleType("hercules_interfaces")
    hercules_interfaces.msg = types.ModuleType("hercules_interfaces.msg")
    hercules_interfaces.msg.GroundTruthState = Message
    hercules_interfaces.msg.ObstacleProxy = Message
    hercules_interfaces.msg.ObstacleProxyArray = Message
    std_msgs = types.ModuleType("std_msgs")
    std_msgs.msg = types.ModuleType("std_msgs.msg")
    std_msgs.msg.Header = Header

    modules = {
        "rclpy": rclpy,
        "rclpy.executors": rclpy.executors,
        "rclpy.node": rclpy.node,
        "rclpy.qos": rclpy.qos,
        "builtin_interfaces": builtin_interfaces,
        "builtin_interfaces.msg": builtin_interfaces.msg,
        "geometry_msgs": geometry_msgs,
        "geometry_msgs.msg": geometry_msgs.msg,
        "hercules_interfaces": hercules_interfaces,
        "hercules_interfaces.msg": hercules_interfaces.msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs.msg,
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)

    path = Path(__file__).parents[1] / "scripts" / "obstacle_observer_node.py"
    spec = importlib.util.spec_from_file_location(
        "obstacle_observer_node_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_clients_and_fov_cache_stay_on_one_worker(monkeypatch):
    module = _module(monkeypatch)
    init_source = inspect.getsource(module.ObstacleObserverNode.__init__)
    assert "self.worker_pool = _capture_worker_pool()" in init_source
    assert "self.camera_fovs: Dict[str, float] = {}" in init_source
    calls = []
    fov_rpc_calls = []

    class Facade:
        def __init__(self, launch_config):
            self.created_thread = threading.get_ident()
            self.connected_thread = None
            self.capture_threads = []

        def connect(self):
            self.connected_thread = threading.get_ident()

    def capture(facade, _detector, agent, _vehicle_type, _state, now, camera_fovs):
        thread_id = threading.get_ident()
        facade.capture_threads.append(thread_id)
        calls.append((agent, thread_id, id(facade), id(camera_fovs)))
        key = f"{agent}:front_center"
        if key not in camera_fovs:
            fov_rpc_calls.append(key)
            camera_fovs[key] = 120.0
        return [], True, {"capture_timestamp": now}, {}

    node = object.__new__(module.ObstacleObserverNode)
    node.capture_lock = threading.Lock()
    node.facades = {}
    node.camera_fovs = {}
    node.states = {}
    node.capture = capture
    node.detectors = {"Drone1": object(), "Drone2": object()}
    node.filter_agent_body_obstacle_proxies = lambda proxies, *_args, **_kwargs: (proxies, None)
    node.facade_type = Facade
    node.launch_type = lambda **kwargs: kwargs
    node.host_ip = "127.0.0.1"
    node.rpc_port = 41451
    node.car_port = 41452
    node.origins = {agent: np.zeros(3) for agent in node.detectors}
    node.uav_radius = 1.0
    node.ugv_radius = 1.25
    node.body_exclusion_margin = 0.5
    node.source = "perception"
    node.get_logger = lambda: types.SimpleNamespace(warning=lambda _message: None)

    def state():
        return types.SimpleNamespace(
            position=[0.0, 0.0, -2.0], velocity=[0.0, 0.0, 0.0], yaw=0.0,
            orientation=[1.0, 0.0, 0.0, 0.0], vehicle_type="drone")

    pool = module._capture_worker_pool()
    try:
        futures = [
            pool.submit(node.capture_one, "Drone1", state()),
            pool.submit(node.capture_one, "Drone1", state()),
            pool.submit(node.capture_one, "Drone2", state()),
        ]
        for future in futures:
            future.result(timeout=2.0)
    finally:
        pool.shutdown(wait=True)

    assert pool._max_workers == 1
    assert len({thread_id for _agent, thread_id, _facade, _cache in calls}) == 1
    assert fov_rpc_calls == ["Drone1:front_center", "Drone2:front_center"]
    assert len({cache_id for _agent, _thread, _facade, cache_id in calls}) == 1
    for facade in node.facades.values():
        assert facade.created_thread == facade.connected_thread
        assert set(facade.capture_threads) == {facade.created_thread}
