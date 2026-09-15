import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "scripts" / "obstacle_observer_node.py"
    spec = importlib.util.spec_from_file_location("obstacle_observer_node", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_preserves_proxy_order_and_geometry():
    module = _module()

    class Proxy:
        obstacle_id = "p0"
        source = "depth_Drone1"
        center = np.array([1.0, 2.0, 3.0])
        radius = 0.5
        velocity = np.array([0.1, 0.2, 0.0])
        timestamp = 4.0
        point_count = 12
        is_planar = False

    message = module.snapshot_message("Drone1", "capture-1", "perception", True, True, 4.0, [Proxy()])
    assert message.header.frame_id == "airsim_world_ned"
    assert message.capture_id == "capture-1"
    assert list(message.proxies[0].center) == [1.0, 2.0, 3.0]
    assert message.proxies[0].has_velocity
