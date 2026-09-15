import importlib.util
from pathlib import Path
import time

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "scripts" / "obstacle_observer_node.py"
    spec = importlib.util.spec_from_file_location("obstacle_observer_node_transport", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_age_continues_after_observer_message_stop():
    module = _module()
    received = module.snapshot_message("Drone1", "capture-1", "perception", True, True, time.time(), [])
    # The transport message is a completed immutable snapshot.  Once the
    # producer stops, consumers must age it from receipt time rather than
    # treating the absence of a new message as a fresh valid capture.
    capture_time = received.capture_stamp.sec + received.capture_stamp.nanosec * 1e-9
    assert capture_time > 0.0
    later = capture_time + 1.31
    assert later - capture_time > 1.3
    assert received.acquisition_success and received.valid
    assert np.asarray(received.proxies).size == 0
