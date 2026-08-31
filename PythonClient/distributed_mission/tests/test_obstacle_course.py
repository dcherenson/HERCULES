import json

import numpy as np
import pytest

from modules.obstacle_course import course_from_tuples, load_course, normalize_course, save_course
from modules.obstacle_detection import truth_obstacle_proxies


def test_course_json_round_trip_preserves_explicit_offsets(tmp_path):
    course = normalize_course({
        "goal": [16.0, 2.0, -1.0],
        "waypoints": [[8.0, 0.0, -1.0]],
        "obstacles": [
            {"id": "wall", "shape": "box", "center": [7.0, -2.0, -2.0], "dimensions": [2.0, 3.0, 4.0]},
            {"id": "ball", "shape": "sphere", "center": [10.0, 1.0, -6.0], "radius": 1.25},
        ],
    })
    path = tmp_path / "course.json"
    save_course(path, course)
    loaded = load_course(path)
    assert loaded == course
    assert json.loads(path.read_text(encoding="utf-8"))["obstacles"][1]["radius"] == 1.25


def test_legacy_course_tuples_are_normalized_to_boxes():
    course = course_from_tuples([(1.0, 2.0, -2.0, (2.0, 3.0, 4.0))])
    assert course["obstacles"][0]["shape"] == "box"
    assert np.allclose(course["obstacles"][0]["center"], [1.0, 2.0, -2.0])


def test_course_validation_rejects_invalid_shape_and_size():
    with pytest.raises(ValueError):
        normalize_course({"obstacles": [{"shape": "capsule", "center": [0, 0, 0], "radius": 1}]})
    with pytest.raises(ValueError):
        normalize_course({"obstacles": [{"shape": "sphere", "center": [0, 0, 0], "radius": 0}]})


def test_truth_sphere_proxy_is_supported_for_uavs_and_ugvs():
    truth = [{"id": "ball", "shape": "sphere", "center": [4.0, 1.0, 0.0], "radius": 1.5}]
    uav = truth_obstacle_proxies(truth, "drone")
    ugv = truth_obstacle_proxies(truth, "ugv", vehicle_z=0.7, vehicle_radius=1.25)
    assert len(uav) == len(ugv) == 1
    assert np.allclose(uav[0].center, [4.0, 1.0, 0.0])
    assert np.isclose(ugv[0].radius, 1.5)
    assert ugv[0].is_planar
