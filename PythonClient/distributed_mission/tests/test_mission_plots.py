import json
import shutil
import subprocess

import numpy as np

from modules.mission_plots import (
    box_vertices,
    compute_collision_clearances,
    generate_mission_plots,
    ned_to_display,
    route_up_xy,
    sensor_view_for_record,
    _topdown_limits,
)


def _record(step, positions, obstacles=None):
    return {
        "step": step,
        "dt": 0.1,
        "states": {
            name: {"position": list(position), "velocity": [0.0, 0.0, 0.0]}
            for name, position in positions.items()
        },
        "vehicle_types": {name: ("drone" if str(name).startswith("Drone") else "ugv") for name in positions},
        "obstacles": obstacles or {},
        "communication_links": [],
    }


def test_collision_clearance_uses_vehicle_and_obstacle_geometry():
    records = [
        _record(
            0,
            {"Drone1": [0.0, 0.0, 0.0], "Husky1": [4.0, 0.0, 0.0]},
            {
                "Drone1": {
                    "proxies": [{"center": [0.0, 3.0, 0.0], "radius": 1.0}]
                }
            },
        )
    ]
    times, clearances = compute_collision_clearances(records, {"Drone1": 1.0, "Husky1": 1.0})

    assert np.allclose(times, [0.0])
    # Drone1 has 1 m clearance to the obstacle and 2 m to Husky1.
    assert np.allclose(clearances["Drone1"], [1.0])
    assert np.allclose(clearances["Husky1"], [2.0])


def test_geometry_helpers_use_ned_to_z_up_and_route_up_frame():
    assert np.allclose(ned_to_display([1.0, 2.0, -3.0]), [1.0, 2.0, 3.0])

    vertices = box_vertices([0.0, 0.0, 0.0], [2.0, 4.0, 6.0])
    assert vertices.shape == (8, 3)
    assert np.allclose(vertices.min(axis=0), [-1.0, -2.0, -3.0])
    assert np.allclose(vertices.max(axis=0), [1.0, 2.0, 3.0])

    assert np.allclose(route_up_xy([[0.0, 0.0], [0.0, 5.0]], np.pi / 2.0), [[0.0, 0.0], [0.0, 5.0]])


def test_sensor_age_is_propagated_from_obstacle_record():
    record = _record(0, {"Drone1": [0.0, 0.0, -1.0]}, {
        "Drone1": {"sensor_view": {"sensor_type": "uav_camera", "age": 0.4}}
    })
    view = sensor_view_for_record(record, "Drone1")
    assert view["sensor_type"] == "uav_camera"
    assert view["age"] == 0.4


def test_topdown_limits_include_goal_location():
    records = [_record(0, {"Drone1": [0.0, 0.0, -1.0]})]
    records[0]["goal"] = [20.0, 20.0, -1.0]
    lower, upper = _topdown_limits(records, ["Drone1"], 0.0, np.zeros(2))
    assert lower[0] <= 20.0 <= upper[0]
    assert lower[1] <= 20.0 <= upper[1]


def test_topdown_limits_include_target_truth_not_noisy_estimate():
    records = [_record(0, {"Drone1": [0.0, 0.0, -1.0]})]
    records[0]["target_truth"] = {
        "name": "Target1", "position": [30.0, 5.0, 0.0],
        "pattern": {"center": [30.0, 5.0, 0.0], "route_heading": 0.0,
                    "longitudinal_span": 10.0, "lateral_span": 8.0},
    }
    records[0]["target_tracking"] = {
        "agents": {"Drone1": {"estimate": {"target_id": "Target1", "position": [-100.0, -100.0], "covariance": [[1.0, 0.0], [0.0, 1.0]]}}}
    }
    lower, upper = _topdown_limits(records, ["Drone1"], 0.0, np.zeros(2))
    # Heading zero makes route-up coordinates [left=y, progress=x].
    assert lower[0] <= 5.0 <= upper[0]
    assert lower[1] <= 30.0 <= upper[1]


def test_generate_mission_plots_writes_images_and_topdown_animations(tmp_path, monkeypatch):
    log_path = tmp_path / "mestres_test.jsonl"
    records = [
        _record(0, {"Drone1": [0.0, 0.0, -1.0], "Husky1": [4.0, 0.0, 0.0]}, {
            "Drone1": {"sensor_view": {"sensor_type": "uav_camera", "position": [0.0, 0.0, -1.0], "orientation_quaternion": [1.0, 0.0, 0.0, 0.0], "horizontal_fov_deg": 90.0, "vertical_fov_deg": 60.0, "range_m": 3.0, "age": 0.0}, "proxies": [{"center": [1.0, 0.0, -1.0], "radius": 0.5}]},
            "Husky1": {"sensor_view": {"sensor_type": "ugv_lidar", "position": [4.0, 0.0, 0.0], "orientation_quaternion": [1.0, 0.0, 0.0, 0.0], "vertical_fov_lower_deg": -10.0, "vertical_fov_upper_deg": 10.0, "range_m": 3.0, "age": 0.0}, "proxies": []},
        }),
        _record(1, {"Drone1": [1.0, 0.0, -1.0], "Husky1": [4.0, 1.0, 0.0]}, {
            "Drone1": {"sensor_view": {"sensor_type": "uav_camera", "position": [1.0, 0.0, -1.0], "orientation_quaternion": [1.0, 0.0, 0.0, 0.0], "horizontal_fov_deg": 90.0, "vertical_fov_deg": 60.0, "range_m": 3.0, "age": 0.1}, "proxies": [{"center": [1.5, 0.0, -1.0], "radius": 0.5}]},
            "Husky1": {"sensor_view": {"sensor_type": "ugv_lidar", "position": [4.0, 1.0, 0.0], "orientation_quaternion": [1.0, 0.0, 0.0, 0.0], "vertical_fov_lower_deg": -10.0, "vertical_fov_upper_deg": 10.0, "range_m": 3.0, "age": 0.1}, "proxies": []},
        }),
    ]
    for record in records:
        record["true_obstacles"] = [{"id": "block0", "shape": "box", "center": [2.0, 0.0, -1.0], "dimensions": [1.0, 2.0, 2.0]}]
        record["target_truth"] = {
            "name": "Target1", "position": [3.0 + record["step"], 0.5, 0.0],
            "collision": {"relevant": False},
            "pattern": {"center": [3.0, 0.5, 0.0], "route_heading": 0.0,
                        "longitudinal_span": 10.0, "lateral_span": 8.0},
        }
        record["targets"] = {"Target1": record["target_truth"]}
        record["target_tracking"] = {
            "agents": {
                "Drone1": {"estimate": {"target_id": "Target1", "position": [3.0 + record["step"], 0.4, 0.0], "covariance": [[0.25, 0.0], [0.0, 0.36]]}},
                "Husky1": {"estimate": {"target_id": "Target1", "position": [3.0 + record["step"], 0.6, 0.0], "covariance": [[0.25, 0.0], [0.0, 0.36]]}},
            }
        }
        record["tracking_communication_links"] = [["Drone1", "Husky1"]]
        record["safety_communication_links"] = []
    with log_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")

    paths = generate_mission_plots(str(log_path), str(tmp_path), {"Drone1": 1.0, "Husky1": 1.0}, animation_fps=5.0)

    assert len(paths) == 4
    assert (tmp_path / "mestres_test_trajectories_3d.png").is_file()
    assert (tmp_path / "mestres_test_collision_clearance.png").is_file()
    animation_path = tmp_path / "mestres_test_topdown.mp4"
    gif_path = tmp_path / "mestres_test_topdown.gif"
    assert animation_path.is_file()
    assert gif_path.is_file()
    assert animation_path.stat().st_size > 0
    assert gif_path.stat().st_size > 0
    if shutil.which("ffprobe") is None:
        raise AssertionError("ffprobe is required for the MP4 regression test")
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,duration", "-of", "json", str(animation_path)
    ], text=True))
    stream = probe["streams"][0]
    assert int(stream["nb_read_frames"]) >= len(records)
    assert float(stream["duration"]) > 0.0
    assert float(stream["duration"]) <= 0.21


def test_collision_plot_accepts_and_marks_authoritative_collision_records(tmp_path):
    records = [
        {
            **_record(0, {"Drone1": [0.0, 0.0, -5.0], "Husky1": [4.0, 0.0, 1.0]}),
            "collisions": {"Drone1": {"has_collided": False}},
        },
        {
            **_record(1, {"Drone1": [0.5, 0.0, -5.0], "Husky1": [4.0, 0.0, 1.0]}),
            "collisions": {"Drone1": {"has_collided": True}},
        },
    ]
    output_path = tmp_path / "collision.png"

    from modules.mission_plots import plot_collision_clearances

    plot_collision_clearances(records, str(output_path), {"Drone1": 1.0, "Husky1": 1.25})

    assert output_path.is_file()

    trajectory_path = tmp_path / "trajectory.png"
    from modules.mission_plots import plot_trajectories_3d

    plot_trajectories_3d(records, str(trajectory_path))
    assert trajectory_path.is_file()
