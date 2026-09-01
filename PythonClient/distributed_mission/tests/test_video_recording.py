import json
import os
import shutil

import numpy as np
import pytest

from modules.video_recording import (
    FollowCameraController,
    analyze_camera_alignment,
    find_recorded_frames,
    load_capture_metadata,
    make_chase_overlay,
    project_world_point,
    render_recordings,
    world_camera_pose_to_vehicle,
)


def test_follow_camera_is_route_aligned_zero_roll_and_expands_immediately():
    controller = FollowCameraController(np.pi / 2.0, minimum_distance=1.0)
    first = controller.update([[0.0, 0.0, -2.0], [0.0, 2.0, -2.0]], 0.1)
    second = controller.update([[0.0, 0.0, -2.0], [0.0, 10.0, -2.0]], 0.1)
    assert first["roll_deg"] == 0.0
    assert second["world_position"][1] < 0.0
    assert np.linalg.norm(np.asarray(second["world_position"]) - np.asarray(second["target"])) > np.linalg.norm(
        np.asarray(first["world_position"]) - np.asarray(first["target"])
    )


def test_world_projection_uses_airsim_forward_right_down_convention():
    center = project_world_point([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 90.0, 1280, 720)
    right = project_world_point([2.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 90.0, 1280, 720)
    behind = project_world_point([-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 90.0, 1280, 720)
    assert np.allclose(center, [640.0, 360.0])
    assert right[0] > center[0]
    assert behind is None


def test_world_chase_pose_is_converted_to_current_vehicle_frame():
    local_position, local_orientation = world_camera_pose_to_vehicle(
        [5.0, 0.0, -2.0], [1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 0.0], np.pi / 2.0
    )
    assert np.allclose(local_position, [-2.0, -4.0, -2.0])
    assert np.allclose(np.linalg.norm(local_orientation), 1.0)
    # The local pose must include the inverse of the vehicle's 90-degree yaw.
    assert np.isclose(abs(local_orientation[3]), np.sin(np.pi / 4.0))


def test_recording_parser_and_media_cleanup(tmp_path):
    pytest.importorskip("PIL")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required")
    from PIL import Image

    staging = tmp_path / "frames"
    staging.mkdir()
    epoch = 1000.0
    streams = [("Drone1", "mission_follow"), ("Drone1", "front_center"), ("Husky1", "front_center")]
    for index in range(3):
        for vehicle, camera in streams:
            image = Image.new("RGB", (64, 36), (index * 40, 80, 120))
            image.save(staging / "img_{}_{}_0_{}.png".format(vehicle, camera, int((epoch + index * 0.1) * 1e9)))
    assert len(find_recorded_frames(str(staging), "Drone1", "mission_follow")) == 3
    records = [
        {"step": index, "dt": 0.1, "timestamp": epoch + index * 0.1,
         "vehicle_types": {"Drone1": "drone", "Husky1": "ugv"},
         "states": {"Drone1": {"position": [index * 0.1, 0.0, -1.0]}, "Husky1": {"position": [index * 0.1, 1.0, 0.0]}},
         "recording": {"chase_camera": {"world_position": [-5.0, 0.0, -3.0], "orientation_quaternion": [1.0, 0.0, 0.0, 0.0]}},
         "obstacles": {}, "true_obstacles": [], "collisions": {}}
        for index in range(3)
    ]
    outputs = render_recordings(str(staging), str(tmp_path), "run", records, "Drone1", "Husky1",
                                width=64, height=36, fps=5.0, gif_height=18, gif_fps=2.0)
    assert len([path for path in outputs if path.endswith((".mp4", ".gif"))]) == 6
    assert all(os.path.getsize(path) > 0 for path in outputs if path.endswith((".mp4", ".gif")))
    manifest = json.loads((tmp_path / "run_recording_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["streams"]) == 3
    assert manifest["playback_speed"] == 2.0
    assert manifest["playback_fps"] == 10.0
    assert not staging.exists()


def test_capture_metadata_round_trips_camera_pose(tmp_path):
    staging = tmp_path / "frames"
    staging.mkdir()
    metadata = {
        "timestamp": 10.0,
        "vehicle": "Drone1",
        "camera": "mission_follow",
        "camera_position": [1.0, 2.0, -3.0],
        "camera_orientation_quaternion": [1.0, 0.0, 0.0, 0.0],
    }
    (staging / "capture_metadata.jsonl").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    result = load_capture_metadata(str(staging), "Drone1", "mission_follow")
    assert result == [metadata]


def test_camera_alignment_report_flags_large_pose_error(tmp_path):
    staging = tmp_path / "frames"
    staging.mkdir()
    (staging / "capture_metadata.jsonl").write_text(json.dumps({
        "timestamp": 0.0,
        "vehicle": "Drone1",
        "camera": "mission_follow",
        "camera_position": [4.0, 0.0, 0.0],
    }) + "\n", encoding="utf-8")
    records = [{
        "timestamp": 0.0,
        "recording": {"chase_camera": {"world_position": [0.0, 0.0, 0.0]}},
    }]
    paths = analyze_camera_alignment(records, str(staging), str(tmp_path), "run", vehicle="Drone1")
    report = json.loads((tmp_path / "run_camera_alignment.json").read_text(encoding="utf-8"))
    assert paths[0].endswith("run_camera_alignment.json")
    assert report["pass"] is False
    assert report["max_position_error_m"] == 4.0


def test_chase_overlay_draws_red_target_marker_separately_from_robot_markers():
    pytest.importorskip("PIL")
    from PIL import Image

    records = [{
        "timestamp": 0.0,
        "states": {
            "Drone1": {"position": [2.0, 0.0, 0.0]},
            "Husky1": {"position": [2.0, 1.0, 0.0]},
        },
        "targets": {"Target1": {"name": "Target1", "position": [2.0, -1.0, 0.0]}},
        "recording": {"chase_camera": {
            "world_position": [0.0, 0.0, 0.0],
            "orientation_quaternion": [1.0, 0.0, 0.0, 0.0],
        }},
    }]
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    make_chase_overlay(records, {"Drone1": "drone", "Husky1": "ugv"}, 100, 100)(image, 0.0)
    pixels = list(image.getdata())
    assert (255, 45, 45) in pixels
    assert (30, 255, 90) in pixels
    assert (40, 150, 255) in pixels
