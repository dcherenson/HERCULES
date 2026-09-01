import json
import os
import shutil

import numpy as np
import pytest

from modules.video_recording import (
    CHASE_GROUND_VEHICLE_MARKER_HEIGHT_M,
    FollowCameraController,
    RecordedFrame,
    analyze_camera_alignment,
    camera_position_to_world,
    find_recorded_frames,
    load_capture_metadata,
    make_chase_overlay,
    project_world_point,
    retime_recorded_frames,
    render_recordings,
    observed_frame_rate,
    _presentation_marker_position,
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


def test_recorded_frames_are_retimed_from_wall_clock_to_mission_time():
    frames = [
        RecordedFrame(100.0, "first.png"),
        RecordedFrame(103.0, "second.png"),
        RecordedFrame(106.0, "third.png"),
    ]
    records = [
        {"timestamp": 0.0, "wall_timestamp": 100.0},
        {"timestamp": 1.0, "wall_timestamp": 103.0},
        {"timestamp": 2.0, "wall_timestamp": 106.0},
    ]
    retimed = retime_recorded_frames(frames, records)
    assert [frame.timestamp for frame in retimed] == [0.0, 1.0, 2.0]
    assert [frame.source_timestamp for frame in retimed] == [100.0, 103.0, 106.0]
    assert observed_frame_rate(frames) == pytest.approx(1.0 / 3.0)


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


def test_vehicle_start_frame_camera_position_is_resolved_to_world_ned():
    records = [{
        "timestamp": 0.0,
        "states": {
            "Drone1": {
                "actor_position": [10.0, -4.0, 2.0],
                "kinematics_position": [8.0, -1.0, 3.0],
            },
        },
    }]
    position, frame, origin = camera_position_to_world(
        [1.0, 2.0, 3.0], "vehicle_start_frame_ned", records, "Drone1"
    )
    assert np.allclose(origin, [2.0, -3.0, -1.0])
    assert np.allclose(position, [3.0, -1.0, 2.0])
    assert frame == "world_ned_from_vehicle_start_frame"


def test_external_chase_camera_position_stays_in_world_ned():
    records = [{
        "timestamp": 0.0,
        "states": {
            "Drone1": {
                "actor_position": [10.0, -4.0, 2.0],
                "kinematics_position": [8.0, -1.0, 3.0],
            },
        },
    }]
    position, frame, origin = camera_position_to_world(
        [4.0, 5.0, 6.0], "external_world_ned", records, "Drone1"
    )
    assert np.allclose(position, [4.0, 5.0, 6.0])
    assert frame == "external_world_ned"
    assert origin is None


def test_ground_vehicle_marker_uses_visual_chassis_anchor_only():
    point = _presentation_marker_position([2.0, 3.0, 4.0], "ugv")
    assert np.allclose(point, [2.0, 3.0, 4.0 - CHASE_GROUND_VEHICLE_MARKER_HEIGHT_M])
    assert np.allclose(
        _presentation_marker_position([2.0, 3.0, 4.0], "drone"),
        [2.0, 3.0, 4.0],
    )


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


def test_chase_overlay_can_disable_all_map_markers():
    pytest.importorskip("PIL")
    from PIL import Image

    records = [{
        "timestamp": 0.0,
        "states": {"Drone1": {"position": [2.0, 0.0, 0.0]}},
        "vehicle_types": {"Drone1": "drone"},
        "targets": {"Target1": {"name": "Target1", "position": [2.0, 1.0, 0.0]}},
        "recording": {"chase_camera": {
            "world_position": [0.0, 0.0, 0.0],
            "orientation_quaternion": [1.0, 0.0, 0.0, 0.0],
        }},
    }]
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    make_chase_overlay(records, {"Drone1": "drone"}, 100, 100, markers_enabled=False)(image, 0.0)
    assert set(image.getdata()) == {(0, 0, 0)}


def test_chase_overlay_interpolates_robot_pose_at_png_capture_time():
    pytest.importorskip("PIL")
    from PIL import Image

    records = [
        {
            "timestamp": 0.0,
            "states": {"Drone1": {"position": [2.0, -1.0, 0.0]}},
            "vehicle_types": {"Drone1": "drone"},
            "recording": {"chase_camera": {
                "world_position": [0.0, 0.0, 0.0],
                "orientation_quaternion": [1.0, 0.0, 0.0, 0.0],
            }},
        },
        {
            "timestamp": 1.0,
            "states": {"Drone1": {"position": [2.0, 1.0, 0.0]}},
            "vehicle_types": {"Drone1": "drone"},
            "recording": {"chase_camera": {
                "world_position": [0.0, 0.0, 0.0],
                "orientation_quaternion": [1.0, 0.0, 0.0, 0.0],
            }},
        },
    ]
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    make_chase_overlay(
        records, {"Drone1": "drone"}, 100, 100,
        camera_metadata=[{
            "timestamp": 0.5,
            "camera_position": [0.0, 0.0, 0.0],
            "camera_orientation_quaternion": [1.0, 0.0, 0.0, 0.0],
        }],
    )(image, 0.5)
    pixels = np.asarray(image)
    marker = (pixels[:, :, 0] == 30) & (pixels[:, :, 1] == 255) & (pixels[:, :, 2] == 90)
    ys, xs = np.where(marker)
    assert len(xs) > 0
    # At the midpoint the robot is [2, 0, 0], which projects to the image
    # center. Holding the t=0 pose would put the ring at x=25 instead.
    assert np.isclose(float(xs.mean()), 50.0, atol=1.0)


def test_chase_overlay_uses_wall_timestamp_for_recorded_pngs():
    pytest.importorskip("PIL")
    from PIL import Image

    records = [
        {
            "timestamp": 0.0,
            "wall_timestamp": 100.0,
            "states": {"Drone1": {"position": [2.0, -1.0, 0.0]}},
            "vehicle_types": {"Drone1": "drone"},
            "recording": {"chase_camera": {
                "world_position": [0.0, 0.0, 0.0],
                "orientation_quaternion": [1.0, 0.0, 0.0, 0.0],
            }},
        },
        {
            "timestamp": 1.0,
            "wall_timestamp": 101.0,
            "states": {"Drone1": {"position": [2.0, 1.0, 0.0]}},
            "vehicle_types": {"Drone1": "drone"},
            "recording": {"chase_camera": {
                "world_position": [0.0, 0.0, 0.0],
                "orientation_quaternion": [1.0, 0.0, 0.0, 0.0],
            }},
        },
    ]
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    make_chase_overlay(
        records, {"Drone1": "drone"}, 100, 100,
        camera_metadata=[{
            "timestamp": 100.5,
            "camera_position": [0.0, 0.0, 0.0],
            "camera_orientation_quaternion": [1.0, 0.0, 0.0, 0.0],
        }],
    )(image, 100.5)
    pixels = np.asarray(image)
    marker = (pixels[:, :, 0] == 30) & (pixels[:, :, 1] == 255) & (pixels[:, :, 2] == 90)
    ys, xs = np.where(marker)
    assert len(xs) > 0
    # The PNG timestamp is Unix time.  It must select the midpoint actor
    # pose, not the final mission record.
    assert np.isclose(float(xs.mean()), 50.0, atol=1.0)
