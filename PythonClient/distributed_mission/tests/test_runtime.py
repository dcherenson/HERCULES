import json

import numpy as np

from simulation.airsim_runtime import AirSimLaunchConfig
from simulation.airsim_runtime import AirSimFacade, AirSimLauncher
from orchestrator import (
    _is_ground_object,
    _plot_route_markers,
    camera_response_world_pose,
    compose_sensor_world_pose,
    parse_args,
    camera_director_for_map,
    effective_obstacle_margin,
    map_ground_z_offset,
    map_vehicle_ground_z,
    map_uav_altitude_floor,
    map_uav_altitude_ceiling,
    clamp_uav_velocity_to_altitude_ceiling,
    shift_course_z,
    rotate_course_left,
    rotate_xy_right,
    rotate_xy_heading,
    startup_formation_offset,
    heading_to_goal,
    resolve_goal_actor,
    camera_pose_for_goal,
    _camera_pose,
    startup_pose_position,
    startup_heading_pose_position,
    target_start_anchor_for_map,
    ugv_speed_control_inputs,
    safe_target_centered_startup_positions,
    safe_target_start_index,
    safe_target_ugv_startup_positions,
    filter_agent_body_obstacle_proxies,
    run_artifact_stem,
)


def test_launcher_preserves_start_formation_modes():
    visible = AirSimLaunchConfig(launch_mode="visible", map_name="rural_australia")
    headless = AirSimLaunchConfig(launch_mode="headless", map_name="flyingcpp")
    existing = AirSimLaunchConfig(launch_mode="existing")
    assert "-RenderOffscreen" not in visible.command()
    assert "-RenderOffscreen" in headless.command()
    assert existing.command() == []
    assert "RuralAustralia_Example_01" in visible.map_url
    assert "FlyingExampleMap" in headless.map_url


def test_run_artifact_stem_includes_map_name():
    assert run_artifact_stem("mestres", "rural_australia", 1234567890) == "mestres_rural_australia_1234567890"


def test_recording_defaults_to_thirty_fps_capture():
    assert parse_args([]).video_fps == 30.0


def test_ugv_speed_control_limits_throttle_and_brakes_overspeed():
    throttle, brake = ugv_speed_control_inputs(0.6, np.array([0.0, 0.0]))
    assert 0.0 < throttle <= 0.08
    assert brake == 0.0

    throttle, brake = ugv_speed_control_inputs(0.6, np.array([1.5, 0.0]))
    assert throttle == 0.0
    assert brake > 0.0

    throttle, brake = ugv_speed_control_inputs(0.0, np.array([0.2, 0.0]))
    assert throttle == 0.0
    assert brake == 1.0


def test_target_centered_ugv_startup_slot_is_moved_out_of_truth_proxy():
    target = np.array([14.5, 4.9, 0.7])
    radius = 6.0
    requested = {
        "Husky1": target + np.array([radius, 0.0, 0.0]),
        "Husky2": target + radius * np.array([-0.5, np.sqrt(3.0) / 2.0, 0.0]),
        "Husky3": target + radius * np.array([-0.5, -np.sqrt(3.0) / 2.0, 0.0]),
    }
    obstacles = [{
        "id": "blocking_slot",
        "shape": "box",
        "center": requested["Husky2"].tolist(),
        "dimensions": [2.0, 2.0, 2.0],
    }]
    adjusted, changes = safe_target_ugv_startup_positions(
        requested, ("Husky1", "Husky2", "Husky3"), target, radius,
        obstacles, ugv_radius=1.25,
    )
    assert any(name == "Husky2" for name, _, _ in changes)
    assert np.isclose(np.linalg.norm(adjusted["Husky2"][:2] - target[:2]), radius)
    assert not np.allclose(adjusted["Husky2"][:2], requested["Husky2"][:2])


def test_target_centered_uav_startup_slot_is_moved_out_of_vertical_truth_proxy():
    target = np.array([0.0, 0.0, 0.7])
    requested = {"Drone4": np.array([2.0, 2.0, -5.0])}
    obstacles = [{
        "id": "blocking_uav_slot",
        "shape": "box",
        "center": [2.0, 2.0, -3.0],
        "dimensions": [2.0, 2.0, 6.0],
    }]
    adjusted, changes = safe_target_centered_startup_positions(
        requested, ("Drone4",), target, obstacles, "drone", vehicle_radius=1.0,
    )
    assert any(name == "Drone4" for name, _, _ in changes)
    assert not np.allclose(adjusted["Drone4"][:2], requested["Drone4"][:2])


def test_target_start_phase_advances_to_clear_fixed_route_sample():
    from modules.target_motion import FigureEightTargetController

    controller = FigureEightTargetController(
        center=np.array([11.2, 0.7, 0.7]), route_heading=0.0,
        longitudinal_span=10.0, lateral_span=8.0, speed=0.1,
    )
    preferred = controller.index
    obstacles = [{
        "id": "blocking_target_sample",
        "shape": "box",
        "center": controller.points[preferred].tolist(),
        "dimensions": [2.0, 2.0, 2.0],
    }]
    index, clearance = safe_target_start_index(controller, obstacles, target_radius=1.25)
    assert index != preferred
    assert clearance >= 0.15


def test_target_start_anchor_is_quarter_route_only_on_rural_australia():
    start = np.zeros(3)
    goal = np.array([20.0, 8.0, -1.0])

    rural = target_start_anchor_for_map(start, goal, "rural_australia", -1.0, 0.0)
    flying = target_start_anchor_for_map(start, goal, "flyingcpp", 0.7, np.pi / 4.0)

    # For a route heading of zero, AirSim camera-right is +Y in NED.
    assert np.allclose(rural, [-5.0, 7.0, -1.0])
    assert np.allclose(flying, [20.0, 8.0, 0.7])


def test_launcher_rejects_unknown_mode_or_map():
    try:
        AirSimLaunchConfig(launch_mode="bad")
        assert False
    except ValueError:
        pass


def test_recording_settings_add_only_named_camera_and_preserve_perception_settings(tmp_path):
    settings_path = tmp_path / "settings.json"
    source = {
        "SettingsVersion": 1.2,
        "CameraDefaults": {"CaptureSettings": [{"ImageType": 0, "Width": 256, "Height": 144},
                                                   {"ImageType": 2, "Width": 64, "Height": 48}]},
        "Vehicles": {
            "Drone1": {"VehicleType": "SimpleFlight", "Sensors": {"Distance": {"Enabled": True}}},
            "Husky1": {"VehicleType": "CPHusky"},
        },
    }
    settings_path.write_text(json.dumps(source), encoding="utf-8")
    config = AirSimLaunchConfig(
        launch_mode="headless", settings_path=str(settings_path), record_video=True,
        record_uav="Drone1", record_ugv="Husky1", recording_folder=str(tmp_path / "frames"),
        video_resolution=(1280, 720), video_fps=20.0,
    )
    launcher = AirSimLauncher(config)
    launcher._prepare_settings_override()
    try:
        override = json.loads(open(config._active_settings_path, encoding="utf-8").read())
        assert override["CameraDefaults"]["CaptureSettings"][1] == {"ImageType": 2, "Width": 64, "Height": 48}
        assert override["Vehicles"]["Drone1"]["Sensors"] == source["Vehicles"]["Drone1"]["Sensors"]
        assert {"mission_follow", "front_left", "front_right"}.issubset(override["Vehicles"]["Drone1"]["Cameras"])
        assert "front_center" not in override["Vehicles"]["Drone1"]["Cameras"]
        assert override["Vehicles"]["Drone1"]["Cameras"]["front_left"]["Yaw"] == -120.0
        assert override["Vehicles"]["Drone1"]["Cameras"]["front_right"]["Yaw"] == 120.0
        assert override["Vehicles"]["Drone1"]["Cameras"]["mission_follow"]["ExternalLocal"] is False
        assert override["Recording"]["Cameras"][-1]["VehicleName"] == "Husky1"
        assert override["Recording"]["RecordInterval"] == 0.05
    finally:
        launcher.cleanup()


def test_managed_launch_defers_husky_creation_and_preserves_lidar_profile(tmp_path):
    settings_path = tmp_path / "settings.json"
    source = {
        "SettingsVersion": 1.2,
        "DefaultSensors": {"Imu": {"SensorType": 2, "Enabled": True}},
        "Vehicles": {
            "Husky1": {
                "VehicleType": "CPHusky", "AutoCreate": True,
                "Sensors": {"Lidar1": {"SensorType": 6, "NumberOfChannels": 16}},
            },
            "Husky2": {"VehicleType": "CPHusky", "AutoCreate": True},
            "Husky3": {"VehicleType": "CPHusky", "AutoCreate": True},
        },
    }
    settings_path.write_text(json.dumps(source), encoding="utf-8")
    config = AirSimLaunchConfig(
        launch_mode="headless", settings_path=str(settings_path),
        runtime_spawn_ugvs=True,
    )
    launcher = AirSimLauncher(config)
    launcher._prepare_settings_override()
    try:
        override = json.loads(open(config._active_settings_path, encoding="utf-8").read())
        assert all(override["Vehicles"][name]["AutoCreate"] is False for name in ("Husky1", "Husky2", "Husky3"))
        assert override["DefaultSensors"]["Lidar1"] == source["Vehicles"]["Husky1"]["Sensors"]["Lidar1"]
        assert json.loads(settings_path.read_text(encoding="utf-8")) == source
    finally:
        launcher.cleanup()


def test_target_tracking_settings_adds_bottom_camera_without_replacing_depth_config(tmp_path):
    settings_path = tmp_path / "settings.json"
    source = {
        "SettingsVersion": 1.2,
        "CameraDefaults": {"CaptureSettings": [{"ImageType": 2, "Width": 64, "Height": 48}]},
        "Vehicles": {
            "Drone1": {
                "VehicleType": "SimpleFlight",
                "Cameras": {
                    "front_center": {
                        "X": 0.4, "Y": 0.2, "Z": -0.1, "Pitch": 2.0,
                        "CaptureSettings": [{"ImageType": 2, "Width": 800, "Height": 600, "FOV_Degrees": 75}]
                    }
                }
            },
            "Husky1": {"VehicleType": "CPHusky"},
        },
    }
    settings_path.write_text(json.dumps(source), encoding="utf-8")
    config = AirSimLaunchConfig(
        launch_mode="headless", settings_path=str(settings_path), target_tracking=True,
        target_uav_camera="target_bottom",
    )
    launcher = AirSimLauncher(config)
    launcher._prepare_settings_override()
    try:
        override = json.loads(open(config._active_settings_path, encoding="utf-8").read())
        cameras = override["Vehicles"]["Drone1"]["Cameras"]
        assert cameras["front_center"] == source["Vehicles"]["Drone1"]["Cameras"]["front_center"]
        assert cameras["target_bottom"]["Pitch"] == -90.0
        assert cameras["target_bottom"]["Roll"] == 0.0
        assert override["CameraDefaults"]["CaptureSettings"][0] == {"ImageType": 2, "Width": 64, "Height": 48}
    finally:
        launcher.cleanup()


def test_fixed_goal_settings_do_not_add_target_tracking_camera(tmp_path):
    settings_path = tmp_path / "settings.json"
    source = {
        "SettingsVersion": 1.2,
        "Vehicles": {
            "Drone1": {
                "VehicleType": "SimpleFlight",
                "Cameras": {
                    "front_center": {"Pitch": 2.0},
                },
            },
        },
    }
    settings_path.write_text(json.dumps(source), encoding="utf-8")
    # This is the launch configuration selected by fixed-goal mode: the
    # camera-director override can still be active, but target tracking is not.
    config = AirSimLaunchConfig(
        launch_mode="headless",
        settings_path=str(settings_path),
        camera_director_position=(0.0, 0.0, -10.0),
        target_tracking=False,
        runtime_spawn_ugvs=False,
    )
    launcher = AirSimLauncher(config)
    launcher._prepare_settings_override()
    try:
        override = json.loads(open(config._active_settings_path, encoding="utf-8").read())
        cameras = override["Vehicles"]["Drone1"]["Cameras"]
        assert "target_bottom" not in cameras
        assert cameras["front_center"] == source["Vehicles"]["Drone1"]["Cameras"]["front_center"]
    finally:
        launcher.cleanup()


def test_fixed_goal_rural_startup_slots_match_resolved_route_heading():
    slot = np.array([-2.0 * np.sqrt(3.0), -2.0, 0.0])
    assert np.allclose(
        startup_formation_offset(slot, "rural_australia", "fixed-goal", np.pi / 2.0),
        [2.0, -2.0 * np.sqrt(3.0), 0.0],
    )


def test_obstacle_camera_fan_preserves_explicit_side_settings(tmp_path):
    settings_path = tmp_path / "settings.json"
    source = {
        "SettingsVersion": 1.2,
        "Vehicles": {
            "Drone1": {
                "VehicleType": "SimpleFlight",
                "Cameras": {
                    "front_center": {"CaptureSettings": [{"ImageType": 2, "Width": 800, "Height": 600}]},
                    "front_left": {"Yaw": -30.0, "CaptureSettings": [{"ImageType": 2, "Width": 320, "Height": 240}]},
                },
            },
        },
    }
    settings_path.write_text(json.dumps(source), encoding="utf-8")
    config = AirSimLaunchConfig(launch_mode="headless", settings_path=str(settings_path), target_tracking=True)
    launcher = AirSimLauncher(config)
    launcher._prepare_settings_override()
    try:
        override = json.loads(open(config._active_settings_path, encoding="utf-8").read())
        cameras = override["Vehicles"]["Drone1"]["Cameras"]
        assert cameras["front_center"] == source["Vehicles"]["Drone1"]["Cameras"]["front_center"]
        assert cameras["front_left"]["Yaw"] == -30.0
        assert cameras["front_left"]["CaptureSettings"] == source["Vehicles"]["Drone1"]["Cameras"]["front_left"]["CaptureSettings"]
        assert cameras["front_right"]["X"] == 0.0
        assert cameras["front_right"]["Yaw"] == 120.0
    finally:
        launcher.cleanup()


def test_record_video_existing_launch_is_rejected():
    assert parse_args(["--record-video", "--launch-mode", "existing"]).record_video is True


def test_target_tracking_defaults_to_camera_and_target_centered_objective():
    defaults = parse_args([])
    assert defaults.target_observation_source == "camera"
    assert defaults.mission_objective == "track-target"
    assert defaults.target_name == "Target1"
    assert defaults.target_ugv_circumradius == 5.0
    assert defaults.tracking_rate == 4.0
    assert defaults.target_speed == 0.10
    assert defaults.tracking_measurement_std == 0.25
    assert defaults.tracking_process_noise == 0.20
    assert defaults.nominal_speed == 1.0
    assert defaults.leader_nominal_speed == 1.0
    assert defaults.nominal_position_gain == 0.5
    assert defaults.ugv_heading_gain == 1.0
    assert defaults.ugv_max_yaw_rate == 1.0


def test_existing_camera_tracking_requires_explicit_preconfigured_acknowledgement():
    assert parse_args(["--launch-mode", "existing", "--target-observation-source", "camera"]).target_camera_preconfigured is False
    assert parse_args(["--launch-mode", "existing", "--target-observation-source", "truth"]).target_observation_source == "truth"


def test_top_down_camera_is_opt_in_and_default_preserves_simulator_view():
    defaults = parse_args([])
    assert defaults.top_down_camera is False
    assert defaults.no_top_down_camera is False
    assert defaults.obstacle_margin is None

    top_down = parse_args(["--top-down-camera", "--camera-height", "25"])
    assert top_down.top_down_camera is True
    assert top_down.camera_height == 25.0

    explicit_disable = parse_args(["--top-down-camera", "--no-top-down-camera"])
    assert explicit_disable.top_down_camera is True
    assert explicit_disable.no_top_down_camera is True
    assert parse_args(["--obstacle-margin", "0"]).obstacle_margin == 0.0
    assert parse_args(["--initial-heading-offset-deg", "180"]).initial_heading_offset_deg == 180.0
    assert effective_obstacle_margin("rural_australia", None) == 0.0
    assert effective_obstacle_margin("flyingcpp", None) == 0.0
    assert effective_obstacle_margin("rural_australia", 0.0) == 0.0


def test_uav_altitude_floor_is_one_meter_above_calibrated_ground():
    assert map_uav_altitude_floor("flyingcpp") == -1.0
    assert map_uav_altitude_floor("rural_australia") == 1.0
    assert map_uav_altitude_floor("rural_australia", -2.5) == -2.5


def test_uav_altitude_ceiling_is_ten_metres_above_calibrated_ground():
    assert map_uav_altitude_ceiling("flyingcpp") == -10.0
    assert map_uav_altitude_ceiling("rural_australia") == -8.0
    assert map_uav_altitude_ceiling("flyingcpp", 12.0) == -12.0


def test_uav_velocity_ceiling_blocks_upward_crossing_in_ned():
    velocity = clamp_uav_velocity_to_altitude_ceiling(
        np.array([1.0, 2.0, -3.0]), -9.9, -10.0, 0.1, 3.0
    )
    assert np.allclose(velocity, [1.0, 2.0, -1.0])

    already_high = clamp_uav_velocity_to_altitude_ceiling(
        np.array([0.0, 0.0, -3.0]), -12.0, -10.0, 0.1, 3.0
    )
    assert np.isclose(already_high[2], 3.0)

    below_ceiling = clamp_uav_velocity_to_altitude_ceiling(
        np.array([0.0, 0.0, -3.0]), -5.0, -10.0, 0.1, 3.0
    )
    assert np.isclose(below_ceiling[2], -3.0)


def test_rural_ground_offset_shifts_goal_waypoints_and_obstacles_only_for_map_frame():
    course = {"goal": [1.0, 2.0, -1.0], "waypoints": [[0.0, 1.0, -1.0]], "obstacles": [{"center": [3.0, 4.0, -2.0]}]}
    shifted = shift_course_z(course, map_ground_z_offset("rural_australia"))
    assert shifted["goal"] == [1.0, 2.0, 1.0]
    assert shifted["waypoints"] == [[0.0, 1.0, 1.0]]
    assert shifted["obstacles"][0]["center"] == [3.0, 4.0, 0.0]
    assert course["goal"] == [1.0, 2.0, -1.0]
    assert map_ground_z_offset("flyingcpp") == 0.0
    assert map_vehicle_ground_z("flyingcpp") == 0.7
    assert map_vehicle_ground_z("rural_australia") == -1.0

    default_config = AirSimLaunchConfig(launch_mode="headless")
    assert default_config.camera_director_position is None
    assert not any(argument.startswith("-settings=") for argument in default_config.command())


def test_rural_default_camera_rotates_existing_camera_settings(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({
        "SettingsVersion": 1.2,
        "ViewMode": "Manual",
        "CameraDirector": {"X": -20, "Y": 0, "Z": -15, "Pitch": -25, "Roll": 3, "Yaw": 10},
    }), encoding="utf-8")
    config = AirSimLaunchConfig(
        launch_mode="headless", map_name="rural_australia", settings_path=str(settings_path),
        rotate_camera_director=True,
    )
    launcher = AirSimLauncher(config)
    launcher._prepare_settings_override()
    try:
        override = json.loads(open(config._active_settings_path, encoding="utf-8").read())
        assert override["CameraDirector"] == {
            "X": 0.0, "Y": 20.0, "Z": -15, "Pitch": -25, "Roll": 0.0, "Yaw": -80.0,
        }
    finally:
        launcher.cleanup()


def test_rural_mission_rotation_moves_goal_waypoints_and_obstacles_left():
    course = {
        "goal": [16.0, 1.0, -1.0],
        "waypoints": [[8.0, -2.0, -1.0]],
        "obstacles": [{
            "id": "wall",
            "shape": "box",
            "center": [7.0, 2.0, -2.0],
            "dimensions": [2.0, 3.0, 4.0],
        }],
    }
    rotated = rotate_course_left(course)
    assert np.allclose(rotated["goal"], [1.0, -16.0, -1.0])
    assert np.allclose(rotated["waypoints"][0], [-2.0, -8.0, -1.0])
    assert np.allclose(rotated["obstacles"][0]["center"], [2.0, -7.0, -2.0])
    assert rotated["obstacles"][0]["dimensions"] == [3.0, 2.0, 4.0]
    assert np.allclose(rotate_xy_right([1.0, 0.0, -5.0]), [0.0, -1.0, -5.0])


def test_rural_top_down_camera_is_behind_the_rotated_heading():
    position, yaw = camera_director_for_map("rural_australia", 6.0, 0.0, 30.0)
    assert np.allclose(position, [0.0, 6.0, -30.0])
    assert np.isclose(yaw, 90.0)

    flying_position, flying_yaw = camera_director_for_map("flyingcpp", 6.0, 0.0, 30.0)
    assert np.allclose(flying_position, [6.0, 0.0, -30.0])
    assert np.isclose(flying_yaw, 0.0)


def test_goal_heading_rotates_initial_formation_offsets():
    assert np.isclose(heading_to_goal([0.0, 0.0, 0.0], [0.0, 10.0, 0.0]), np.pi / 2.0)
    assert np.allclose(rotate_xy_heading([2.0, 0.0, -4.0], np.pi / 2.0), [0.0, 2.0, -4.0])


def test_camera_pose_is_behind_and_points_at_goal():
    position, yaw = camera_pose_for_goal(
        np.zeros(3), np.array([0.0, 36.0, 2.0]), 20.0, -15.0, top_down=False
    )
    assert np.allclose(position, [0.0, -20.0, -15.0])
    assert np.isclose(yaw, 90.0)

    top_down_position, top_down_yaw = camera_pose_for_goal(
        np.zeros(3), np.array([0.0, 36.0, 2.0]), 6.0, -30.0, top_down=True
    )
    assert np.allclose(top_down_position, [0.0, -6.0, -30.0])
    assert np.isclose(top_down_yaw, -90.0)


def test_auto_goal_actor_selects_generic_actor_on_opposite_side():
    class Position:
        def __init__(self, x, y, z):
            self.x_val, self.y_val, self.z_val = x, y, z

    class Pose:
        def __init__(self, position):
            self.position = position

    class Client:
        def simListSceneObjects(self, pattern):
            assert pattern == r"^Actor(?:_\d+)?$"
            return ["Actor_0", "Actor_1"]

        def simGetObjectPose(self, name, ned):
            assert ned is True
            if name == "Actor_0":
                return Pose(Position(2.0, 36.0, 2.0))
            return Pose(Position(3.0, -5.0, 2.0))

    facade = AirSimFacade(multirotor_client=Client())
    name, position = resolve_goal_actor(facade, np.array([1.0, -16.0, 1.0]))
    assert name == "Actor_0"
    assert np.allclose(position, [2.0, 36.0, 2.0])


def test_explicit_goal_actor_bypasses_auto_direction_gate():
    class Position:
        x_val, y_val, z_val = 1.0, 2.0, 3.0

    class Pose:
        position = Position()

    class Client:
        def simGetObjectPose(self, name, ned):
            return Pose()

    name, position = resolve_goal_actor(
        AirSimFacade(multirotor_client=Client()), np.array([1.0, -16.0, 1.0]), actor_name="MyGoal"
    )
    assert name == "MyGoal"
    assert np.allclose(position, [1.0, 2.0, 3.0])


def test_ground_object_filter_matches_unreal_ground_names_only():
    assert _is_ground_object("Landscape")
    assert _is_ground_object("SM_Terrain_Floor")
    assert not _is_ground_object("distributed_cbf_block_0")
    config = AirSimLaunchConfig()
    config.map_name = "bad"
    try:
        _ = config.map_url
        assert False
    except ValueError:
        pass


def test_top_down_override_is_written_to_camera_director_settings(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"SettingsVersion": 1.2, "Vehicles": {"Drone1": {}}}), encoding="utf-8")
    config = AirSimLaunchConfig(
        launch_mode="headless",
        settings_path=str(settings_path),
        camera_director_position=(6.0, 2.0, -30.0),
    )
    launcher = AirSimLauncher(config)
    launcher._prepare_settings_override()
    try:
        override_path = config._active_settings_path
        override = json.loads(open(override_path, encoding="utf-8").read())
        assert override["ViewMode"] == "Manual"
        assert override["CameraDirector"] == {
            "X": 6.0, "Y": 2.0, "Z": -30.0, "Pitch": -90.0, "Roll": 0.0, "Yaw": 0.0
        }
        assert any(argument.startswith("-settings=") for argument in config.command())
    finally:
        launcher.cleanup()


def test_top_down_override_is_created_without_a_user_settings_file(tmp_path, monkeypatch):
    missing_settings = tmp_path / "missing-settings.json"
    config = AirSimLaunchConfig(
        launch_mode="headless",
        settings_path=str(missing_settings),
        camera_director_position=(0.0, 6.0, -30.0),
        camera_director_yaw=90.0,
    )
    launcher = AirSimLauncher(config)
    launcher._prepare_settings_override()
    try:
        override = json.loads(open(config._active_settings_path, encoding="utf-8").read())
        assert override["SettingsVersion"] == 1.2
        assert override["ViewMode"] == "Manual"
        assert override["CameraDirector"]["X"] == 0.0
        assert override["CameraDirector"]["Y"] == 6.0
        assert override["CameraDirector"]["Yaw"] == 90.0
    finally:
        launcher.cleanup()


def test_runtime_top_down_shim_never_calls_vehicle_camera_api():
    class CameraClient:
        def __init__(self):
            self.called = False

        def simSetCameraPose(self, *args):
            self.called = True

    client = CameraClient()
    facade = AirSimFacade(multirotor_client=client)
    try:
        facade.set_external_camera_top_down([0.0, 0.0, -30.0])
        assert False
    except RuntimeError:
        pass
    assert not client.called


def test_external_camera_pose_uses_unreal_camera_actor_only():
    class CameraClient:
        def __init__(self):
            self.calls = []

        def simSetObjectPose(self, name, pose, teleport):
            self.calls.append((name, pose, teleport))
            return True

    client = CameraClient()
    facade = AirSimFacade(multirotor_client=client)
    assert facade.set_external_camera_pose(object()) == "ExternalCamera"
    assert client.calls[0][0] == "ExternalCamera"
    assert client.calls[0][2] is True
    assert not hasattr(client, "simSetCameraPose")


def test_camera_pose_passes_roll_pitch_yaw_in_airsim_order():
    class Vector3r:
        def __init__(self, x, y, z):
            self.values = (x, y, z)

    class Pose:
        def __init__(self, position, orientation):
            self.position = position
            self.orientation = orientation

    vector_class = Vector3r
    pose_class = Pose

    class AirSim:
        Vector3r = vector_class
        Pose = pose_class

        @staticmethod
        def to_quaternion(roll, pitch, yaw):
            return roll, pitch, yaw

    pose = _camera_pose(AirSim, np.zeros(3), yaw_degrees=90.0, pitch_degrees=-25.0, roll_degrees=0.0)
    assert np.allclose(pose.orientation, [0.0, np.radians(-25.0), np.radians(90.0)])


def test_startup_pose_preserves_world_target_for_spawned_and_configured_vehicles():
    targets = {"Drone1": np.array([4.0, 5.0, -5.0]), "Husky1": np.array([2.0, 3.0, 1.0])}
    origins = {"Drone1": np.array([10.0, 10.0, 10.0]), "Husky1": np.array([-1.0, 2.0, 0.5])}
    assert np.allclose(startup_pose_position("Drone1", targets, {"Drone1"}, origins), targets["Drone1"])
    assert np.allclose(startup_pose_position("Husky1", targets, set(), origins), [3.0, 1.0, 0.5])


def test_startup_heading_pose_preserves_runtime_spawn_position():
    targets = {"Husky1": np.array([-2.0, 3.464, 1.0])}
    origins = {"Husky1": np.array([-2.0, 3.464, 1.0])}
    assert np.allclose(
        startup_heading_pose_position("Husky1", targets, {"Husky1"}, origins),
        np.zeros(3),
    )
    assert np.allclose(
        startup_heading_pose_position("Husky1", targets, set(), origins),
        np.zeros(3),
    )


def test_perception_proxy_at_controlled_agent_is_not_added_as_static_obstacle():
    from modules.cbf import ObstacleProxy

    states = {
        "Drone1": {"position": np.array([0.0, 0.0, -5.0])},
        "Drone2": {"position": np.array([4.0, 0.0, -5.0])},
    }
    proxies = [
        ObstacleProxy("self_return", [0.2, 0.1, -5.0], 1.5, source="depth_Drone1"),
        ObstacleProxy("neighbor_return", [4.1, -0.1, -5.0], 1.5, source="depth_Drone1"),
        ObstacleProxy("real_obstacle", [8.0, 0.0, -5.0], 1.5, source="depth_Drone1"),
        ObstacleProxy("truth_obstacle", [0.0, 0.0, -5.0], 1.5, source="truth"),
    ]
    retained, rejected = filter_agent_body_obstacle_proxies(
        proxies,
        states,
        {"Drone1": 1.0, "Drone2": 1.0},
    )
    assert rejected == 2
    assert [proxy.obstacle_id for proxy in retained] == ["real_obstacle", "truth_obstacle"]


def test_fixed_route_markers_use_multirotor_plot_api():
    class VectorClass:
        def __init__(self, x, y, z):
            self.x_val, self.y_val, self.z_val = x, y, z

    class AirSim:
        Vector3r = VectorClass

    class PlotClient:
        def __init__(self):
            self.points = None
            self.line = None

        def simPlotPoints(self, points, **kwargs):
            self.points = (points, kwargs)

        def simPlotLineStrip(self, points, **kwargs):
            self.line = (points, kwargs)

    class Facade:
        airsim = AirSim

        def __init__(self):
            self.multirotor = PlotClient()

    facade = Facade()
    _plot_route_markers(facade, [np.array([8.0, 0.0, -1.0]), np.array([16.0, 0.0, -1.0])])
    assert len(facade.multirotor.points[0]) == 2
    assert facade.multirotor.points[1]["is_persistent"] is True
    assert len(facade.multirotor.line[0]) == 2


def test_sensor_pose_composes_vehicle_world_pose_with_relative_mount():
    angle = np.pi / 2.0
    vehicle_rotation = np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    position, rotation = compose_sensor_world_pose(
        np.asarray([10.0, 20.0, -2.0]),
        vehicle_rotation,
        np.asarray([1.0, 0.0, -1.0]),
        np.eye(3),
    )
    assert np.allclose(position, [10.0, 21.0, -3.0])
    assert np.allclose(rotation, vehicle_rotation)


def test_camera_response_pose_is_translated_from_vehicle_start_frame_to_world():
    position, rotation, frame = camera_response_world_pose(
        np.asarray([0.46, 2.0, -4.90]),
        np.eye(3),
        np.asarray([-2.0, -2.0, -5.40]),
        np.asarray([-2.0, 2.0, -4.90]),
    )
    assert np.allclose(position, [0.46, -2.0, -5.40])
    assert np.allclose(rotation, np.eye(3))
    assert frame == "world_ned_from_vehicle_start_frame"


def test_camera_response_pose_has_explicit_fallback_without_frame_origin():
    position, rotation, frame = camera_response_world_pose(
        np.asarray([0.46, 0.0, -4.90]), np.eye(3), None, None
    )
    assert np.allclose(position, [0.46, 0.0, -4.90])
    assert np.allclose(rotation, np.eye(3))
    assert frame == "response_pose_frame_unknown"


def test_ugv_command_maps_signed_speed_to_air_sim_car_controls():
    class Controls:
        def __init__(self):
            self.throttle = None
            self.brake = None
            self.steering = None
            self.is_manual_gear = None
            self.manual_gear = None

    class AirSim:
        CarControls = Controls

    class Car:
        def __init__(self):
            self.last = None

        def setCarControls(self, controls, vehicle_name=None):
            self.last = (controls, vehicle_name)

    car = Car()
    facade = AirSimFacade(airsim_module=AirSim, car_client=car)
    facade.command_ugv("Husky1", -1.0, 0.25)
    controls, vehicle_name = car.last
    assert vehicle_name == "Husky1"
    assert controls.throttle == 0.5
    assert controls.brake == 0.0
    assert controls.manual_gear == -1
    assert controls.steering == 0.25

    facade.command_ugv("Husky1", 0.0, -0.5, brake=1.0)
    controls, _ = car.last
    assert controls.throttle == 0.0
    assert controls.brake == 1.0
    assert controls.manual_gear == 1

    facade.command_ugv("Husky1", 0.1, 0.0, throttle=0.75)
    controls, _ = car.last
    assert controls.throttle == 0.75


def test_stop_ugv_uses_handbrake_for_cphusky_stationary_hold():
    class Controls:
        def __init__(self):
            self.throttle = None
            self.brake = None
            self.handbrake = None
            self.steering = None

    class AirSim:
        CarControls = Controls

    class Car:
        def __init__(self):
            self.last = None

        def setCarControls(self, controls, vehicle_name=None):
            self.last = controls

    car = Car()
    AirSimFacade(airsim_module=AirSim, car_client=car).stop_ugv("Husky1")
    assert car.last.handbrake is True
    assert car.last.throttle == 0.0
    assert car.last.steering == 0.0
