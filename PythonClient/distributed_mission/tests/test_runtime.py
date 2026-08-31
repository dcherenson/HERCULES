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
    shift_course_z,
    rotate_course_left,
    rotate_xy_right,
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


def test_launcher_rejects_unknown_mode_or_map():
    try:
        AirSimLaunchConfig(launch_mode="bad")
        assert False
    except ValueError:
        pass


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
    assert effective_obstacle_margin("rural_australia", None) == 1.0
    assert effective_obstacle_margin("flyingcpp", None) == 0.0
    assert effective_obstacle_margin("rural_australia", 0.0) == 0.0


def test_rural_ground_offset_shifts_goal_waypoints_and_obstacles_only_for_map_frame():
    course = {"goal": [1.0, 2.0, -1.0], "waypoints": [[0.0, 1.0, -1.0]], "obstacles": [{"center": [3.0, 4.0, -2.0]}]}
    shifted = shift_course_z(course, map_ground_z_offset("rural_australia"))
    assert shifted["goal"] == [1.0, 2.0, 1.0]
    assert shifted["waypoints"] == [[0.0, 1.0, 1.0]]
    assert shifted["obstacles"][0]["center"] == [3.0, 4.0, 0.0]
    assert course["goal"] == [1.0, 2.0, -1.0]
    assert map_ground_z_offset("flyingcpp") == 0.0

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
            "X": 0.0, "Y": 20.0, "Z": -15, "Pitch": -25, "Roll": 3, "Yaw": -80.0,
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
