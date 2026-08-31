import json

import numpy as np

from simulation.airsim_runtime import AirSimLaunchConfig
from simulation.airsim_runtime import AirSimFacade, AirSimLauncher
from orchestrator import _is_ground_object, compose_sensor_world_pose


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
