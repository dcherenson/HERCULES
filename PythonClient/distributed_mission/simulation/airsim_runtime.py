"""Shared AirSim launcher and Hero-mode client facade."""

from __future__ import annotations

import atexit
import json
import os
import queue
import socket
import subprocess
import threading
import time
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass
class AirSimLaunchConfig:
    launch_mode: str = "visible"
    map_name: str = "flyingcpp"
    unreal_editor_path: str = "/Users/Shared/Epic Games/UE_5.2/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
    uproject_path: Optional[str] = None
    project_root: Optional[str] = None
    host: str = "127.0.0.1"
    multirotor_port: int = 41451
    car_port: int = 41452
    resolution: tuple = (800, 600)
    startup_timeout: float = 120.0
    suppress_unreal_output: bool = True
    settings_path: Optional[str] = None
    camera_director_position: Optional[tuple] = None
    camera_director_yaw: float = 0.0

    def __post_init__(self) -> None:
        if self.launch_mode not in {"visible", "headless", "existing"}:
            raise ValueError("launch_mode must be visible, headless, or existing")
        if self.project_root is None:
            self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        if self.uproject_path is None:
            self.uproject_path = os.path.join(self.project_root, "Unreal", "Environments", "Blocks", "Blocks.uproject")
        if self.camera_director_position is not None:
            self.camera_director_position = tuple(float(value) for value in self.camera_director_position)
            if len(self.camera_director_position) != 3:
                raise ValueError("camera_director_position must contain three values")

    @property
    def map_url(self) -> str:
        maps = {
            "flyingcpp": "/Game/FlyingCPP/Maps/FlyingExampleMap",
            "rural_australia": "/Game/RuralAustralia/Maps/RuralAustralia_Example_01",
        }
        if self.map_name not in maps:
            raise ValueError("map_name must be flyingcpp or rural_australia")
        return maps[self.map_name] + "?game=/Script/AirSim.AirSimGameMode"

    def command(self) -> Sequence[str]:
        if self.launch_mode == "existing":
            return []
        command = [
            self.unreal_editor_path,
            self.uproject_path,
            self.map_url,
            "-game",
            "-windowed",
            "-resx={}".format(self.resolution[0]),
            "-resy={}".format(self.resolution[1]),
        ]
        settings_path = getattr(self, "_active_settings_path", None) or self.settings_path
        if settings_path:
            command.append("-settings={}".format(settings_path))
        if self.launch_mode == "headless":
            command.append("-RenderOffscreen")
        return command


class AirSimLauncher:
    def __init__(self, config: Optional[AirSimLaunchConfig] = None):
        self.config = config or AirSimLaunchConfig()
        self.process: Optional[subprocess.Popen] = None
        self._temporary_settings_path: Optional[str] = None

    def _prepare_settings_override(self) -> None:
        if self.config.launch_mode == "existing" or self.config.camera_director_position is None:
            return
        source_path = self.config.settings_path
        if source_path is None:
            source_path = os.path.join(os.path.expanduser("~"), "Documents", "AirSim", "settings.json")
        if not os.path.isfile(source_path):
            # AirSim can still launch with its defaults. Do not silently move a
            # vehicle camera just because a user's settings file is absent.
            return
        with open(source_path, "r", encoding="utf-8") as source:
            settings = json.load(source)
        settings["ViewMode"] = "Manual"
        camera_director = dict(settings.get("CameraDirector") or {})
        position = self.config.camera_director_position
        camera_director.update({
            "X": position[0],
            "Y": position[1],
            "Z": position[2],
            "Pitch": -90.0,
            "Roll": 0.0,
            "Yaw": float(self.config.camera_director_yaw),
        })
        settings["CameraDirector"] = camera_director
        temporary = tempfile.NamedTemporaryFile(
            mode="w", suffix="_airsim_top_down_settings.json", delete=False, encoding="utf-8"
        )
        with temporary:
            json.dump(settings, temporary, indent=2)
        self._temporary_settings_path = temporary.name
        self.config._active_settings_path = temporary.name

    def launch(self) -> None:
        if self.config.launch_mode != "existing":
            self._prepare_settings_override()
            stdout = subprocess.DEVNULL if self.config.suppress_unreal_output else None
            stderr = subprocess.DEVNULL if self.config.suppress_unreal_output else None
            self.process = subprocess.Popen(list(self.config.command()), stdout=stdout, stderr=stderr)
            atexit.register(self.cleanup)
        self.wait_for_rpc()

    def wait_for_rpc(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((self.config.host, self.config.multirotor_port), timeout=1.0):
                    return
            except OSError:
                time.sleep(1.0)
        raise TimeoutError("AirSim RPC server did not open port {} within {} seconds".format(self.config.multirotor_port, self.config.startup_timeout))

    def cleanup(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
        if self._temporary_settings_path:
            try:
                os.unlink(self._temporary_settings_path)
            except FileNotFoundError:
                pass
            self._temporary_settings_path = None
            if hasattr(self.config, "_active_settings_path"):
                delattr(self.config, "_active_settings_path")


class AirSimFacade:
    """One shared simulator interface for all agents and sensors.

    Hero mode requires separate RPC services. The facade hides that detail and
    creates one MultirotorClient and one CarClient exactly once.
    """

    def __init__(self, config: Optional[AirSimLaunchConfig] = None, airsim_module: Any = None, multirotor_client: Any = None, car_client: Any = None):
        self.config = config or AirSimLaunchConfig()
        self.airsim = airsim_module
        self.multirotor = multirotor_client
        self.car = car_client
        self._vehicle_names = set()

    def connect(self) -> None:
        if self.airsim is None:
            import hercules_cosysairsim as airsim
            self.airsim = airsim
        if self.multirotor is None:
            self.multirotor = self.airsim.MultirotorClient(port=self.config.multirotor_port)
        if self.car is None:
            self.car = self.airsim.CarClient(port=self.config.car_port)
        self.multirotor.confirmConnection()
        self.car.confirmConnection()
        try:
            self._vehicle_names = set(self.multirotor.listVehicles())
        except Exception:
            self._vehicle_names = set()

    def spawn_vehicle(self, name: str, vehicle_type: str, pose: Any) -> None:
        if name in self._vehicle_names:
            return
        vehicle_name = "simpleflight" if vehicle_type == "drone" else "cphusky"
        try:
            self.multirotor.simAddVehicle(name, vehicle_name, pose)
            self._vehicle_names.add(name)
        except Exception:
            # AirSim reports an error when the requested vehicle already exists.
            # Existing vehicles are intentionally reusable between runs.
            pass

    def spawn_object(self, name: str, asset_name: str, pose: Any, scale: Any) -> bool:
        result = self.multirotor.simSpawnObject(name, asset_name, pose, scale, False, False)
        return bool(result) if result is not None else True

    def set_vehicle_pose(self, name: str, pose: Any) -> None:
        self.multirotor.simSetVehiclePose(pose, True, vehicle_name=name)

    def delete_object(self, name: str) -> None:
        self.multirotor.simDestroyObject(name)

    def enable(self, name: str, vehicle_type: str, enabled: bool = True) -> None:
        client = self.multirotor if vehicle_type == "drone" else self.car
        client.enableApiControl(enabled, name)
        if enabled and vehicle_type == "drone":
            client.armDisarm(True, name)

    def state(self, name: str) -> Dict[str, Any]:
        kinematics = self.multirotor.simGetGroundTruthKinematics(vehicle_name=name)
        position = np.array([kinematics.position.x_val, kinematics.position.y_val, kinematics.position.z_val], dtype=float)
        velocity = np.array([kinematics.linear_velocity.x_val, kinematics.linear_velocity.y_val, kinematics.linear_velocity.z_val], dtype=float)
        angular_velocity = getattr(kinematics, "angular_velocity", None)
        yaw_rate = float(getattr(angular_velocity, "z_val", 0.0)) if angular_velocity is not None else 0.0
        yaw = 0.0
        if hasattr(self.airsim, "quaternion_to_euler_angles"):
            _, _, yaw = self.airsim.quaternion_to_euler_angles(kinematics.orientation)
        return {"position": position, "velocity": velocity, "yaw": float(yaw), "yaw_rate": yaw_rate, "kinematics": kinematics, "timestamp": time.time()}

    def collision_info(self, name: str) -> Dict[str, Any]:
        """Return authoritative AirSim collision state for one vehicle."""

        try:
            info = self.multirotor.simGetCollisionInfo(vehicle_name=name)
            return {
                "available": True,
                "has_collided": bool(getattr(info, "has_collided", False)),
                "object_name": str(getattr(info, "object_name", "")),
                "object_id": int(getattr(info, "object_id", -1)),
                "penetration_depth": float(getattr(info, "penetration_depth", 0.0)),
                "time_stamp": float(getattr(info, "time_stamp", 0.0)),
            }
        except Exception as error:
            return {"available": False, "has_collided": False, "error": str(error)}

    def set_external_camera_top_down(
        self,
        position: np.ndarray,
        yaw: float = 0.0,
    ) -> None:
        """Deprecated compatibility shim; configure the launch before connecting.

        AirSim's legacy empty-vehicle camera selector resolves to a vehicle
        camera in Hero mode. The top-down pose is therefore applied through
        ``CameraDirector`` in :class:`AirSimLaunchConfig` before Unreal starts.
        """

        raise RuntimeError(
            "top-down camera must be configured in AirSimLaunchConfig before launch; "
            "runtime camera pose would alter a vehicle-mounted sensor"
        )

    def command_uav(self, name: str, velocity: np.ndarray, duration: float) -> None:
        vector = np.asarray(velocity, dtype=float)
        self.multirotor.moveByVelocityAsync(float(vector[0]), float(vector[1]), float(vector[2]), duration, vehicle_name=name)

    def command_ugv(self, name: str, speed: float, steering: float, duration: float = 0.1) -> None:
        controls = self.airsim.CarControls()
        controls.throttle = float(np.clip(speed / 2.0, 0.0, 1.0))
        controls.steering = float(np.clip(steering, -1.0, 1.0))
        controls.is_manual_gear = True
        controls.manual_gear = 1
        self.car.setCarControls(controls, vehicle_name=name)

    def stop_ugv(self, name: str) -> None:
        controls = self.airsim.CarControls()
        controls.throttle = 0.0
        controls.brake = 1.0
        self.car.setCarControls(controls, vehicle_name=name)

    def pause(self, value: bool) -> None:
        self.multirotor.simPause(value)

    def continue_for(self, duration: float) -> None:
        self.multirotor.simContinueForTime(duration)

    def close(self, names: Sequence[tuple]) -> None:
        for name, vehicle_type in names:
            try:
                if vehicle_type == "ugv":
                    self.stop_ugv(name)
                self.enable(name, vehicle_type, False)
            except Exception:
                pass


class AsyncJsonlWriter:
    """Write diagnostics off the control thread with bounded buffering."""

    def __init__(self, path: str, flush_interval: float = 1.0, max_queue: int = 256):
        self.path = path
        self.flush_interval = flush_interval
        self._queue = queue.Queue(maxsize=max_queue)
        self._stop = object()
        self.dropped_records = 0
        self._thread = threading.Thread(target=self._run, name="cbf-jsonl-writer", daemon=True)
        self._thread.start()

    def write(self, record: Dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self.dropped_records += 1

    def close(self) -> None:
        self._queue.put(self._stop)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        with open(self.path, "w", encoding="utf-8", buffering=1) as output:
            last_flush = time.monotonic()
            while True:
                try:
                    item = self._queue.get(timeout=self.flush_interval)
                except queue.Empty:
                    output.flush()
                    last_flush = time.monotonic()
                    continue
                if item is self._stop:
                    while True:
                        try:
                            item = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        output.write(json.dumps(item, default=_json_default) + "\n")
                    output.flush()
                    return
                output.write(json.dumps(item, default=_json_default) + "\n")
                if time.monotonic() - last_flush >= self.flush_interval:
                    output.flush()
                    last_flush = time.monotonic()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)
