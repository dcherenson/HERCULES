#!/usr/bin/env python3
"""Capture ROS mission camera streams and export separate presentation files."""

from __future__ import annotations

import os
import socket
from pathlib import Path
import sys
import time
from typing import Any, Dict

import numpy as np
import rclpy
from hercules_interfaces.msg import GroundTruthState
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


AGENTS = ["Drone1", "Drone2", "SimpleFlight",
          "Husky1", "Husky2", "Husky3", "Target1"]
CONTROLLED = AGENTS[:-1]


def _source_modules():
    roots = [Path(os.environ.get("HERCULES_REPO_ROOT", "/workspaces/hercules"))]
    roots.extend(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "PythonClient" / "distributed_mission").is_dir()
    )
    for root in roots:
        for path in (root / "PythonClient", root / "PythonClient" / "distributed_mission"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
    from modules.mission_plots import plot_topdown_animation, load_mission_records  # pylint: disable=import-outside-toplevel
    from modules.video_recording import (  # pylint: disable=import-outside-toplevel
        AirSimFrameRecorder, FollowCameraController, render_recordings,
    )
    return plot_topdown_animation, load_mission_records, AirSimFrameRecorder, FollowCameraController, render_recordings


class RosVideoRecorder(Node):
    """Keep camera capture off the mission control callback."""

    def __init__(self) -> None:
        super().__init__("ros_video_recorder")
        self.log_path = str(self.declare_parameter("log_path", "").value)
        self.output_dir = Path(str(self.declare_parameter("video_output_dir", "").value))
        staging_value = str(self.declare_parameter("video_staging_dir", "").value)
        self.staging_dir = Path(staging_value) if staging_value else None
        self.record_uav = str(self.declare_parameter("record_uav", "Drone1").value)
        self.record_ugv = str(self.declare_parameter("record_ugv", "Husky1").value)
        self.video_fps = float(self.declare_parameter("video_fps", 30.0).value)
        self.video_width = int(self.declare_parameter("video_width", 1280).value)
        self.video_height = int(self.declare_parameter("video_height", 720).value)
        self.gif_fps = float(self.declare_parameter("gif_fps", 10.0).value)
        self.gif_height = int(self.declare_parameter("gif_height", 540).value)
        self.playback_speed = float(self.declare_parameter("playback_speed", 2.0).value)
        self.duration = float(self.declare_parameter("duration_sec", 30.0).value)
        self.startup_timeout = float(self.declare_parameter("video_startup_timeout_sec", 30.0).value)
        self.finish_grace = float(self.declare_parameter("video_finish_grace_sec", 3.0).value)
        self.postprocess_timeout = float(self.declare_parameter("video_postprocess_timeout_sec", 20.0).value)
        self.rpc_port = int(self.declare_parameter("video_rpc_port", 41451).value)
        self.rpc_host = str(self.declare_parameter("video_rpc_host", "127.0.0.1").value or "127.0.0.1")
        self.car_port = int(self.declare_parameter("video_car_port", 41452).value)
        self.route_heading = float(self.declare_parameter("route_heading_rad", 0.0).value)
        self.keep_frames = bool(self.declare_parameter("video_keep_frames", False).value)
        if self.record_uav not in CONTROLLED or self.record_ugv not in CONTROLLED:
            raise ValueError("record_uav and record_ugv must name controlled vehicles")
        if self.video_fps <= 0 or self.gif_fps <= 0 or self.playback_speed <= 0:
            raise ValueError("video rates and playback speed must be positive")
        if self.video_width <= 0 or self.video_height <= 0 or self.gif_height <= 0:
            raise ValueError("video dimensions must be positive")
        if not self.log_path or not str(self.output_dir):
            raise ValueError("log_path and video_output_dir are required")
        if self.staging_dir is None:
            self.staging_dir = self.output_dir / "recording_frames"

        self.states: Dict[str, GroundTruthState] = {}
        self.state_subscriptions = [self.create_subscription(
            GroundTruthState, f"/hercules_mission/ground_truth/{name}",
            lambda message, name=name: self.states.__setitem__(name, message), 10)
            for name in AGENTS]
        self.started_at: float | None = None
        self.created_at = time.monotonic()
        self.finished = False
        self.client: Any = None
        self.recorder: Any = None
        self.camera: Any = None
        self.camera_base_offset = np.zeros(3, dtype=float)
        self.air = None
        self.rendering = False
        self.timer = self.create_timer(0.05, self.tick)
        self.get_logger().info(
            f"ROS media recorder waiting for seven states; outputs={self.output_dir}")

    def _start(self) -> None:
        plotter, loader, recorder_type, camera_type, render_type = _source_modules()
        del plotter, loader, render_type
        try:
            import hercules_cosysairsim as airsim  # pylint: disable=import-outside-toplevel
            self.air = airsim
            rpc_host = self.rpc_host
            if rpc_host in {"host.docker.internal", "docker.for.mac.host.internal"}:
                try:
                    addresses = socket.getaddrinfo(rpc_host, None, socket.AF_INET, socket.SOCK_STREAM)
                    if addresses:
                        rpc_host = str(addresses[0][4][0])
                except OSError:
                    pass
            try:
                self.client = airsim.MultirotorClient(ip=rpc_host, port=self.rpc_port)
            except TypeError:
                self.client = airsim.MultirotorClient(port=self.rpc_port)
            self.client.confirmConnection()
            # AirSim applies simSetCameraPose relative to the detached
            # camera's configured base pose in this build.  The Python
            # recorder measures that offset from an image response and
            # subtracts it from each desired world-NED chase pose.
            response = self.client.simGetImages([
                airsim.ImageRequest("mission_follow", airsim.ImageType.Scene, False, True)
            ], vehicle_name=self.record_uav)
            if response:
                position = getattr(response[0], "camera_position", None)
                if position is not None:
                    candidate = np.asarray(
                        [position.x_val, position.y_val, position.z_val], dtype=float)
                    if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
                        self.camera_base_offset = candidate
        except Exception as error:
            raise RuntimeError("AirSim camera recorder could not connect: {}".format(error)) from error
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.camera = camera_type(self.route_heading, aspect=self.video_width / self.video_height)
        streams = [(self.record_uav, "mission_follow"),
                   (self.record_uav, "front_center"),
                   (self.record_ugv, "front_center")]
        self.recorder = recorder_type(airsim, self.rpc_port, streams,
                                      str(self.staging_dir), self.video_fps,
                                      host=self.rpc_host,
                                      endpoint_ports={self.record_uav: self.rpc_port,
                                                      self.record_ugv: self.car_port})
        self.recorder.start()
        self.started_at = time.monotonic()
        self.get_logger().info("ROS camera recording started")

    def _set_chase_pose(self) -> None:
        if self.camera is None or self.client is None:
            return
        positions = [np.asarray(self.states[name].position, dtype=float) for name in AGENTS]
        chase = self.camera.update(positions, 0.05)
        position = chase["world_position"]
        quaternion = chase["orientation_quaternion"]
        pose = self.air.Pose(
            self.air.Vector3r(float(position[0]), float(position[1]), float(position[2])),
            self.air.Quaternionr(float(quaternion[1]), float(quaternion[2]),
                                 float(quaternion[3]), float(quaternion[0])),
        )
        try:
            command_position = np.asarray(position, dtype=float) - self.camera_base_offset
            command_pose = self.air.Pose(
                self.air.Vector3r(float(command_position[0]), float(command_position[1]),
                                  float(command_position[2])),
                self.air.Quaternionr(float(quaternion[1]), float(quaternion[2]),
                                     float(quaternion[3]), float(quaternion[0])),
            )
            self.client.simSetCameraPose(
                "mission_follow", command_pose, vehicle_name=self.record_uav)
        except Exception:
            # Older builds expose the detached camera as an object actor only.
            try:
                self.client.simSetObjectPose("ExternalCamera", pose, True)
            except Exception as error:
                self.get_logger().warning(f"chase camera pose update failed: {error}")

    def tick(self) -> None:
        if self.finished:
            return
        if self.started_at is None:
            if all(name in self.states for name in AGENTS):
                try:
                    self._start()
                except Exception as error:
                    self.get_logger().error(str(error))
                    self._finish()
                    return
            elif time.monotonic() - self.created_at > self.startup_timeout:
                self.get_logger().error("ROS media recorder timed out waiting for states")
                self._finish()
            return
        self._set_chase_pose()
        if time.monotonic() - self.started_at >= self.duration + self.finish_grace:
            self._finish()

    def _render(self) -> None:
        plotter, loader, _, _, render_type = _source_modules()
        deadline = time.monotonic() + self.postprocess_timeout
        records = []
        while time.monotonic() < deadline:
            try:
                records = loader(self.log_path)
            except (OSError, ValueError):
                records = []
            if records:
                break
            time.sleep(0.25)
        if not records:
            self.get_logger().error("ROS media export skipped: mission log is empty")
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        topdown_mp4 = self.output_dir / "topdown.mp4"
        topdown_gif = self.output_dir / "topdown.gif"
        plotter(records, str(topdown_mp4), str(topdown_gif),
                fps=1.0 / max(float(records[0].get("dt", 0.1)), 1e-6),
                playback_speed=self.playback_speed,
                route_heading=self.route_heading)
        outputs = render_type(
            str(self.staging_dir), str(self.output_dir), "mission", records,
            self.record_uav, self.record_ugv, width=self.video_width,
            height=self.video_height, fps=self.video_fps,
            gif_height=self.gif_height, gif_fps=self.gif_fps,
            keep_frames=self.keep_frames, playback_speed=self.playback_speed,
            map_name="rural_australia",
        )
        self.get_logger().info(f"ROS media export complete: {outputs}")

    def _finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        self.timer.cancel()
        if self.recorder is not None:
            self.recorder.stop()
        try:
            self._render()
        except Exception as error:
            self.get_logger().error(f"ROS media export failed: {error}")
        if rclpy.ok():
            rclpy.shutdown()

    def destroy_node(self):
        if not self.finished:
            self._finish()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = RosVideoRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
