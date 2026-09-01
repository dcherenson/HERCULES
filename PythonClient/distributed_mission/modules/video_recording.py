"""Post-run media recording and presentation overlays.

The simulator recorder writes timestamped PNGs.  This module turns those
frames into constant-rate media after the control loop has finished, so media
encoding cannot affect CBF timing.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class RecordedFrame:
    timestamp: float
    path: str


class AirSimFrameRecorder:
    """Capture selected Unreal SceneCapture cameras off the control thread.

    The AirSim recording RPC is present in this fork but can report an active
    state without producing files in headless Hero mode.  This worker uses the
    same AirSim image API and named cameras directly, retaining Unreal's
    rendering path while making the output observable and recoverable.
    """

    def __init__(self, airsim_module: Any, port: int, streams: Sequence[Tuple[str, str]],
                 staging_dir: str, fps: float):
        self.airsim_module = airsim_module
        self.port = int(port)
        self.streams = list(streams)
        self.staging_dir = staging_dir
        self.fps = float(fps)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.capture_count = 0
        self.error_count = 0
        self.metadata_path = os.path.join(staging_dir, "capture_metadata.jsonl")
        self._metadata_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        os.makedirs(self.staging_dir, exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8"):
            pass
        self._thread = threading.Thread(target=self._run, name="airsim-video-capture", daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        return {"captures": self.capture_count, "errors": self.error_count}

    def _run(self) -> None:
        try:
            client = self.airsim_module.MultirotorClient(port=self.port)
        except Exception:
            self.error_count += 1
            return
        period = 1.0 / max(self.fps, 1e-6)
        deadline = time.monotonic()
        while not self._stop.is_set():
            for vehicle, camera in self.streams:
                if self._stop.is_set():
                    break
                try:
                    response = client.simGetImages([
                        self.airsim_module.ImageRequest(camera, self.airsim_module.ImageType.Scene, False, True)
                    ], vehicle_name=vehicle)[0]
                    data = getattr(response, "image_data_uint8", b"")
                    if data:
                        # Wall time matches the mission JSONL timestamps. The
                        # AirSim response timestamp is simulator-clock based
                        # and is not guaranteed to share that epoch.
                        timestamp = time.time_ns()
                        path = os.path.join(
                            self.staging_dir,
                            "img_{}_{}_0_{}.png".format(vehicle, camera, timestamp),
                        )
                        with open(path, "wb") as output:
                            output.write(bytes(data))
                        metadata: Dict[str, Any] = {
                            "timestamp": timestamp / 1e9,
                            "vehicle": vehicle,
                            "camera": camera,
                            "path": path,
                            "width": int(getattr(response, "width", 0) or 0),
                            "height": int(getattr(response, "height", 0) or 0),
                        }
                        response_position = getattr(response, "camera_position", None)
                        response_orientation = getattr(response, "camera_orientation", None)
                        if response_position is not None:
                            metadata["camera_position"] = [
                                float(response_position.x_val),
                                float(response_position.y_val),
                                float(response_position.z_val),
                            ]
                        if response_orientation is not None:
                            metadata["camera_orientation_quaternion"] = [
                                float(response_orientation.w_val),
                                float(response_orientation.x_val),
                                float(response_orientation.y_val),
                                float(response_orientation.z_val),
                            ]
                        with self._metadata_lock:
                            with open(self.metadata_path, "a", encoding="utf-8") as metadata_file:
                                metadata_file.write(json.dumps(metadata) + "\n")
                        self.capture_count += 1
                except Exception:
                    self.error_count += 1
            deadline += period
            self._stop.wait(max(0.0, deadline - time.monotonic()))


def _quat_matrix(values: Any) -> np.ndarray:
    q = np.asarray(values if values is not None else [1.0, 0.0, 0.0, 0.0], dtype=float)
    if q.shape != (4,):
        raise ValueError("orientation must be [w, x, y, z]")
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        return np.eye(3)
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_multiply(first: Any, second: Any) -> np.ndarray:
    """Multiply normalized-or-not quaternions in [w, x, y, z] order."""

    w1, x1, y1, z1 = np.asarray(first, dtype=float).reshape(4)
    w2, x2, y2, z2 = np.asarray(second, dtype=float).reshape(4)
    result = np.asarray([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)
    result /= max(np.linalg.norm(result), 1e-12)
    return result


def world_camera_pose_to_vehicle(
    camera_position: Any,
    camera_orientation: Any,
    vehicle_position: Any,
    vehicle_yaw: float,
) -> Tuple[List[float], List[float]]:
    """Convert a desired world-NED camera pose to AirSim vehicle-local NED.

    ``simSetCameraPose`` applies a pose relative to the selected vehicle.
    The chase controller works in world NED, so both translation and
    orientation must be transformed by the vehicle's current planar pose.
    """

    camera = np.asarray(camera_position, dtype=float).reshape(3)
    vehicle = np.asarray(vehicle_position, dtype=float).reshape(3)
    yaw = float(vehicle_yaw)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    vehicle_rotation = np.asarray([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])
    local_position = vehicle_rotation.T @ (camera - vehicle)
    vehicle_quaternion = np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    inverse_vehicle_quaternion = vehicle_quaternion * np.asarray([1.0, -1.0, -1.0, -1.0])
    local_orientation = _quat_multiply(inverse_vehicle_quaternion, camera_orientation)
    return local_position.tolist(), local_orientation.tolist()


def look_at_quaternion(camera_position: Any, target: Any) -> List[float]:
    """Return a zero-roll AirSim quaternion looking from camera to target."""

    camera = np.asarray(camera_position, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)
    forward = target - camera
    norm = np.linalg.norm(forward)
    if norm <= 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    forward /= norm
    world_up = np.array([0.0, 0.0, -1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) <= 1e-9:
        right = np.array([0.0, 1.0, 0.0])
    else:
        right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.column_stack((forward, right, down))
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = 2.0 * math.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1e-12))
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = 2.0 * math.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 1e-12))
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = 2.0 * math.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 1e-12))
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    values = np.asarray([w, x, y, z], dtype=float)
    values /= max(np.linalg.norm(values), 1e-12)
    return values.tolist()


class FollowCameraController:
    """Route-aligned, adaptive-distance chase camera."""

    def __init__(self, route_heading: float, aspect: float = 16.0 / 9.0, horizontal_fov: float = 90.0,
                 margin: float = 0.10, elevation_deg: float = 25.0, minimum_distance: float = 12.0,
                 smoothing_tau: float = 0.35):
        self.route_heading = float(route_heading)
        self.aspect = float(aspect)
        self.horizontal_fov = float(horizontal_fov)
        self.margin = float(margin)
        self.elevation_deg = float(elevation_deg)
        self.minimum_distance = float(minimum_distance)
        self.smoothing_tau = float(smoothing_tau)
        self._position: Optional[np.ndarray] = None
        self._target: Optional[np.ndarray] = None

    @property
    def vertical_fov(self) -> float:
        return float(2.0 * np.degrees(np.arctan(np.tan(np.radians(self.horizontal_fov) / 2.0) / self.aspect)))

    def update(self, positions: Iterable[Any], dt: float) -> Dict[str, Any]:
        points = np.asarray(list(positions), dtype=float).reshape((-1, 3))
        if len(points) == 0:
            raise ValueError("at least one position is required")
        target = (np.min(points, axis=0) + np.max(points, axis=0)) / 2.0
        radius = float(np.max(np.linalg.norm(points - target, axis=1)))
        half_fov = min(np.radians(self.horizontal_fov), np.radians(self.vertical_fov)) / 2.0
        distance = max(self.minimum_distance, radius * (1.0 + self.margin) / max(np.sin(half_fov), 1e-6))
        heading = np.array([np.cos(self.route_heading), np.sin(self.route_heading), 0.0])
        elevation = np.radians(self.elevation_deg)
        offset = np.array([
            -heading[0] * np.cos(elevation),
            -heading[1] * np.cos(elevation),
            -np.sin(elevation),
        ]) * distance
        candidate = target + offset

        if self._position is None or self._target is None:
            self._position = candidate
            self._target = target
        else:
            alpha = 1.0 - np.exp(-max(float(dt), 0.0) / max(self.smoothing_tau, 1e-6))
            previous_distance = np.linalg.norm(self._position - self._target)
            # Expand immediately when the team spreads out; smooth only the
            # inward return so no robot is lost at the edge of the frame.
            if distance >= previous_distance:
                self._position = candidate
            else:
                self._position = self._position + alpha * (candidate - self._position)
            self._target = self._target + alpha * (target - self._target)

        return {
            "world_position": self._position.tolist(),
            "target": self._target.tolist(),
            "orientation_quaternion": look_at_quaternion(self._position, self._target),
            "horizontal_fov_deg": self.horizontal_fov,
            "vertical_fov_deg": self.vertical_fov,
            "roll_deg": 0.0,
            "route_heading_rad": self.route_heading,
        }


def project_world_point(point: Any, camera_position: Any, camera_orientation: Any,
                        horizontal_fov_deg: float, width: int, height: int) -> Optional[Tuple[float, float]]:
    """Project a world-NED point to image pixels for an AirSim camera."""

    relative = _quat_matrix(camera_orientation).T @ (
        np.asarray(point, dtype=float).reshape(3) - np.asarray(camera_position, dtype=float).reshape(3)
    )
    if not np.all(np.isfinite(relative)) or relative[0] <= 1e-6:
        return None
    horizontal = math.tan(math.radians(float(horizontal_fov_deg)) / 2.0)
    vertical_fov = 2.0 * math.atan(horizontal * float(height) / float(width))
    vertical = math.tan(vertical_fov / 2.0)
    x = (0.5 + relative[1] / relative[0] / (2.0 * horizontal)) * width
    y = (0.5 + relative[2] / relative[0] / (2.0 * vertical)) * height
    return float(x), float(y)


_FRAME_PATTERN = re.compile(r"^img_(?P<vehicle>.+)_(?P<camera>mission_follow|front_center)_0_(?P<timestamp>\d+)\.png$")


def find_recorded_frames(root: str, vehicle: str, camera: str) -> List[RecordedFrame]:
    frames: List[RecordedFrame] = []
    for path in Path(root).rglob("*.png"):
        match = _FRAME_PATTERN.match(path.name)
        if not match or match.group("vehicle") != vehicle or match.group("camera") != camera:
            continue
        frames.append(RecordedFrame(float(match.group("timestamp")) / 1e9, str(path)))
    return sorted(frames, key=lambda item: item.timestamp)


def load_capture_metadata(root: str, vehicle: str, camera: str) -> List[Mapping[str, Any]]:
    """Load image-response pose diagnostics for one recorded stream."""

    path = Path(root) / "capture_metadata.jsonl"
    if not path.is_file():
        return []
    result: List[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("vehicle") == vehicle and item.get("camera") == camera:
            result.append(item)
    return sorted(result, key=lambda item: float(item.get("timestamp", 0.0)))


def analyze_camera_alignment(records: Sequence[Mapping[str, Any]], staging_dir: str,
                             output_dir: str, stem: str, vehicle: str = "Drone1",
                             camera: str = "mission_follow", tolerance_m: float = 1.0) -> List[str]:
    """Write a repeatable commanded-vs-image-response chase-camera report."""

    metadata = load_capture_metadata(staging_dir, vehicle, camera)
    if records:
        chase_records = [
            record for record in records
            if (record.get("recording", {}).get("chase_camera", {}).get("world_position") is not None)
        ]
    else:
        chase_records = []
    comparisons = []
    for item in metadata:
        observed = item.get("camera_position")
        if observed is None or not chase_records:
            continue
        index = min(
            range(len(chase_records)),
            key=lambda i: abs(float(chase_records[i].get("timestamp", 0.0)) - float(item.get("timestamp", 0.0))),
        )
        commanded = chase_records[index]["recording"]["chase_camera"]["world_position"]
        error = float(np.linalg.norm(np.asarray(observed, dtype=float) - np.asarray(commanded, dtype=float)))
        comparisons.append({
            "capture_timestamp": float(item.get("timestamp", 0.0)),
            "record_timestamp": float(chase_records[index].get("timestamp", 0.0)),
            "observed_world_position": observed,
            "commanded_world_position": commanded,
            "position_error_m": error,
        })
    errors = [item["position_error_m"] for item in comparisons]
    report = {
        "available": bool(comparisons),
        "tolerance_m": float(tolerance_m),
        "captures_compared": len(comparisons),
        "max_position_error_m": max(errors) if errors else None,
        "mean_position_error_m": float(np.mean(errors)) if errors else None,
        "pass": bool(comparisons) and max(errors) <= float(tolerance_m),
        "comparisons": comparisons,
    }
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, stem + "_camera_alignment.json")
    markdown_path = os.path.join(output_dir, stem + "_camera_alignment.md")
    with open(json_path, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    with open(markdown_path, "w", encoding="utf-8") as output:
        output.write("# Chase camera alignment\n\n")
        output.write("- Captures compared: {}\n".format(report["captures_compared"]))
        output.write("- Maximum position error: {} m\n".format(report["max_position_error_m"]))
        output.write("- Tolerance: {} m\n".format(report["tolerance_m"]))
        output.write("- Result: **{}**\n\n".format("PASS" if report["pass"] else "FAIL/UNAVAILABLE"))
        output.write("The observed pose comes from the `simGetImages` response for the exact chase PNG.\n")
    return [json_path, markdown_path]


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg is required for mission video output")
    return executable


def _load_rgb(path: str, width: int, height: int) -> bytes:
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return image.tobytes()


def _encode_frames(frames: Sequence[RecordedFrame], output_path: str, fps: float, width: int, height: int,
                   start_time: float, duration: float, source_fps: Optional[float] = None,
                   overlay: Optional[Callable[[Any, float], None]] = None) -> Dict[str, Any]:
    if not frames:
        raise ValueError("no recorded frames available")
    from PIL import Image

    # ``source_fps`` controls how many mission-time samples are rendered;
    # ``fps`` controls playback.  Keeping them separate allows a 10 Hz
    # capture to play at 20 fps and therefore at 2x real time.
    timeline_fps = float(source_fps if source_fps is not None else fps)
    # Avoid turning an exact duration such as 1.2 s at 5 fps into seven
    # frames because of a binary floating-point value of 6.000000000000001.
    frame_count = max(1, int(math.ceil(max(duration, 1.0 / timeline_fps) * timeline_fps - 1e-9)))
    process = subprocess.Popen([
        _ffmpeg(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "{}x{}".format(width, height),
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output_path,
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    source_index = 0
    previous_source_index = -1
    duplicates = 0
    try:
        for index in range(frame_count):
            timestamp = start_time + index / timeline_fps
            while source_index + 1 < len(frames) and frames[source_index + 1].timestamp <= timestamp:
                source_index += 1
            if index > 0 and source_index == previous_source_index:
                duplicates += 1
            with Image.open(frames[source_index].path) as source:
                image = source.convert("RGB")
                if image.size != (width, height):
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                if overlay is not None:
                    overlay(image, timestamp)
                process.stdin.write(image.tobytes())
            previous_source_index = source_index
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    except Exception:
        if process.stdin:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise RuntimeError("ffmpeg failed for {}: {}".format(output_path, stderr[-1000:]))
    return {
        "frames": frame_count,
        "duplicates": duplicates,
        "dropped_source_frames": max(0, len(frames) - frame_count),
        "capture_fps": timeline_fps,
        "playback_fps": float(fps),
    }


def encode_gif(mp4_path: str, gif_path: str, fps: float, height: int) -> None:
    ffmpeg = _ffmpeg()
    palette = gif_path + ".palette.png"
    subprocess.run([
        ffmpeg, "-y", "-i", mp4_path, "-vf", "fps={},scale=-1:{}:flags=lanczos,palettegen".format(fps, height), palette,
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        subprocess.run([
            ffmpeg, "-y", "-i", mp4_path, "-i", palette,
            "-lavfi", "fps={},scale=-1:{}:flags=lanczos[x];[x][1:v]paletteuse".format(fps, height), gif_path,
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    finally:
        try:
            os.unlink(palette)
        except FileNotFoundError:
            pass


def verify_mp4(path: str) -> Dict[str, Any]:
    probe = shutil.which("ffprobe")
    if not probe:
        raise RuntimeError("ffprobe is required to verify mission video output")
    result = subprocess.check_output([
        probe, "-v", "error", "-count_frames", "-show_entries",
        "format=duration:stream=width,height,nb_read_frames,duration", "-of", "json", path,
    ], text=True)
    data = json.loads(result)
    streams = data.get("streams") or []
    if not streams or os.path.getsize(path) <= 0:
        raise RuntimeError("invalid or empty MP4: {}".format(path))
    return data


def _draw_chase_marker(image: Any, center: Tuple[float, float], color: Tuple[int, int, int]) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    x, y = center
    radius = 16
    bounds = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(bounds, outline=(0, 0, 0), width=5)
    draw.ellipse(bounds, outline=color, width=3)


def make_chase_overlay(records: Sequence[Mapping[str, Any]], vehicle_types: Mapping[str, str],
                       width: int, height: int,
                       camera_metadata: Optional[Sequence[Mapping[str, Any]]] = None) -> Callable[[Any, float], None]:
    indexed = sorted(records, key=lambda record: float(record.get("timestamp", 0.0)))
    metadata = sorted(camera_metadata or [], key=lambda item: float(item.get("timestamp", 0.0)))
    cursor = 0
    metadata_cursor = 0

    def overlay(image: Any, timestamp: float) -> None:
        nonlocal cursor, metadata_cursor
        while cursor + 1 < len(indexed) and float(indexed[cursor + 1].get("timestamp", 0.0)) <= timestamp:
            cursor += 1
        record = indexed[cursor]
        chase = record.get("recording", {}).get("chase_camera", {})
        measured = chase.get("measured") or {}
        camera_position = measured.get("world_position") or chase.get("world_position")
        orientation = measured.get("world_orientation_quaternion") or chase.get("orientation_quaternion")
        while (metadata_cursor + 1 < len(metadata)
               and float(metadata[metadata_cursor + 1].get("timestamp", 0.0)) <= timestamp):
            metadata_cursor += 1
        if metadata:
            observed = metadata[metadata_cursor]
            camera_position = observed.get("camera_position") or camera_position
            orientation = observed.get("camera_orientation_quaternion") or orientation
        if camera_position is None or orientation is None:
            return
        for name, state in (record.get("states") or {}).items():
            if not isinstance(state, Mapping):
                continue
            point = state.get("actor_position") or state.get("position")
            projected = project_world_point(point, camera_position, orientation, 90.0, width, height) if point is not None else None
            if projected is None:
                continue
            if -32 <= projected[0] <= width + 32 and -32 <= projected[1] <= height + 32:
                color = (30, 255, 90) if vehicle_types.get(name) == "drone" else (40, 150, 255)
                _draw_chase_marker(image, projected, color)
        # Targets are intentionally not part of vehicle_types or formation
        # state, but their truth actor pose is logged separately and gets a
        # distinct red screen-space marker on the chase stream.
        target_entries = record.get("targets")
        if not isinstance(target_entries, Mapping):
            target = record.get("target_truth") or record.get("target")
            target_entries = {str(target.get("name", "Target1")): target} if isinstance(target, Mapping) else {}
        for target in target_entries.values():
            if not isinstance(target, Mapping):
                continue
            point = target.get("actor_position") or target.get("position")
            projected = project_world_point(point, camera_position, orientation, 90.0, width, height) if point is not None else None
            if projected is not None and -32 <= projected[0] <= width + 32 and -32 <= projected[1] <= height + 32:
                _draw_chase_marker(image, projected, (255, 45, 45))

    return overlay


def render_recordings(staging_dir: str, output_dir: str, stem: str, records: Sequence[Mapping[str, Any]],
                      uav: str, ugv: str, width: int = 1280, height: int = 720, fps: float = 20.0,
                      gif_height: int = 540, gif_fps: float = 10.0, keep_frames: bool = False,
                      capture_stats: Optional[Tuple[int, int]] = None,
                      playback_speed: float = 2.0) -> List[str]:
    """Encode selected AirSim SceneCapture streams and return generated paths."""

    os.makedirs(output_dir, exist_ok=True)
    if not records:
        return []
    if fps <= 0.0 or gif_fps <= 0.0 or playback_speed <= 0.0:
        raise ValueError("capture and playback rates must be positive")
    times = [float(record.get("timestamp", 0.0)) for record in records]
    start_time = min(times)
    dt = float(records[0].get("dt", 0.1))
    first_step = float(records[0].get("step", 0.0))
    last_step = float(records[-1].get("step", len(records) - 1))
    duration = max(dt, (last_step - first_step + 1.0) * dt)
    vehicle_types = records[0].get("vehicle_types") or {}
    streams = [
        ("chase", uav, "mission_follow", True),
        ("{}_fpv".format(uav), uav, "front_center", False),
        ("{}_fpv".format(ugv), ugv, "front_center", False),
    ]
    outputs: List[str] = []
    playback_fps = float(fps) * float(playback_speed)
    manifest: Dict[str, Any] = {
        "staging_dir": staging_dir,
        "start_time": start_time,
        "mission_duration": duration,
        "playback_speed": float(playback_speed),
        "capture_fps": float(fps),
        "playback_fps": playback_fps,
        "gif_playback_fps": float(gif_fps) * float(playback_speed),
        "streams": [],
    }
    if capture_stats is not None:
        manifest["capture_worker"] = {
            "captures": int(capture_stats[0]),
            "errors": int(capture_stats[1]),
        }
    all_mp4_valid = True
    for label, vehicle, camera, is_chase in streams:
        frames = find_recorded_frames(staging_dir, vehicle, camera)
        if not frames:
            print("Warning: no recorded frames for {} / {}".format(vehicle, camera), flush=True)
            manifest["streams"].append({"label": label, "vehicle": vehicle, "camera": camera, "available": False})
            if is_chase:
                all_mp4_valid = False
            continue
        mp4_path = os.path.join(output_dir, "{}_{}.mp4".format(stem, label))
        camera_metadata = load_capture_metadata(staging_dir, vehicle, camera) if is_chase else []
        overlay = make_chase_overlay(records, vehicle_types, width, height, camera_metadata) if is_chase else None
        stats = _encode_frames(frames, mp4_path, playback_fps, width, height, start_time, duration, fps, overlay)
        try:
            verification = verify_mp4(mp4_path)
        except Exception:
            all_mp4_valid = False
            raise
        gif_path = os.path.join(output_dir, "{}_{}.gif".format(stem, label))
        encode_gif(mp4_path, gif_path, gif_fps * playback_speed, gif_height)
        outputs.extend([mp4_path, gif_path])
        manifest["streams"].append({
            "label": label, "vehicle": vehicle, "camera": camera, "available": True,
            "source_frames": len(frames), "output": mp4_path, "gif": gif_path,
            "metadata_frames": len(camera_metadata) if is_chase else 0,
            "stats": stats, "ffprobe": verification,
        })
    manifest_path = os.path.join(output_dir, stem + "_recording_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
    outputs.append(manifest_path)
    if all_mp4_valid and not keep_frames:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return outputs
