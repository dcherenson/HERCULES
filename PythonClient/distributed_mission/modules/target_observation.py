"""AirSim target observations used by the distributed tracker.

The worker uses AirSim's named-object detector to establish visibility and a
DepthPerspective image to recover the target center in the sensor frame. It
is intentionally independent of the obstacle detector: target observations
are cached asynchronously so image RPCs do not consume the CBF deadline.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .obstacle_detection import decode_depth_response
from .target_tracking import TargetMeasurement


def _quaternion_matrix(quaternion: Any) -> np.ndarray:
    values = np.asarray([
        float(getattr(quaternion, "w_val", 1.0)),
        float(getattr(quaternion, "x_val", 0.0)),
        float(getattr(quaternion, "y_val", 0.0)),
        float(getattr(quaternion, "z_val", 0.0)),
    ], dtype=float)
    norm = np.linalg.norm(values)
    if norm <= 1e-12:
        return np.eye(3)
    w, x, y, z = values / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _vector3(value: Any) -> np.ndarray:
    return np.asarray([
        float(getattr(value, "x_val", np.nan)),
        float(getattr(value, "y_val", np.nan)),
        float(getattr(value, "z_val", np.nan)),
    ], dtype=float)


def _bbox_values(detection: Any) -> Optional[Tuple[float, float, float, float]]:
    box = getattr(detection, "box2D", None)
    if box is None:
        return None
    minimum = getattr(box, "min", None)
    maximum = getattr(box, "max", None)
    if minimum is None or maximum is None:
        return None
    values = np.asarray([
        float(getattr(minimum, "x_val", np.nan)),
        float(getattr(minimum, "y_val", np.nan)),
        float(getattr(maximum, "x_val", np.nan)),
        float(getattr(maximum, "y_val", np.nan)),
    ], dtype=float)
    if not np.all(np.isfinite(values)):
        return None
    return tuple(values.tolist())


def backproject_target_roi(
    depth: np.ndarray,
    bbox: Tuple[float, float, float, float],
    horizontal_fov_rad: float,
    target_radius: float = 1.25,
    min_range: float = 0.4,
    max_range: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Estimate a target center and covariance from a depth ROI.

    The central 60% of a detected box is used to reduce background leakage.
    The near-surface median is shifted outward by the configured target
    radius, producing an approximate body center rather than a front surface.
    """

    image = np.asarray(depth, dtype=float)
    if image.ndim != 2:
        raise ValueError("depth must be two-dimensional")
    height, width = image.shape
    x0, y0, x1, y1 = bbox
    x0, x1 = sorted((max(0.0, x0), min(float(width - 1), x1)))
    y0, y1 = sorted((max(0.0, y0), min(float(height - 1), y1)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("target bounding box is empty")
    span_x = x1 - x0
    span_y = y1 - y0
    left = int(np.floor(x0 + 0.2 * span_x))
    right = int(np.ceil(x1 - 0.2 * span_x)) + 1
    top = int(np.floor(y0 + 0.2 * span_y))
    bottom = int(np.ceil(y1 - 0.2 * span_y)) + 1
    roi = image[top:bottom, left:right]
    valid = np.isfinite(roi) & (roi >= float(min_range)) & (roi <= float(max_range))
    if not np.any(valid):
        raise ValueError("target depth ROI has no finite in-range values")
    # Use the near half of the surface returns so foliage/background behind a
    # target does not move the estimate away from the actor.
    values = roi[valid]
    cutoff = float(np.percentile(values, 60.0))
    near = valid & (roi <= cutoff)
    ys, xs = np.nonzero(near)
    if len(xs) == 0:
        ys, xs = np.nonzero(valid)
    ranges = roi[ys, xs]
    depth_value = float(np.median(ranges))
    u = float(np.median(xs + left))
    v = float(np.median(ys + top))
    fx = width / (2.0 * np.tan(float(horizontal_fov_rad) / 2.0))
    fy = fx
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    ray = np.array([1.0, (u - cx) / fx, (v - cy) / fy], dtype=float)
    ray /= max(np.linalg.norm(ray), 1e-12)
    point = depth_value * ray + float(max(0.0, target_radius)) * ray
    spread = float(np.median(np.abs(ranges - np.median(ranges))) * 1.4826)
    position_std = max(0.25, spread, 0.02 * depth_value)
    covariance = np.eye(2, dtype=float) * position_std ** 2
    return point, covariance, {
        "bbox": [float(x0), float(y0), float(x1), float(y1)],
        "roi_samples": int(len(ranges)),
        "depth_median_m": depth_value,
        "depth_spread_m": spread,
        "image_width": int(width),
        "image_height": int(height),
    }


def truth_target_measurement(
    target_id: str,
    target_position: np.ndarray,
    ego_position: np.ndarray,
    timestamp: float,
    measurement_std: float,
    max_range: float,
    rng: np.random.Generator,
    capture_id: str,
) -> Optional[TargetMeasurement]:
    """Create a deterministic, range-gated noisy truth measurement."""

    target = np.asarray(target_position, dtype=float).reshape(3)
    ego = np.asarray(ego_position, dtype=float).reshape(3)
    if np.linalg.norm(target[:2] - ego[:2]) > float(max_range):
        return None
    std = max(1e-6, float(measurement_std))
    noisy = target[:2] + rng.normal(0.0, std, size=2)
    return TargetMeasurement(
        target_id=target_id,
        position=noisy,
        covariance=np.eye(2) * std ** 2,
        timestamp=float(timestamp),
        source="truth",
        capture_id=capture_id,
        sensor="truth",
        visible=True,
        metadata={"truth_position": target.tolist(), "range_m": float(np.linalg.norm(target[:2] - ego[:2]))},
    )


@dataclass
class TargetObservationWorker:
    """Round-robin asynchronous camera capture for all tracking agents."""

    airsim_module: Any
    port: int
    agent_cameras: Mapping[str, str]
    target_id: str = "Target1"
    target_actor_pattern: str = "Target1*"
    sensing_range: float = 100.0
    measurement_std: float = 0.5
    horizontal_fov_deg: float = 120.0
    target_radius: float = 1.25
    rate_hz: float = 4.0

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Dict[str, TargetMeasurement] = {}
        self._capture_sequence = 0
        self.capture_count = 0
        self.error_count = 0
        self._fov_by_camera: Dict[Tuple[str, str], float] = {}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="target-observation-worker", daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        return {"captures": self.capture_count, "errors": self.error_count}

    def snapshot(self) -> Dict[str, TargetMeasurement]:
        with self._lock:
            return dict(self._latest)

    def _configure(self, client: Any, agent: str, camera: str) -> None:
        image_type = self.airsim_module.ImageType.DepthPerspective
        try:
            client.simSetDetectionFilterRadius(camera, image_type, float(self.sensing_range) * 100.0, vehicle_name=agent)
            client.simClearDetectionMeshNames(camera, image_type, vehicle_name=agent)
            client.simAddDetectionFilterMeshName(camera, image_type, self.target_actor_pattern, vehicle_name=agent)
        except TypeError:
            # Older client bindings do not expose vehicle_name as a keyword.
            client.simSetDetectionFilterRadius(camera, image_type, float(self.sensing_range) * 100.0, agent)
            client.simClearDetectionMeshNames(camera, image_type, agent)
            client.simAddDetectionFilterMeshName(camera, image_type, self.target_actor_pattern, agent)

    def _capture(self, client: Any, agent: str, camera: str) -> TargetMeasurement:
        self._configure(client, agent, camera)
        image_type = self.airsim_module.ImageType.DepthPerspective
        detections = client.simGetDetections(camera, image_type, vehicle_name=agent) or []
        selected = None
        for detection in detections:
            name = str(getattr(detection, "name", ""))
            if name == self.target_id or name.startswith(self.target_id):
                selected = detection
                break
        request = self.airsim_module.ImageRequest(camera, image_type, True, False)
        responses = client.simGetImages([request], vehicle_name=agent)
        if not responses:
            raise RuntimeError("empty target depth response")
        response = responses[0]
        capture_time = time.time()
        self._capture_sequence += 1
        capture_id = "target_capture_{:06d}_{}".format(self._capture_sequence, agent)
        if selected is None:
            return TargetMeasurement(
                target_id=self.target_id,
                position=np.zeros(2),
                covariance=np.eye(2) * float(self.measurement_std) ** 2,
                timestamp=capture_time,
                valid=False,
                source="camera",
                capture_id=capture_id,
                sensor=camera,
                visible=False,
                metadata={"detections": len(detections)},
            )
        bbox = _bbox_values(selected)
        if bbox is None:
            raise RuntimeError("target detection had no valid 2D box")
        depth = decode_depth_response(response)
        fov_key = (agent, camera)
        horizontal_fov_deg = self._fov_by_camera.get(fov_key, self.horizontal_fov_deg)
        try:
            camera_info = client.simGetCameraInfo(camera, vehicle_name=agent)
            camera_fov = float(getattr(camera_info, "fov", np.nan))
            if np.isfinite(camera_fov) and 1.0 < camera_fov < 179.0:
                horizontal_fov_deg = camera_fov
                self._fov_by_camera[fov_key] = camera_fov
        except Exception:
            pass
        sensor_point, covariance, metadata = backproject_target_roi(
            depth,
            bbox,
            np.deg2rad(horizontal_fov_deg),
            self.target_radius,
            max_range=self.sensing_range,
        )
        response_position = getattr(response, "camera_position", None)
        response_orientation = getattr(response, "camera_orientation", None)
        if response_position is None:
            raise RuntimeError("target depth response omitted camera position")
        camera_position = _vector3(response_position)
        camera_rotation = _quaternion_matrix(response_orientation)
        try:
            kinematics = client.simGetGroundTruthKinematics(vehicle_name=agent)
            actor_pose = client.simGetObjectPose(agent, True)
            actor_position = _vector3(getattr(actor_pose, "position"))
            kinematics_position = _vector3(getattr(kinematics, "position"))
            camera_position = camera_position + actor_position - kinematics_position
        except Exception:
            # The response pose is still useful on builds that already report
            # world coordinates, so preserve it rather than dropping capture.
            pass
        world_point = camera_position + camera_rotation @ sensor_point
        metadata.update({
            "detection_name": str(getattr(selected, "name", "")),
            "horizontal_fov_deg": float(horizontal_fov_deg),
            "response_camera_position": camera_position.tolist(),
            "response_camera_orientation": [
                float(getattr(response_orientation, "w_val", 1.0)),
                float(getattr(response_orientation, "x_val", 0.0)),
                float(getattr(response_orientation, "y_val", 0.0)),
                float(getattr(response_orientation, "z_val", 0.0)),
            ],
            "vehicle_actor_position": actor_position.tolist() if "actor_position" in locals() else None,
            "vehicle_kinematics_position": kinematics_position.tolist() if "kinematics_position" in locals() else None,
        })
        return TargetMeasurement(
            target_id=self.target_id,
            position=world_point[:2],
            covariance=covariance,
            timestamp=capture_time,
            source="camera",
            capture_id=capture_id,
            sensor=camera,
            visible=True,
            metadata=dict(metadata, position_frame="world_ned", point_frame="camera_local_ned"),
        )

    def _run(self) -> None:
        try:
            client = self.airsim_module.MultirotorClient(port=int(self.port))
        except Exception:
            self.error_count += 1
            return
        period = 1.0 / max(float(self.rate_hz), 1e-6)
        deadline = time.monotonic()
        while not self._stop.is_set():
            for agent, camera in self.agent_cameras.items():
                if self._stop.is_set():
                    break
                try:
                    measurement = self._capture(client, str(agent), str(camera))
                    with self._lock:
                        self._latest[str(agent)] = measurement
                    self.capture_count += 1
                except Exception:
                    self.error_count += 1
            deadline += period
            self._stop.wait(max(0.0, deadline - time.monotonic()))
