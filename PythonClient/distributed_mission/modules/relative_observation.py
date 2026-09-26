"""AirSim camera peer observations for cooperative localization.

The target observer and this worker deliberately share the same geometry
helpers: AirSim detection metadata is used only to associate an actor, while
the reported measurement comes from the depth ROI and camera pose.  The
worker keeps captures asynchronous so a slow image RPC cannot block mission
control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import socket
import threading
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .obstacle_detection import decode_depth_response
from .target_observation import (
    _bbox_values,
    _quaternion_matrix,
    _usable_target_bbox,
    _vector3,
    backproject_target_roi,
)


@dataclass
class RelativeObservation:
    """One body-frame polar peer observation with Cartesian metadata."""

    observer_id: str
    observed_id: str
    relative_position: np.ndarray
    covariance: np.ndarray
    range: float
    bearing: float
    timestamp: float
    capture_id: str
    sensor: str
    valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


def _yaw_from_quaternion(value: Any) -> float:
    values = np.asarray([
        float(getattr(value, "w_val", 1.0)),
        float(getattr(value, "x_val", 0.0)),
        float(getattr(value, "y_val", 0.0)),
        float(getattr(value, "z_val", 0.0)),
    ], dtype=float)
    norm = np.linalg.norm(values)
    if norm <= 1e-12 or not np.all(np.isfinite(values)):
        return 0.0
    w, x, y, z = values / norm
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _polar_covariance(relative_position: np.ndarray, covariance: np.ndarray,
                      range_floor: float, bearing_floor: float) -> np.ndarray:
    x, y = np.asarray(relative_position, dtype=float).reshape(2)
    radius = max(float(np.hypot(x, y)), 1e-9)
    jacobian = np.array([[x / radius, y / radius],
                         [-y / (radius * radius), x / (radius * radius)]], dtype=float)
    result = jacobian @ np.asarray(covariance, dtype=float).reshape(2, 2) @ jacobian.T
    result = 0.5 * (result + result.T)
    result[0, 0] = max(result[0, 0], float(range_floor) ** 2)
    result[1, 1] = max(result[1, 1], float(bearing_floor) ** 2)
    return result


def _controlled_actor_id(name: Any, allowed: Sequence[str]) -> Optional[str]:
    """Resolve a detection to one configured actor without prefix aliases.

    AirSim normally reports the actor name verbatim.  A few builds report a
    generated component suffix (for example ``Drone1_body``), so a suffix is
    accepted only after an explicit separator.  In particular, ``Drone10``
    can never be associated with ``Drone1`` merely because it shares a
    prefix.
    """

    detection_name = str(name or "")
    configured = {str(item) for item in allowed}
    if detection_name in configured:
        return detection_name
    for actor_id in sorted(configured, key=len, reverse=True):
        if detection_name.startswith(actor_id) and len(detection_name) > len(actor_id):
            separator = detection_name[len(actor_id)]
            if separator in "_-. :":
                return actor_id
    return None


def _lift_position_covariance(covariance: np.ndarray, depth_std: float) -> np.ndarray:
    """Lift the 2-D ROI covariance into camera 3-D before rotating it.

    ``backproject_target_roi`` reports uncertainty in the camera x/y
    projection.  Treating that matrix as if it were already world x/y loses
    uncertainty whenever a camera is pitched or rolled.  The depth component
    uses the ROI's depth-spread/range floor estimate, which is conservative
    without silently turning an anisotropic lateral covariance into an
    isotropic 3-D matrix.
    """

    planar = np.asarray(covariance, dtype=float).reshape(2, 2)
    planar = 0.5 * (planar + planar.T)
    if not np.all(np.isfinite(planar)):
        return np.full((3, 3), np.nan, dtype=float)
    depth_variance = max(float(depth_std) ** 2, 1e-12)
    result = np.zeros((3, 3), dtype=float)
    result[:2, :2] = planar
    result[2, 2] = depth_variance
    return result


@dataclass
class RelativeObservationWorker:
    """Round-robin depth-camera observer for all controlled agents.

    ``agent_cameras`` should use ``localization_front`` for UAVs and
    ``front_center`` for Huskies.  Detection filters are configured with the
    peer actor names, but Target1 and the observing vehicle are always
    rejected before a measurement is produced.
    """

    airsim_module: Any
    port: int
    agent_cameras: Mapping[str, Any]
    agent_ids: Sequence[str]
    sensing_range: float = 100.0
    range_std: float = 0.25
    bearing_std_rad: float = np.deg2rad(1.0)
    horizontal_fov_deg: float = 120.0
    rate_hz: float = 2.0
    host: str = "127.0.0.1"
    endpoint_ports: Optional[Mapping[str, int]] = None

    def __post_init__(self) -> None:
        self.host = str(self.host or "127.0.0.1")
        self.endpoint_ports = dict(self.endpoint_ports or {})
        self.agent_ids = tuple(str(item) for item in self.agent_ids)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Dict[Tuple[str, str], RelativeObservation] = {}
        self._capture_sequence = 0
        self.capture_count = 0
        self.error_count = 0
        self.visible_count = 0
        self.invalid_count = 0
        self.capture_timestamps = []
        self.rpc_durations = []
        self._fov_by_camera: Dict[Tuple[str, str], float] = {}

    def _port_for_agent(self, agent: str) -> int:
        if agent in self.endpoint_ports:
            return int(self.endpoint_ports[agent])
        return 41452 if agent.lower().startswith(("husky", "ugv", "car")) else int(self.port)

    def _cameras_for_agent(self, agent: str) -> Tuple[str, ...]:
        value = self.agent_cameras.get(agent, "localization_front")
        if isinstance(value, str):
            return (value,)
        return tuple(str(camera) for camera in value)

    def _client_for_agent(self, agent: str) -> Any:
        client_type = getattr(self.airsim_module, "MultirotorClient", None)
        if client_type is None:
            raise RuntimeError("AirSim module does not provide MultirotorClient")
        host = self.host
        if host in {"host.docker.internal", "docker.for.mac.host.internal"}:
            try:
                addresses = socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
                if addresses:
                    host = str(addresses[0][4][0])
            except OSError:
                pass
        try:
            return client_type(ip=host, port=self._port_for_agent(agent))
        except TypeError:
            return client_type(port=self._port_for_agent(agent))

    def _configure(self, client: Any, observer: str, camera: str) -> None:
        image_type = self.airsim_module.ImageType.DepthPerspective
        radius_cm = float(self.sensing_range) * 100.0
        try:
            client.simSetDetectionFilterRadius(camera, image_type, radius_cm, vehicle_name=observer)
            # AirSim stores these filters per camera/image type.  The target
            # observer shares Husky front_center, so additive filters avoid a
            # clear/add race; exact actor-ID association below separates the
            # two consumers safely.
            for peer in self.agent_ids:
                if peer != observer and peer != "Target1":
                    client.simAddDetectionFilterMeshName(camera, image_type, peer + "*", vehicle_name=observer)
        except TypeError:
            client.simSetDetectionFilterRadius(camera, image_type, radius_cm, observer)
            for peer in self.agent_ids:
                if peer != observer and peer != "Target1":
                    client.simAddDetectionFilterMeshName(camera, image_type, peer + "*", observer)

    def _invalid(self, observer: str, capture_id: str, timestamp: float,
                 camera: str, reason: str) -> RelativeObservation:
        return RelativeObservation(
            observer, "", np.zeros(2), np.diag([self.range_std ** 2, self.bearing_std_rad ** 2]),
            0.0, 0.0, timestamp, capture_id, camera, valid=False,
            metadata={"rejection_reason": reason},
        )

    def _capture(self, client: Any, observer: str, camera: str) -> RelativeObservation:
        self._configure(client, observer, camera)
        image_type = self.airsim_module.ImageType.DepthPerspective
        try:
            detections = client.simGetDetections(camera, image_type, vehicle_name=observer) or []
        except TypeError:
            detections = client.simGetDetections(camera, image_type, observer) or []
        selected = None
        selected_id = ""
        allowed = {peer for peer in self.agent_ids if peer != observer and peer != "Target1"}
        for detection in detections:
            name = str(getattr(detection, "name", ""))
            candidate = _controlled_actor_id(name, allowed)
            if candidate is not None:
                selected, selected_id = detection, candidate
                break
        self._capture_sequence += 1
        capture_id = f"relative_capture_{self._capture_sequence:08d}_{observer}"
        capture_time = time.time()
        request = self.airsim_module.ImageRequest(camera, image_type, True, False)
        try:
            responses = client.simGetImages([request], vehicle_name=observer)
        except TypeError:
            responses = client.simGetImages([request], observer)
        if not responses:
            raise RuntimeError("empty peer depth response")
        if selected is None:
            return self._invalid(observer, capture_id, capture_time, camera, "peer_not_detected")
        bbox = _bbox_values(selected)
        if bbox is None:
            return self._invalid(observer, capture_id, capture_time, camera, "invalid_detection_bbox")
        depth = decode_depth_response(responses[0])
        usable, reason = _usable_target_bbox(bbox, int(depth.shape[1]), int(depth.shape[0]))
        if not usable:
            return self._invalid(observer, capture_id, capture_time, camera, reason)
        fov = self._fov_by_camera.get((observer, camera), self.horizontal_fov_deg)
        try:
            info = client.simGetCameraInfo(camera, vehicle_name=observer)
            candidate_fov = float(getattr(info, "fov", np.nan))
            if np.isfinite(candidate_fov) and 1.0 < candidate_fov < 179.0:
                fov = candidate_fov
                self._fov_by_camera[(observer, camera)] = candidate_fov
        except Exception:
            pass
        sensor_point, sensor_covariance, metadata = backproject_target_roi(
            depth, bbox, np.deg2rad(fov), target_radius=0.0,
            position_std_floor=self.range_std, max_range=self.sensing_range,
        )
        sensor_point = np.asarray(sensor_point, dtype=float).reshape(3)
        camera_position = _vector3(getattr(responses[0], "camera_position", None))
        camera_orientation = getattr(responses[0], "camera_orientation", None)
        camera_rotation = _quaternion_matrix(camera_orientation)
        ego_position = camera_position.copy()
        ego_rotation = camera_rotation.copy()
        try:
            kinematics = client.simGetGroundTruthKinematics(vehicle_name=observer)
            ego_position = _vector3(getattr(kinematics, "position"))
            if getattr(kinematics, "orientation", None) is not None:
                ego_rotation = _quaternion_matrix(getattr(kinematics, "orientation"))
        except Exception:
            pass
        if (not np.all(np.isfinite(sensor_point)) or
                not np.all(np.isfinite(camera_position)) or
                not np.all(np.isfinite(camera_rotation)) or
                not np.all(np.isfinite(ego_position)) or
                not np.all(np.isfinite(ego_rotation))):
            return self._invalid(observer, capture_id, capture_time, camera, "peer_pose_invalid")
        world_point = camera_position + camera_rotation @ sensor_point
        delta_world = world_point - ego_position
        # AirSim quaternions map body/NED vectors into world/NED.  Project the
        # full 3-D delta through the inverse vehicle attitude before dropping
        # the vertical component; this preserves pitched/rolled camera
        # uncertainty instead of truncating the camera matrix to 2-D.
        world_to_body = ego_rotation.T
        relative_body = world_to_body @ delta_world
        relative_position = relative_body[:2]
        distance = float(np.linalg.norm(relative_position))
        if (not np.all(np.isfinite(relative_position)) or distance <= 1e-6 or
                distance > float(self.sensing_range)):
            return self._invalid(observer, capture_id, capture_time, camera, "peer_range_invalid")
        depth_std = float(self.range_std)
        depth_spread = float(metadata.get("depth_spread_m", np.nan))
        depth_median = float(metadata.get("depth_median_m", np.nan))
        if np.isfinite(depth_spread):
            depth_std = max(depth_std, depth_spread)
        if np.isfinite(depth_median):
            depth_std = max(depth_std, 0.02 * abs(depth_median))
        sensor_covariance_3d = _lift_position_covariance(sensor_covariance, depth_std)
        camera_covariance_world = camera_rotation @ sensor_covariance_3d @ camera_rotation.T
        body_covariance_3d = world_to_body @ camera_covariance_world @ world_to_body.T
        body_covariance = body_covariance_3d[:2, :2]
        covariance = _polar_covariance(relative_position, body_covariance,
                                       self.range_std, self.bearing_std_rad)
        if not np.all(np.isfinite(covariance)):
            return self._invalid(observer, capture_id, capture_time, camera, "peer_covariance_invalid")
        metadata.update({"detection_name": str(getattr(selected, "name", "")),
                         "sensor": camera, "position_frame": "body_ned",
                         "observer_id": observer, "observed_id": selected_id})
        return RelativeObservation(
            observer, selected_id, relative_position, covariance, distance,
            float(np.arctan2(relative_position[1], relative_position[0])),
            capture_time, capture_id, camera, valid=True, metadata=metadata,
        )

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="relative-observation-worker", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def snapshot(self) -> Dict[Tuple[str, str], RelativeObservation]:
        with self._lock:
            return dict(self._latest)

    def diagnostics(self) -> Dict[str, float | int]:
        with self._lock:
            timestamps = np.asarray(self.capture_timestamps, dtype=float)
            intervals = np.diff(timestamps) if len(timestamps) > 1 else np.empty(0)
            durations = np.asarray(self.rpc_durations, dtype=float)
            return {
                "captures": int(self.capture_count), "visible": int(self.visible_count),
                "invalid": int(self.invalid_count), "errors": int(self.error_count),
                "mean_capture_rate_hz": float(1.0 / np.mean(intervals)) if len(intervals) else 0.0,
                "mean_rpc_duration_sec": float(np.mean(durations)) if len(durations) else 0.0,
            }

    def _run(self) -> None:
        period = 1.0 / max(float(self.rate_hz), 1e-6)
        deadline = time.monotonic()
        clients: Dict[str, Any] = {}
        while not self._stop.is_set():
            for observer in self.agent_cameras:
                for camera in self._cameras_for_agent(observer):
                    if self._stop.is_set():
                        break
                    started = time.monotonic()
                    try:
                        client = clients.get(observer)
                        if client is None:
                            client = self._client_for_agent(observer)
                            clients[observer] = client
                        value = self._capture(client, observer, camera)
                        with self._lock:
                            if value.valid:
                                self._latest[(observer, value.observed_id)] = value
                            self.capture_count += 1
                            self.visible_count += int(value.valid)
                            self.invalid_count += int(not value.valid)
                            self.capture_timestamps.append(time.time())
                            self.rpc_durations.append(time.monotonic() - started)
                    except Exception:
                        with self._lock:
                            self.error_count += 1
                            self.rpc_durations.append(time.monotonic() - started)
            deadline += period
            self._stop.wait(max(0.0, deadline - time.monotonic()))
        for client in clients.values():
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
