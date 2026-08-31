"""Local, memoryless obstacle proxy generation from depth and LiDAR points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .cbf import ObstacleProxy


@dataclass
class PerceptionConfig:
    top_n: int = 5
    min_range: float = 0.4
    max_range: float = 30.0
    ground_band: float = 0.25
    voxel_size: float = 0.20
    cluster_eps: float = 0.65
    cluster_min_samples: int = 8
    max_proxy_radius: float = 2.0
    fit_padding: float = 0.25
    stale_after: float = 0.25
    depth_stride: int = 2
    max_points: int = 5000


@dataclass
class DetectionDiagnostics:
    """Intermediate point-cloud stages captured for offline diagnosis."""

    stage_points: Dict[str, np.ndarray] = field(default_factory=dict)
    cluster_labels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    proxy_labels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    stage_counts: Dict[str, int] = field(default_factory=dict)


class ObstacleDetector:
    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        self._projection_cache = {}

    def detect(
        self,
        points_world: Optional[np.ndarray],
        ego_position: np.ndarray,
        timestamp: float = 0.0,
        source: str = "unknown",
        is_planar: bool = False,
        ground_z: Optional[float] = None,
    ) -> List[ObstacleProxy]:
        proxies, _ = self.detect_with_diagnostics(
            points_world, ego_position, timestamp, source, is_planar, ground_z
        )
        return proxies

    def detect_with_diagnostics(
        self,
        points_world: Optional[np.ndarray],
        ego_position: np.ndarray,
        timestamp: float = 0.0,
        source: str = "unknown",
        is_planar: bool = False,
        ground_z: Optional[float] = None,
    ) -> Tuple[List[ObstacleProxy], DetectionDiagnostics]:
        """Detect proxies and retain bounded-independent intermediate stages."""

        diagnostics = DetectionDiagnostics()
        if points_world is None:
            diagnostics.stage_counts = {"input": 0, "finite_range": 0, "ground_filtered": 0, "voxelized": 0, "clusters": 0, "noise": 0, "proxies": 0}
            return [], diagnostics
        points = np.asarray(points_world, dtype=float)
        if points.size == 0:
            diagnostics.stage_counts = {"input": 0, "finite_range": 0, "ground_filtered": 0, "voxelized": 0, "clusters": 0, "noise": 0, "proxies": 0}
            return [], diagnostics
        points = points.reshape((-1, 3))
        diagnostics.stage_points["input"] = points.copy()
        diagnostics.stage_counts["input"] = len(points)
        ego = np.asarray(ego_position, dtype=float).reshape(3)
        distances = np.linalg.norm(points - ego, axis=1)
        valid = np.all(np.isfinite(points), axis=1)
        valid &= distances >= self.config.min_range
        valid &= distances <= self.config.max_range
        points = points[valid]
        diagnostics.stage_points["finite_range"] = points.copy()
        diagnostics.stage_counts["finite_range"] = len(points)
        if len(points) == 0:
            diagnostics.stage_counts.update({"ground_filtered": 0, "voxelized": 0, "clusters": 0, "noise": 0, "proxies": 0})
            return [], diagnostics

        if ground_z is not None:
            points = points[np.abs(points[:, 2] - ground_z) > self.config.ground_band]
        if is_planar:
            points = points[np.abs(points[:, 2] - ego[2]) < 3.0]
        diagnostics.stage_points["ground_filtered"] = points.copy()
        diagnostics.stage_counts["ground_filtered"] = len(points)
        if len(points) == 0:
            diagnostics.stage_counts.update({"voxelized": 0, "clusters": 0, "noise": 0, "proxies": 0})
            return [], diagnostics

        points = self._voxel_downsample(points)
        if len(points) > self.config.max_points:
            keep = np.linspace(0, len(points) - 1, self.config.max_points, dtype=int)
            points = points[keep]
        diagnostics.stage_points["voxelized"] = points.copy()
        diagnostics.stage_counts["voxelized"] = len(points)
        labels = self._cluster_labels(points, is_planar)
        diagnostics.cluster_labels = labels.copy()
        diagnostics.proxy_labels = np.full(len(points), -1, dtype=int)
        diagnostics.stage_counts["clusters"] = len([label for label in set(labels) if label >= 0])
        diagnostics.stage_counts["noise"] = int(np.sum(labels < 0))
        clusters = [points[labels == label] for label in sorted(set(labels)) if label >= 0]
        cluster_indices = [np.flatnonzero(labels == label) for label in sorted(set(labels)) if label >= 0]
        proxy_entries = []
        for index, (cluster, indices) in enumerate(zip(clusters, cluster_indices)):
            for patch_index, patch_indices in enumerate(self._split_cluster_indices(indices, points)):
                patch = points[patch_indices]
                if len(patch) == 0:
                    continue
                center = np.mean(patch, axis=0)
                radius = float(np.max(np.linalg.norm(patch - center, axis=1)) + self.config.fit_padding)
                clearance = float(np.linalg.norm(center - ego) - radius)
                proxy_entries.append((clearance, ObstacleProxy(
                    obstacle_id=f"{source}_{index}_{patch_index}",
                    center=center,
                    radius=radius,
                    source=source,
                    timestamp=timestamp,
                    point_count=len(patch),
                    is_planar=is_planar,
                ), patch_indices))

        proxy_entries.sort(key=lambda entry: entry[0])
        selected_entries = proxy_entries[: self.config.top_n]
        proxies = [entry[1] for entry in selected_entries]
        for proxy_index, (_, _, patch_indices) in enumerate(selected_entries):
            diagnostics.proxy_labels[patch_indices] = proxy_index
        diagnostics.stage_counts["proxies"] = len(proxies)
        return proxies, diagnostics

    def depth_to_world(
        self,
        depth: np.ndarray,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,
        horizontal_fov_rad: float,
        stride: int = 1,
    ) -> np.ndarray:
        """Back-project an AirSim DepthPerspective image into world points."""
        camera_points = self.depth_to_sensor(depth, horizontal_fov_rad, stride)
        return camera_points @ np.asarray(camera_rotation, dtype=float).T + np.asarray(camera_position, dtype=float)

    def depth_to_sensor(
        self,
        depth: np.ndarray,
        horizontal_fov_rad: float,
        stride: int = 1,
    ) -> np.ndarray:
        """Back-project DepthPerspective pixels into the local camera frame."""

        image = np.asarray(depth, dtype=float)
        if image.ndim != 2:
            raise ValueError("depth must be a two-dimensional array")
        height, width = image.shape
        stride = max(1, int(stride))
        fx = width / (2.0 * np.tan(horizontal_fov_rad / 2.0))
        fy = fx
        key = (height, width, round(float(horizontal_fov_rad), 9), stride)
        rays = self._projection_cache.get(key)
        if rays is None:
            u, v = np.meshgrid(np.arange(0, width, stride), np.arange(0, height, stride))
            rays = np.stack(((u - (width - 1) / 2.0) / fx, (v - (height - 1) / 2.0) / fy), axis=-1)
            self._projection_cache[key] = rays.reshape((-1, 2))
        z = image[::stride, ::stride].reshape(-1)
        rays = np.asarray(rays).reshape((-1, 2))
        x = rays[:, 0] * z
        y = rays[:, 1] * z
        # AirSim camera coordinates are forward/right/down.
        camera_points = np.stack((z, x, y), axis=-1)
        if len(camera_points) > self.config.max_points:
            keep = np.linspace(0, len(camera_points) - 1, self.config.max_points, dtype=int)
            camera_points = camera_points[keep]
        return camera_points

    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        size = self.config.voxel_size
        keys = np.floor(points / size).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        return points[np.sort(first)]

    def _cluster(self, points: np.ndarray, planar: bool) -> List[np.ndarray]:
        labels = self._cluster_labels(points, planar)
        return [points[labels == label] for label in sorted(set(labels)) if label >= 0]

    def _cluster_labels(self, points: np.ndarray, planar: bool) -> np.ndarray:
        features = points[:, :2] if planar else points
        try:
            from sklearn.cluster import DBSCAN
            labels = DBSCAN(eps=self.config.cluster_eps, min_samples=self.config.cluster_min_samples).fit_predict(features)
        except ImportError:
            labels = self._grid_components(features)
        return np.asarray(labels, dtype=int)

    def _grid_components(self, features: np.ndarray) -> np.ndarray:
        """Small dependency-free fallback used only when sklearn is absent."""
        if len(features) == 0:
            return np.empty(0, dtype=int)
        labels = -np.ones(len(features), dtype=int)
        current = 0
        for index in range(len(features)):
            if labels[index] >= 0:
                continue
            labels[index] = current
            frontier = [index]
            while frontier:
                source = frontier.pop()
                neighbors = np.where(np.linalg.norm(features - features[source], axis=1) <= self.config.cluster_eps)[0]
                for neighbor in neighbors:
                    if labels[neighbor] < 0:
                        labels[neighbor] = current
                        frontier.append(int(neighbor))
            current += 1
        return labels

    def _split_cluster(self, points: np.ndarray) -> List[np.ndarray]:
        if len(points) <= 1:
            return [points]
        radius = np.max(np.linalg.norm(points - np.mean(points, axis=0), axis=1))
        if radius <= self.config.max_proxy_radius or len(points) < 4:
            return [points]
        centered = points - np.mean(points, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        order = np.argsort(centered @ axis)
        midpoint = len(order) // 2
        return self._split_cluster(points[order[:midpoint]]) + self._split_cluster(points[order[midpoint:]])

    def _split_cluster_indices(self, indices: np.ndarray, points: np.ndarray) -> List[np.ndarray]:
        """Split a cluster while preserving indices for diagnostics."""

        cluster = points[indices]
        if len(cluster) <= 1:
            return [indices]
        radius = np.max(np.linalg.norm(cluster - np.mean(cluster, axis=0), axis=1))
        if radius <= self.config.max_proxy_radius or len(cluster) < 4:
            return [indices]
        centered = cluster - np.mean(cluster, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        order = np.argsort(centered @ axis)
        midpoint = len(order) // 2
        return self._split_cluster_indices(indices[order[:midpoint]], points) + self._split_cluster_indices(indices[order[midpoint:]], points)


def decode_depth_response(response: object) -> np.ndarray:
    """Decode an AirSim float DepthPerspective response."""
    width = int(getattr(response, "width"))
    height = int(getattr(response, "height"))
    values = np.asarray(getattr(response, "image_data_float"), dtype=float)
    if width <= 0 or height <= 0 or values.size != width * height:
        raise ValueError("invalid AirSim depth response")
    return values.reshape((height, width))
