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
    # Keep the local proxy conservative but bounded. The vehicle radius and
    # logged age margin are added by the CBF layer; larger mixed-surface
    # spheres make the formation avoidance problem infeasible.
    max_proxy_radius: float = 2.0
    fit_padding: float = 0.25
    # LiDAR sees the near surface of a solid obstacle. Move planar proxy
    # centers a bounded distance away from the sensor toward the occupied
    # volume; zero disables this geometric surface correction.
    planar_surface_offset: float = 0.75
    # Sparse foliage often forms a long, irregular patch. Anchoring its proxy
    # at the patch midpoint can place the CBF obstacle behind the first
    # measured return. This remains opt-in so the established FlyingCPP box
    # behavior is unchanged.
    planar_use_nearest_surface: bool = False
    # Some sparse planar patches have a distant centroid but a close visible
    # surface. Rank those patches by their nearest return when requested.
    rank_by_surface_distance: bool = False
    # A DBSCAN/grid component can be valid noise-free geometry but still be
    # too small to represent a collision obstacle. Keep this separate from
    # the clustering core threshold so production tuning can reject tiny
    # foliage fragments without changing the clustering algorithm itself.
    min_proxy_points: int = 1
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


def truth_obstacle_proxies(
    truth_obstacles: Sequence[Dict[str, object]],
    vehicle_type: str,
    timestamp: float = 0.0,
    vehicle_z: Optional[float] = None,
    vehicle_radius: float = 0.0,
) -> List[ObstacleProxy]:
    """Convert spawned truth boxes into fixed spherical CBF test proxies.

    The CBF interface accepts spherical obstacles, while the orchestrator's
    truth set records boxes. UGV proxies use one horizontal bounding-circle
    radius because their model is planar. UAV boxes are represented by a
    bounded vertical stack of horizontal spheres, which approximates the box
    volume without turning a tall wall into one excessively large sphere.
    This helper is intentionally only for explicit truth-obstacle test mode.
    """

    proxies: List[ObstacleProxy] = []
    for obstacle in truth_obstacles or []:
        shape = str(obstacle.get("shape", "box")).lower()
        if shape not in {"box", "sphere"}:
            continue
        try:
            center = np.asarray(obstacle["center"], dtype=float).reshape(3)
        except (KeyError, TypeError, ValueError):
            continue
        if not np.all(np.isfinite(center)):
            continue
        obstacle_id = "truth_" + str(obstacle.get("id", len(proxies)))
        if shape == "sphere":
            try:
                radius = float(obstacle["radius"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(radius) or radius <= 0.0:
                continue
            if vehicle_type != "drone" and vehicle_z is not None:
                if center[2] + radius < float(vehicle_z) - max(0.0, float(vehicle_radius)):
                    continue
            proxies.append(ObstacleProxy(
                obstacle_id=obstacle_id,
                center=center,
                radius=radius,
                source="truth",
                timestamp=float(timestamp),
                point_count=0,
                is_planar=vehicle_type == "ugv",
            ))
            continue
        try:
            dimensions = np.asarray(obstacle["dimensions"], dtype=float).reshape(3)
        except (KeyError, TypeError, ValueError):
            continue
        if not np.all(np.isfinite(dimensions)):
            continue
        half_dimensions = np.abs(dimensions) / 2.0
        # The UGV CBF is planar, but a box floating well above the Husky must
        # not become a false XY obstacle. Keep ground/floating boxes only when
        # their vertical extent overlaps the vehicle's modeled footprint.
        if vehicle_type != "drone" and vehicle_z is not None:
            obstacle_bottom = center[2] - half_dimensions[2]
            obstacle_top = center[2] + half_dimensions[2]
            if obstacle_top < float(vehicle_z) - max(0.0, float(vehicle_radius)):
                continue
        radius = float(np.linalg.norm(half_dimensions[:2]))
        if vehicle_type != "drone":
            centers = [(center, obstacle_id)]
        else:
            # Adjacent slice spheres overlap vertically, so a UAV cannot
            # pass through a gap in the box approximation while changing
            # altitude. The number of slices is bounded by the box height.
            vertical_height = 2.0 * half_dimensions[2]
            slice_count = max(1, int(np.ceil(vertical_height / max(2.0 * radius, 1e-6))) + 1)
            z_values = np.linspace(center[2] - half_dimensions[2], center[2] + half_dimensions[2], slice_count)
            centers = [(center + np.array([0.0, 0.0, z - center[2]]), "{}_z{:02d}".format(obstacle_id, index))
                       for index, z in enumerate(z_values)]
        for proxy_center, proxy_id in centers:
            proxies.append(ObstacleProxy(
                obstacle_id=proxy_id,
                center=proxy_center,
                radius=radius,
                source="truth",
                timestamp=float(timestamp),
                point_count=0,
                is_planar=vehicle_type == "ugv",
            ))
    return proxies


def estimate_ground_z(
    points_world: np.ndarray,
    ego_position: np.ndarray,
    search_below: float = 0.75,
    search_above: float = 1.5,
    bin_size: float = 0.20,
    min_range: float = 0.4,
    max_range: float = 30.0,
    min_separation_above_ego: float = 0.0,
) -> Optional[float]:
    """Estimate a locally dominant horizontal ground return for a planar sensor.

    AirSim uses NED coordinates, and the map ground is not required to be at
    ``z=0``.  A UGV LiDAR cloud commonly contains a dense horizontal return;
    selecting its dominant quantized height near the vehicle removes that
    return without assuming a global map altitude.  The estimate is only used
    for ground rejection, never as an obstacle or a control state.
    """

    points = np.asarray(points_world, dtype=float).reshape((-1, 3))
    ego = np.asarray(ego_position, dtype=float).reshape(3)
    if points.size == 0 or not np.all(np.isfinite(ego)):
        return None
    finite = np.all(np.isfinite(points), axis=1)
    ranges = np.linalg.norm(points - ego, axis=1)
    finite &= ranges >= float(min_range)
    finite &= ranges <= float(max_range)
    points = points[finite]
    if len(points) == 0:
        return None
    heights = points[:, 2]
    nearby = heights[(heights >= ego[2] - float(search_below)) & (heights <= ego[2] + float(search_above))]
    nearby = nearby[nearby >= ego[2] + float(min_separation_above_ego)]
    if len(nearby) == 0:
        return None
    size = max(float(bin_size), 1e-3)
    bins = np.floor(nearby / size).astype(np.int64)
    unique, counts = np.unique(bins, return_counts=True)
    if min_separation_above_ego > 0.0:
        # An airborne view can contain a dense return from another vehicle or
        # an obstacle. The ground is the lowest plausible horizontal plane in
        # NED, so prefer the highest supported candidate after excluding
        # returns too close to the vehicle. A weak candidate is not used.
        support = counts >= max(3, int(np.max(counts) * 0.10))
        candidates = unique[support]
        if len(candidates) == 0:
            return None
        selected = int(np.max(candidates))
    else:
        # Prefer the densest return and use its actual median, which is less
        # sensitive to bin-edge effects on gently uneven terrain.
        selected = int(unique[int(np.argmax(counts))])
    in_bin = nearby[bins == selected]
    return float(np.median(in_bin))


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
        small_patch_count = 0
        clusters = [points[labels == label] for label in sorted(set(labels)) if label >= 0]
        cluster_indices = [np.flatnonzero(labels == label) for label in sorted(set(labels)) if label >= 0]
        proxy_entries = []
        for index, (cluster, indices) in enumerate(zip(clusters, cluster_indices)):
            for patch_index, patch_indices in enumerate(self._split_cluster_indices(indices, points)):
                patch = points[patch_indices]
                if len(patch) == 0:
                    continue
                if len(patch) < max(1, int(self.config.min_proxy_points)):
                    small_patch_count += 1
                    continue
                center = np.mean(patch, axis=0)
                # UGV constraints are planar. Vertical LiDAR returns from a
                # wall or a ground edge must not enlarge the XY safety disk;
                # doing so made a 2 m wide course block look substantially
                # wider than its actual horizontal footprint. UAV depth
                # proxies retain the full 3D fit.
                if is_planar:
                    xy_min = np.min(patch[:, :2], axis=0)
                    xy_max = np.max(patch[:, :2], axis=0)
                    if self.config.planar_use_nearest_surface:
                        nearest_index = int(np.argmin(np.linalg.norm(patch[:, :2] - ego[:2], axis=1)))
                        center[:2] = patch[nearest_index, :2]
                    else:
                        center[:2] = 0.5 * (xy_min + xy_max)
                    line_of_sight = center[:2] - ego[:2]
                    line_norm = float(np.linalg.norm(line_of_sight))
                    offset = min(max(0.0, float(self.config.planar_surface_offset)), line_norm)
                    if line_norm > 1e-9 and offset > 0.0:
                        center[:2] += offset * line_of_sight / line_norm
                    if self.config.planar_use_nearest_surface:
                        radius = float(np.max(np.linalg.norm(patch[:, :2] - center[:2], axis=1)) + self.config.fit_padding)
                    else:
                        radius = float(0.5 * np.linalg.norm(xy_max - xy_min) + self.config.fit_padding)
                else:
                    radius = float(np.max(np.linalg.norm(patch - center, axis=1)) + self.config.fit_padding)
                # ``max_proxy_radius`` is a safety-model limit as well as a
                # split threshold. Without the cap, a depth discontinuity can
                # make one proxy several metres wide and render a perfectly
                # valid CBF problem infeasible.
                radius = min(radius, float(self.config.max_proxy_radius))
                clearance = float(np.linalg.norm(center - ego) - radius)
                if self.config.rank_by_surface_distance:
                    if is_planar:
                        ranking_distance = float(np.min(np.linalg.norm(patch[:, :2] - ego[:2], axis=1)))
                    else:
                        ranking_distance = float(np.min(np.linalg.norm(patch - ego, axis=1)))
                    ranking_key = ranking_distance
                else:
                    ranking_key = clearance
                proxy_entries.append((ranking_key, ObstacleProxy(
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
        diagnostics.stage_counts["small_patches"] = small_patch_count
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
        # AirSim's DepthPerspective image uses the camera-frame lateral axis
        # directly: increasing pixel ``u`` maps to local +Y. Keeping this
        # sign explicit is important because mirroring it puts obstacles on
        # the opposite side of the vehicle and makes the CBF react to a proxy
        # that is not the colliding object.
        # DepthPerspective is Euclidean range along each viewing ray, rather
        # than forward-axis depth.  Normalize the camera ray before scaling;
        # without this correction off-axis returns are displaced increasingly
        # far from the camera and static proxies appear to move between
        # captures as the vehicle changes viewpoint.
        directions = np.stack((np.ones_like(z), rays[:, 0], rays[:, 1]), axis=-1)
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        camera_points = directions * z[:, None]
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
