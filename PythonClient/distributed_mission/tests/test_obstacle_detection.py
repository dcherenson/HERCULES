import numpy as np

from modules.cbf import ObstacleProxy
from modules.obstacle_detection import ObstacleDetector, PerceptionConfig, estimate_ground_z, truth_obstacle_proxies


def test_detector_returns_nearest_five_clusters():
    chunks = []
    for index in range(7):
        center = np.array([2.0 + index * 2.0, 0.0, -5.0])
        chunks.append(center + np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]))
    points = np.vstack(chunks)
    detector = ObstacleDetector(PerceptionConfig(top_n=5, cluster_min_samples=2, ground_band=0.01, voxel_size=0.05))
    proxies = detector.detect(points, np.zeros(3), source="synthetic")
    assert len(proxies) == 5
    clearances = [np.linalg.norm(proxy.center) - proxy.radius for proxy in proxies]
    assert clearances == sorted(clearances)


def test_proxy_radius_respects_the_configured_safety_bound():
    points = np.vstack([
        np.array([5.0, 0.0, -5.0]) + np.random.default_rng(2).normal(0.0, 2.0, (40, 3)),
    ])
    detector = ObstacleDetector(PerceptionConfig(cluster_min_samples=2, max_proxy_radius=0.75))
    proxies = detector.detect(points, np.zeros(3), source="synthetic")
    assert proxies and all(proxy.radius <= 0.75 for proxy in proxies)


def test_planar_proxy_radius_ignores_vertical_lidar_extent():
    points = np.vstack([
        np.array([5.0, 0.0, -2.0]) + np.array([0.0, 0.0, z])
        for z in np.linspace(-3.0, 3.0, 20)
    ])
    points = np.vstack((points, np.array([
        [4.0, -1.0, -2.0], [4.0, 1.0, -2.0],
        [6.0, -1.0, -2.0], [6.0, 1.0, -2.0],
    ])))
    detector = ObstacleDetector(PerceptionConfig(
        cluster_min_samples=2, cluster_eps=1.1, voxel_size=0.05, fit_padding=0.25,
    ))
    proxies = detector.detect(points, np.zeros(3), source="lidar", is_planar=True)
    assert proxies
    assert proxies[0].radius < 1.6


def test_planar_proxy_center_is_corrected_away_from_sensor_surface():
    points = np.asarray([
        [5.0, -1.0, -1.0], [5.0, 1.0, -1.0],
        [6.0, -1.0, -1.0], [6.0, 1.0, -1.0],
    ])
    detector = ObstacleDetector(PerceptionConfig(
        cluster_min_samples=2, cluster_eps=1.1, voxel_size=0.05,
        fit_padding=0.0, planar_surface_offset=1.0,
    ))
    proxies = detector.detect(points, np.asarray([0.0, 0.0, 0.0]), is_planar=True)
    assert proxies
    assert proxies[0].center[0] > 5.0


def test_depth_backprojection_has_forward_axis():
    detector = ObstacleDetector()
    depth = np.ones((3, 3), dtype=float) * 2.0
    points = detector.depth_to_world(depth, np.zeros(3), np.eye(3), np.pi / 2)
    center = points[4]
    assert np.allclose(center, [2.0, 0.0, 0.0], atol=1e-6)
    # A pixel to the right has positive local Y in the returned camera pose.
    assert points[5, 1] > 0.0
    assert points[3, 1] < 0.0


def test_depth_backprojection_uses_euclidean_ray_range():
    detector = ObstacleDetector()
    depth = np.ones((3, 3), dtype=float) * 2.0
    points = detector.depth_to_sensor(depth, np.pi / 2.0)
    # The corner ray is farther off-axis, so a Euclidean range of 2 m must
    # still produce a point whose norm is 2 m rather than a forward depth of
    # 2 m plus lateral components.
    assert np.isclose(np.linalg.norm(points[0]), 2.0, atol=1e-6)
    assert points[0, 0] < 2.0


def test_invalid_and_empty_sensor_frames_are_safe_empty_results():
    detector = ObstacleDetector()
    assert detector.detect(np.empty((0, 3)), np.zeros(3)) == []
    assert detector.detect(None, np.zeros(3)) == []


def test_ground_height_is_estimated_from_dominant_local_return():
    ego = np.array([0.0, 0.0, 1.7])
    ground = np.column_stack((np.linspace(-5.0, 5.0, 40), np.zeros(40), np.full(40, 2.02)))
    clutter = np.array([[1.0, 1.0, 1.3], [2.0, -1.0, 2.8]])
    assert np.isclose(estimate_ground_z(np.vstack((ground, clutter)), ego), 2.02)


def test_ground_height_returns_none_when_no_local_plane_is_visible():
    points = np.array([[0.0, 0.0, -4.0], [1.0, 1.0, -3.0]])
    assert estimate_ground_z(points, np.array([0.0, 0.0, 1.7])) is None


def test_truth_box_proxy_uses_planar_or_vertical_stack_geometry():
    truth = [{"id": "box", "shape": "box", "center": [1.0, 2.0, 3.0], "dimensions": [2.0, 4.0, 6.0]}]
    uav = truth_obstacle_proxies(truth, "drone")
    ugv = truth_obstacle_proxies(truth, "ugv")
    assert np.isclose(uav[0].radius, np.sqrt(5.0))
    assert len(uav) == 3
    assert np.isclose(min(item.center[2] for item in uav), 0.0)
    assert np.isclose(max(item.center[2] for item in uav), 6.0)
    assert np.isclose(ugv[0].radius, np.sqrt(5.0))
    assert uav[0].source == "truth" and ugv[0].is_planar


def test_truth_ugv_proxies_ignore_boxes_above_or_below_vehicle_footprint():
    truth = [
        {"id": "ground", "shape": "box", "center": [4.0, 0.0, -1.0], "dimensions": [2.0, 2.0, 2.0]},
        {"id": "floating", "shape": "box", "center": [4.0, 0.0, -5.0], "dimensions": [2.0, 2.0, 1.0]},
    ]
    proxies = truth_obstacle_proxies(truth, "ugv", vehicle_z=0.7, vehicle_radius=1.25)
    assert [item.obstacle_id for item in proxies] == ["truth_ground"]


def test_ground_height_can_be_estimated_from_an_airborne_view():
    ego = np.array([0.0, 0.0, -5.0])
    ground = np.column_stack((np.linspace(-10.0, 10.0, 80), np.zeros(80), np.full(80, 2.02)))
    block = np.column_stack((np.full(12, 5.0), np.linspace(-1.0, 1.0, 12), np.full(12, 0.0)))
    nearby_vehicle = np.column_stack((np.linspace(-1.0, 1.0, 120), np.zeros(120), np.full(120, -3.5)))
    assert np.isclose(estimate_ground_z(
        np.vstack((ground, block, nearby_vehicle)), ego, search_below=2.0,
        search_above=12.0, min_separation_above_ego=2.0,
    ), 2.02)
