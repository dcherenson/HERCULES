import numpy as np

from modules.obstacle_detection import ObstacleDetector, PerceptionConfig


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


def test_depth_backprojection_has_forward_axis():
    detector = ObstacleDetector()
    depth = np.ones((3, 3), dtype=float) * 2.0
    points = detector.depth_to_world(depth, np.zeros(3), np.eye(3), np.pi / 2)
    center = points[4]
    assert np.allclose(center, [2.0, 0.0, 0.0], atol=1e-6)


def test_invalid_and_empty_sensor_frames_are_safe_empty_results():
    detector = ObstacleDetector()
    assert detector.detect(np.empty((0, 3)), np.zeros(3)) == []
    assert detector.detect(None, np.zeros(3)) == []
