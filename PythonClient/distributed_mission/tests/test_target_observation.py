import numpy as np

from modules.target_observation import TargetObservationWorker, backproject_target_roi, truth_target_measurement


def test_target_roi_backprojection_returns_forward_camera_point_and_covariance():
    depth = np.full((10, 20), 10.0)
    point, covariance, metadata = backproject_target_roi(
        depth, (7.0, 3.0, 13.0, 7.0), np.deg2rad(90.0), target_radius=1.0
    )
    assert point[0] > 10.0
    assert abs(point[1]) < 0.7
    assert abs(point[2]) < 0.7
    assert covariance.shape == (2, 2)
    assert metadata["roi_samples"] > 0


def test_truth_observation_is_seeded_and_range_gated():
    first = truth_target_measurement(
        "Target1", [10.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, 0.5, 100.0,
        np.random.default_rng(12), "capture_a"
    )
    second = truth_target_measurement(
        "Target1", [10.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, 0.5, 100.0,
        np.random.default_rng(12), "capture_a"
    )
    assert first is not None and second is not None
    assert np.allclose(first.position, second.position)
    assert truth_target_measurement(
        "Target1", [101.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0, 0.5, 100.0,
        np.random.default_rng(12), "capture_b"
    ) is None


class _Point:
    def __init__(self, x, y, z=0.0):
        self.x_val, self.y_val, self.z_val = x, y, z


class _Box:
    def __init__(self):
        self.min = _Point(1.0, 1.0)
        self.max = _Point(3.0, 3.0)


class _Detection:
    def __init__(self, name):
        self.name = name
        self.box2D = _Box()


class _Quaternion:
    def __init__(self, w, x, y, z):
        self.w_val, self.x_val, self.y_val, self.z_val = w, x, y, z


class _Response:
    width = 4
    height = 4
    image_data_float = [5.0] * 16
    camera_position = _Point(10.0, 20.0, -5.0)
    camera_orientation = _Quaternion(1.0, 0.0, 0.0, 0.0)


class _ImageType:
    DepthPerspective = 2


class _ImageRequest:
    def __init__(self, *args, **kwargs):
        pass


class _AirSim:
    ImageType = _ImageType
    ImageRequest = _ImageRequest


class _Kinematics:
    position = _Point(0.0, 0.0, 0.0)


class _Client:
    def __init__(self, detections):
        self.detections = detections
        self.filters = []

    def simSetDetectionFilterRadius(self, *args, **kwargs):
        self.filters.append(("radius", args, kwargs))

    def simClearDetectionMeshNames(self, *args, **kwargs):
        self.filters.append(("clear", args, kwargs))

    def simAddDetectionFilterMeshName(self, *args, **kwargs):
        self.filters.append(("mesh", args, kwargs))

    def simGetDetections(self, *args, **kwargs):
        return self.detections

    def simGetImages(self, *args, **kwargs):
        return [_Response()]

    def simGetGroundTruthKinematics(self, **kwargs):
        return _Kinematics()

    def simGetObjectPose(self, *args, **kwargs):
        return type("Pose", (), {"position": _Point(0.0, 0.0, 0.0)})()


def test_worker_rejects_non_target_names_and_keeps_capture_id():
    worker = TargetObservationWorker(_AirSim, 41451, {"Drone1": "target_bottom"})
    invalid = worker._capture(_Client([_Detection("Tree_01")]), "Drone1", "target_bottom")
    assert not invalid.valid
    assert invalid.visible is False
    assert invalid.capture_id.startswith("target_capture_")

    valid = worker._capture(_Client([_Detection("Target1")]), "Drone1", "target_bottom")
    assert valid.valid
    assert valid.visible
    assert valid.capture_id != invalid.capture_id
    assert valid.metadata["position_frame"] == "world_ned"
