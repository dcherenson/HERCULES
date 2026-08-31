import json

import numpy as np

from modules.obstacle_detection import DetectionDiagnostics, ObstacleDetector, PerceptionConfig
from modules.perception_diagnostics import (
    PerceptionTraceStore,
    analyze_perception_records,
    generate_perception_diagnostics,
    load_trace_sidecar,
)


def _record(step, capture_id, proxy_center, age=0.1, sensor_position=None):
    sensor_position = sensor_position or [0.0, 0.0, 0.0]
    return {
        "step": step,
        "dt": 0.1,
        "states": {"Drone1": {"position": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}},
        "obstacles": {
            "Drone1": {
                "age": age,
                "sensor_view": {
                    "capture_id": capture_id,
                    "sensor_type": "uav_camera",
                    "position": sensor_position,
                    "orientation_quaternion": [1.0, 0.0, 0.0, 0.0],
                },
                "proxies": [{"center": proxy_center, "radius": 0.5}],
            }
        },
    }


def test_detector_excludes_zero_returns_before_clustering():
    detector = ObstacleDetector(PerceptionConfig(cluster_min_samples=1, voxel_size=0.01))
    proxies, diagnostics = detector.detect_with_diagnostics(
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), np.zeros(3), source="test"
    )
    assert len(proxies) == 1
    assert diagnostics.stage_counts["input"] == 2
    assert diagnostics.stage_counts["finite_range"] == 1


def test_trace_sidecar_round_trip_is_bounded(tmp_path):
    store = PerceptionTraceStore(sample_limit=2)
    diagnostics = DetectionDiagnostics(
        stage_points={
            "input": np.arange(15, dtype=float).reshape((5, 3)),
            "voxelized": np.arange(15, dtype=float).reshape((5, 3)),
        },
        cluster_labels=np.asarray([0, 0, 1, 1, -1]),
        proxy_labels=np.asarray([0, 0, 1, 1, -1]),
        stage_counts={"input": 5, "voxelized": 5, "clusters": 2, "noise": 1, "proxies": 2},
    )
    capture_id = store.add(
        "capture_000001_Drone1", "Drone1", "uav_camera",
        np.arange(30, dtype=float).reshape((10, 3)),
        np.arange(30, dtype=float).reshape((10, 3)),
        diagnostics,
    )
    path = tmp_path / "trace.npz"
    store.save(str(path))
    metadata, archive = load_trace_sidecar(str(path))
    try:
        assert capture_id in metadata["captures"]
        keys = metadata["captures"][capture_id]["point_keys"]
        assert archive[keys["raw_sensor"]].shape == (2, 3)
        assert archive[keys["voxelized"]].shape == (2, 3)
        assert archive[keys["cluster_labels"]].shape == (2,)
    finally:
        archive.close()


def test_analyzer_ignores_cached_frames_and_flags_corruption():
    stable = [
        _record(0, "capture_0", [2.0, 0.0, 0.0]),
        _record(1, "capture_0", [2.0, 0.0, 0.0]),
        _record(2, "capture_1", [2.1, 0.0, 0.0]),
    ]
    stable_report = analyze_perception_records(stable)
    assert stable_report["agents"]["Drone1"]["capture_count"] == 2
    assert stable_report["agents"]["Drone1"]["anomalies"] == []

    corrupted = [
        _record(0, "capture_0", [1.0, 0.0, 0.0], age=-0.1, sensor_position=[3.0, 0.0, 0.0]),
        _record(1, "capture_1", [7.0, 0.0, 0.0], age=-0.1, sensor_position=[3.0, 0.0, 0.0]),
        _record(2, "capture_2", [-1.0, 0.0, 0.0], age=-0.1, sensor_position=[3.0, 0.0, 0.0]),
    ]
    report = analyze_perception_records(corrupted)
    anomalies = report["agents"]["Drone1"]["anomalies"]
    assert "sensor-to-vehicle offset exceeds 2 m" in anomalies
    assert "negative sensor age" in anomalies
    assert "UAV proxy behind camera" in anomalies
    assert "proxy association changes by more than 5 m" in anomalies


def test_diagnostic_report_outputs_json_markdown_and_timeline(tmp_path):
    records = [_record(0, "capture_0", [2.0, 0.0, 0.0])]
    log_path = tmp_path / "run.jsonl"
    with log_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")
    paths = generate_perception_diagnostics(str(log_path), None, str(tmp_path))
    assert (tmp_path / "run_perception_report.json").is_file()
    assert (tmp_path / "run_perception_report.md").is_file()
    assert (tmp_path / "run_perception_timeline.png").is_file()
    assert set(paths) == {
        str(tmp_path / "run_perception_report.json"),
        str(tmp_path / "run_perception_report.md"),
        str(tmp_path / "run_perception_timeline.png"),
    }
