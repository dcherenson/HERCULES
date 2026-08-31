"""Capture storage and offline diagnostics for distributed obstacle perception."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .mission_plots import box_vertices, ned_to_display, sensor_view_for_record
from .obstacle_detection import DetectionDiagnostics


TRACE_STAGES = ("raw_sensor", "world_input", "input", "finite_range", "ground_filtered", "voxelized")


class PerceptionTraceStore:
    """Keep bounded capture samples in memory and write one compressed sidecar."""

    def __init__(self, sample_limit: int = 512):
        self.sample_limit = max(1, int(sample_limit))
        self.metadata: Dict[str, dict] = {}
        self.arrays: Dict[str, np.ndarray] = {}

    @staticmethod
    def _safe_key(capture_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", str(capture_id))

    def _sample(self, values: Any) -> Tuple[np.ndarray, np.ndarray]:
        array = np.asarray(values)
        if array.ndim == 1 and array.size == 0:
            array = array.reshape((0,))
        if array.ndim == 1:
            array = array.reshape((-1, 1))
        if len(array) <= self.sample_limit:
            indices = np.arange(len(array), dtype=int)
        else:
            indices = np.linspace(0, len(array) - 1, self.sample_limit, dtype=int)
        return np.asarray(array[indices]), indices

    def add(
        self,
        capture_id: str,
        agent_name: str,
        sensor_type: str,
        raw_sensor_points: np.ndarray,
        world_points: np.ndarray,
        diagnostics: DetectionDiagnostics,
    ) -> str:
        """Store one capture once and return its stable ID."""

        capture_id = str(capture_id)
        if capture_id in self.metadata:
            return capture_id
        diagnostics = diagnostics or DetectionDiagnostics()
        safe = self._safe_key(capture_id)
        stage_arrays = {
            "raw_sensor": np.asarray(raw_sensor_points, dtype=np.float32).reshape((-1, 3)),
            "world_input": np.asarray(world_points, dtype=np.float32).reshape((-1, 3)),
        }
        stage_arrays.update({name: np.asarray(values) for name, values in diagnostics.stage_points.items()})
        point_keys = {}
        sampled_indices = None
        for stage_name in TRACE_STAGES:
            values = stage_arrays.get(stage_name)
            if values is None:
                continue
            sampled, indices = self._sample(values)
            key = safe + "__" + stage_name
            self.arrays[key] = np.asarray(sampled, dtype=np.float32)
            point_keys[stage_name] = key
            if stage_name == "voxelized":
                sampled_indices = indices
        for label_name in ("cluster_labels", "proxy_labels"):
            labels = getattr(diagnostics, label_name, np.empty(0, dtype=int))
            if sampled_indices is not None and len(labels) == diagnostics.stage_counts.get("voxelized", len(labels)):
                labels = np.asarray(labels)[sampled_indices]
            else:
                labels = np.asarray(labels)
            key = safe + "__" + label_name
            self.arrays[key] = np.asarray(labels, dtype=np.int32)
            point_keys[label_name] = key
        self.metadata[capture_id] = {
            "agent": str(agent_name),
            "sensor_type": str(sensor_type),
            "stage_counts": {str(name): int(value) for name, value in diagnostics.stage_counts.items()},
            "point_keys": point_keys,
        }
        return capture_id

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        arrays = dict(self.arrays)
        arrays["__metadata__"] = np.asarray(json.dumps({"sample_limit": self.sample_limit, "captures": self.metadata}))
        np.savez_compressed(path, **arrays)
        return path


def load_trace_sidecar(path: str) -> Tuple[dict, Any]:
    """Load sidecar metadata and an open NumPy archive."""

    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["__metadata__"].item()))
    return metadata, archive


def _quaternion_to_matrix(quaternion: Any) -> np.ndarray:
    values = np.asarray(quaternion if quaternion is not None else [1.0, 0.0, 0.0, 0.0], dtype=float)
    norm = np.linalg.norm(values)
    if values.shape != (4,) or norm <= 1e-12:
        return np.eye(3)
    w, x, y, z = values / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _fresh_captures(records: Sequence[Mapping], agent_name: str) -> List[Tuple[int, Mapping, dict, str]]:
    captures = []
    previous_id = object()
    for index, record in enumerate(records):
        view = sensor_view_for_record(record, agent_name)
        capture_id = view.get("capture_id")
        if capture_id is None:
            capture_id = "legacy-{}".format(index)
        if capture_id == previous_id:
            continue
        previous_id = capture_id
        captures.append((index, record, view, str(capture_id)))
    return captures


def _match_centers(previous: np.ndarray, current: np.ndarray, gate: float = 5.0) -> List[Tuple[int, int, float]]:
    distances = np.linalg.norm(previous[:, None, :] - current[None, :, :], axis=2)

    @lru_cache(maxsize=None)
    def best(previous_index: int, used_current: int) -> Tuple[int, float, Tuple[Tuple[int, int], ...]]:
        if previous_index >= len(previous):
            return 0, 0.0, ()
        best_result = best(previous_index + 1, used_current)  # Leave this proxy unmatched.
        for current_index in range(len(current)):
            if used_current & (1 << current_index) or distances[previous_index, current_index] > gate:
                continue
            count, cost, pairs = best(previous_index + 1, used_current | (1 << current_index))
            candidate = (count + 1, cost + float(distances[previous_index, current_index]), ((previous_index, current_index),) + pairs)
            if candidate[0] > best_result[0] or (candidate[0] == best_result[0] and candidate[1] < best_result[1]):
                best_result = candidate
        return best_result

    _, _, pairs = best(0, 0)
    return [(first, second, float(distances[first, second])) for first, second in pairs]


def analyze_perception_records(records: Sequence[Mapping], sidecar_metadata: Mapping | None = None) -> dict:
    """Compute capture-level stability and coordinate-frame diagnostics."""

    if not records:
        return {"agents": {}, "summary": {"anomalies": 0}}
    names = sorted({str(name) for record in records for name in (record.get("states") or {})})
    report = {"agents": {}, "summary": {"anomalies": 0}}
    sidecar_captures = (sidecar_metadata or {}).get("captures", {})
    for name in names:
        captures = _fresh_captures(records, name)
        jumps = []
        radius_changes = []
        offsets = []
        ages = []
        behind_count = 0
        negative_age_count = 0
        zero_count = 0
        zero_entered_count = 0
        previous_proxies = None
        unmatched_jumps = []
        for _, record, view, capture_id in captures:
            obstacle_data = (record.get("obstacles") or {}).get(name, {})
            proxies = obstacle_data.get("proxies", []) if isinstance(obstacle_data, Mapping) else []
            current_centers = np.asarray([proxy["center"] for proxy in proxies], dtype=float).reshape((-1, 3)) if proxies else np.empty((0, 3))
            current_radii = np.asarray([float(proxy.get("radius", 0.0)) for proxy in proxies], dtype=float)
            if previous_proxies is not None:
                previous_centers, previous_radii = previous_proxies
                matches = _match_centers(previous_centers, current_centers)
                matched_previous = {first for first, _, _ in matches}
                matched_current = {second for _, second, _ in matches}
                for first, second, distance in matches:
                    jumps.append(distance)
                    if first < len(previous_radii) and second < len(current_radii):
                        radius_changes.append(abs(previous_radii[first] - current_radii[second]))
                for first in set(range(len(previous_centers))) - matched_previous:
                    if len(current_centers):
                        unmatched_jumps.append(float(np.min(np.linalg.norm(current_centers - previous_centers[first], axis=1))))
                for second in set(range(len(current_centers))) - matched_current:
                    if len(previous_centers):
                        unmatched_jumps.append(float(np.min(np.linalg.norm(previous_centers - current_centers[second], axis=1))))
            previous_proxies = (current_centers, current_radii)
            try:
                age = float(obstacle_data.get("age", view.get("age", 0.0)))
                ages.append(age)
                negative_age_count += int(age < 0.0)
            except (TypeError, ValueError):
                pass
            stage_counts = sidecar_captures.get(capture_id, {}).get("stage_counts", {})
            zero_count += int(stage_counts.get("zero_returns", 0))
            zero_entered_count += int(stage_counts.get("zero_returns_entered_detector", 0))
            state = (record.get("states") or {}).get(name, {})
            vehicle_position = view.get("vehicle_position", state.get("position"))
            if view.get("position") is not None and vehicle_position is not None:
                offsets.append(float(np.linalg.norm(np.asarray(view["position"], dtype=float) - np.asarray(vehicle_position, dtype=float))))
            if view.get("sensor_type") == "uav_camera" and view.get("position") is not None:
                rotation = _quaternion_to_matrix(view.get("orientation_quaternion"))
                sensor_position = np.asarray(view["position"], dtype=float)
                for center in current_centers:
                    behind_count += int((rotation.T @ (center - sensor_position))[0] < 0.0)
        anomalies = []
        if offsets and max(offsets) > 2.0:
            anomalies.append("sensor-to-vehicle offset exceeds 2 m")
        if negative_age_count:
            anomalies.append("negative sensor age")
        if zero_entered_count:
            anomalies.append("zero LiDAR returns reached the detector")
        if behind_count:
            anomalies.append("UAV proxy behind camera")
        if jumps and max(jumps) > 5.0:
            anomalies.append("proxy jump exceeds 5 m")
        if unmatched_jumps and max(unmatched_jumps) > 5.0:
            anomalies.append("proxy association changes by more than 5 m")
        if radius_changes and max(radius_changes) > 2.0:
            anomalies.append("proxy radius change exceeds 2 m")
        agent_report = {
            "capture_count": len(captures),
            "proxy_count_min": min((len((r.get("obstacles") or {}).get(name, {}).get("proxies", [])) for _, r, _, _ in captures), default=0),
            "proxy_count_max": max((len((r.get("obstacles") or {}).get(name, {}).get("proxies", [])) for _, r, _, _ in captures), default=0),
            "max_matched_center_jump_m": max(jumps, default=0.0),
            "max_unmatched_center_jump_m": max(unmatched_jumps, default=0.0),
            "max_matched_radius_change_m": max(radius_changes, default=0.0),
            "max_sensor_vehicle_offset_m": max(offsets, default=0.0),
            "max_sensor_age_s": max(ages, default=0.0),
            "min_sensor_age_s": min(ages, default=0.0),
            "uav_behind_camera_proxy_count": behind_count,
            "negative_age_count": negative_age_count,
            "zero_return_count": zero_count,
            "zero_returns_entered_detector": zero_entered_count,
            "anomalies": anomalies,
            "worst_capture_id": captures[min(range(len(captures)), key=lambda i: offsets[i] if i < len(offsets) else 0.0)][3] if captures and offsets else (captures[0][3] if captures else None),
        }
        report["agents"][name] = agent_report
        report["summary"]["anomalies"] += len(anomalies)
    return report


def _write_timeline(records: Sequence[Mapping], report: Mapping, output_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(report.get("agents", {}))
    times = np.asarray([float(record.get("step", index)) * float(record.get("dt", 0.1)) for index, record in enumerate(records)])
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for name in names:
        captures = _fresh_captures(records, name)
        x = []
        count = []
        age = []
        offset = []
        for index, record, view, _ in captures:
            x.append(times[index] if index < len(times) else float(index))
            data = (record.get("obstacles") or {}).get(name, {})
            count.append(len(data.get("proxies", [])))
            age.append(float(data.get("age", view.get("age", np.nan))))
            state = (record.get("states") or {}).get(name, {})
            vehicle_position = view.get("vehicle_position", state.get("position"))
            offset.append(float(np.linalg.norm(np.asarray(view["position"]) - np.asarray(vehicle_position))) if view.get("position") is not None and vehicle_position is not None else np.nan)
        axes[0].plot(x, count, marker=".", label=name)
        axes[1].plot(x, age, marker=".", label=name)
        axes[2].plot(x, offset, marker=".", label=name)
    axes[0].set_ylabel("proxy count")
    axes[1].set_ylabel("sensor age (s)")
    axes[2].set_ylabel("sensor offset (m)")
    axes[2].set_xlabel("mission time (s)")
    axes[0].set_title("Obstacle-perception capture diagnostics")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        if names:
            axis.legend(fontsize="x-small", ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def _write_worst_capture_plot(record: Mapping, name: str, capture_id: str, metadata: Mapping, archive: Any, output_path: str) -> str | None:
    capture = metadata.get("captures", {}).get(capture_id, {})
    keys = capture.get("point_keys", {})
    if "raw_sensor" not in keys and "world_input" not in keys:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    def plot_points(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape((-1, 3))
        finite = np.all(np.isfinite(points), axis=1)
        # Ignore invalid depth sentinels and unbounded legacy values in the
        # human-readable overlay while retaining their stage counts in JSON.
        finite &= np.linalg.norm(points, axis=1) <= 35.0
        return points[finite]

    if "raw_sensor" in keys:
        points = plot_points(archive[keys["raw_sensor"]])
        if len(points):
            axes[0].scatter(points[:, 0], points[:, 1], s=2, alpha=0.35, label="raw sensor")
    axes[0].scatter([0], [0], color="red", marker="x", label="sensor origin")
    axes[0].set_title(name + " sensor frame")
    axes[0].set_xlabel("forward/local X")
    axes[0].set_ylabel("right/local Y")
    if "world_input" in keys:
        points = plot_points(archive[keys["world_input"]])
        if len(points):
            axes[1].scatter(points[:, 0], points[:, 1], s=2, alpha=0.35, label="world input")
    data = (record.get("obstacles") or {}).get(name, {})
    proxies = data.get("proxies", [])
    if proxies:
        centers = np.asarray([proxy["center"] for proxy in proxies])
        axes[1].scatter(centers[:, 0], centers[:, 1], color="tab:orange", marker="o", label="proxy centers")
    for obstacle in record.get("true_obstacles") or []:
        if obstacle.get("shape", "box") != "box":
            continue
        vertices = box_vertices(obstacle["center"], obstacle["dimensions"])
        low = vertices.min(axis=0)
        high = vertices.max(axis=0)
        axes[1].add_patch(plt.Rectangle((low[0], low[1]), high[0] - low[0], high[1] - low[1], color="gray", alpha=0.3))
    state = (record.get("states") or {}).get(name, {})
    if state.get("position") is not None:
        position = np.asarray(state["position"])
        axes[1].scatter([position[0]], [position[1]], color="red", marker="x", label="vehicle")
    axes[1].set_title(name + " world frame; capture " + capture_id)
    axes[1].set_xlabel("X (NED m)")
    axes[1].set_ylabel("Y (NED m)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="x-small")
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def generate_perception_diagnostics(log_path: str, sidecar_path: str | None, output_dir: str | None = None) -> List[str]:
    """Generate machine-readable and visual reports from a mission log."""

    from .mission_plots import load_mission_records

    records = load_mission_records(log_path)
    if not records:
        return []
    target_dir = output_dir or os.path.dirname(os.path.abspath(log_path))
    os.makedirs(target_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(log_path))[0]
    metadata = {}
    archive = None
    if sidecar_path and os.path.isfile(sidecar_path):
        metadata, archive = load_trace_sidecar(sidecar_path)
    report = analyze_perception_records(records, metadata)
    json_path = os.path.join(target_dir, stem + "_perception_report.json")
    md_path = os.path.join(target_dir, stem + "_perception_report.md")
    timeline_path = os.path.join(target_dir, stem + "_perception_timeline.png")
    with open(json_path, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    with open(md_path, "w", encoding="utf-8") as output:
        output.write("# Perception diagnostics\n\n")
        output.write("Anomalies are evaluated per fresh sensor capture; cached control cycles are excluded from jump metrics.\n\n")
        for name, data in sorted(report["agents"].items()):
            output.write("## {}\n\n".format(name))
            output.write("- Captures: {}\n- Proxy count: {}–{}\n- Maximum matched center jump: {:.3f} m\n- Maximum sensor offset: {:.3f} m\n- Sensor age range: {:.3f}–{:.3f} s\n".format(
                data["capture_count"], data["proxy_count_min"], data["proxy_count_max"],
                data["max_matched_center_jump_m"], data["max_sensor_vehicle_offset_m"],
                data["min_sensor_age_s"], data["max_sensor_age_s"],
            ))
            output.write("- Anomalies: {}\n\n".format(", ".join(data["anomalies"]) or "none"))
        if not archive:
            output.write("\nPoint sidecar unavailable; point-cloud overlays were skipped.\n")
    _write_timeline(records, report, timeline_path)
    paths = [json_path, md_path, timeline_path]
    if archive is not None:
        try:
            for name, data in sorted(report["agents"].items()):
                capture_id = data.get("worst_capture_id")
                if not capture_id:
                    continue
                record = next((record for record in records if sensor_view_for_record(record, name).get("capture_id", "legacy-{}".format(record.get("step", 0))) == capture_id), None)
                if record is None:
                    continue
                plot_path = os.path.join(target_dir, stem + "_{}_perception_debug.png".format(name))
                if _write_worst_capture_plot(record, name, capture_id, metadata, archive, plot_path):
                    paths.append(plot_path)
        finally:
            archive.close()
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path")
    parser.add_argument("--sidecar", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    for path in generate_perception_diagnostics(args.log_path, args.sidecar, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
