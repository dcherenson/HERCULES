"""Post-run plots for distributed mission diagnostics.

The orchestrator writes one JSON object per control cycle. This module keeps
plotting independent from AirSim so logs can be inspected after a run, even
when the simulator is no longer available.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def ned_to_display(points: Any) -> np.ndarray:
    """Convert AirSim NED coordinates to the physical Z-up display frame."""

    values = np.asarray(points, dtype=float).copy()
    if values.shape[-1] != 3:
        raise ValueError("expected coordinates with a final dimension of 3")
    values[..., 2] *= -1.0
    return values


def box_vertices(center: Any, dimensions: Any) -> np.ndarray:
    """Return the eight NED-frame vertices of an axis-aligned box."""

    center = np.asarray(center, dtype=float)
    dimensions = np.asarray(dimensions, dtype=float)
    if center.shape != (3,) or dimensions.shape != (3,):
        raise ValueError("center and dimensions must each contain three values")
    half = dimensions / 2.0
    return np.asarray([
        center + [sx * half[0], sy * half[1], sz * half[2]]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ])


def _quaternion_to_matrix(quaternion: Any) -> np.ndarray:
    """Convert a ``[w, x, y, z]`` quaternion to a rotation matrix."""

    values = np.asarray(quaternion if quaternion is not None else [1.0, 0.0, 0.0, 0.0], dtype=float)
    if values.shape != (4,):
        raise ValueError("quaternion must contain [w, x, y, z]")
    norm = np.linalg.norm(values)
    if norm <= 1e-12:
        return np.eye(3)
    w, x, y, z = values / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _transform_sensor_points(position: Any, orientation_quaternion: Any, points: Any) -> np.ndarray:
    position = np.asarray(position, dtype=float)
    local_points = np.asarray(points, dtype=float)
    world_points = local_points @ _quaternion_to_matrix(orientation_quaternion).T + position
    return ned_to_display(world_points)


def uav_frustum_segments(
    position: Any,
    orientation_quaternion: Any,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
    range_m: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return display-frame line segments for a camera frustum.

    AirSim camera coordinates are treated as forward ``+X``, right ``+Y`` and
    down ``+Z``.  The camera response orientation is a world-NED quaternion.
    """

    distance = float(range_m)
    if distance <= 0.0:
        return []
    half_horizontal = np.tan(np.radians(float(horizontal_fov_deg)) / 2.0)
    half_vertical = np.tan(np.radians(float(vertical_fov_deg)) / 2.0)
    far_corners = np.asarray([
        [distance, -distance * half_horizontal, -distance * half_vertical],
        [distance, distance * half_horizontal, -distance * half_vertical],
        [distance, distance * half_horizontal, distance * half_vertical],
        [distance, -distance * half_horizontal, distance * half_vertical],
    ])
    points = _transform_sensor_points(position, orientation_quaternion, far_corners)
    apex = ned_to_display(np.asarray(position, dtype=float))
    segments = [(apex, corner) for corner in points]
    segments.extend((points[index], points[(index + 1) % 4]) for index in range(4))
    return segments


def lidar_scan_segments(
    position: Any,
    orientation_quaternion: Any,
    horizontal_fov_start_deg: float,
    horizontal_fov_end_deg: float,
    vertical_fov_lower_deg: float,
    vertical_fov_upper_deg: float,
    range_m: float,
    azimuth_samples: int = 16,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return display-frame wireframe segments for a LiDAR scan volume."""

    distance = float(range_m)
    if distance <= 0.0 or azimuth_samples < 3:
        return []
    start = float(horizontal_fov_start_deg)
    end = float(horizontal_fov_end_deg)
    span = end - start
    closed = abs(span) >= 359.999
    count = max(3, int(azimuth_samples))
    angles = np.linspace(start, end, count + (1 if closed else 0), endpoint=not closed)
    if closed:
        angles = angles[:-1]
    points = []
    for elevation in (float(vertical_fov_lower_deg), float(vertical_fov_upper_deg)):
        elevation_rad = np.radians(elevation)
        points.append(np.asarray([
            [distance * np.cos(elevation_rad) * np.cos(np.radians(angle)),
             distance * np.cos(elevation_rad) * np.sin(np.radians(angle)),
             distance * np.sin(elevation_rad)]
            for angle in angles
        ]))
    rings = [_transform_sensor_points(position, orientation_quaternion, ring) for ring in points]
    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    for ring in rings:
        for index in range(len(ring) - 1 + int(closed)):
            segments.append((ring[index % len(ring)], ring[(index + 1) % len(ring)]))
    for lower, upper in zip(rings[0], rings[1]):
        segments.append((lower, upper))
    return segments


def sphere_wireframe_segments(center: Any, radius: float, latitude_samples: int = 5, longitude_samples: int = 8) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return display-frame line segments for a compact wireframe sphere."""

    radius = float(radius)
    if radius <= 0.0:
        return []
    center = ned_to_display(np.asarray(center, dtype=float))
    latitudes = np.linspace(-np.pi / 2.0, np.pi / 2.0, max(3, int(latitude_samples)))
    longitudes = np.linspace(0.0, 2.0 * np.pi, max(4, int(longitude_samples)), endpoint=False)
    rings = []
    for latitude in latitudes:
        ring = np.asarray([
            center + radius * np.asarray([np.cos(latitude) * np.cos(longitude),
                                           np.cos(latitude) * np.sin(longitude),
                                           np.sin(latitude)])
            for longitude in longitudes
        ])
        rings.append(ring)
    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    for ring in rings:
        for index in range(len(ring)):
            segments.append((ring[index], ring[(index + 1) % len(ring)]))
    for lower, upper in zip(rings[0], rings[-1]):
        segments.append((lower, upper))
    for ring_index in range(len(rings) - 1):
        for longitude_index in range(len(longitudes)):
            segments.append((rings[ring_index][longitude_index], rings[ring_index + 1][longitude_index]))
    return segments


def sensor_view_for_record(record: Mapping, vehicle_name: str) -> Dict[str, Any]:
    """Return the cached sensor view and its propagated age from a log record."""

    obstacle_data = (record.get("obstacles") or {}).get(vehicle_name, {})
    sensor_view = obstacle_data.get("sensor_view", {}) if isinstance(obstacle_data, Mapping) else {}
    return dict(sensor_view) if isinstance(sensor_view, Mapping) else {}


def load_mission_records(log_path: str) -> List[dict]:
    """Load JSONL diagnostics records in file order."""

    records: List[dict] = []
    with open(log_path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("Invalid JSON in {} at line {}".format(log_path, line_number)) from error
            if not isinstance(record, dict):
                raise ValueError("Expected an object in {} at line {}".format(log_path, line_number))
            records.append(record)
    return records


def extract_trajectories(records: Sequence[Mapping]) -> Dict[str, np.ndarray]:
    """Return each vehicle trajectory as an ``N x 3`` NumPy array."""

    positions: Dict[str, List[np.ndarray]] = {}
    for record in records:
        for name, state in (record.get("states") or {}).items():
            position = state.get("position") if isinstance(state, Mapping) else None
            if position is None or len(position) != 3:
                continue
            positions.setdefault(str(name), []).append(np.asarray(position, dtype=float))
    return {name: np.vstack(values) for name, values in positions.items() if values}


def _record_times(records: Sequence[Mapping], default_dt: float = 0.1) -> np.ndarray:
    """Build a time axis in seconds, preferring simulation step and ``dt``."""

    if not records:
        return np.empty(0, dtype=float)
    try:
        dt = float(records[0].get("dt", default_dt))
    except (TypeError, ValueError):
        dt = default_dt
    steps = []
    for index, record in enumerate(records):
        try:
            steps.append(float(record.get("step", index)))
        except (TypeError, ValueError):
            steps.append(float(index))
    return (np.asarray(steps, dtype=float) - steps[0]) * dt


def compute_collision_clearances(
    records: Sequence[Mapping],
    vehicle_radii: Mapping[str, float] | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute each vehicle's minimum clearance to another vehicle or obstacle.

    A clearance is geometric distance minus the participating safety radii.
    Zero means contact and a negative value means the logged geometry overlaps.
    Obstacle clearances use the obstacle proxy radius already recorded by the
    perception/CBF pipeline, including any age margin.
    """

    if not records:
        return np.empty(0, dtype=float), {}
    if vehicle_radii is None:
        saved_radii = records[0].get("vehicle_radii", {})
        vehicle_radii = {str(name): float(radius) for name, radius in saved_radii.items()}

    names = sorted({str(name) for record in records for name in (record.get("states") or {})})
    clearances = {name: np.full(len(records), np.nan, dtype=float) for name in names}

    for index, record in enumerate(records):
        states = record.get("states") or {}
        positions = {}
        for name, state in states.items():
            if isinstance(state, Mapping) and state.get("position") is not None:
                positions[str(name)] = np.asarray(state["position"], dtype=float)

        for name, position in positions.items():
            minimum = np.inf
            own_radius = float(vehicle_radii.get(name, 0.0))
            for other_name, other_position in positions.items():
                if other_name == name:
                    continue
                other_radius = float(vehicle_radii.get(other_name, 0.0))
                minimum = min(minimum, float(np.linalg.norm(position - other_position)) - own_radius - other_radius)

            obstacle_data = (record.get("obstacles") or {}).get(name, {})
            proxies = obstacle_data.get("proxies", []) if isinstance(obstacle_data, Mapping) else []
            for proxy in proxies:
                center = proxy.get("center")
                if center is None:
                    continue
                try:
                    obstacle_radius = float(proxy.get("radius", 0.0))
                    clearance = float(np.linalg.norm(position - np.asarray(center, dtype=float)) - own_radius - obstacle_radius)
                except (TypeError, ValueError):
                    continue
                minimum = min(minimum, clearance)
            if np.isfinite(minimum):
                clearances[name][index] = minimum
    return _record_times(records), clearances


def _set_equal_3d_axes(axis, trajectories: Mapping[str, np.ndarray]) -> None:
    points = [trajectory for trajectory in trajectories.values() if len(trajectory)]
    if not points:
        return
    all_points = np.vstack(points)
    lower = np.nanmin(all_points, axis=0)
    upper = np.nanmax(all_points, axis=0)
    center = (lower + upper) / 2.0
    half_range = max(float(np.max(upper - lower)) / 2.0, 0.5)
    axis.set_xlim(center[0] - half_range, center[0] + half_range)
    axis.set_ylim(center[1] - half_range, center[1] + half_range)
    axis.set_zlim(center[2] - half_range, center[2] + half_range)


def _collision_is_relevant(collision: Mapping) -> bool:
    return bool(collision.get("relevant", collision.get("has_collided", False)))


def _box_faces(vertices: np.ndarray) -> List[List[np.ndarray]]:
    indices = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
               (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
    return [[vertices[index] for index in face] for face in indices]


def _animation_axis_limits(records: Sequence[Mapping], names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    points: List[np.ndarray] = []
    for record in records:
        for name in names:
            state = (record.get("states") or {}).get(name, {})
            position = state.get("position") if isinstance(state, Mapping) else None
            if position is not None and len(position) == 3:
                points.append(ned_to_display(position))
        for obstacle in record.get("true_obstacles") or []:
            if obstacle.get("shape", "box") != "box":
                continue
            try:
                points.extend(ned_to_display(box_vertices(obstacle["center"], obstacle["dimensions"])))
            except (KeyError, TypeError, ValueError):
                continue
        for obstacle_data in (record.get("obstacles") or {}).values():
            proxies = obstacle_data.get("proxies", []) if isinstance(obstacle_data, Mapping) else []
            for proxy in proxies:
                try:
                    center = np.asarray(proxy["center"], dtype=float)
                    radius = float(proxy.get("radius", 0.0))
                    points.extend((ned_to_display(center + offset) for offset in (
                        [-radius, 0, 0], [radius, 0, 0], [0, -radius, 0],
                        [0, radius, 0], [0, 0, -radius], [0, 0, radius])))
                except (KeyError, TypeError, ValueError):
                    continue
    if not points:
        return np.asarray([-10.0, -10.0, -10.0]), np.asarray([10.0, 10.0, 10.0])
    all_points = np.vstack(points)
    lower = np.nanmin(all_points, axis=0)
    upper = np.nanmax(all_points, axis=0)
    span = np.maximum(upper - lower, 1.0)
    padding = np.maximum(0.1 * span, 1.0)
    return lower - padding, upper + padding


def _plot_segments(
    axis,
    segments: Sequence[Tuple[np.ndarray, np.ndarray]],
    color: Any,
    alpha: float = 0.5,
    linewidth: float = 0.8,
    linestyle: str = "-",
) -> None:
    for start, end in segments:
        points = np.vstack((start, end))
        axis.plot(
            points[:, 0], points[:, 1], points[:, 2],
            color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle,
        )


def plot_perception_animation_3d(
    records: Sequence[Mapping],
    output_path: str,
    fps: float | None = None,
) -> str:
    """Render a post-run MP4 of trajectories, FOVs, beliefs, and true boxes."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if not records:
        raise ValueError("cannot animate an empty mission log")
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is required to render the perception animation")
    try:
        dt = float(records[0].get("dt", 0.1))
    except (TypeError, ValueError):
        dt = 0.1
    effective_fps = float(fps) if fps is not None else (1.0 / max(dt, 1e-6))
    if effective_fps <= 0.0:
        raise ValueError("animation fps must be positive")

    names = sorted({str(name) for record in records for name in (record.get("states") or {})})
    lower, upper = _animation_axis_limits(records, names)
    colors = {name: plt.get_cmap("tab10")(index % 10) for index, name in enumerate(names)}
    figure = plt.figure(figsize=(12, 8))
    axis = figure.add_subplot(111, projection="3d")
    times = _record_times(records, dt)
    has_any_sensor_view = any(sensor_view_for_record(record, name) for record in records for name in names)
    has_any_truth = any(record.get("true_obstacles") is not None for record in records)
    true_obstacles = next((record.get("true_obstacles") for record in records if record.get("true_obstacles") is not None), [])

    def draw_frame(frame_index: int) -> None:
        axis.cla()
        for obstacle in true_obstacles or []:
            if obstacle.get("shape", "box") != "box":
                continue
            try:
                vertices = ned_to_display(box_vertices(obstacle["center"], obstacle["dimensions"]))
            except (KeyError, TypeError, ValueError):
                continue
            axis.add_collection3d(Poly3DCollection(
                _box_faces(vertices), facecolors="gray", edgecolors="dimgray", alpha=0.32, linewidths=0.8
            ))

        for name in names:
            history = []
            current_position = None
            for record in records[:frame_index + 1]:
                state = (record.get("states") or {}).get(name, {})
                position = state.get("position") if isinstance(state, Mapping) else None
                if position is not None and len(position) == 3:
                    current_position = np.asarray(position, dtype=float)
                    history.append(current_position)
            if not history:
                continue
            trajectory = ned_to_display(np.vstack(history))
            color = colors[name]
            axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=color, linewidth=1.8)
            axis.scatter(*trajectory[-1], color=color, s=35, depthshade=False)
            axis.text(*trajectory[-1], " " + name, color=color, fontsize=8)
            collision = (records[frame_index].get("collisions") or {}).get(name, {})
            if isinstance(collision, Mapping) and _collision_is_relevant(collision):
                axis.scatter(*trajectory[-1], color="red", marker="x", s=75, linewidths=2.0, depthshade=False)

            obstacle_data = (records[frame_index].get("obstacles") or {}).get(name, {})
            proxies = obstacle_data.get("proxies", []) if isinstance(obstacle_data, Mapping) else []
            for proxy in proxies:
                try:
                    _plot_segments(
                        axis,
                        sphere_wireframe_segments(proxy["center"], proxy.get("radius", 0.0)),
                        color,
                        0.38,
                        0.65,
                        ":",
                    )
                except (KeyError, TypeError, ValueError):
                    continue

            sensor_view = sensor_view_for_record(records[frame_index], name)
            if sensor_view:
                try:
                    sensor_type = sensor_view.get("sensor_type")
                    if sensor_type == "uav_camera":
                        segments = uav_frustum_segments(
                            sensor_view["position"], sensor_view.get("orientation_quaternion"),
                            sensor_view["horizontal_fov_deg"], sensor_view["vertical_fov_deg"], sensor_view["range_m"],
                        )
                        _plot_segments(axis, segments, "deepskyblue", 0.65, 0.9)
                except (KeyError, TypeError, ValueError):
                    pass

        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_zlim(lower[2], upper[2])
        try:
            axis.set_box_aspect(upper - lower)
        except (AttributeError, TypeError):
            pass
        axis.view_init(elev=45.0, azim=-60.0)
        mission_time = times[frame_index] if frame_index < len(times) else frame_index * dt
        axis.set_title("Distributed mission perception (step {}; t = {:.2f} s)".format(
            records[frame_index].get("step", frame_index), mission_time
        ))
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_zlabel("Altitude Z-up (m; -AirSim NED Z)")
        missing = []
        if not has_any_sensor_view:
            missing.append("sensor FOV unavailable in this log")
        if not has_any_truth:
            missing.append("true obstacle geometry unavailable in this log")
        if missing:
            axis.text2D(0.02, 0.96, " | ".join(missing), transform=axis.transAxes, color="darkred", fontsize=9)
        legend_handles = [Line2D([0], [0], color=colors[name], lw=2, label=name) for name in names]
        legend_handles.extend([
            Line2D([0], [0], color="gray", lw=5, alpha=0.45, label="true obstacle"),
            Line2D([0], [0], color="black", lw=1, linestyle=":", alpha=0.65, label="agent obstacle estimate"),
            Line2D([0], [0], color="deepskyblue", lw=1.5, alpha=0.8, label="UAV camera FOV"),
            Line2D([0], [0], color="red", marker="x", linestyle="None", markersize=8, label="AirSim collision"),
        ])
        if legend_handles:
            axis.legend(handles=legend_handles, loc="upper left", fontsize="x-small")
        ages = []
        for name in names:
            sensor_view = sensor_view_for_record(records[frame_index], name)
            if "age" in sensor_view:
                try:
                    ages.append("{}: {:.2f}s".format(name, float(sensor_view["age"])))
                except (TypeError, ValueError):
                    pass
        if ages:
            axis.text2D(0.02, 0.02, "Sensor age — " + ", ".join(ages), transform=axis.transAxes, fontsize=8)

    draw_frame(0)
    movie = animation.FuncAnimation(figure, draw_frame, frames=len(records), interval=1000.0 / effective_fps, blit=False)
    writer = animation.FFMpegWriter(
        fps=effective_fps,
        codec="libx264",
        bitrate=1800,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    movie.save(output_path, writer=writer, dpi=110)
    plt.close(figure)
    return output_path


def plot_trajectories_3d(records: Sequence[Mapping], output_path: str) -> str:
    """Save a 3D trajectory plot for every vehicle in the log."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trajectories = extract_trajectories(records)
    # AirSim reports NED coordinates: positive Z points downward. Convert only
    # the display coordinate so physical altitude appears upward in the plot.
    display_trajectories = {
        name: trajectory * np.array([1.0, 1.0, -1.0])
        for name, trajectory in trajectories.items()
    }
    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")
    colors = plt.get_cmap("tab10")
    collision_label_added = False
    for index, (name, trajectory) in enumerate(sorted(display_trajectories.items())):
        color = colors(index % 10)
        axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=name, color=color, linewidth=1.8)
        axis.scatter(*trajectory[0], color=color, marker="o", s=30)
        axis.scatter(*trajectory[-1], color=color, marker="X", s=45)
        collision_positions = []
        for record in records:
            collision = (record.get("collisions") or {}).get(name, {})
            state = (record.get("states") or {}).get(name, {})
            position = state.get("position") if isinstance(state, Mapping) else None
            if collision.get("relevant", collision.get("has_collided", False)) and position is not None and len(position) == 3:
                collision_positions.append(np.asarray(position, dtype=float) * np.array([1.0, 1.0, -1.0]))
        if collision_positions:
            points = np.vstack(collision_positions)
            axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color="red",
                marker="x",
                s=70,
                linewidths=2.0,
                label="AirSim collision" if not collision_label_added else None,
                zorder=5,
            )
            collision_label_added = True
    axis.set_title("Distributed mission vehicle trajectories")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Altitude Z-up (m; -AirSim NED Z)")
    if display_trajectories:
        axis.legend(loc="best", fontsize="small")
    _set_equal_3d_axes(axis, display_trajectories)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_collision_clearances(
    records: Sequence[Mapping],
    output_path: str,
    vehicle_radii: Mapping[str, float] | None = None,
) -> str:
    """Save one combined minimum-collision-clearance plot for all vehicles."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, clearances = compute_collision_clearances(records, vehicle_radii)
    figure, axis = plt.subplots(figsize=(11, 6))
    for name, values in sorted(clearances.items()):
        axis.plot(times, values, label=name, linewidth=1.6)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.2, label="collision/contact")
    collision_times = []
    collision_values = []
    for index, record in enumerate(records):
        for collision in (record.get("collisions") or {}).values():
            if isinstance(collision, Mapping) and collision.get("relevant", collision.get("has_collided", False)):
                collision_times.append(times[index])
                collision_values.append(0.0)
    if collision_times:
        axis.scatter(
            collision_times,
            collision_values,
            color="red",
            marker="x",
            s=55,
            linewidths=2.0,
            label="AirSim collision",
            zorder=5,
        )
    axis.set_title("Minimum estimated collision clearance by vehicle")
    axis.set_xlabel("Mission time (s)")
    axis.set_ylabel("Minimum estimated clearance (m)")
    axis.grid(True, alpha=0.3)
    if clearances:
        axis.legend(loc="best", fontsize="small", ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def generate_mission_plots(
    log_path: str,
    output_dir: str | None = None,
    vehicle_radii: Mapping[str, float] | None = None,
    animation_fps: float | None = None,
    include_animation: bool = True,
) -> List[str]:
    """Generate standard plots and, unless disabled, the perception MP4."""

    records = load_mission_records(log_path)
    if not records:
        return []
    target_dir = output_dir or os.path.dirname(os.path.abspath(log_path))
    os.makedirs(target_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(log_path))[0]
    trajectory_path = os.path.join(target_dir, stem + "_trajectories_3d.png")
    clearance_path = os.path.join(target_dir, stem + "_collision_clearance.png")
    plot_trajectories_3d(records, trajectory_path)
    plot_collision_clearances(records, clearance_path, vehicle_radii)
    paths = [trajectory_path, clearance_path]
    if include_animation:
        animation_path = os.path.join(target_dir, stem + "_perception_3d.mp4")
        plot_perception_animation_3d(records, animation_path, animation_fps)
        paths.append(animation_path)
    return paths
