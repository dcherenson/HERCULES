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


def route_up_xy(points: Any, heading: float, origin: Any = (0.0, 0.0)) -> np.ndarray:
    """Rotate NED XY coordinates so the route heading points up the plot."""

    values = np.asarray(points, dtype=float)
    if values.shape[-1] < 2:
        raise ValueError("points must contain at least two coordinates")
    origin = np.asarray(origin, dtype=float).reshape(2)
    delta = values[..., :2] - origin
    cosine, sine = np.cos(float(heading)), np.sin(float(heading))
    return np.stack((-sine * delta[..., 0] + cosine * delta[..., 1],
                     cosine * delta[..., 0] + sine * delta[..., 1]), axis=-1)


def _uav_fov_footprint(sensor_view: Mapping[str, Any]) -> Optional[np.ndarray]:
    try:
        position = np.asarray(sensor_view["position"], dtype=float).reshape(3)
        rotation = _quaternion_to_matrix(sensor_view.get("orientation_quaternion"))
        distance = float(sensor_view["range_m"])
        horizontal = np.tan(np.radians(float(sensor_view["horizontal_fov_deg"])) / 2.0)
        vertical = np.tan(np.radians(float(sensor_view.get("vertical_fov_deg", 60.0))) / 2.0)
    except (KeyError, TypeError, ValueError):
        return None
    if distance <= 0.0:
        return None
    corners = np.asarray([
        [distance, -distance * horizontal, -distance * vertical],
        [distance, distance * horizontal, -distance * vertical],
        [distance, distance * horizontal, distance * vertical],
        [distance, -distance * horizontal, distance * vertical],
    ])
    return corners @ rotation.T + position


def _truth_xy_extent(obstacles: Sequence[Mapping[str, Any]]) -> List[np.ndarray]:
    extent: List[np.ndarray] = []
    for obstacle in obstacles:
        try:
            center = np.asarray(obstacle["center"], dtype=float).reshape(3)
            if obstacle.get("shape", "box") == "sphere":
                radius = float(obstacle["radius"])
                extent.extend(center[:2] + offset for offset in (
                    [-radius, 0.0], [radius, 0.0], [0.0, -radius], [0.0, radius]
                ))
            else:
                half = np.asarray(obstacle["dimensions"], dtype=float).reshape(3)[:2] / 2.0
                extent.extend(center[:2] + offset for offset in (
                    [-half[0], -half[1]], [-half[0], half[1]],
                    [half[0], -half[1]], [half[0], half[1]],
                ))
        except (KeyError, TypeError, ValueError):
            continue
    return extent


def _topdown_limits(records: Sequence[Mapping], names: Sequence[str], heading: float, origin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points: List[np.ndarray] = []
    for record in records:
        for name in names:
            state = (record.get("states") or {}).get(name, {})
            position = state.get("position") if isinstance(state, Mapping) else None
            if position is not None:
                points.append(route_up_xy(np.asarray(position, dtype=float), heading, origin))
    goal = next((record.get("goal") for record in records if record.get("goal") is not None), None)
    if goal is not None:
        try:
            points.append(route_up_xy(np.asarray(goal, dtype=float), heading, origin))
        except (TypeError, ValueError):
            pass
    truth = next((record.get("true_obstacles") for record in records if record.get("true_obstacles") is not None), [])
    if truth:
        points.extend(route_up_xy(point, heading, origin) for point in _truth_xy_extent(truth))
    if not points:
        return np.asarray([-10.0, -10.0]), np.asarray([10.0, 10.0])
    values = np.vstack(points)
    lower, upper = np.nanmin(values, axis=0), np.nanmax(values, axis=0)
    span = np.maximum(upper - lower, 1.0)
    half = max(float(np.max(span)) / 2.0, 5.0)
    center = (lower + upper) / 2.0
    padding = max(1.0, 0.08 * half)
    return center - half - padding, center + half + padding


def plot_topdown_animation(records: Sequence[Mapping], mp4_path: str, gif_path: str,
                           fps: float | None = None, playback_speed: float = 2.0) -> Tuple[str, str]:
    """Render the compact route-up top-down presentation animation."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Polygon

    if not records:
        raise ValueError("cannot animate an empty mission log")
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is required to render the top-down animation")
    names = sorted({str(name) for record in records for name in (record.get("states") or {})})
    types = records[0].get("vehicle_types") or {}
    initial_positions = [
        np.asarray((records[0].get("states") or {})[name].get("position"), dtype=float)
        for name in names if (records[0].get("states") or {}).get(name, {}).get("position") is not None
    ]
    origin = np.mean(np.asarray(initial_positions), axis=0)[:2] if initial_positions else np.zeros(2)
    first_position = np.mean(np.asarray(initial_positions), axis=0) if initial_positions else np.zeros(3)
    goal = np.asarray(records[0].get("goal", first_position + [0.0, 1.0, 0.0]), dtype=float)
    heading = float(np.arctan2(goal[1] - first_position[1], goal[0] - first_position[0]))
    lower, upper = _topdown_limits(records, names, heading, origin)
    source_fps = max(float(fps if fps is not None else 1.0 / max(float(records[0].get("dt", 0.1)), 1e-6)), 1.0)
    if playback_speed <= 0.0:
        raise ValueError("playback_speed must be positive")
    output_fps = source_fps * float(playback_speed)
    first_step = float(records[0].get("step", 0))
    last_step = float(records[-1].get("step", len(records) - 1))
    duration = max((last_step - first_step + 1.0) * float(records[0].get("dt", 0.1)), 1.0 / source_fps)

    def frame_indices() -> List[int]:
        # Render one frame for each source mission-time sample, then write at
        # the faster output rate. This shortens playback instead of merely
        # resampling the same mission interval at a denser frame rate.
        count = max(len(records), int(np.ceil(duration * source_fps - 1e-9)))
        times = np.arange(count, dtype=float) / source_fps
        record_times = _record_times(records)
        return [min(len(records) - 1, int(np.searchsorted(record_times, value, side="right") - 1)) for value in times]

    colors = {name: plt.get_cmap("tab10")(index % 10) for index, name in enumerate(names)}

    def render(output_path: str, output_fps: float, dpi: int, writer: Any) -> None:
        figure, axis = plt.subplots(figsize=(7.2, 7.2), dpi=dpi)
        indices = frame_indices()

        def draw(frame_index: int) -> None:
            axis.clear()
            record = records[frame_index]
            truth = next((item.get("true_obstacles") for item in records if item.get("true_obstacles") is not None), [])
            for obstacle in truth or []:
                try:
                    center = route_up_xy(np.asarray(obstacle["center"], dtype=float), heading, origin)
                    if obstacle.get("shape", "box") == "sphere":
                        radius = float(obstacle["radius"])
                        axis.add_patch(Circle(center, radius, color="gray", alpha=0.28, linewidth=1.0))
                    else:
                        dimensions = np.asarray(obstacle["dimensions"], dtype=float)
                        corners = route_up_xy(np.asarray([
                            np.asarray(obstacle["center"], dtype=float) + [sx * dimensions[0] / 2.0, sy * dimensions[1] / 2.0, 0.0]
                            for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1))
                        ]), heading, origin)
                        axis.add_patch(Polygon(corners, closed=True, color="gray", alpha=0.28, linewidth=1.0))
                except (KeyError, TypeError, ValueError):
                    continue
            goal_point = route_up_xy(goal, heading, origin)
            axis.scatter(
                [goal_point[0]], [goal_point[1]], marker="*", s=190,
                facecolor="gold", edgecolor="black", linewidth=1.0,
                zorder=10,
            )
            axis.annotate(
                "GOAL", (goal_point[0], goal_point[1]), xytext=(6, 6),
                textcoords="offset points", color="black", fontsize="small",
                fontweight="bold", zorder=11,
            )
            for name in names:
                state = (record.get("states") or {}).get(name, {})
                position = state.get("position") if isinstance(state, Mapping) else None
                if position is None:
                    continue
                trajectory = []
                for previous in records[:frame_index + 1]:
                    previous_state = (previous.get("states") or {}).get(name, {})
                    if previous_state.get("position") is not None:
                        trajectory.append(route_up_xy(np.asarray(previous_state["position"], dtype=float), heading, origin))
                if trajectory:
                    trail = np.asarray(trajectory)
                    axis.plot(trail[:, 0], trail[:, 1], color=colors[name], linewidth=2.0)
                    current = trail[-1]
                    axis.scatter([current[0]], [current[1]], color=colors[name], s=36, zorder=6)
                obstacle_data = (record.get("obstacles") or {}).get(name, {})
                for proxy in obstacle_data.get("proxies", []) if isinstance(obstacle_data, Mapping) else []:
                    try:
                        proxy_center = route_up_xy(np.asarray(proxy["center"], dtype=float), heading, origin)
                        axis.add_patch(Circle(proxy_center, float(proxy.get("radius", 0.0)), fill=False,
                                              edgecolor=colors[name], alpha=0.35, linestyle=":", linewidth=1.0))
                    except (KeyError, TypeError, ValueError):
                        continue
                if types.get(name) == "drone":
                    view = sensor_view_for_record(record, name)
                    footprint = _uav_fov_footprint(view) if view.get("sensor_type") == "uav_camera" else None
                    if footprint is not None:
                        footprint_xy = route_up_xy(footprint, heading, origin)
                        axis.add_patch(Polygon(footprint_xy[:, :2], closed=True, facecolor="deepskyblue",
                                              edgecolor="deepskyblue", alpha=0.10, linewidth=1.0))
            for link in record.get("communication_links") or []:
                if len(link) != 2 or link[0] not in names or link[1] not in names:
                    continue
                first = (record.get("states") or {}).get(link[0], {}).get("position")
                second = (record.get("states") or {}).get(link[1], {}).get("position")
                if first is not None and second is not None:
                    segment = route_up_xy(np.asarray([first, second], dtype=float), heading, origin)
                    axis.plot(segment[:, 0], segment[:, 1], color="black", alpha=0.35, linestyle="--", linewidth=0.8)
            for name in names:
                collision = (record.get("collisions") or {}).get(name, {})
                state = (record.get("states") or {}).get(name, {})
                if collision.get("relevant", collision.get("has_collided", False)) and state.get("position") is not None:
                    point = route_up_xy(np.asarray(state["position"], dtype=float), heading, origin)
                    axis.scatter([point[0]], [point[1]], color="red", marker="x", s=75, linewidths=2.0, zorder=8)
            axis.set_xlim(lower[0], upper[0])
            axis.set_ylim(lower[1], upper[1])
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("Route-left (m)")
            axis.set_ylabel("Progress to goal (m)")
            mission_time = float(record.get("step", frame_index)) * float(record.get("dt", 0.1))
            axis.set_title("Distributed mission — top down (t = {:.1f} s)".format(mission_time))
            axis.legend(handles=[
                Line2D([0], [0], color="deepskyblue", lw=6, alpha=0.25, label="UAV camera FOV"),
                Line2D([0], [0], marker="*", color="gold", markeredgecolor="black", linestyle="None", markersize=11, label="goal"),
                Line2D([0], [0], color="gray", lw=6, alpha=0.35, label="true obstacle"),
                Line2D([0], [0], color="black", lw=1, linestyle="--", alpha=0.5, label="communication"),
                Line2D([0], [0], color="black", lw=1, linestyle=":", alpha=0.5, label="obstacle estimate"),
                Line2D([0], [0], color="red", marker="x", linestyle="None", markersize=7, label="collision"),
            ], loc="upper left", fontsize="x-small")

        movie = animation.FuncAnimation(figure, draw, frames=indices, interval=1000.0 / output_fps, blit=False)
        movie.save(output_path, writer=writer, dpi=dpi)
        plt.close(figure)

    os.makedirs(os.path.dirname(os.path.abspath(mp4_path)), exist_ok=True)
    render(mp4_path, output_fps, 100, animation.FFMpegWriter(fps=output_fps, codec="libx264", bitrate=1600, extra_args=["-pix_fmt", "yuv420p"]))
    from modules.video_recording import verify_mp4

    verify_mp4(mp4_path)
    # Encode the GIF from the verified MP4 path through ffmpeg.  This keeps
    # the Matplotlib renderer independent of PillowWriter and guarantees the
    # same palette conversion used by the simulator video artifacts.
    from modules.video_recording import encode_gif

    encode_gif(mp4_path, gif_path, output_fps, 540)
    return mp4_path, gif_path


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
    playback_speed: float = 2.0,
) -> List[str]:
    """Generate standard plots and, unless disabled, the top-down animations."""

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
        topdown_mp4 = os.path.join(target_dir, stem + "_topdown.mp4")
        topdown_gif = os.path.join(target_dir, stem + "_topdown.gif")
        plot_topdown_animation(records, topdown_mp4, topdown_gif, animation_fps, playback_speed)
        paths.extend([topdown_mp4, topdown_gif])
    return paths
