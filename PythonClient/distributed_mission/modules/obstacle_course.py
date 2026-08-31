"""Serializable obstacle-course descriptions used by the editor and simulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


DEFAULT_GOAL = [16.0, 1.0, -1.0]


def _finite_vector(value: Any, length: int, field_name: str) -> List[float]:
    try:
        vector = np.asarray(value, dtype=float).reshape(length)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must contain {} numeric values".format(field_name, length)) from error
    if not np.all(np.isfinite(vector)):
        raise ValueError("{} must contain only finite values".format(field_name))
    return vector.tolist()


def normalize_obstacle(obstacle: Mapping[str, Any], index: int = 0) -> Dict[str, Any]:
    """Validate and normalize one editor/orchestrator obstacle record."""

    if not isinstance(obstacle, Mapping):
        raise ValueError("obstacle {} must be an object".format(index))
    shape = str(obstacle.get("shape", "box")).strip().lower()
    if shape not in {"box", "sphere"}:
        raise ValueError("obstacle {} shape must be box or sphere".format(index))
    result: Dict[str, Any] = {
        "id": str(obstacle.get("id", "obstacle_{}".format(index))),
        "shape": shape,
        "center": _finite_vector(obstacle.get("center"), 3, "obstacle center"),
    }
    if shape == "box":
        dimensions = np.asarray(_finite_vector(obstacle.get("dimensions"), 3, "box dimensions"), dtype=float)
        if np.any(dimensions <= 0.0):
            raise ValueError("box dimensions must be positive")
        result["dimensions"] = dimensions.tolist()
    else:
        try:
            radius = float(obstacle.get("radius"))
        except (TypeError, ValueError) as error:
            raise ValueError("sphere radius must be numeric") from error
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("sphere radius must be positive and finite")
        result["radius"] = radius
    for field in ("asset_name", "label"):
        if field in obstacle:
            result[field] = str(obstacle[field])
    return result


def normalize_course(course: Any) -> Dict[str, Any]:
    """Return a validated course mapping with explicit NED offsets."""

    if isinstance(course, Mapping):
        obstacles_value = course.get("obstacles", [])
        goal_value = course.get("goal", DEFAULT_GOAL)
        waypoints_value = course.get("waypoints", [])
        version = int(course.get("version", 1))
    elif isinstance(course, Sequence) and not isinstance(course, (str, bytes)):
        obstacles_value = course
        goal_value = DEFAULT_GOAL
        waypoints_value = []
        version = 1
    else:
        raise ValueError("course must be an object or obstacle list")
    if not isinstance(obstacles_value, Sequence) or isinstance(obstacles_value, (str, bytes)):
        raise ValueError("course obstacles must be a list")
    if not isinstance(waypoints_value, Sequence) or isinstance(waypoints_value, (str, bytes)):
        raise ValueError("course waypoints must be a list")
    return {
        "version": version,
        "goal": _finite_vector(goal_value, 3, "course goal"),
        "waypoints": [_finite_vector(value, 3, "course waypoint") for value in waypoints_value],
        "obstacles": [normalize_obstacle(value, index) for index, value in enumerate(obstacles_value)],
    }


def course_from_tuples(obstacles: Iterable[Sequence[Any]], goal: Sequence[float] = DEFAULT_GOAL) -> Dict[str, Any]:
    """Convert the legacy ``(x, y, z, dimensions)`` course format."""

    records = []
    for index, values in enumerate(obstacles):
        try:
            x, y, z, dimensions = values
        except (TypeError, ValueError) as error:
            raise ValueError("legacy obstacle {} must be (x, y, z, dimensions)".format(index)) from error
        records.append({
            "id": "obstacle_{}".format(index),
            "shape": "box",
            "center": [x, y, z],
            "dimensions": dimensions,
        })
    return normalize_course({"goal": goal, "obstacles": records})


def load_course(path: str | Path) -> Dict[str, Any]:
    """Load and validate an editor-generated JSON course."""

    with Path(path).open("r", encoding="utf-8") as source:
        return normalize_course(json.load(source))


def save_course(path: str | Path, course: Any) -> Dict[str, Any]:
    """Validate and write a course, returning the normalized data."""

    normalized = normalize_course(course)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        json.dump(normalized, output, indent=2)
        output.write("\n")
    return normalized
