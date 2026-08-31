#!/usr/bin/env python3
"""Small interactive top-down editor for AirSim NED obstacle courses.

The editor deliberately edits explicit obstacle offsets; it does not plan a
route or alter any controller parameters.  Boxes and spheres can be dragged
in X/Y and their size/height can be adjusted with the keyboard.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

import numpy as np

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(TOOL_DIR)
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from modules.obstacle_course import course_from_tuples, load_course, save_course


class ObstacleCourseEditor:
    """Matplotlib mouse/keyboard editor for a normalized course mapping."""

    def __init__(self, course: Dict[str, Any], output_path: str, box_dimensions: np.ndarray, sphere_radius: float):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        self.plt = plt
        self.Rectangle = Rectangle
        self.Circle = Circle
        self.course = course
        self.output_path = output_path
        self.box_dimensions = np.asarray(box_dimensions, dtype=float)
        self.sphere_radius = float(sphere_radius)
        self.selected: Optional[int] = None
        self.dragging = False
        self.mode = "select"
        self.cursor = np.zeros(2)
        self.figure, self.axis = plt.subplots(figsize=(11, 8))
        self.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw()

    def _set_cursor(self, event: Any) -> bool:
        if event.xdata is None or event.ydata is None:
            return False
        self.cursor[:] = [float(event.xdata), float(event.ydata)]
        return True

    def _hit_test(self, x: float, y: float) -> Optional[int]:
        for index in range(len(self.course["obstacles"]) - 1, -1, -1):
            obstacle = self.course["obstacles"][index]
            center = np.asarray(obstacle["center"], dtype=float)
            if obstacle["shape"] == "sphere":
                if np.linalg.norm(np.asarray([x, y]) - center[:2]) <= float(obstacle["radius"]):
                    return index
            else:
                half = np.asarray(obstacle["dimensions"], dtype=float)[:2] / 2.0
                if abs(x - center[0]) <= half[0] and abs(y - center[1]) <= half[1]:
                    return index
        return None

    def _new_obstacle(self, shape: str, x: float, y: float) -> None:
        index = len(self.course["obstacles"])
        if shape == "sphere":
            obstacle = {
                "id": "obstacle_{}".format(index), "shape": "sphere",
                "center": [x, y, -self.box_dimensions[2] / 2.0], "radius": self.sphere_radius,
            }
        else:
            obstacle = {
                "id": "obstacle_{}".format(index), "shape": "box",
                "center": [x, y, -self.box_dimensions[2] / 2.0],
                "dimensions": self.box_dimensions.tolist(),
            }
        self.course["obstacles"].append(obstacle)
        self.selected = index
        self.mode = "select"

    def on_press(self, event: Any) -> None:
        if event.button != 1 or not self._set_cursor(event):
            return
        if self.mode in {"add_box", "add_sphere"}:
            self._new_obstacle("box" if self.mode == "add_box" else "sphere", *self.cursor)
            self.redraw()
            return
        self.selected = self._hit_test(*self.cursor)
        self.dragging = self.selected is not None
        self.redraw()

    def on_release(self, event: Any) -> None:
        if event.button == 1:
            self.dragging = False

    def on_motion(self, event: Any) -> None:
        if not self.dragging or self.selected is None or not self._set_cursor(event):
            return
        center = self.course["obstacles"][self.selected]["center"]
        center[0], center[1] = float(self.cursor[0]), float(self.cursor[1])
        self.redraw()

    def _change_size(self, amount: float) -> None:
        if self.selected is None:
            return
        obstacle = self.course["obstacles"][self.selected]
        if obstacle["shape"] == "sphere":
            obstacle["radius"] = max(0.25, float(obstacle["radius"]) + amount)
        else:
            dimensions = np.asarray(obstacle["dimensions"], dtype=float)
            dimensions[:2] = np.maximum(0.5, dimensions[:2] + amount)
            obstacle["dimensions"] = dimensions.tolist()

    def on_key(self, event: Any) -> None:
        key = event.key
        if key in {"b", "s"}:
            self.mode = "add_box" if key == "b" else "add_sphere"
        elif key in {"delete", "backspace"} and self.selected is not None:
            del self.course["obstacles"][self.selected]
            self.selected = None
        elif key in {"+", "=", "]"}:
            self._change_size(0.5 if key != "]" else 0.25)
        elif key in {"-", "["}:
            self._change_size(-0.5 if key == "-" else -0.25)
        elif key in {"up", "down", "left", "right"} and self.selected is not None:
            delta = {"up": [0.0, 0.25], "down": [0.0, -0.25], "left": [-0.25, 0.0], "right": [0.25, 0.0]}[key]
            center = self.course["obstacles"][self.selected]["center"]
            center[0] += delta[0]
            center[1] += delta[1]
        elif key in {"u", "n"} and self.selected is not None:
            center = self.course["obstacles"][self.selected]["center"]
            # AirSim NED Z is positive downward: physical up decreases Z.
            center[2] += -0.5 if key == "u" else 0.5
        elif key in {"w", "enter"}:
            self.save()
        elif key == "q":
            self.save()
            self.plt.close(self.figure)
            return
        self.redraw()

    def save(self) -> None:
        save_course(self.output_path, self.course)
        print("Saved obstacle course: {}".format(os.path.abspath(self.output_path)), flush=True)

    def redraw(self) -> None:
        self.axis.clear()
        for index, obstacle in enumerate(self.course["obstacles"]):
            center = np.asarray(obstacle["center"], dtype=float)
            selected = index == self.selected
            if obstacle["shape"] == "sphere":
                patch = self.Circle(center[:2], float(obstacle["radius"]), facecolor="tab:orange", alpha=0.35,
                                    edgecolor="crimson" if selected else "darkorange", linewidth=2.5 if selected else 1.2)
            else:
                dimensions = np.asarray(obstacle["dimensions"], dtype=float)
                patch = self.Rectangle(center[:2] - dimensions[:2] / 2.0, dimensions[0], dimensions[1],
                                       facecolor="tab:blue", alpha=0.30,
                                       edgecolor="crimson" if selected else "navy", linewidth=2.5 if selected else 1.2)
            self.axis.add_patch(patch)
            self.axis.text(center[0], center[1], str(obstacle.get("id", index)), ha="center", va="center", fontsize=8)
        goal = np.asarray(self.course["goal"], dtype=float)
        self.axis.scatter(goal[0], goal[1], marker="*", s=180, color="red", label="goal")
        for waypoint in self.course.get("waypoints", []):
            point = np.asarray(waypoint, dtype=float)
            self.axis.scatter(point[0], point[1], marker="x", s=100, color="purple")
        if self.course["obstacles"]:
            centers = np.asarray([item["center"][:2] for item in self.course["obstacles"]], dtype=float)
            lower, upper = centers.min(axis=0) - 5.0, centers.max(axis=0) + 5.0
            self.axis.set_xlim(lower[0], upper[0])
            self.axis.set_ylim(lower[1], upper[1])
        else:
            self.axis.set_xlim(-5.0, 20.0)
            self.axis.set_ylim(-10.0, 10.0)
        self.axis.set_aspect("equal", adjustable="box")
        self.axis.grid(True, alpha=0.3)
        self.axis.set_xlabel("AirSim NED X (m)")
        self.axis.set_ylabel("AirSim NED Y (m)")
        self.axis.set_title(
            "Obstacle course editor — {} | b: add box, s: add sphere, drag: move, arrows: nudge, "
            "+/-: resize, u/n: altitude, delete, w: save, q: save & quit".format(self.mode)
        )
        self.axis.legend(loc="upper right")
        self.figure.canvas.draw_idle()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="existing course JSON to edit")
    parser.add_argument("--output", default="obstacle_course.json", help="course JSON written by the editor")
    parser.add_argument("--empty", action="store_true", help="start with no obstacles instead of the default FlyingCPP course")
    parser.add_argument("--goal", nargs=3, type=float, metavar=("X", "Y", "Z"), help="override goal in AirSim NED")
    parser.add_argument("--box-dimensions", nargs=3, type=float, default=[2.0, 3.0, 4.0], metavar=("X", "Y", "Z"))
    parser.add_argument("--sphere-radius", type=float, default=1.5)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.input:
        course = load_course(args.input)
    elif args.empty:
        course = course_from_tuples([], goal=args.goal or [16.0, 1.0, -1.0])
    else:
        from orchestrator import BLOCK_COURSE

        course = course_from_tuples(BLOCK_COURSE, goal=args.goal or [16.0, 1.0, -1.0])
    if args.goal:
        course["goal"] = [float(value) for value in args.goal]
    if args.sphere_radius <= 0.0 or np.any(np.asarray(args.box_dimensions) <= 0.0):
        raise SystemExit("default obstacle sizes must be positive")
    editor = ObstacleCourseEditor(course, args.output, np.asarray(args.box_dimensions), args.sphere_radius)
    editor.plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
