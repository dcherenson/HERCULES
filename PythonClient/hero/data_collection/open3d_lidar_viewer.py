#!/usr/bin/env python3
"""
open3d_lidar_viewer.py

Plays LiDAR .npy frames from a directory in chronological order, where the
filename (without extension) is the timestamp (seconds).

This version avoids Window.set_on_tick() by running a background playback loop
and posting updates onto the GUI thread.

Requirements:
  pip install open3d==0.18.0 numpy

How to use:
  1) Edit LIDAR_PATH and (optionally) START_TIMESTAMP_S below.
  2) Run:
       python3 open3d_lidar_viewer.py

Keys:
  Space : pause/resume playback
  D     : next frame (also pauses)
  A     : previous frame (also pauses)
  Q     : quit

Notes:
  - Point size is controlled by POINT_SIZE (increase to make points larger).
  - The current .npy filename is shown in the top-left label and in the window title.
"""

from __future__ import annotations

import glob
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


# ============================================================
# USER SETTINGS (edit these only)
# ============================================================
LIDAR_PATH = "/media/sgarimella34/SSD2/raw_data_hercules/ausenv_lidarfix_TEST1_2ugvuav/Drone1/"

# Start playing from the first frame whose timestamp >= this value (seconds).
# Set to None to start from the beginning.
START_TIMESTAMP_S: Optional[float] = 0.0  # e.g. 1700000123.5

FPS = 10.0
LOOP = True

POINT_SIZE = 3.0
FRAME_SIZE_M = 1.0

VOXEL_M = 0.0
MIN_RANGE_M = None
MAX_RANGE_M = None

BACKGROUND = [0.0, 0.0, 0.0, 1.0]  # RGBA
# ============================================================


def _parse_timestamp_from_stem(stem: str) -> Tuple[int, float, str]:
    """
    Sorting key for filenames:
      - Prefer numeric timestamp from full stem, else first numeric substring, else lexicographic.
    Returns (kind, ts, stem) where kind=0 means numeric timestamp is available.
    """
    s = stem.strip()

    try:
        return (0, float(int(s)), stem)
    except Exception:
        pass
    try:
        return (0, float(s), stem)
    except Exception:
        pass

    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        try:
            return (0, float(m.group(1)), stem)
        except Exception:
            pass

    return (1, 0.0, stem)


def _timestamp_from_filename(path: str) -> Optional[float]:
    kind, ts, _ = _parse_timestamp_from_stem(Path(path).stem)
    return ts if kind == 0 else None


def find_npy_files(path: str) -> List[str]:
    p = Path(path).expanduser()

    if p.is_file() and p.suffix == ".npy":
        return [str(p)]

    candidates: List[str] = []
    if p.is_dir():
        if p.name == "lidar":
            candidates = glob.glob(str(p / "*.npy"))
        else:
            candidates = glob.glob(str(p / "**" / "lidar" / "*.npy"), recursive=True)
            if not candidates:
                candidates = glob.glob(str(p / "**" / "*.npy"), recursive=True)

    files: List[str] = []
    for f in candidates:
        try:
            arr = np.load(f, mmap_mode="r")
            if arr.ndim == 2 and arr.shape[1] == 3:
                files.append(f)
        except Exception:
            continue

    files.sort(key=lambda fp: _parse_timestamp_from_stem(Path(fp).stem))
    return files


def start_index_from_timestamp(files: List[str], start_ts_s: Optional[float]) -> int:
    if not files or start_ts_s is None:
        return 0

    target = float(start_ts_s)

    # If everything parseable, do binary search.
    ts_list: List[float] = []
    for f in files:
        ts = _timestamp_from_filename(f)
        if ts is None:
            ts_list = []
            break
        ts_list.append(ts)

    if not ts_list:
        # Linear fallback
        for i, f in enumerate(files):
            ts = _timestamp_from_filename(f)
            if ts is not None and ts >= target:
                return i
        return 0

    lo, hi = 0, len(ts_list) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if ts_list[mid] >= target:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1

    return best if best is not None else (len(files) - 1)


def build_point_cloud(
    pts: np.ndarray,
    voxel_m: Optional[float] = None,
    min_range_m: Optional[float] = None,
    max_range_m: Optional[float] = None,
) -> o3d.geometry.PointCloud:
    pts = np.asarray(pts, dtype=np.float64)

    mask = np.isfinite(pts).all(axis=1)

    ranges = np.linalg.norm(pts, axis=1)
    if min_range_m is not None:
        mask &= ranges >= float(min_range_m)
    if max_range_m is not None:
        mask &= ranges <= float(max_range_m)

    pts = pts[mask]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    if voxel_m is not None and voxel_m > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_m))

    # Simple Z-based coloring
    if len(pcd.points) > 0:
        z = np.asarray(pcd.points)[:, 2]
        if z.size > 100:
            z_min, z_max = np.percentile(z, [5, 95])
        else:
            z_min, z_max = float(z.min()), float(z.max())
        if z_max <= z_min:
            z_max = z_min + 1e-6
        t = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
        colors = np.stack([t, 1.0 - t, 0.5 * np.ones_like(t)], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def _key_equals(event_key, *candidates) -> bool:
    """
    Some Open3D builds use gui.KeyName enums; others may deliver int keycodes.
    This helper checks both styles safely.
    """
    for c in candidates:
        try:
            if event_key == c:
                return True
        except Exception:
            pass
    return False


class LidarPlayerApp:
    def __init__(self, files: List[str], start_idx: int = 0):
        if not files:
            raise SystemExit("No LiDAR .npy files found.")

        self.files = files
        self.idx = int(max(0, min(start_idx, len(files) - 1)))
        self.dt = 1.0 / max(1e-6, float(FPS))

        self._running = True
        self._paused = False

        app = gui.Application.instance
        app.initialize()

        title = f"Open3D LiDAR Player [{Path(self.files[self.idx]).name}]"
        self.window = app.create_window(title, 1280, 800)

        # Widgets
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background(BACKGROUND)

        self.label = gui.Label("")
        self._update_label_and_title()

        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        self.window.set_on_key(self._on_key)

        self.window.add_child(self.label)
        self.window.add_child(self.scene_widget)

        # Materials
        self.pcd_mat = rendering.MaterialRecord()
        self.pcd_mat.shader = "defaultUnlit"
        self.pcd_mat.point_size = float(POINT_SIZE)

        self.frame_mat = rendering.MaterialRecord()
        self.frame_mat.shader = "defaultLit"

        # Coordinate frame
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(FRAME_SIZE_M))
        self.scene_widget.scene.add_geometry("axes", self.frame, self.frame_mat)

        # First frame (at start index)
        self._load_and_show(self.idx, reset_camera=True)

        # Start playback loop in background thread
        gui.Application.instance.run_in_thread(self._player_loop)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        em = int(self.window.theme.font_size)

        label_h = int(1.6 * em)
        self.label.frame = gui.Rect(r.x + 8, r.y + 6, r.width - 16, label_h)
        self.scene_widget.frame = gui.Rect(r.x, r.y + label_h + 8, r.width, r.height - label_h - 8)

    def _on_close(self):
        self._running = False
        return True

    def _update_label_and_title(self):
        name = Path(self.files[self.idx]).name
        self.label.text = f"[{self.idx + 1}/{len(self.files)}] {name}"
        self.window.title = f"Open3D LiDAR Player [{name}]"

    def _load_and_show(self, new_idx: int, reset_camera: bool = False):
        if not self._running:
            return

        self.idx = int(new_idx) % len(self.files)

        pts = np.load(self.files[self.idx])
        pcd = build_point_cloud(
            pts,
            voxel_m=VOXEL_M,
            min_range_m=MIN_RANGE_M,
            max_range_m=MAX_RANGE_M,
        )

        if self.scene_widget.scene.has_geometry("pcd"):
            self.scene_widget.scene.remove_geometry("pcd")
        self.scene_widget.scene.add_geometry("pcd", pcd, self.pcd_mat)

        self._update_label_and_title()

        if reset_camera:
            bbox = pcd.get_axis_aligned_bounding_box()
            if bbox.is_empty():
                bbox = self.frame.get_axis_aligned_bounding_box()
            self.scene_widget.setup_camera(60.0, bbox, bbox.get_center())

    def _step(self, delta: int):
        if not self.files:
            return
        self._load_and_show(self.idx + delta, reset_camera=False)

    def _on_key(self, event: gui.KeyEvent) -> bool:
        """
        IMPORTANT: Must return a Python bool in some Open3D builds.
        Returning EventCallbackResult can crash (pybind cast error + heap corruption).
        """
        if event.type != gui.KeyEvent.Type.DOWN:
            return False

        k = event.key

        # Space: pause/resume
        if _key_equals(k, gui.KeyName.SPACE, ord(" ")):
            self._paused = not self._paused
            return True

        # D: next frame (and pause)
        if _key_equals(k, gui.KeyName.D, ord("D"), ord("d")):
            self._paused = True
            self._step(+1)
            return True

        # A: previous frame (and pause)
        if _key_equals(k, gui.KeyName.A, ord("A"), ord("a")):
            self._paused = True
            self._step(-1)
            return True

        # Q: quit
        if _key_equals(k, gui.KeyName.Q, ord("Q"), ord("q")):
            self._running = False
            self.window.close()
            return True

        return False

    def _player_loop(self):
        # Background thread: sleep, then post UI updates onto the main thread.
        while self._running:
            time.sleep(self.dt)

            if not self._running:
                break
            if self._paused:
                continue

            next_idx = self.idx + 1
            if next_idx >= len(self.files):
                if LOOP:
                    next_idx = 0
                else:
                    self._running = False
                    gui.Application.instance.post_to_main_thread(self.window, self.window.close)
                    break

            def _ui_update():
                self._load_and_show(next_idx, reset_camera=False)

            gui.Application.instance.post_to_main_thread(self.window, _ui_update)

    def run(self):
        gui.Application.instance.run()


def main():
    files = find_npy_files(LIDAR_PATH)
    if not files:
        raise SystemExit(
            "No LiDAR .npy files found.\n"
            "Set LIDAR_PATH to a lidar folder, a vehicle folder, or a dataset root that contains **/lidar/*.npy."
        )

    start_idx = start_index_from_timestamp(files, START_TIMESTAMP_S)
    app = LidarPlayerApp(files, start_idx=start_idx)
    app.run()


if __name__ == "__main__":
    main()
