#!/usr/bin/env python3
"""
visualize_imu_odom.py

Plots 3D trajectories from one or two TXT files in this format:

Line 1: integer sample count (optional, will be ignored if present)
Then each line:
  t  x  y  z  qw  qx  qy  qz

No CLI args: set parameters in USER PARAMETERS below.

To plot only one:
- Set the other filename to "" or None.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


# =========================
# USER PARAMETERS (EDIT ME)
# =========================

DATA_DIR = Path("/home/sgarimella34/Documents/raw_data_hercules/Husky2")

# Set to "" or None to disable that stream.
IMU_FILENAME: Optional[str] = "recovered_odom.txt"
# IMU_FILENAME: Optional[str] = ""
GT_FILENAME: Optional[str] = "odom.txt"  
# GT_FILENAME: Optional[str] = ""

PLOT_TITLE = "Trajectory (3D) with Orientation"

# Labels and line styles
IMU_LABEL = "Recovered / IMU Odom"
GT_LABEL = "Ground Truth Odom"
IMU_LINESTYLE = "-"
GT_LINESTYLE = "-"
IMU_LINEWIDTH = 2.5
GT_LINEWIDTH = 2.0

# Colors (requested)
IMU_COLOR = "orange"
GT_COLOR = "blue"

# Orientation triads
DRAW_ORIENTATION = True
TRIAD_EVERY_N = 5000
TRIAD_SCALE = 3.0
TRIAD_LINEWIDTH = 1.0
TRIAD_ALPHA = 0.35  # lower so it doesn't overpower the trajectory

# View
ELEVATION_DEG = 25
AZIMUTH_DEG = -65

# Output
SAVE_FIGURE = False
OUT_PNG = DATA_DIR / "trajectory_3d.png"
SHOW_PLOT = True

# Debug prints
PRINT_DEBUG = True


# =========================
# LOADING
# =========================

def _looks_like_int_only_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if any(c.isspace() for c in s):
        return False
    if "." in s or "e" in s.lower():
        return False
    try:
        int(s)
        return True
    except ValueError:
        return False


def load_odom_like_txt(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    lines = path.read_text(errors="ignore").splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]

    if not lines:
        raise ValueError(f"Empty file: {path}")

    # If first line is a single integer count, drop it.
    if _looks_like_int_only_line(lines[0]):
        lines = lines[1:]

    if not lines:
        raise ValueError(f"No data lines after header/count in: {path}")

    rows: List[List[float]] = []
    for ln in lines:
        parts = [p for p in ln.replace(",", " ").split() if p]
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue

    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 3:
        raise ValueError(f"Unexpected data shape {data.shape} in: {path}")

    return data


def extract_xyz_quat_wxyz(data: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Supports:
      [t, x, y, z, qw, qx, qy, qz]  (8 cols)
      [x, y, z, qw, qx, qy, qz]     (7 cols)
      [t, x, y, z]                  (4 cols, no quat)
      [x, y, z]                     (3 cols, no quat)
    Returns:
      pos: Nx3
      quat_wxyz: Nx4 or None
    """
    M = data.shape[1]

    if M >= 8:
        pos = data[:, 1:4]
        quat = data[:, 4:8]  # w x y z
        return pos, quat

    if M == 7:
        pos = data[:, 0:3]
        quat = data[:, 3:7]  # w x y z
        return pos, quat

    if M >= 4:
        pos = data[:, 1:4]
        return pos, None

    pos = data[:, 0:3]
    return pos, None


# =========================
# ORIENTATION
# =========================

def quat_wxyz_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n <= 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
            [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )


def _plot_triad_rgb(ax, p: np.ndarray, R: np.ndarray, scale: float):
    """
    Explicit triad colors so you never get the default blue quiver problem:
      X axis: red
      Y axis: green
      Z axis: blue
    """
    ex = R @ np.array([1.0, 0.0, 0.0])
    ey = R @ np.array([0.0, 1.0, 0.0])
    ez = R @ np.array([0.0, 0.0, 1.0])

    ax.quiver(
        p[0], p[1], p[2], ex[0], ex[1], ex[2],
        length=scale, normalize=True, linewidth=TRIAD_LINEWIDTH, alpha=TRIAD_ALPHA,
        color="red",
    )
    ax.quiver(
        p[0], p[1], p[2], ey[0], ey[1], ey[2],
        length=scale, normalize=True, linewidth=TRIAD_LINEWIDTH, alpha=TRIAD_ALPHA,
        color="green",
    )
    ax.quiver(
        p[0], p[1], p[2], ez[0], ez[1], ez[2],
        length=scale, normalize=True, linewidth=TRIAD_LINEWIDTH, alpha=TRIAD_ALPHA,
        color="blue",
    )


def _set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# =========================
# MAIN
# =========================

def _resolve_path(filename: Optional[str]) -> Optional[Path]:
    if filename is None:
        return None
    if isinstance(filename, str) and filename.strip() == "":
        return None
    return (DATA_DIR / filename).resolve()


def main():
    imu_path = _resolve_path(IMU_FILENAME)
    gt_path = _resolve_path(GT_FILENAME)

    if PRINT_DEBUG:
        print(f"DATA_DIR: {DATA_DIR}")
        print(f"IMU_PATH: {imu_path}")
        print(f"GT_PATH:  {gt_path} (disabled if None)")

    if imu_path is None and gt_path is None:
        raise ValueError("Both IMU_FILENAME and GT_FILENAME are empty/None. Set at least one.")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    plotted_any = False

    if imu_path is not None:
        imu_data = load_odom_like_txt(imu_path)
        imu_pos, imu_quat = extract_xyz_quat_wxyz(imu_data)

        # Force orange line
        ax.plot(
            imu_pos[:, 0], imu_pos[:, 1], imu_pos[:, 2],
            linestyle=IMU_LINESTYLE, linewidth=IMU_LINEWIDTH,
            color=IMU_COLOR, label=IMU_LABEL
        )
        plotted_any = True

        if DRAW_ORIENTATION and (imu_quat is not None):
            for i in range(0, imu_pos.shape[0], max(1, TRIAD_EVERY_N)):
                qw, qx, qy, qz = imu_quat[i]
                R = quat_wxyz_to_R(float(qw), float(qx), float(qy), float(qz))
                _plot_triad_rgb(ax, imu_pos[i], R, TRIAD_SCALE)

    if gt_path is not None:
        gt_data = load_odom_like_txt(gt_path)
        gt_pos, gt_quat = extract_xyz_quat_wxyz(gt_data)

        # Force blue line
        ax.plot(
            gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
            linestyle=GT_LINESTYLE, linewidth=GT_LINEWIDTH,
            color=GT_COLOR, label=GT_LABEL
        )
        plotted_any = True

        if DRAW_ORIENTATION and (gt_quat is not None):
            for i in range(0, gt_pos.shape[0], max(1, TRIAD_EVERY_N)):
                qw, qx, qy, qz = gt_quat[i]
                R = quat_wxyz_to_R(float(qw), float(qx), float(qy), float(qz))
                _plot_triad_rgb(ax, gt_pos[i], R, TRIAD_SCALE)

    if not plotted_any:
        raise RuntimeError("Nothing was plotted (check file paths).")

    ax.set_title(PLOT_TITLE)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)
    ax.view_init(elev=ELEVATION_DEG, azim=AZIMUTH_DEG)
    ax.legend(loc="upper right")
    _set_axes_equal(ax)

    if SAVE_FIGURE:
        fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")

    if SHOW_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
