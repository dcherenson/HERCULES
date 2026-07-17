#!/usr/bin/env python3
"""
imu_from_GTodometry.py

Generate synthetic IMU from odometry:
  odom lines: t x y z qw qx qy qz
  output IMU: t ax ay az gx gy gz

OPTIONAL (in-code flag):
  Append world-frame orientation quaternion (x y z w) from pose_world_frame.txt:
  output: t ax ay az gx gy gz qx qy qz qw
  and output filename becomes *_9axis.txt

Important behavior:
- accel + gyro are still computed ONLY from odom.txt (same spline + RotationSpline method as before)
- the absolute quaternion appended is interpolated ONLY from pose_world_frame.txt (RotationSpline)
- pose_world_frame.txt is loaded from the SAME directory as the odom.txt you pass in
"""

import argparse
import os
from typing import Tuple, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline


# ============================================================
# IN-CODE SETTINGS
# ============================================================

# If True: append world-frame quaternion (x y z w) from pose_world_frame.txt
#          and write to *_9axis.txt
ADD_WORLD_QUATERNION = True

# The pose file name expected to live next to the odom.txt you pass in
POSE_WORLD_BASENAME = "pose_world_frame.txt"

# Must match your recovery script's GRAVITY_WORLD.
# If your world Z is "down" (NED-like): use +9.80665 in +Z.
# If your world Z is "up" (ENU-like):  use -9.80665 in Z.
WORLD_Z_IS_DOWN = True
G_MAG = 9.80665
GRAVITY_WORLD = np.array([0.0, 0.0, +G_MAG if WORLD_Z_IS_DOWN else -G_MAG], dtype=float)


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


def load_odom(odom_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = []
    with open(odom_file, "r") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                lines.append(ln)

    if not lines:
        raise ValueError(f"Empty odom file: {odom_file}")

    if _looks_like_int_only_line(lines[0]):
        lines = lines[1:]

    ts = []
    pos = []
    quat_xyzw = []

    for ln in lines:
        parts = ln.split()
        if len(parts) < 8:
            continue
        t = float(parts[0])
        x, y, z = map(float, parts[1:4])
        qw, qx, qy, qz = map(float, parts[4:8])

        ts.append(t)
        pos.append([x, y, z])
        quat_xyzw.append([qx, qy, qz, qw])  # SciPy: [x,y,z,w]

    ts = np.asarray(ts, dtype=float)
    pos = np.asarray(pos, dtype=float)
    quat_xyzw = np.asarray(quat_xyzw, dtype=float)

    if ts.size < 3:
        raise ValueError(f"Not enough samples in {odom_file}")

    # Enforce strictly increasing time
    if np.any(np.diff(ts) <= 0):
        keep = np.ones_like(ts, dtype=bool)
        keep[1:] = np.diff(ts) > 0
        ts = ts[keep]
        pos = pos[keep]
        quat_xyzw = quat_xyzw[keep]

    # Normalize quaternions and enforce sign continuity for spline stability
    quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    for i in range(1, quat_xyzw.shape[0]):
        if np.dot(quat_xyzw[i - 1], quat_xyzw[i]) < 0.0:
            quat_xyzw[i] *= -1.0

    return ts, pos, quat_xyzw


def load_pose_world_quat(pose_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads pose_world_frame.txt.
    Expected per valid row: t x y z qw qx qy qz
    Returns: ts_pose, quat_xyzw (SciPy order: [x y z w])
    """
    lines = []
    with open(pose_file, "r") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                lines.append(ln)

    if not lines:
        raise ValueError(f"Empty pose file: {pose_file}")

    if _looks_like_int_only_line(lines[0]):
        lines = lines[1:]

    ts = []
    quat_xyzw = []

    for ln in lines:
        parts = ln.split()
        if len(parts) < 8:
            continue

        t = float(parts[0])
        qw, qx, qy, qz = map(float, parts[4:8])

        ts.append(t)
        quat_xyzw.append([qx, qy, qz, qw])  # convert to SciPy order

    ts = np.asarray(ts, dtype=float)
    quat_xyzw = np.asarray(quat_xyzw, dtype=float)

    if ts.size < 3:
        raise ValueError(f"Not enough usable pose samples in {pose_file}")

    # Enforce strictly increasing time
    if np.any(np.diff(ts) <= 0):
        keep = np.ones_like(ts, dtype=bool)
        keep[1:] = np.diff(ts) > 0
        ts = ts[keep]
        quat_xyzw = quat_xyzw[keep]

    # Normalize + sign continuity (same idea as odom loader)
    quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    for i in range(1, quat_xyzw.shape[0]):
        if np.dot(quat_xyzw[i - 1], quat_xyzw[i]) < 0.0:
            quat_xyzw[i] *= -1.0

    return ts, quat_xyzw


def compute_synthetic_imu(
    ts_odom: np.ndarray,
    pos: np.ndarray,
    quat_xyzw: np.ndarray,
    ts_out: np.ndarray,
    add_noise: bool,
    accel_noise_density: float,
    gyro_noise_density: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # ---- Position spline with CLAMPED endpoint velocities ----
    dt0 = ts_odom[1] - ts_odom[0]
    dt1 = ts_odom[-1] - ts_odom[-2]
    if dt0 <= 0 or dt1 <= 0:
        raise ValueError("Non-increasing timestamps in odom")

    v0 = (pos[1] - pos[0]) / dt0
    v1 = (pos[-1] - pos[-2]) / dt1

    pos_spline = CubicSpline(
        ts_odom,
        pos,
        axis=0,
        bc_type=((1, v0), (1, v1)),  # clamped first-derivative endpoints
    )
    acc_world = pos_spline(ts_out, 2)

    # World specific force
    spec_force_world = acc_world - GRAVITY_WORLD

    # ---- Orientation spline (from odom) ----
    rots_in = Rotation.from_quat(quat_xyzw)
    rot_spline = RotationSpline(ts_odom, rots_in)
    rots_out = rot_spline(ts_out)

    # Accel into body: f_b = R^T f_w
    acc_body = rots_out.inv().apply(spec_force_world)

    # ---- Body angular velocity consistent with rot_new = rot * Exp(omega*dt) ----
    t = ts_out
    dt = np.diff(t)
    omega_body = np.zeros((t.shape[0], 3), dtype=float)

    for i in range(t.shape[0] - 1):
        dti = float(dt[i])
        if not np.isfinite(dti) or dti <= 0.0:
            omega_body[i] = omega_body[i - 1] if i > 0 else 0.0
            continue
        delta = rots_out[i].inv() * rots_out[i + 1]
        omega_body[i] = delta.as_rotvec() / dti

    omega_body[-1] = omega_body[-2]

    # ---- Optional noise (density -> per-sample sigma) ----
    if add_noise:
        dt_mean = float(np.mean(dt[np.isfinite(dt) & (dt > 0)]))
        if dt_mean <= 0.0:
            dt_mean = 1.0 / 200.0

        sigma_a = accel_noise_density / np.sqrt(dt_mean)
        sigma_g = gyro_noise_density / np.sqrt(dt_mean)

        acc_body += np.random.normal(scale=sigma_a, size=acc_body.shape)
        omega_body += np.random.normal(scale=sigma_g, size=omega_body.shape)

    return acc_body, omega_body


def compute_world_quat_at_times(
    ts_pose: np.ndarray,
    quat_pose_xyzw: np.ndarray,
    ts_out: np.ndarray,
) -> np.ndarray:
    """
    Uses RotationSpline (same method as odom orientation) to interpolate
    the world-frame orientation, then returns quaternion as (x y z w).
    """
    rots_in = Rotation.from_quat(quat_pose_xyzw)
    rot_spline = RotationSpline(ts_pose, rots_in)
    rots_out = rot_spline(ts_out)
    return rots_out.as_quat()  # SciPy returns [x y z w]


def _make_9axis_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    if ext == "":
        return base + "_9axis"
    if base.endswith("_9axis"):
        return path
    return base + "_9axis" + ext


def save_imu(
    ts: np.ndarray,
    acc: np.ndarray,
    omega: np.ndarray,
    out_file: str,
    world_quat_xyzw: Optional[np.ndarray] = None,
):
    with open(out_file, "w") as f:
        if world_quat_xyzw is None:
            for t, a, w in zip(ts, acc, omega):
                f.write(
                    f"{t:.6f} "
                    f"{a[0]:.6f} {a[1]:.6f} {a[2]:.6f} "
                    f"{w[0]:.6f} {w[1]:.6f} {w[2]:.6f}\n"
                )
        else:
            if world_quat_xyzw.shape[0] != ts.shape[0]:
                raise ValueError("world_quat_xyzw length does not match ts length")
            for t, a, w, q in zip(ts, acc, omega, world_quat_xyzw):
                f.write(
                    f"{t:.6f} "
                    f"{a[0]:.6f} {a[1]:.6f} {a[2]:.6f} "
                    f"{w[0]:.6f} {w[1]:.6f} {w[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
                )


def main():
    p = argparse.ArgumentParser(description="Generate synthetic IMU by differentiating odometry")
    p.add_argument("odom_file", help="Path to odom.txt (t x y z qw qx qy qz)")
    p.add_argument("imu_out", help="Output IMU txt (t ax ay az gx gy gz)")
    p.add_argument(
        "--imu-rate",
        type=float,
        default=None,
        help="Output IMU frequency in Hz; if unset, uses odom timestamps",
    )
    p.add_argument("--add-noise", action="store_true", help="Add white noise to accel/gyro")
    p.add_argument(
        "--accel-noise-density",
        type=float,
        default=0.02,
        help="Accel noise density [m/s^2/sqrt(Hz)]",
    )
    p.add_argument(
        "--gyro-noise-density",
        type=float,
        default=0.001,
        help="Gyro noise density [rad/s/sqrt(Hz)]",
    )
    args = p.parse_args()

    ts_odom, pos, quat_xyzw = load_odom(args.odom_file)

    if args.imu_rate:
        t0, t1 = ts_odom[0], ts_odom[-1]
        dt = 1.0 / float(args.imu_rate)
        ts_out = np.arange(t0, t1 + 0.5 * dt, dt)
    else:
        ts_out = ts_odom

    acc_body, omega_body = compute_synthetic_imu(
        ts_odom,
        pos,
        quat_xyzw,
        ts_out,
        add_noise=args.add_noise,
        accel_noise_density=args.accel_noise_density,
        gyro_noise_density=args.gyro_noise_density,
    )

    out_path = args.imu_out
    world_quat_xyzw = None

    if ADD_WORLD_QUATERNION:
        # Load pose_world_frame.txt from the SAME directory as the provided odom file
        odom_dir = os.path.dirname(os.path.abspath(args.odom_file))
        pose_path = os.path.join(odom_dir, POSE_WORLD_BASENAME)

        if not os.path.isfile(pose_path):
            raise FileNotFoundError(f"Expected pose file next to odom.txt: {pose_path}")

        ts_pose, quat_pose_xyzw = load_pose_world_quat(pose_path)

        # Interpolate orientation at the SAME ts_out used for IMU
        world_quat_xyzw = compute_world_quat_at_times(ts_pose, quat_pose_xyzw, ts_out)

        out_path = _make_9axis_path(out_path)

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    save_imu(ts_out, acc_body, omega_body, out_path, world_quat_xyzw=world_quat_xyzw)

    if world_quat_xyzw is None:
        print(f"Written synthetic IMU ({len(ts_out)} samples) to {out_path}")
    else:
        print(f"Written synthetic IMU+world_quat ({len(ts_out)} samples) to {out_path}")


if __name__ == "__main__":
    main()
