#!/usr/bin/env python3
"""
imu_from_GTodometry.py

Generate synthetic IMU from odometry:
  odom lines: t x y z qw qx qy qz
  output IMU: t ax ay az gx gy gz

acc_body is specific force in body frame: f_b = R^T (a_w - g_w)
omega_body is body angular velocity (rad/s) in body frame, consistent with:
  rot_new = rot * Exp(omega_body * dt)

  USAGE:

python3 /home/sgarimella34/multi-robot-coordination/Cosys-AirSim/PythonClient/hero/data_collection/imu_from_GTodometry.py ./odom.txt ./synthetic_imu.txt --imu-rate 200.0

"""

import argparse
import os
from typing import Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline


# =========================
# USER FRAME SETTINGS
# =========================
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
    # Estimate endpoint velocities from odom to avoid vertical bias from "natural" spline.
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

    # ---- Orientation spline ----
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


def save_imu(ts: np.ndarray, acc: np.ndarray, omega: np.ndarray, out_file: str):
    with open(out_file, "w") as f:
        for t, a, w in zip(ts, acc, omega):
            f.write(
                f"{t:.6f} "
                f"{a[0]:.6f} {a[1]:.6f} {a[2]:.6f} "
                f"{w[0]:.6f} {w[1]:.6f} {w[2]:.6f}\n"
            )


def main():
    p = argparse.ArgumentParser(description="Generate synthetic IMU by differentiating odometry")
    p.add_argument("odom_file", help="Path to odom.txt (t x y z qw qx qy qz)")
    p.add_argument("imu_out", help="Output IMU txt (t ax ay az gx gy gz)")
    p.add_argument("--imu-rate", type=float, default=None,
                   help="Output IMU frequency in Hz; if unset, uses odom timestamps")
    p.add_argument("--add-noise", action="store_true", help="Add white noise to accel/gyro")
    p.add_argument("--accel-noise-density", type=float, default=0.02,
                   help="Accel noise density [m/s^2/√Hz]")
    p.add_argument("--gyro-noise-density", type=float, default=0.001,
                   help="Gyro noise density [rad/s/√Hz]")
    args = p.parse_args()

    ts_odom, pos, quat_xyzw = load_odom(args.odom_file)

    if args.imu_rate:
        t0, t1 = ts_odom[0], ts_odom[-1]
        dt = 1.0 / float(args.imu_rate)
        ts_out = np.arange(t0, t1 + 0.5 * dt, dt)
    else:
        ts_out = ts_odom

    acc_body, omega_body = compute_synthetic_imu(
        ts_odom, pos, quat_xyzw, ts_out,
        add_noise=args.add_noise,
        accel_noise_density=args.accel_noise_density,
        gyro_noise_density=args.gyro_noise_density,
    )

    out_dir = os.path.dirname(args.imu_out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    save_imu(ts_out, acc_body, omega_body, args.imu_out)
    print(f"Written synthetic IMU ({len(ts_out)} samples) to {args.imu_out}")


if __name__ == "__main__":
    main()
