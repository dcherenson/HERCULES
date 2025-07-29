#!/usr/bin/env python3
"""
NOTE!!! Run inside herculesvenv python3 venv with numpy<2.0
imu_from_GTodometry.py

Given an odometry file (t px py pz qw qx qy qz), fit splines to
position and orientation, compute:

  • linear acceleration = second derivative of position  
  • angular velocity   = first derivative of orientation  

Optionally add white noise (Kalibr-style), and write:

  t ax ay az gx gy gz

to a text file with the same convention as the real IMU.
"""

import argparse
import os

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

def load_odom(odom_file):
    """Load odom.txt → (ts, positions, quaternions)."""
    ts, pos, ori = [], [], []
    with open(odom_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            t = float(parts[0])
            ts.append(t)
            x, y, z = map(float, parts[1:4])
            pos.append([x, y, z])
            qw, qx, qy, qz = map(float, parts[4:8])
            # SciPy expects [x,y,z,w]
            ori.append([qx, qy, qz, qw])
    return np.array(ts), np.array(pos), np.array(ori)

def compute_synthetic_imu(ts_odom, pos, ori, ts_out,
                          add_noise,
                          accel_noise_density,
                          gyro_noise_density):
    """
    1) Fit CubicSpline on ts_odom→pos → world-frame accel (natural BC)
    2) Subtract gravity vector [0,0,-9.80665] m/s² → specific force
    3) Rotate into body frame using orientation spline
    4) Compute angular velocity via quaternion spline derivative
    5) Optionally add white noise
    """
    # a) world-frame acceleration
    pos_spline = CubicSpline(ts_odom, pos, axis=0, bc_type='natural')
    acc_world = pos_spline(ts_out, 2)

    # b) gravity compensation (ENU convention: z up)
    g = np.array([0.0, 0.0, 9.80665])
    spec_force_world = acc_world - g

    # c) rotate into body frame
    rots = Rotation.from_quat(ori)
    rot_spline = RotationSpline(ts_odom, rots)
    rots_out = rot_spline(ts_out)
    acc_body = rots_out.inv().apply(spec_force_world)

    # d) body-frame angular velocity
    omega_body = rot_spline(ts_out, 1)

    # e) optional white noise
    if add_noise:
        dt = np.mean(np.diff(ts_out))
        sigma_a = accel_noise_density * np.sqrt(1.0 / dt)
        sigma_g = gyro_noise_density * np.sqrt(1.0 / dt)
        acc_body  += np.random.normal(scale=sigma_a, size=acc_body.shape)
        omega_body += np.random.normal(scale=sigma_g, size=omega_body.shape)

    return acc_body, omega_body

def save_imu(ts, acc, omega, out_file):
    """Write lines: t ax ay az gx gy gz"""
    with open(out_file, 'w') as f:
        for t, a, w in zip(ts, acc, omega):
            f.write(f"{t:.6f} "
                    f"{a[0]:.6f} {a[1]:.6f} {a[2]:.6f} "
                    f"{w[0]:.6f} {w[1]:.6f} {w[2]:.6f}\n")

def main():
    p = argparse.ArgumentParser(
        description="Generate synthetic IMU by differentiating odometry"
    )
    p.add_argument("odom_file",
                   help="Path to odom.txt (t px py pz qw qx qy qz)")
    p.add_argument("imu_out",
                   help="Output IMU txt (t ax ay az gx gy gz)")
    p.add_argument("--imu-rate", type=float, default=None,
                   help="Output IMU frequency in Hz; if unset, uses odom timestamps")
    p.add_argument("--add-noise", action="store_true",
                   help="Add white noise to accel/gyro")
    p.add_argument("--accel-noise-density", type=float, default=0.02,
                   help="Accel noise density [m/s^2/√Hz]")
    p.add_argument("--gyro-noise-density", type=float, default=0.001,
                   help="Gyro noise density [rad/s/√Hz]")
    args = p.parse_args()

    # load odometry
    ts_odom, pos, ori = load_odom(args.odom_file)

    # timestamps
    if args.imu_rate:
        t0, t1 = ts_odom[0], ts_odom[-1]
        dt = 1.0 / args.imu_rate
        ts_out = np.arange(t0, t1 + dt*0.5, dt)
    else:
        ts_out = ts_odom

    # compute synthetic IMU
    acc, omega = compute_synthetic_imu(
        ts_odom, pos, ori, ts_out,
        add_noise=args.add_noise,
        accel_noise_density=args.accel_noise_density,
        gyro_noise_density=args.gyro_noise_density
    )

    # Apply 180° rotation about X: (x,y,z) -> (x,-y,-z)
    R = np.array([1.0, -1.0, -1.0])
    # acc   = acc * R
    # omega = omega * R

    # —— INDIVIDUAL SIGN CONTROL FOR ACCELERATIONS —— #
    # Set to True to negate that axis, False to keep original
    invert_x = False
    invert_y = False
    invert_z = False

    if invert_x:
        acc[:, 0] = -acc[:, 0]
    if invert_y:
        acc[:, 1] = -acc[:, 1]
    if invert_z:
        acc[:, 2] = -acc[:, 2]

    # angular velocities unchanged
    omega_fixed = omega

    # ensure output directory exists
    out_dir = os.path.dirname(args.imu_out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # save IMU with per-axis sign control
    save_imu(ts_out, acc, omega_fixed, args.imu_out)
    print(f"Written synthetic IMU ({len(ts_out)} samples) to {args.imu_out}")

if __name__ == "__main__":
    main()
