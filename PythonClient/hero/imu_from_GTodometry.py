#!/usr/bin/env python3
"""
generate_synthetic_imu.py

Given an odometry file (t px py pz qw qx qy qz), fit splines to
position and orientation, compute:

  • linear acceleration = second derivative of position
  • angular velocity   = first derivative of orientation

Optionally add white noise (Kalibr-style), and write:

  t ax ay az gx gy gz

to a text file with the same convention as the real IMU.


EXAMPLE USAGE
python3 generate_synthetic_imu.py \
  /media/.../raw_data/odom.txt \
  /media/.../raw_data/synthetic_imu.txt \
  --add-noise \
  --accel-noise-density 0.02 \
  --gyro-noise-density 0.001

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
            # positions
            x, y, z = map(float, parts[1:4])
            pos.append([x, y, z])
            # odom.txt gives qw,qx,qy,qz → reorder to x,y,z,w
            qw, qx, qy, qz = map(float, parts[4:8])
            ori.append([qx, qy, qz, qw])
    ts = np.array(ts)
    pos = np.array(pos)
    ori = np.array(ori)
    return ts, pos, ori

def compute_synthetic_imu(ts, pos, ori, add_noise,
                          accel_noise_density,
                          gyro_noise_density):
    """
    Fit cubic splines over ts→pos and ts→ori, then:
      accel(t) = d²pos/dt²
      omega(t) = angular rate from RotationSpline derivative

    Noise densities in [units/√Hz] → per-sample sigma ≈ density*√(1/Δt).
    """
    # position spline and acceleration
    pos_spline = CubicSpline(ts, pos, axis=0)
    acc = pos_spline(ts, 2)  # second derivative

    # orientation spline and angular velocity
    rots = Rotation.from_quat(ori)
    rot_spline = RotationSpline(ts, rots)
    # derivative=1 gives angular velocity [rad/s] in body frame
    omega = rot_spline(ts, 1)

    if add_noise:
        dt = np.mean(np.diff(ts))
        sigma_a = accel_noise_density * np.sqrt(1.0 / dt)
        sigma_g = gyro_noise_density * np.sqrt(1.0 / dt)
        acc   += np.random.normal(scale=sigma_a, size=acc.shape)
        omega += np.random.normal(scale=sigma_g, size=omega.shape)

    return acc, omega

def save_imu(ts, acc, omega, out_file):
    """Write t ax ay az gx gy gz lines to out_file."""
    with open(out_file, 'w') as f:
        for t, a, w in zip(ts, acc, omega):
            f.write(
                f"{t:.6f} "
                f"{a[0]:.6f} {a[1]:.6f} {a[2]:.6f} "
                f"{w[0]:.6f} {w[1]:.6f} {w[2]:.6f}\n"
            )

def main():
    p = argparse.ArgumentParser(
        description="Generate synthetic IMU by differentiating odometry"
    )
    p.add_argument("odom_file",
                   help="Path to odom.txt (t px py pz qw qx qy qz)")
    p.add_argument("imu_out",
                   help="Path to write synthetic IMU txt (t ax ay az gx gy gz)")
    p.add_argument("--add-noise", action="store_true",
                   help="Add white noise to accel/gyro")
    p.add_argument("--accel-noise-density", type=float, default=0.02,
                   help="Accel noise density [m/s^2/√Hz]")
    p.add_argument("--gyro-noise-density", type=float, default=0.001,
                   help="Gyro noise density [rad/s/√Hz]")
    args = p.parse_args()

    ts, pos, ori = load_odom(args.odom_file)
    acc, omega = compute_synthetic_imu(
        ts, pos, ori,
        add_noise=args.add_noise,
        accel_noise_density=args.accel_noise_density,
        gyro_noise_density=args.gyro_noise_density
    )
    os.makedirs(os.path.dirname(args.imu_out), exist_ok=True)
    save_imu(ts, acc, omega, args.imu_out)
    print(f"Written synthetic IMU to {args.imu_out}")

if __name__ == "__main__":
    main()
