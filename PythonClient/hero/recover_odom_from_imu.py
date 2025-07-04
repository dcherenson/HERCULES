#!/usr/bin/env python3
"""
imu_to_odometry_corrected.py

Recover NED-frame odometry (t, x y z qw qx qy qz) from AirSim synthetic IMU:
    t, ax, ay, az, gx, gy, gz.

Features:
  • Correct NED gravity sign: add +9.80665 m/s² down
  • Trapezoidal integration (2nd-order) for velocity & position
  • Quaternion updates via exponential map, normalized each step
"""

import sys
import numpy as np
from scipy.spatial.transform import Rotation

# Gravity vector in NED frame: +Z is down
GRAVITY_NED = np.array([0.0, 0.0, 9.80665])

def load_imu(path: str):
    """Load IMU file: columns [t ax ay az gx gy gz]."""
    data = np.loadtxt(path)
    t         = data[:, 0]
    acc_body  = data[:, 1:4]
    omega_body= data[:, 4:7]
    return t, acc_body, omega_body

def integrate_ned_trapezoid(t: np.ndarray,
                             acc_body: np.ndarray,
                             omega_body: np.ndarray):
    """
    Integrate IMU to recover pos & orientation in NED.

    Returns:
      pos   : (N,3) in NED [north, east, down]
      quats : (N,4) [w, x, y, z]
    """
    n = len(t)
    dt = np.diff(t, prepend=t[0])

    pos   = np.zeros((n, 3))
    vel   = np.zeros((n, 3))
    quats = np.zeros((n, 4))

    # start with identity orientation
    rot = Rotation.identity()
    quats[0] = [1.0, 0.0, 0.0, 0.0]

    # Precompute world-frame accelerations from IMU (spec-force)
    # (we'll add gravity back in below)
    for i in range(1, n):
        # 1) update orientation via exponential map
        delta_q = Rotation.from_rotvec(omega_body[i-1] * dt[i])
        rot_new = rot * delta_q

        # 2) rotate body-frame specific force into world, then add gravity
        a_prev = rot.apply(acc_body[i-1]) + GRAVITY_NED
        a_curr = rot_new.apply(acc_body[i]) + GRAVITY_NED

        # 3) integrate velocity (trapezoidal)
        vel[i] = vel[i-1] + 0.5 * (a_prev + a_curr) * dt[i]

        # 4) integrate position (trapezoidal on velocity)
        pos[i] = pos[i-1] + 0.5 * (vel[i-1] + vel[i]) * dt[i]

        # 5) normalize and record quaternion [w, x, y, z]
        q_xyzw = rot_new.as_quat()  # [x,y,z,w]
        q_xyzw /= np.linalg.norm(q_xyzw)
        quats[i] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

        # rotate forward
        rot = rot_new

    return pos, quats

def save_odom(t: np.ndarray,
              pos: np.ndarray,
              quats: np.ndarray,
              path: str):
    """Write odometry file: t x y z qw qx qy qz."""
    with open(path, 'w') as f:
        for ti, p, q in zip(t, pos, quats):
            f.write(f"{ti:.6f} "
                    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

def main():
    if len(sys.argv) != 3:
        print("Usage:\n  python3 imu_to_odometry_corrected.py "
              "synthetic_imu.txt recovered_odom.txt")
        sys.exit(1)

    imu_path   = sys.argv[1]
    odom_out   = sys.argv[2]

    t, acc_body, omega_body = load_imu(imu_path)
    pos, quats = integrate_ned_trapezoid(t, acc_body, omega_body)
    save_odom(t, pos, quats, odom_out)
    print(f"Recovered odometry written to {odom_out}")

if __name__ == "__main__":
    main()
