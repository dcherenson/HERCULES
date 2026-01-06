#!/usr/bin/env python3
"""
recover_odom_from_imu.py

Recover odometry (t, x y z qw qx qy qz) from IMU:
    t, ax, ay, az, gx, gy, gz

Critical: you MUST set the initial attitude. For your synthetic pipeline, the
best option is to initialize from the first pose in the ground-truth odom file
that you used to generate the synthetic IMU.

This script assumes your IMU is:
  - acc_body is specific force in body frame (a_world - g, rotated into body)
  - omega_body is body angular velocity (rad/s)
and that you want a world frame consistent with your odom output.

No fancy alignment is done beyond the provided initialization.
"""

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

# -------------------------
# USER SETTINGS (EDIT ME)
# -------------------------

IMU_PATH = "/home/sgarimella34/Documents/raw_data_hercules/Drone2/synthetic_imu.txt"
OUT_ODOM_PATH = "/home/sgarimella34/Documents/raw_data_hercules/Drone2/recovered_odom.txt"

# If True, we initialize from the first pose in this odom file:
INIT_FROM_GT_ODOM = True
GT_ODOM_INIT_PATH = "/home/sgarimella34/Documents/raw_data_hercules/Drone2/odom.txt"

# If INIT_FROM_GT_ODOM is False, use these:
INIT_POS = np.array([0.0, 0.0, 0.0])          # x y z
INIT_VEL = np.array([0.0, 0.0, 0.0])          # vx vy vz
INIT_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])  # qw qx qy qz

# Gravity in your chosen world frame.
# IMPORTANT: This must match the frame used in the IMU synthesis.
# If your odom positions are NED (z down), then gravity is +Z:
GRAVITY_WORLD = np.array([0.0, 0.0, 9.80665])

# Safety: drop / clamp weird time steps
MAX_DT = 0.1   # seconds, tune if needed (for example if IMU is 200 Hz, dt ~ 0.005)

# -------------------------
# IO
# -------------------------

def load_imu(path: str):
    data = np.loadtxt(path)
    t = data[:, 0]
    acc_body = data[:, 1:4]
    omega_body = data[:, 4:7]
    return t, acc_body, omega_body

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

def load_first_pose_from_odom(odom_path: str):
    """
    Odom format:
      optional first line: integer count
      then: t x y z qw qx qy qz
    Returns: (pos0, quat_wxyz0)
    """
    p = Path(odom_path)
    lines = p.read_text(errors="ignore").splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        raise ValueError(f"Empty odom file: {odom_path}")

    if _looks_like_int_only_line(lines[0]):
        lines = lines[1:]
    if not lines:
        raise ValueError(f"No pose lines in odom file: {odom_path}")

    parts = lines[0].split()
    if len(parts) < 8:
        raise ValueError(f"First pose line has too few columns: {lines[0]}")

    x, y, z = map(float, parts[1:4])
    qw, qx, qy, qz = map(float, parts[4:8])
    return np.array([x, y, z], dtype=float), np.array([qw, qx, qy, qz], dtype=float)

# -------------------------
# INTEGRATION
# -------------------------

def integrate_trapezoid(t, acc_body, omega_body, pos0, vel0, quat_wxyz0):
    n = len(t)
    if n < 2:
        raise ValueError("IMU file too short")

    pos = np.zeros((n, 3), dtype=float)
    vel = np.zeros((n, 3), dtype=float)
    quats = np.zeros((n, 4), dtype=float)

    pos[0] = pos0
    vel[0] = vel0

    # Initialize orientation from given quaternion
    qw, qx, qy, qz = quat_wxyz0
    rot = Rotation.from_quat([qx, qy, qz, qw])  # SciPy expects [x y z w]
    q0_xyzw = rot.as_quat()
    q0_xyzw /= np.linalg.norm(q0_xyzw)
    quats[0] = [q0_xyzw[3], q0_xyzw[0], q0_xyzw[1], q0_xyzw[2]]

    dt = np.diff(t, prepend=t[0])

    for i in range(1, n):
        dti = float(dt[i])

        # Handle bad timestamps that can cause "shoot off into space"
        if not np.isfinite(dti) or dti <= 0.0:
            # Just copy state forward
            pos[i] = pos[i - 1]
            vel[i] = vel[i - 1]
            quats[i] = quats[i - 1]
            continue

        if dti > MAX_DT:
            # Clamp large gaps to avoid huge integration jumps
            dti = MAX_DT

        # 1) orientation update: assumes omega_body is in body frame
        delta_q = Rotation.from_rotvec(omega_body[i - 1] * dti)
        rot_new = rot * delta_q

        # 2) accel: body specific force -> world accel by rotating and adding gravity
        a_prev = rot.apply(acc_body[i - 1]) + GRAVITY_WORLD
        a_curr = rot_new.apply(acc_body[i]) + GRAVITY_WORLD

        # 3) trapezoidal integrate velocity
        vel[i] = vel[i - 1] + 0.5 * (a_prev + a_curr) * dti

        # 4) trapezoidal integrate position
        pos[i] = pos[i - 1] + 0.5 * (vel[i - 1] + vel[i]) * dti

        # 5) record quaternion [w x y z]
        q_xyzw = rot_new.as_quat()
        q_xyzw /= np.linalg.norm(q_xyzw)
        quats[i] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

        rot = rot_new

    return pos, quats

def save_odom(t, pos, quats, path: str):
    with open(path, "w") as f:
        for ti, p, q in zip(t, pos, quats):
            f.write(
                f"{ti:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
            )

def main():
    t, acc_body, omega_body = load_imu(IMU_PATH)

    if INIT_FROM_GT_ODOM:
        pos0, quat0 = load_first_pose_from_odom(GT_ODOM_INIT_PATH)
        vel0 = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        pos0 = INIT_POS.astype(float)
        vel0 = INIT_VEL.astype(float)
        quat0 = INIT_QUAT_WXYZ.astype(float)

    pos, quats = integrate_trapezoid(t, acc_body, omega_body, pos0, vel0, quat0)
    save_odom(t, pos, quats, OUT_ODOM_PATH)
    print(f"Wrote recovered odom to: {OUT_ODOM_PATH}")

if __name__ == "__main__":
    main()
