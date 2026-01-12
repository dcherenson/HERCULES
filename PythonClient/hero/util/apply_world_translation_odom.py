#!/usr/bin/env python3

"""
This code is used for transforming the ground-truth odometry for each robot, which is relative to the robot's starting pose, into the 
world NED frame using the X and Y initial pose from the settings.json file.
"""


import json
from pathlib import Path
from typing import Dict, Optional, Tuple

# -------------------------------
# USER-DEFINED PATHS
# -------------------------------
# Base folder containing Drone1/, Drone2/, Husky1/, Husky2/, settings.json
BASE_DIR = Path("/home/sgarimella34/Documents/raw_data_hercules")
SETTINGS_PATH = BASE_DIR / "settings.json"
# -------------------------------


def load_settings(settings_path: Path) -> Dict[str, Dict[str, float]]:
    with settings_path.open("r") as f:
        settings = json.load(f)
    vehicles = settings.get("Vehicles", {})
    init_poses = {}
    for name, v in vehicles.items():
        X = float(v.get("X", 0.0))
        Y = float(v.get("Y", 0.0))
        init_poses[name] = {"X": X, "Y": Y}
    return init_poses


def parse_line(line: str) -> Optional[Tuple[float, float, float, float, float, float, float, float]]:
    parts = line.strip().split()
    if len(parts) < 8:
        return None
    try:
        return tuple(map(float, parts[:8]))
    except ValueError:
        return None


def format_line(vals: Tuple[float, float, float, float, float, float, float, float]) -> str:
    return " ".join(f"{v:.6f}" for v in vals) + "\n"


def process_robot(robot_dir: Path, init_pose: Dict[str, float]) -> None:
    odom_path = robot_dir / "odom.txt"
    if not odom_path.exists():
        print(f"[WARN] Missing {odom_path}, skipping {robot_dir.name}")
        return

    out_path = robot_dir / "pose_world_frame.txt"

    X_off = init_pose["X"]
    Y_off = init_pose["Y"]

    lines_out = []
    skipped = 0
    total = 0

    with odom_path.open("r") as f:
        for line in f:
            total += 1
            parsed = parse_line(line)
            if parsed is None:
                skipped += 1
                continue
            t, x, y, z, wrot, xrot, yrot, zrot = parsed
            x_w = x + X_off
            y_w = y + Y_off
            lines_out.append(format_line((t, x_w, y_w, z, wrot, xrot, yrot, zrot)))

    with out_path.open("w") as f:
        f.writelines(lines_out)

    print(f"[OK] {robot_dir.name}: wrote {out_path.name} ({len(lines_out)} lines; skipped {skipped} of {total})")


def main():
    if not SETTINGS_PATH.exists():
        print(f"[ERROR] settings.json not found at: {SETTINGS_PATH}")
        return

    init_poses = load_settings(SETTINGS_PATH)
    subdirs = [p for p in BASE_DIR.iterdir() if p.is_dir()]

    for sd in sorted(subdirs):
        if sd.name in init_poses:
            process_robot(sd, init_poses[sd.name])


if __name__ == "__main__":
    main()
