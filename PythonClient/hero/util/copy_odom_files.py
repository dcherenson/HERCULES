#!/usr/bin/env python3

import os
import shutil

# -----------------------------
# CONFIGURATION (edit if needed)
# -----------------------------

# Source root directory (where the new odom.txt files live)
SRC_ROOT = "/media/sgarimella34/hercules-collect/raw_data_hercules/city_cslam_ugv1uav1_test1/"

# Destination root directory (where you want to copy them to)
DST_ROOT = "/home/sgarimella34/Documents/raw_data_hercules"

# Subdirectories to process
AGENTS = ["Drone1", "Drone2", "Husky1", "Husky2"]

# Filename to copy
FILENAME = "odom.txt"

# -----------------------------
# COPY LOGIC
# -----------------------------

def main():
    for agent in AGENTS:
        src_file = os.path.join(SRC_ROOT, agent, FILENAME)
        dst_file = os.path.join(DST_ROOT, agent, FILENAME)

        if not os.path.isfile(src_file):
            print(f"[WARNING] Source file not found: {src_file}")
            continue

        if not os.path.isdir(os.path.join(DST_ROOT, agent)):
            print(f"[WARNING] Destination directory missing: {os.path.join(DST_ROOT, agent)}")
            continue

        try:
            shutil.copy2(src_file, dst_file)
            print(f"[OK] Copied {src_file} -> {dst_file}")
        except Exception as e:
            print(f"[ERROR] Failed to copy for {agent}: {e}")

if __name__ == "__main__":
    main()
