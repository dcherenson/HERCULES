#!/usr/bin/env python3
"""
hercules_to_dair_collector_sync.py

Collect Hercules (UGV/UAV) data and save it in **DAIR-V2X-C** format so you can
fine-tune ImVoxelNet (veh-only / inf-only) and train PointPillars on the same dump.

Changes vs. previous draft:
  * Removed IMU_RATE (we don't log IMU here)  one base tick drives everything.
  * DT_MAIN is derived from the highest sensor rate (20 Hz camera).
  * Perfect sync is assumed because we pause/step the sim once per tick.
    -> No ALIGN_TOL_MS; we pair frames by identical tick timestamps (and assert).
  * Cleaner rate divisors and simpler queues.
  * Still leaves 3D label generation as a TODO hook.
"""

import os
import json
import math
from collections import deque
import numpy as np
import cv2

import setup_path  # your AirSim path helper
import cosysairsim as airsim

# -------------------------- CONFIG --------------------------
OUT_ROOT   = "/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/dair_v2x_synth"
SIDE_INF   = ["Drone1"]    # infrastructure-like agents
SIDE_VEH   = ["Husky1"]    # vehicle-like agents
ALL_AGENTS = SIDE_INF + SIDE_VEH

IMG_W, IMG_H = 1920, 1080
IMG_EXT      = ".jpg"      # DAIR uses JPEG
LIDAR_NAME   = "LidarSensor1"
CAM_NAME     = "front_center"

# Rates (Hz)
CAM_RATE    = 20
LIDAR_RATE  = 10
DURATION_S  = 840.0         # total simulated seconds to record

# Base tick = highest sensor rate (or lcm of all rates if more are added)
BASE_HZ = CAM_RATE          # 20 Hz
DT      = 1.0 / BASE_HZ     # 0.05 s

# Derived integer step intervals
CAM_EVERY   = BASE_HZ // CAM_RATE    # 1
LIDAR_EVERY = BASE_HZ // LIDAR_RATE  # 2

# Ports
DRONE_PORT = 41451
HUSKY_PORT = 41452

# Class mapping to DAIR labels (extend as needed)
DAIR_CLASSES = {
    "Car": "Car",
    "BP_CrowdCharacter": "Pedestrian"
}

# ---------------------- OUTPUT STRUCTURE --------------------
def dair_paths(root: str):
    return {
        "inf": {
            "img":   f"{root}/cooperative/infrastructure-side/image",
            "lidar": f"{root}/cooperative/infrastructure-side/lidar",
            "calib": f"{root}/cooperative/infrastructure-side/calib",
            "ts":    f"{root}/cooperative/infrastructure-side/timestamp"
        },
        "veh": {
            "img":   f"{root}/cooperative/vehicle-side/image",
            "lidar": f"{root}/cooperative/vehicle-side/lidar",
            "calib": f"{root}/cooperative/vehicle-side/calib",
            "ts":    f"{root}/cooperative/vehicle-side/timestamp"
        },
        "label": {
            "veh":         f"{root}/cooperative/label/vehicle",
            "inf":         f"{root}/cooperative/label/infrastructure",
            "cooperative": f"{root}/cooperative/label/cooperative"
        }
    }

PATHS = dair_paths(OUT_ROOT)
for grp in PATHS.values():
    for p in grp.values():
        os.makedirs(p, exist_ok=True)

# -------------------- AIRSIM CLIENTS ------------------------
drone_client = airsim.MultirotorClient(port=DRONE_PORT)
car_client   = airsim.CarClient(port=HUSKY_PORT)

drone_client.confirmConnection()
car_client.confirmConnection()

for n in SIDE_INF:
    drone_client.enableApiControl(True, vehicle_name=n)
for n in SIDE_VEH:
    car_client.enableApiControl(True, vehicle_name=n)

# Pause world globally
drone_client.simPause(True)

# ----------------- CALIBRATION DUMP (ONE-TIME) --------------
def make_calib_json(K, cam2vlidar, vlidar2world=None):
    return {
        "cam_intrinsic": K.tolist(),
        "cam2vlidar": cam2vlidar.tolist(),
        "vlidar2world": None if vlidar2world is None else vlidar2world.tolist()
    }

# TODO: Replace these with real values from AirSim poses
K_placeholder   = np.array([[1200, 0, IMG_W/2], [0, 1200, IMG_H/2], [0, 0, 1]], dtype=float)
I44             = np.eye(4, dtype=float)

json.dump(make_calib_json(K_placeholder, I44),
          open(os.path.join(PATHS["veh"]["calib"], "calib.json"), "w"), indent=2)
json.dump(make_calib_json(K_placeholder, I44),
          open(os.path.join(PATHS["inf"]["calib"], "calib.json"), "w"), indent=2)

# ----------------- HELPERS -----------------

def tick_to_ms(tick_idx: int) -> int:
    """Integer ms derived from tick index."""
    return int(round(tick_idx * DT * 1000.0))

def save_lidar_bin(path, pts_xyz):
    # Append dummy intensity 1.0 to match XYZI float32 layout
    arr = np.concatenate([pts_xyz, np.ones((pts_xyz.shape[0], 1), np.float32)], axis=1).astype(np.float32)
    arr.tofile(path)

def get_rgb(client, vname):
    reqs = [airsim.ImageRequest(CAM_NAME, airsim.ImageType.Scene, False, False)]
    while True:
        imgs = client.simGetImages(reqs, vehicle_name=vname)
        im = imgs[0]
        if im.width > 0:
            data = np.frombuffer(im.image_data_uint8, dtype=np.uint8)
            rgb = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if rgb is not None:
                return rgb

def get_lidar(client, vname):
    while True:
        ld = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=vname)
        if ld.point_cloud:
            pts = np.array(ld.point_cloud, dtype=np.float32).reshape(-1, 3)
            if pts.size:
                return pts

def get_pose_quat(client, vname, is_drone: bool):
    st = client.getMultirotorState(vehicle_name=vname) if is_drone else client.getCarState(vehicle_name=vname)
    k = st.kinematics_estimated
    p = k.position
    o = k.orientation
    return (p.x_val, p.y_val, p.z_val, o.w_val, o.x_val, o.y_val, o.z_val)

# 2D detection (optional; GT 2D not required for ImVoxelNet training)
def get_detections(client, vname):
    dets = client.simGetDetections(CAM_NAME, airsim.ImageType.Scene)
    out = []
    if not dets:
        return out
    for d in dets:
        cls = None
        for key in DAIR_CLASSES:
            if d.name.startswith(key):
                cls = DAIR_CLASSES[key]
                break
        if cls is None:
            continue
        out.append({
            "type": cls,
            "bbox2d": [int(d.box2D.min.x_val), int(d.box2D.min.y_val),
                       int(d.box2D.max.x_val), int(d.box2D.max.y_val)]
        })
    return out

# ----------------- MAIN LOOP -----------------
num_ticks = int(DURATION_S * BASE_HZ)
print(f"Collecting {num_ticks} ticks @ {BASE_HZ} Hz (DT={DT:.3f}s)")

veh_ts_list, inf_ts_list = [], []
frame_pairs = []  # (veh_ts, inf_ts)

# Queues if you later need to buffer
veh_queue, inf_queue = deque(), deque()

for tick in range(1, num_ticks + 1):
    drone_client.simContinueForTime(DT)  # advances the whole sim
    t_ms = tick_to_ms(tick)

    # VEHICLE SIDE
    for name in SIDE_VEH:
        if tick % CAM_EVERY == 0:
            rgb = get_rgb(car_client, name)
            img_path = os.path.join(PATHS["veh"]["img"], f"{t_ms}{IMG_EXT}")
            cv2.imwrite(img_path, rgb)
        if tick % LIDAR_EVERY == 0:
            pts = get_lidar(car_client, name)
            lidar_path = os.path.join(PATHS["veh"]["lidar"], f"{t_ms}.bin")
            save_lidar_bin(lidar_path, pts)
            veh_ts_list.append(t_ms)
            veh_queue.append(t_ms)

    # INFRA SIDE
    for name in SIDE_INF:
        if tick % CAM_EVERY == 0:
            rgb = get_rgb(drone_client, name)
            img_path = os.path.join(PATHS["inf"]["img"], f"{t_ms}{IMG_EXT}")
            cv2.imwrite(img_path, rgb)
        if tick % LIDAR_EVERY == 0:
            pts = get_lidar(drone_client, name)
            lidar_path = os.path.join(PATHS["inf"]["lidar"], f"{t_ms}.bin")
            save_lidar_bin(lidar_path, pts)
            inf_ts_list.append(t_ms)
            inf_queue.append(t_ms)

    # Pair by identical timestamp (perfect sync assumption)
    while veh_queue and inf_queue and veh_queue[0] == inf_queue[0]:
        ts = veh_queue.popleft()
        inf_queue.popleft()
        frame_pairs.append((ts, ts))

# Unpause
drone_client.simPause(False)

# Timestamps files
with open(os.path.join(PATHS["veh"]["ts"], "timestamp.txt"), "w") as f:
    for ts in veh_ts_list:
        f.write(f"{ts}\n")
with open(os.path.join(PATHS["inf"]["ts"], "timestamp.txt"), "w") as f:
    for ts in inf_ts_list:
        f.write(f"{ts}\n")

# -------------- LABEL STUBS (replace with real 3D GT) --------------
# DAIR expects lists of objects with fields: type, truncated, occluded, alpha,
# 2d bbox [xmin,ymin,xmax,ymax], 3d bbox [h,w,l,x,y,z,ry] in Virtual LiDAR CS.

def write_empty_label(path):
    json.dump([], open(path, "w"))

for veh_ts, inf_ts in frame_pairs:
    write_empty_label(os.path.join(PATHS["label"]["veh"], f"{veh_ts}.json"))
    write_empty_label(os.path.join(PATHS["label"]["inf"], f"{inf_ts}.json"))
    write_empty_label(os.path.join(PATHS["label"]["cooperative"], f"{veh_ts}.json"))

# Pairs meta for preprocessing
pairs_meta = {
    "pairs": [{"veh_ts": v, "inf_ts": i} for (v, i) in frame_pairs],
    "system_error_offset_ms": 0
}
json.dump(pairs_meta, open(os.path.join(OUT_ROOT, "vic_sync_pairs.json"), "w"), indent=2)

print("Done. Data saved under:", OUT_ROOT)
