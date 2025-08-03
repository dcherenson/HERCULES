#!/usr/bin/env python3
"""
collect_sim_to_dair_kitti.py

Robust data collection + labeling pipeline for Hercules (Cosys-AirSim) simulator
to produce DAIR-V2X / KITTI-compatible data.

Labels are generated in the same tick as the sensor data capture to preserve sync.
"""

import os
import json
import math
import time
import traceback
from collections import deque

import numpy as np
import cv2

import setup_path  # ensure AirSim / Cosys-AirSim is on PYTHONPATH
import cosysairsim as airsim  # your custom fork

# ------------------ USER CONFIG ------------------

SETTINGS_JSON_PATH = "/home/sgarimella34/Documents/AirSim/settings.json"
OUT_ROOT = "/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets/dair_v2x_synth"

SIDE_INF = ["Drone1"]
SIDE_VEH = ["Husky1"]

CAM_NAME = "front_center"
LIDAR_NAME = "LidarSensor1"

IMG_EXT = ".png"

CAM_RATE = 20
LIDAR_RATE = 10
DURATION_S = 300.0  # seconds

BASE_HZ = CAM_RATE
DT = 1.0 / BASE_HZ
CAM_EVERY = BASE_HZ // CAM_RATE
LIDAR_EVERY = BASE_HZ // LIDAR_RATE

VEHICLE_DET_RADIUS_CM = int(120 * 100)
INF_DET_RADIUS_CM = int(300 * 100)

DRONE_PORT = 41451
HUSKY_PORT = 41452

# Objects: prefix in detection name -> class and physical size (length, width, height)
OBJECTS = [
    {"actor_name": "Car", "class": "Car", "size_lwh": [4.5, 1.8, 1.6]},
    {"actor_name": "SK_Survival_Character", "class": "Pedestrian", "size_lwh": [0.6, 0.6, 1.7]},
]

DAIR_CLASSES = {"Car": "Car", "SK_Survival_Character": "Pedestrian"}

# ------------------ MESH NAME FILTERS (edit these) ------------------
# Wildcard patterns passed to AirSim's simAddDetectionFilterMeshName.
# If empty or None, falls back to detecting everything ("*").
MESH_FILTERS = {
    "veh": ["Sportscar", "Sedan2", "SK_Survival_Character"],
    "inf": ["Sportscar", "Sedan2", "SK_Survival_Character"],
}

# ------------------ OUTPUT PATHS ------------------

def dair_paths(root: str):
    return {
        "inf": {
            "img":   f"{root}/cooperative/infrastructure-side/image",
            "lidar": f"{root}/cooperative/infrastructure-side/lidar",
            "calib": f"{root}/cooperative/infrastructure-side/calib",
            "ts":    f"{root}/cooperative/infrastructure-side/timestamp",
            "kitti_label": f"{root}/cooperative/infrastructure-side/kitti_label"
        },
        "veh": {
            "img":   f"{root}/cooperative/vehicle-side/image",
            "lidar": f"{root}/cooperative/vehicle-side/lidar",
            "calib": f"{root}/cooperative/vehicle-side/calib",
            "ts":    f"{root}/cooperative/vehicle-side/timestamp",
            "kitti_label": f"{root}/cooperative/vehicle-side/kitti_label"
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

# ------------------ AIRSIM CLIENTS ------------------

drone_client = airsim.MultirotorClient(port=DRONE_PORT)
car_client = airsim.CarClient(port=HUSKY_PORT)

print("[STARTUP] connecting to AirSim clients...", flush=True)
drone_client.confirmConnection()
car_client.confirmConnection()
print("[STARTUP] connected to AirSim clients", flush=True)

for n in SIDE_INF:
    drone_client.enableApiControl(True, vehicle_name=n)
for n in SIDE_VEH:
    car_client.enableApiControl(True, vehicle_name=n)

# Pause world; we'll advance deterministically per tick.
drone_client.simPause(True)

# ------------------ TRANSFORMS / UTILS ------------------

def load_settings(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"settings.json not found at {path}")
    with open(path, "r") as f:
        return json.load(f)

def euler_to_rot_matrix(yaw_deg, pitch_deg, roll_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0,              0,             1]
    ], dtype=float)
    Ry = np.array([
        [ math.cos(pitch), 0, math.sin(pitch)],
        [ 0,               1, 0             ],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ], dtype=float)
    Rx = np.array([
        [1, 0,               0              ],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll),  math.cos(roll)]
    ], dtype=float)
    return Rz @ Ry @ Rx

def quat_to_rot_matrix(w, x, y, z):
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ], dtype=float)

def build_homogeneous(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def invert_homogeneous(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def wrap_to_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi

def extract_yaw_from_virtual_lidar_rotation(R):
    return math.atan2(R[1, 0], R[0, 0])

def extract_yaw_from_camera_rotation(R):
    heading = R @ np.array([0, 0, 1], dtype=float)
    return math.atan2(heading[0], heading[2])

# AirSim NED → DAIR-V2X Virtual LiDAR adjustment (z-up, leveled)
F_VL = np.diag([1.0, -1.0, -1.0])

# ------------------ CALIBRATION ------------------

def build_sensor_calibrations(settings, vehicle_name):
    vehicle_cfg = settings["Vehicles"].get(vehicle_name)
    if vehicle_cfg is None:
        raise RuntimeError(f"Vehicle {vehicle_name} not in settings.json")
    cam_cfg = vehicle_cfg["Cameras"][CAM_NAME]
    capture = None
    for cap in cam_cfg.get("CaptureSettings", []):
        if cap.get("ImageType", -1) == 0:
            capture = cap
            break
    if capture is None:
        raise RuntimeError(f"No scene capture for camera {CAM_NAME} on {vehicle_name}")
    width = capture["Width"]
    height = capture["Height"]
    fov_deg = capture["FOV_Degrees"]
    fov_rad = math.radians(fov_deg)
    fx = fy = width / (2 * math.tan(fov_rad / 2))
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=float)

    cam_X = cam_cfg.get("X", 0.0)
    cam_Y = cam_cfg.get("Y", 0.0)
    cam_Z = cam_cfg.get("Z", 0.0)
    cam_Pitch = cam_cfg.get("Pitch", 0.0)
    cam_Roll = cam_cfg.get("Roll", 0.0)
    cam_Yaw = cam_cfg.get("Yaw", 0.0)
    R_cam = euler_to_rot_matrix(cam_Yaw, cam_Pitch, cam_Roll)
    T_cam_to_vehicle = build_homogeneous(R_cam, np.array([cam_X, cam_Y, cam_Z], dtype=float))

    lidar_cfg = vehicle_cfg["Sensors"][LIDAR_NAME]
    lidar_X = lidar_cfg.get("X", 0.0)
    lidar_Y = lidar_cfg.get("Y", 0.0)
    lidar_Z = lidar_cfg.get("Z", 0.0)
    lidar_Roll = lidar_cfg.get("Roll", 0.0)
    lidar_Pitch = lidar_cfg.get("Pitch", 0.0)
    lidar_Yaw = lidar_cfg.get("Yaw", 0.0)
    R_lidar = euler_to_rot_matrix(lidar_Yaw, lidar_Pitch, lidar_Roll)
    T_lidar_to_vehicle = build_homogeneous(R_lidar, np.array([lidar_X, lidar_Y, lidar_Z], dtype=float))

    return {
        "K": K,
        "image_size": (width, height),
        "T_cam_to_vehicle": T_cam_to_vehicle,
        "T_lidar_to_vehicle": T_lidar_to_vehicle
    }

def get_vehicle_world_transform(client, vehicle_name, is_drone: bool):
    if is_drone:
        st = client.getMultirotorState(vehicle_name=vehicle_name)
    else:
        st = client.getCarState(vehicle_name=vehicle_name)
    kin = st.kinematics_estimated
    pos = kin.position
    ori = kin.orientation
    R = quat_to_rot_matrix(ori.w_val, ori.x_val, ori.y_val, ori.z_val)
    T_world_vehicle = build_homogeneous(R, np.array([pos.x_val, pos.y_val, pos.z_val], dtype=float))
    return T_world_vehicle

def get_object_world_pose(client, name):
    pose = client.simGetObjectPose(name)
    if pose is None:
        return None, None
    p = pose.position
    o = pose.orientation
    R = quat_to_rot_matrix(o.w_val, o.x_val, o.y_val, o.z_val)
    t = np.array([p.x_val, p.y_val, p.z_val], dtype=float)
    return R, t

# ------------------ SENSOR ACCESS ------------------

def set_detection_filters(client, vehicle_name, radius_cm, mesh_names=None):
    client.simClearDetectionMeshNames(CAM_NAME, airsim.ImageType.Scene, vehicle_name=vehicle_name)
    client.simSetDetectionFilterRadius(CAM_NAME, airsim.ImageType.Scene, radius_cm, vehicle_name=vehicle_name)
    if not mesh_names:
        client.simAddDetectionFilterMeshName(CAM_NAME, airsim.ImageType.Scene, "*", vehicle_name=vehicle_name)
    else:
        for pattern in mesh_names:
            client.simAddDetectionFilterMeshName(CAM_NAME, airsim.ImageType.Scene, pattern, vehicle_name=vehicle_name)

def get_detections_api(client, vehicle_name):
    dets = client.simGetDetections(CAM_NAME, airsim.ImageType.Scene, vehicle_name=vehicle_name)
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
            cls = d.name
        bbox2d = [
            float(d.box2D.min.x_val),
            float(d.box2D.min.y_val),
            float(d.box2D.max.x_val),
            float(d.box2D.max.y_val)
        ]
        out.append({
            "name": d.name,
            "type": cls,
            "bbox2d": bbox2d,
            "relative_pose": d.relative_pose
        })
    return out

def get_image(client, vehicle_name):
    resp = client.simGetImages([airsim.ImageRequest(CAM_NAME, airsim.ImageType.Scene, False, True)],
                               vehicle_name=vehicle_name)[0]
    if resp.compress:
        data = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    else:
        img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        img_rgba = img1d.reshape(resp.height, resp.width, 4)
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        return img_bgr

def get_lidar_points_with_retry(client, vehicle_name, max_retries=3, backoff_s=0.001):
    for attempt in range(max_retries):
        ld = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=vehicle_name)
        if ld.point_cloud:
            pts = np.array(ld.point_cloud, dtype=np.float32).reshape(-1, 3)
            if pts.size:
                return pts
        time.sleep(backoff_s)
    return None

def save_lidar_bin(path, pts_xyz):
    arr = np.concatenate([pts_xyz, np.ones((pts_xyz.shape[0], 1), np.float32)], axis=1).astype(np.float32)
    arr.tofile(path)

# ------------------ LABEL CONSTRUCTION ------------------

def compute_truncated_state(bbox2d, img_w, img_h):
    xmin, ymin, xmax, ymax = bbox2d
    overflow_x = max(0, -xmin) + max(0, xmax - img_w)
    overflow_y = max(0, -ymin) + max(0, ymax - img_h)
    if overflow_x > 0 and overflow_y > 0:
        return "1" if overflow_x >= overflow_y else "2"
    if overflow_x > 0:
        return "1"
    if overflow_y > 0:
        return "2"
    return "0"

def compute_occluded_state():
    return "0"

def build_label_entry_dair(obj_cls, size_lwh, alpha, bbox2d, loc_vl, ry, img_size):
    length, width, height = size_lwh
    h = height
    w = width
    l = length
    truncated = compute_truncated_state(bbox2d, img_size[0], img_size[1]) if bbox2d is not None else "0"
    occluded = compute_occluded_state()
    return {
        "type": obj_cls,
        "truncated_state": truncated,
        "occluded_state": occluded,
        "alpha": alpha if alpha is not None else 0.0,
        "2d_box": {
            "xmin": float(bbox2d[0]) if bbox2d is not None else 0.0,
            "ymin": float(bbox2d[1]) if bbox2d is not None else 0.0,
            "xmax": float(bbox2d[2]) if bbox2d is not None else 0.0,
            "ymax": float(bbox2d[3]) if bbox2d is not None else 0.0
        } if bbox2d is not None else None,
        "3d_dimensions": {"h": float(h), "w": float(w), "l": float(l)},
        "3d_location": {"x": float(loc_vl[0]), "y": float(loc_vl[1]), "z": float(loc_vl[2])},
        "rotation": float(ry)
    }

def build_kitti_label_line(obj_type, truncated, occluded, alpha, bbox2d, dims, loc_cam, rotation_y):
    left, top, right, bottom = bbox2d
    h, w, l = dims
    x, y, z = loc_cam
    return f"{obj_type} {truncated:.2f} {occluded} {alpha:.6f} {left:.2f} {top:.2f} {right:.2f} {bottom:.2f} {h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {rotation_y:.6f}"

# ------------------ MAIN ------------------

def main():
    print("[ENTRY] Starting main()", flush=True)

    if not os.path.isfile(SETTINGS_JSON_PATH):
        print(f"[FATAL] settings.json not found at {SETTINGS_JSON_PATH}", flush=True)
        return
    try:
        settings = load_settings(SETTINGS_JSON_PATH)
        print(f"[OK] loaded settings.json from {SETTINGS_JSON_PATH}", flush=True)
    except Exception as e:
        print(f"[FATAL] failed to parse settings.json: {e}", flush=True)
        return

    try:
        veh_calib = build_sensor_calibrations(settings, SIDE_VEH[0])
        inf_calib = build_sensor_calibrations(settings, SIDE_INF[0])
        print("[OK] built calibrations", flush=True)
    except Exception as e:
        print(f"[FATAL] calibration build failed: {e}", flush=True)
        traceback.print_exc()
        return

    set_detection_filters(car_client, SIDE_VEH[0], VEHICLE_DET_RADIUS_CM, mesh_names=MESH_FILTERS["veh"])
    set_detection_filters(drone_client, SIDE_INF[0], INF_DET_RADIUS_CM, mesh_names=MESH_FILTERS["inf"])
    print("[OK] detection filters set", flush=True)

    # Dump static calib JSONs (initial world pose)
    T_world_veh_init = get_vehicle_world_transform(car_client, SIDE_VEH[0], is_drone=False)
    T_world_inf_init = get_vehicle_world_transform(drone_client, SIDE_INF[0], is_drone=True)

    # Vehicle-side calibration dump
    T_world_lidar_veh_raw = T_world_veh_init @ veh_calib["T_lidar_to_vehicle"]
    R_raw_veh = T_world_lidar_veh_raw[:3, :3]
    t_raw_veh = T_world_lidar_veh_raw[:3, 3]
    R_virtual_veh = F_VL @ R_raw_veh @ F_VL
    yaw_veh = extract_yaw_from_virtual_lidar_rotation(R_virtual_veh)
    R_level_veh = np.array([
        [math.cos(yaw_veh), -math.sin(yaw_veh), 0],
        [math.sin(yaw_veh),  math.cos(yaw_veh), 0],
        [0,                 0,                1]
    ], dtype=float)
    T_world_virtual_lidar_veh = build_homogeneous(R_level_veh, t_raw_veh)
    T_world_cam_veh = T_world_veh_init @ veh_calib["T_cam_to_vehicle"]
    cam2vlidar_virtual = invert_homogeneous(T_world_cam_veh) @ T_world_virtual_lidar_veh
    json.dump({
        "cam_intrinsic": veh_calib["K"].tolist(),
        "cam2vlidar": cam2vlidar_virtual.tolist(),
        "vlidar2world": T_world_virtual_lidar_veh.tolist()
    }, open(os.path.join(PATHS["veh"]["calib"], "calib.json"), "w"), indent=2)

    # Infrastructure-side calibration dump
    T_world_lidar_inf_raw = T_world_inf_init @ inf_calib["T_lidar_to_vehicle"]
    R_raw_inf = T_world_lidar_inf_raw[:3, :3]
    t_raw_inf = T_world_lidar_inf_raw[:3, 3]
    R_virtual_inf = F_VL @ R_raw_inf @ F_VL
    yaw_inf = extract_yaw_from_virtual_lidar_rotation(R_virtual_inf)
    R_level_inf = np.array([
        [math.cos(yaw_inf), -math.sin(yaw_inf), 0],
        [math.sin(yaw_inf),  math.cos(yaw_inf), 0],
        [0,                 0,                1]
    ], dtype=float)
    T_world_virtual_lidar_inf = build_homogeneous(R_level_inf, t_raw_inf)
    T_world_cam_inf = T_world_inf_init @ inf_calib["T_cam_to_vehicle"]
    cam2vlidar_virtual_inf = invert_homogeneous(T_world_cam_inf) @ T_world_virtual_lidar_inf
    json.dump({
        "cam_intrinsic": inf_calib["K"].tolist(),
        "cam2vlidar": cam2vlidar_virtual_inf.tolist(),
        "vlidar2world": T_world_virtual_lidar_inf.tolist()
    }, open(os.path.join(PATHS["inf"]["calib"], "calib.json"), "w"), indent=2)

    # Sanity write test label to prove writeability
    debug_label = [{
        "type": "Car",
        "truncated_state": "0",
        "occluded_state": "0",
        "alpha": 0.0,
        "2d_box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
        "3d_dimensions": {"h": 1.0, "w": 1.0, "l": 1.0},
        "3d_location": {"x": 0.0, "y": 0.0, "z": 0.0},
        "rotation": 0.0
    }]
    try:
        with open(os.path.join(PATHS["label"]["veh"], "debug_test.json"), "w") as f:
            json.dump(debug_label, f)
        with open(os.path.join(PATHS["label"]["inf"], "debug_test.json"), "w") as f:
            json.dump(debug_label, f)
        print("[DIAG] wrote debug_test.json to label directories", flush=True)
    except Exception as e:
        print(f"[DIAG][ERROR] failed debug label write: {e}", flush=True)

    # --- MAIN TICK LOOP (capture + per-pair label generation) ---
    num_ticks = int(DURATION_S * BASE_HZ)
    print(f"[COLLECT] Collecting {num_ticks} ticks @ {BASE_HZ}Hz (DT={DT:.3f}s)", flush=True)

    veh_ts_list = []
    inf_ts_list = []
    frame_pairs = []

    for tick in range(1, num_ticks + 1):
        try:
            drone_client.simContinueForTime(DT)
        except Exception as e:
            print(f"[ERROR] sim step failed at tick {tick}: {e}", flush=True)
            continue

        t_ms = int(round(tick * DT * 1000.0))

        if tick % 100 == 0:
            print(f"[COLLECT] tick {tick}/{num_ticks}, veh_ts={len(veh_ts_list)}, inf_ts={len(inf_ts_list)}", flush=True)

        # --- capture vehicle sensors ---
        veh_got_lidar = False
        for name in SIDE_VEH:
            if tick % CAM_EVERY == 0:
                img = get_image(car_client, name)
                if img is not None:
                    cv2.imwrite(os.path.join(PATHS["veh"]["img"], f"{t_ms}{IMG_EXT}"), img)
            if tick % LIDAR_EVERY == 0:
                pts = get_lidar_points_with_retry(car_client, name)
                if pts is not None and pts.size:
                    save_lidar_bin(os.path.join(PATHS["veh"]["lidar"], f"{t_ms}.bin"), pts)
                    veh_ts_list.append(t_ms)
                    veh_got_lidar = True
                else:
                    print(f"[WARN] vehicle lidar missing at tick {tick} (t_ms={t_ms})", flush=True)

        # --- capture infrastructure sensors ---
        inf_got_lidar = False
        for name in SIDE_INF:
            if tick % CAM_EVERY == 0:
                img = get_image(drone_client, name)
                if img is not None:
                    cv2.imwrite(os.path.join(PATHS["inf"]["img"], f"{t_ms}{IMG_EXT}"), img)
            if tick % LIDAR_EVERY == 0:
                pts = get_lidar_points_with_retry(drone_client, name)
                if pts is not None and pts.size:
                    save_lidar_bin(os.path.join(PATHS["inf"]["lidar"], f"{t_ms}.bin"), pts)
                    inf_ts_list.append(t_ms)
                    inf_got_lidar = True
                else:
                    print(f"[WARN] infra lidar missing at tick {tick} (t_ms={t_ms})", flush=True)

        # --- if both sides have lidar this tick, treat as a synchronized pair and generate labels ---
        if veh_got_lidar and inf_got_lidar:
            frame_pairs.append((t_ms, t_ms))
            # get current poses
            T_world_veh = get_vehicle_world_transform(car_client, SIDE_VEH[0], is_drone=False)
            T_world_inf = get_vehicle_world_transform(drone_client, SIDE_INF[0], is_drone=True)

            # Vehicle virtual lidar (leveled)
            T_world_lidar_veh_raw = T_world_veh @ veh_calib["T_lidar_to_vehicle"]
            R_raw_veh = T_world_lidar_veh_raw[:3, :3]
            t_raw_veh = T_world_lidar_veh_raw[:3, 3]
            R_virtual_veh = F_VL @ R_raw_veh @ F_VL
            yaw_veh = extract_yaw_from_virtual_lidar_rotation(R_virtual_veh)
            R_level_veh = np.array([
                [math.cos(yaw_veh), -math.sin(yaw_veh), 0],
                [math.sin(yaw_veh),  math.cos(yaw_veh), 0],
                [0,                 0,                1]
            ], dtype=float)
            T_world_virtual_lidar_veh = build_homogeneous(R_level_veh, t_raw_veh)
            inv_T_world_virtual_lidar_veh = invert_homogeneous(T_world_virtual_lidar_veh)

            # Infra virtual lidar
            T_world_lidar_inf_raw = T_world_inf @ inf_calib["T_lidar_to_vehicle"]
            R_raw_inf = T_world_lidar_inf_raw[:3, :3]
            t_raw_inf = T_world_lidar_inf_raw[:3, 3]
            R_virtual_inf = F_VL @ R_raw_inf @ F_VL
            yaw_inf = extract_yaw_from_virtual_lidar_rotation(R_virtual_inf)
            R_level_inf = np.array([
                [math.cos(yaw_inf), -math.sin(yaw_inf), 0],
                [math.sin(yaw_inf),  math.cos(yaw_inf), 0],
                [0,                0,                 1]
            ], dtype=float)
            T_world_virtual_lidar_inf = build_homogeneous(R_level_inf, t_raw_inf)
            inv_T_world_virtual_lidar_inf = invert_homogeneous(T_world_virtual_lidar_inf)

            # Camera transforms
            T_world_cam_veh = T_world_veh @ veh_calib["T_cam_to_vehicle"]
            inv_T_world_cam_veh = invert_homogeneous(T_world_cam_veh)
            T_world_cam_inf = T_world_inf @ inf_calib["T_cam_to_vehicle"]
            inv_T_world_cam_inf = invert_homogeneous(T_world_cam_inf)

            # Detections
            veh_dets = get_detections_api(car_client, SIDE_VEH[0])
            inf_dets = get_detections_api(drone_client, SIDE_INF[0])
            print(f"[Frame {t_ms}] Vehicle detections: {[d['name'] for d in veh_dets]}", flush=True)
            print(f"[Frame {t_ms}] Infra detections: {[d['name'] for d in inf_dets]}", flush=True)

            veh_det_dict = {d["name"]: d for d in veh_dets}
            inf_det_dict = {d["name"]: d for d in inf_dets}

            # Vehicle-side label construction
            veh_labels = []
            kitti_lines_veh = []
            for obj in OBJECTS:
                det = None
                for name in veh_det_dict:
                    if name.startswith(obj["actor_name"]) or obj["actor_name"] in name:
                        det = veh_det_dict[name]
                        break
                R_obj_world, t_obj_world = get_object_world_pose(car_client, det["name"] if det else obj["actor_name"])
                if R_obj_world is None:
                    continue
                p_world = np.hstack([t_obj_world, 1.0]).reshape(4, 1)

                obj_in_vl = inv_T_world_virtual_lidar_veh @ p_world
                loc_vl = obj_in_vl[:3, 0]
                R_lidar_obj = (R_level_veh.T) @ R_obj_world
                ry = extract_yaw_from_virtual_lidar_rotation(R_lidar_obj)

                alpha = None
                bbox2d = None
                if det:
                    rel_pose = det["relative_pose"]
                    pos_cam = np.array([
                        rel_pose.position.x_val,
                        rel_pose.position.y_val,
                        rel_pose.position.z_val
                    ], dtype=float)
                    ori = rel_pose.orientation
                    R_cam_obj = quat_to_rot_matrix(ori.w_val, ori.x_val, ori.y_val, ori.z_val)
                    rotation_y_cam = extract_yaw_from_camera_rotation(R_cam_obj)
                    alpha = wrap_to_pi(rotation_y_cam - math.atan2(pos_cam[0], pos_cam[2]))
                    bbox2d = det["bbox2d"]

                    obj_in_cam = inv_T_world_cam_veh @ p_world
                    loc_cam = obj_in_cam[:3, 0]
                    R_world_cam = T_world_cam_veh[:3, :3]
                    R_camera_obj = R_world_cam.T @ R_obj_world
                    rotation_y = extract_yaw_from_camera_rotation(R_camera_obj)
                    length, width, height = obj["size_lwh"]
                    h = height; w = width; l = length
                    kitti_line = build_kitti_label_line(
                        obj["class"],
                        0.0,
                        0,
                        alpha if alpha is not None else 0.0,
                        bbox2d,
                        (h, w, l),
                        (loc_cam[0], loc_cam[1], loc_cam[2]),
                        rotation_y
                    )
                    kitti_lines_veh.append(kitti_line)

                entry = build_label_entry_dair(obj["class"], obj["size_lwh"], alpha if alpha is not None else 0.0,
                                              bbox2d, loc_vl, ry, veh_calib["image_size"])
                veh_labels.append(entry)

            # Infra-side label construction
            inf_labels = []
            for obj in OBJECTS:
                det = None
                for name in inf_det_dict:
                    if name.startswith(obj["actor_name"]) or obj["actor_name"] in name:
                        det = inf_det_dict[name]
                        break
                R_obj_world, t_obj_world = get_object_world_pose(drone_client, det["name"] if det else obj["actor_name"])
                if R_obj_world is None:
                    continue
                p_world = np.hstack([t_obj_world, 1.0]).reshape(4, 1)

                obj_in_vl = inv_T_world_virtual_lidar_inf @ p_world
                loc_vl = obj_in_vl[:3, 0]
                R_lidar_obj = (R_level_inf.T) @ R_obj_world
                ry = extract_yaw_from_virtual_lidar_rotation(R_lidar_obj)

                alpha = None
                bbox2d = None
                if det:
                    rel_pose = det["relative_pose"]
                    pos_cam = np.array([
                        rel_pose.position.x_val,
                        rel_pose.position.y_val,
                        rel_pose.position.z_val
                    ], dtype=float)
                    ori = rel_pose.orientation
                    R_cam_obj = quat_to_rot_matrix(ori.w_val, ori.x_val, ori.y_val, ori.z_val)
                    rotation_y_cam = extract_yaw_from_camera_rotation(R_cam_obj)
                    alpha = wrap_to_pi(rotation_y_cam - math.atan2(pos_cam[0], pos_cam[2]))
                    bbox2d = det["bbox2d"]

                entry = build_label_entry_dair(obj["class"], obj["size_lwh"], alpha if alpha is not None else 0.0,
                                              bbox2d, loc_vl, ry, inf_calib["image_size"])
                inf_labels.append(entry)

            coop_labels = veh_labels + inf_labels

            # Write out labels immediately for this pair
            try:
                with open(os.path.join(PATHS["label"]["veh"], f"{t_ms}.json"), "w") as f:
                    json.dump(veh_labels, f, indent=2)
                with open(os.path.join(PATHS["label"]["inf"], f"{t_ms}.json"), "w") as f:
                    json.dump(inf_labels, f, indent=2)
                with open(os.path.join(PATHS["label"]["cooperative"], f"{t_ms}.json"), "w") as f:
                    json.dump(coop_labels, f, indent=2)
                if kitti_lines_veh:
                    kt_path = os.path.join(PATHS["veh"]["kitti_label"], f"{t_ms}.txt")
                    with open(kt_path, "w") as f:
                        for line in kitti_lines_veh:
                            f.write(line + "\n")
                print(f"[INFO] Wrote labels for pair veh_ts={t_ms}, inf_ts={t_ms}: "
                      f"vehicle_objs={len(veh_labels)}, infra_objs={len(inf_labels)}, kitti_lines={len(kitti_lines_veh)}", flush=True)
            except Exception as e:
                print(f"[ERROR] failed writing label JSONs for timestamp {t_ms}: {e}", flush=True)
                traceback.print_exc()
                continue

    # after loop: unpause world and dump metadata
    drone_client.simPause(False)

    # Write timestamp files
    with open(os.path.join(PATHS["veh"]["ts"], "timestamp.txt"), "w") as f:
        for ts in veh_ts_list:
            f.write(f"{ts}\n")
    with open(os.path.join(PATHS["inf"]["ts"], "timestamp.txt"), "w") as f:
        for ts in inf_ts_list:
            f.write(f"{ts}\n")

    # Write pairing file
    pairs_meta = {
        "pairs": [{"veh_ts": v, "inf_ts": i} for (v, i) in frame_pairs],
        "system_error_offset_ms": 0
    }
    json.dump(pairs_meta, open(os.path.join(OUT_ROOT, "vic_sync_pairs.json"), "w"), indent=2)

    print("[DONE] Finished. Data stored under:", OUT_ROOT, flush=True)
    print(f"Total pairs: {len(frame_pairs)}", flush=True)
    print(f"Collected vehicle lidar timestamps: {len(veh_ts_list)}, infra: {len(inf_ts_list)}", flush=True)

if __name__ == "__main__":
    main()
