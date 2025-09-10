#!/usr/bin/env python3
import os, json, time, math
import numpy as np
import setup_path
import cosysairsim as airsim
import cv2
from Hercules2D3DDetector import Hercules2D3DDetector as H

# =========================
# Config
# =========================
SELECTED_VEHICLE = "Husky1"      # "Husky1" (UGV) or "Drone1" (UAV)
CAMERA_NAME_OVERRIDE = None      # e.g., "front_center" or None

# Sim step control
ADVANCE_DT_SECONDS = 0.10  # 0.1s per timestep
N_STEPS = 5000

# DAIR-V2X-C-style root
DAIRV2X_C_ROOT = "/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets/dair_v2x_synth/cooperative-vehicle-infrastructure"

# AirSim settings
SETTINGS_JSON_PATH = "/home/sgarimella34/Documents/AirSim/settings.json"

# Names used on both sides
CAM_NAME   = "front_center"
LIDAR_NAME = "LidarSensor1"

# Vehicle names per side in your sim
VEHICLE_SIDE_NAME = "Husky1"
INFRA_SIDE_NAME   = "Drone1"

# =========================
# Helpers: vehicle config
# =========================
def _configure_from_vehicle(vehicle_name: str):
    name_l = (vehicle_name or "").lower()
    if name_l.startswith("husky"):
        platform = "ugv"
        H.CLIENT_CLASS = airsim.CarClient
        H.PORT = 41452
    else:
        platform = "drone"
        H.CLIENT_CLASS = airsim.MultirotorClient
        H.PORT = 41451

    H.VEHICLE_NAME = vehicle_name
    if CAMERA_NAME_OVERRIDE:
        H.CAMERA_NAME = CAMERA_NAME_OVERRIDE

    print(
        f"[CFG] platform={platform}, vehicle={H.VEHICLE_NAME}, "
        f"client={H.CLIENT_CLASS.__name__}, port={H.PORT}, "
        f"camera={getattr(H, 'CAMERA_NAME', 'front_center')}"
    )

def _advance_sim_once(ctrl, dt_sec: float):
    ctrl.simPause(True)
    if hasattr(ctrl, "simContinueForTime"):
        ctrl.simContinueForTime(float(dt_sec))
        ctrl.simPause(True); return
    # Fallback if needed:
    ctrl.simPause(False)
    time.sleep(max(0.0, float(dt_sec)))
    ctrl.simPause(True)

def _client_for_vehicle(vehicle_name: str):
    name_l = (vehicle_name or "").lower()
    if name_l.startswith("husky"):
        cli = airsim.CarClient(port=41452)
    else:
        cli = airsim.MultirotorClient(port=41451)
    cli.confirmConnection()
    return cli

# =========================
# Calib from settings.json only (DAIR format)
# =========================
def _deg2rad(d): return d * math.pi / 180.0

def _R_from_rpy_deg(roll_deg, pitch_deg, yaw_deg):
    r, p, y = map(_deg2rad, (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=float)
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]], dtype=float)
    return Rz @ Ry @ Rx  # AirSim NED: roll X, pitch Y, yaw Z

def _T_from_xyzrpy(x, y, z, roll_deg, pitch_deg, yaw_deg):
    T = np.eye(4, dtype=float)
    T[:3, :3] = _R_from_rpy_deg(roll_deg, pitch_deg, yaw_deg)
    T[:3,  3] = np.array([x, y, z], dtype=float)
    return T

def _load_settings(path):
    with open(path, "r") as f:
        return json.load(f)

def T_cam_lidar_from_settings(vehicle_name, cam_name, lidar_name, settings_path):
    """
    Build lidar→camera extrinsics purely from settings.json (vehicle->sensor mounts).
    AirSim stores sensor poses as vehicle→sensor. So:
      T_cam_lidar = inv(T_vehicle→camera) @ T_vehicle→lidar
    """
    js = _load_settings(settings_path)
    v = js["Vehicles"][vehicle_name]

    # Camera mount (vehicle→camera)
    c = v["Cameras"][cam_name]
    T_v_c = _T_from_xyzrpy(
        c.get("X", 0.0), c.get("Y", 0.0), c.get("Z", 0.0),
        c.get("Roll", 0.0), c.get("Pitch", 0.0), c.get("Yaw", 0.0)
    )

    # LiDAR mount (vehicle→lidar)
    l = v["Sensors"][lidar_name]
    T_v_l = _T_from_xyzrpy(
        l.get("X", 0.0), l.get("Y", 0.0), l.get("Z", 0.0),
        l.get("Roll", 0.0), l.get("Pitch", 0.0), l.get("Yaw", 0.0)
    )

    # lidar→camera in AirSim’s native frames (x fwd, y right, z down)
    T_c_l = np.linalg.inv(T_v_c) @ T_v_l
    return T_c_l

def _save_dair_lidar2cam_json(path, T_cam_lidar):
    """
    Save DAIR-style lidar->camera calibration:
      {
        "rotation":    3x3,
        "translation": [tx, ty, tz]
      }
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    R = T_cam_lidar[:3, :3]
    t = T_cam_lidar[:3, 3]
    obj = {
        "rotation": R.tolist(),
        "translation": [float(t[0]), float(t[1]), float(t[2])]
    }
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[calib] Wrote {path}")

def _compute_static_lidar2cam_mats():
    """
    Compute once from settings.json:
      - T_cam_lidar for VEHICLE side (in DAIR Virtual LiDAR basis)
      - T_cam_lidar for INFRA side  (in DAIR Virtual LiDAR basis)
    """
    T_c_l_veh = T_cam_lidar_from_settings(VEHICLE_SIDE_NAME, CAM_NAME, LIDAR_NAME, SETTINGS_JSON_PATH)
    T_c_l_inf = T_cam_lidar_from_settings(INFRA_SIDE_NAME,   CAM_NAME, LIDAR_NAME, SETTINGS_JSON_PATH)

    # Basis change on the LiDAR side only: AirSim (x fwd, y right, z down)
    # -> DAIR Virtual LiDAR (x fwd, y left, z up)
    B = np.eye(4, dtype=float)
    B[1,1] = -1.0  # flip Y
    B[2,2] = -1.0  # flip Z

    # Map from **DAIR Virtual LiDAR** → camera frame
    T_c_vl_veh = T_c_l_veh @ B
    T_c_vl_inf = T_c_l_inf @ B
    return T_c_vl_veh, T_c_vl_inf

def _write_per_frame_calib(frame_id: str, T_c_vl_veh: np.ndarray, T_c_vl_inf: np.ndarray):
    """
    Write DAIR-style lidar->camera JSONs for BOTH sides using the provided frame_id.
    """
    veh_dir = os.path.join(DAIRV2X_C_ROOT, "vehicle-side", "calib", "lidar_to_camera")
    inf_dir = os.path.join(DAIRV2X_C_ROOT, "infrastructure-side", "calib", "virtuallidar_to_camera")
    veh_json = os.path.join(veh_dir, f"{frame_id}.json")
    inf_json = os.path.join(inf_dir, f"{frame_id}.json")
    _save_dair_lidar2cam_json(veh_json, T_c_vl_veh)
    _save_dair_lidar2cam_json(inf_json, T_c_vl_inf)

# =========================
# NEW: Save RGB & LiDAR (DAIR/KitTI style)
# =========================
def _ensure_dirs():
    paths = [
        os.path.join(DAIRV2X_C_ROOT, "vehicle-side", "image"),
        os.path.join(DAIRV2X_C_ROOT, "vehicle-side", "velodyne"),
        os.path.join(DAIRV2X_C_ROOT, "infrastructure-side", "image"),
        os.path.join(DAIRV2X_C_ROOT, "infrastructure-side", "velodyne"),
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)

def _save_png_from_response(resp, path_png):
    arr = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode image for {path_png}")
    cv2.imwrite(path_png, img)

def _capture_rgb_png(ctrl, vehicle_name: str, camera_name: str, path_png: str):
    req = [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, True)]
    resp = ctrl.simGetImages(req, vehicle_name=vehicle_name)[0]
    _save_png_from_response(resp, path_png)

def _capture_lidar_bin(ctrl, vehicle_name: str, lidar_name: str, path_bin: str):
    """
    Save LiDAR as KITTI-style .bin: Nx4 float32 [x, y, z, intensity].
    AirSim LiDAR points are in sensor-local frame (NED-like x fwd, y right, z down).
    Convert to DAIR/KITTI frame (x fwd, y left, z up) by flipping Y and Z.
    Intensity is set to 1.0 if not provided.
    """
    try:
        ld = ctrl.getLidarData(lidar_name, vehicle_name)
    except TypeError:
        ld = ctrl.getLidarData(lidar_name)
    pts = np.array(ld.point_cloud, dtype=np.float32).reshape(-1, 3) if ld and len(ld.point_cloud) >= 3 else np.zeros((0,3), dtype=np.float32)

    if pts.shape[0] > 0:
        # AirSim -> DAIR/KITTI basis: flip Y and Z
        pts[:, 1] *= -1.0
        pts[:, 2] *= -1.0
        intensity = np.ones((pts.shape[0], 1), dtype=np.float32)
        out = np.hstack([pts.astype(np.float32), intensity])
    else:
        out = np.zeros((0,4), dtype=np.float32)

    out.tofile(path_bin)

def _save_rgb_and_lidar_for_frame(frame_id: str, ctrl_veh, ctrl_inf):
    # Ensure dirs exist
    _ensure_dirs()
    veh_img = os.path.join(DAIRV2X_C_ROOT, "vehicle-side", "image",    f"{frame_id}.png")
    veh_bin = os.path.join(DAIRV2X_C_ROOT, "vehicle-side", "velodyne", f"{frame_id}.bin")
    inf_img = os.path.join(DAIRV2X_C_ROOT, "infrastructure-side", "image",    f"{frame_id}.png")
    inf_bin = os.path.join(DAIRV2X_C_ROOT, "infrastructure-side", "velodyne", f"{frame_id}.bin")

    # Capture while paused (same instant)
    _capture_rgb_png(ctrl_veh, VEHICLE_SIDE_NAME, CAM_NAME, veh_img)
    _capture_lidar_bin(ctrl_veh, VEHICLE_SIDE_NAME, LIDAR_NAME, veh_bin)

    _capture_rgb_png(ctrl_inf, INFRA_SIDE_NAME, CAM_NAME, inf_img)
    _capture_lidar_bin(ctrl_inf, INFRA_SIDE_NAME, LIDAR_NAME, inf_bin)

# =========================
# Main
# =========================
def main():
    # Informational header based on SELECTED_VEHICLE
    _configure_from_vehicle(SELECTED_VEHICLE)

    # Create real control clients for BOTH sides (to manage pause/step)
    ctrl_veh = _client_for_vehicle(VEHICLE_SIDE_NAME)
    ctrl_inf = _client_for_vehicle(INFRA_SIDE_NAME)
    print("Connected to simulator (both sides). Starting detection loop...")

    # FrozenClient factories to block any internal unpause during detector.run()
    def make_frozen_client(ctrl):
        class FrozenClient:
            def __init__(self, *args, **kwargs):
                self._c = ctrl
            def simPause(self, is_paused: bool):
                if is_paused:
                    return self._c.simPause(True)
                return None  # swallow unpause attempts
            def __getattr__(self, name):
                return getattr(self._c, name)
        return FrozenClient

    FrozenVeh = make_frozen_client(ctrl_veh)
    FrozenInf = make_frozen_client(ctrl_inf)

    detector = H()

    # Pre-compute static lidar->camera mats (from settings.json, with DAIR basis)
    T_c_vl_veh, T_c_vl_inf = _compute_static_lidar2cam_mats()

    # Start paused on both controllers
    ctrl_veh.simPause(True)
    ctrl_inf.simPause(True)

    for t in range(N_STEPS):
        # KITTI-style frame id: zero-padded counter starting at 0
        frame_id = f"{t:06d}"

        # Write DAIR-style per-frame calib JSONs (same rigid transform each time)
        _write_per_frame_calib(frame_id, T_c_vl_veh, T_c_vl_inf)

        # Save RGB and LiDAR for BOTH sides at the SAME paused instant
        _save_rgb_and_lidar_for_frame(frame_id, ctrl_veh, ctrl_inf)

        print(f"\n=== Processing timestep {t} (frame_id={frame_id}) ===")
        ctrl_veh.simPause(True); ctrl_inf.simPause(True)

        # VEHICLE SIDE (visualization only; still paused)
        print(f"--- Vehicle-side: {VEHICLE_SIDE_NAME} ---")
        H.VEHICLE_NAME = VEHICLE_SIDE_NAME
        H.PORT = 41452  # informational; FrozenVeh ignores port
        H.CLIENT_CLASS = FrozenVeh
        if CAMERA_NAME_OVERRIDE:
            H.CAMERA_NAME = CAMERA_NAME_OVERRIDE
        detector.run()  # blocks until you close all VEHICLE windows

        # Still paused
        ctrl_veh.simPause(True); ctrl_inf.simPause(True)

        # INFRASTRUCTURE SIDE (visualization only; still paused)
        print(f"--- Infrastructure-side: {INFRA_SIDE_NAME} ---")
        H.VEHICLE_NAME = INFRA_SIDE_NAME
        H.PORT = 41451  # informational; FrozenInf ignores port
        H.CLIENT_CLASS = FrozenInf
        if CAMERA_NAME_OVERRIDE:
            H.CAMERA_NAME = CAMERA_NAME_OVERRIDE
        detector.run()  # blocks until you close all INFRA windows

        # Ensure paused after both sides
        ctrl_veh.simPause(True); ctrl_inf.simPause(True)

        # Advance exactly one step using ONLY the DRONE (infrastructure) client,
        # then pause again, unless last timestep
        if t < N_STEPS - 1:
            _advance_sim_once(ctrl_inf, ADVANCE_DT_SECONDS)  # step via DRONE client only
            ctrl_veh.simPause(True)  # keep both explicitly paused
        else:
            ctrl_veh.simPause(True); ctrl_inf.simPause(True)
            print("Completed all timesteps.")

if __name__ == "__main__":
    main()
