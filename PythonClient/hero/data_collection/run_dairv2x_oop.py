#!/usr/bin/env python3
import os, json, math, time, traceback
import numpy as np
import cv2
import setup_path
import cosysairsim as airsim

from Hercules2D3DDetector import Hercules2D3DDetector as H

# ===================== CONFIG =====================
DRONE_PORT = 41451
HUSKY_PORT = 41452

SIDE_INF = ["Drone1"]   # UAV / infrastructure
SIDE_VEH = ["Husky1"]   # UGV / vehicle

CAM_NAME   = H.CAMERA_NAME         # use your class default ("front_center")
LIDAR_NAME = "LidarSensor1"

# DAIR-V2X-C defaults (10 Hz cam/lidar)
CAM_RATE   = 10
LIDAR_RATE = 10
DURATION_S = 300.0

OUT_ROOT   = "/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets/dair_v2x_synth/"

IMG_EXT    = ".png"
BASE_HZ    = CAM_RATE
DT         = 1.0 / BASE_HZ
CAM_EVERY  = BASE_HZ // CAM_RATE
LIDAR_EVERY = BASE_HZ // LIDAR_RATE

# ------------------ OUTPUT LAYOUT ------------------
def dair_paths(root):
    return {
        "inf": {
            "img":         f"{root}/cooperative/infrastructure-side/image",
            "depth":       f"{root}/cooperative/infrastructure-side/depth",
            "seg":         f"{root}/cooperative/infrastructure-side/seg",
            "lidar":       f"{root}/cooperative/infrastructure-side/lidar",
            "calib":       f"{root}/cooperative/infrastructure-side/calib",
            "ts":          f"{root}/cooperative/infrastructure-side/timestamp",
            "kitti_label": f"{root}/cooperative/infrastructure-side/kitti_label"
        },
        "veh": {
            "img":         f"{root}/cooperative/vehicle-side/image",
            "depth":       f"{root}/cooperative/vehicle-side/depth",
            "seg":         f"{root}/cooperative/vehicle-side/seg",
            "lidar":       f"{root}/cooperative/vehicle-side/lidar",
            "calib":       f"{root}/cooperative/vehicle-side/calib",
            "ts":          f"{root}/cooperative/vehicle-side/timestamp",
            "kitti_label": f"{root}/cooperative/vehicle-side/kitti_label"
        },
        "label": {
            "veh":         f"{root}/cooperative/label/vehicle",
            "inf":         f"{root}/cooperative/label/infrastructure",
            "cooperative": f"{root}/cooperative/label/cooperative"
        }
    }

PATHS = dair_paths(OUT_ROOT)
for group in PATHS.values():
    for p in group.values():
        os.makedirs(p, exist_ok=True)

# ------------------ HELPERS ------------------
def get_images(client, vehicle_name):
    """Return (BGR uint8), (depth_m float32), (seg BGR uint8)."""
    reqs = [
        airsim.ImageRequest(CAM_NAME, airsim.ImageType.Scene,       False, False),
        airsim.ImageRequest(CAM_NAME, airsim.ImageType.DepthPlanar, True,  False),
        airsim.ImageRequest(CAM_NAME, airsim.ImageType.Segmentation,False, False),
    ]
    while True:
        imgs = client.simGetImages(reqs, vehicle_name=vehicle_name)
        if len(imgs) != 3: 
            continue
        scene, depth, seg = imgs
        if scene.width <= 0 or scene.height <= 0:
            continue
        rgb = np.frombuffer(scene.image_data_uint8, dtype=np.uint8).reshape(scene.height, scene.width, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_m = np.array(depth.image_data_float, dtype=np.float32).reshape(depth.height, depth.width)
        seg_rgb = np.frombuffer(seg.image_data_uint8, dtype=np.uint8).reshape(seg.height, seg.width, 3)
        seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)
        return bgr, depth_m, seg_bgr

def get_lidar_points(client, vehicle_name):
    ld = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=vehicle_name)
    if ld and ld.point_cloud:
        pts = np.array(ld.point_cloud, dtype=np.float32).reshape(-1, 3)
        if pts.size:
            return pts
    return None

def save_lidar_bin(path, xyz):
    """KITTI .bin with x,y,z,1.0 float32."""
    N = xyz.shape[0]
    arr = np.hstack([xyz.astype(np.float32), np.ones((N,1), dtype=np.float32)])
    arr.astype(np.float32).tofile(path)

def build_calib_json(client, side_name):
    """
    Get K and P from your class using FOV + a single image for size.
    AirSim CameraInfo has FOV and pose but not width/height, so we read one frame for w,h.
    """
    img, depth, seg = get_images(client, side_name)
    h, w = img.shape[:2]

    info = client.simGetCameraInfo(CAM_NAME, vehicle_name=side_name)
    K, _vfov = H.compute_intrinsics_from_horizontal_fov(info.fov, w, h)  # class static method
    P = np.array(info.proj_mat.matrix, dtype=np.float64).reshape((4,4))
    if not np.isfinite(P).all() or np.allclose(P, 0):
        P = np.eye(4, dtype=float); P[:3,:3] = K  # fallback

    return {"K": K, "P": P, "image_size": (w, h)}

def write_calib_json(path, calib):
    out = {
        "K": calib["K"].tolist(),
        "P": calib["P"].tolist(),
        "image_size": list(map(int, calib["image_size"])),
        "camera_name": CAM_NAME,
        "lidar_name":  LIDAR_NAME,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def kitti_json_from_result(res, cam_pose, P, img_size):
    """
    Build a DAIR-V2X/KITTI-style entry from one processed target.
    Prefer the tight 2D box; fall back to amodal; if neither present, rebuild
    from world cuboid corners using THE CLASS projector (H.project_world_points_to_image),
    which matches AirSim’s 4x4 OpenGL-style P convention.
    """
    if not res.get("found", False):
        return None

    # Prefer boxes from your class if you later expose them
    box = res.get("tight_bbox_xyxy") or res.get("amodal_bbox_xyxy")

    # If class didn’t return a 2D box, rebuild from corners using H.project_world_points_to_image
    if box is None:
        corners_w = res.get("corners_w", None)
        if corners_w is None:
            return None

        w_img, h_img = img_size  # do NOT shadow the class alias H
        pts2d, depth_forward, valid = H.project_world_points_to_image(
            np.asarray(corners_w, dtype=float), cam_pose, P, w_img, h_img
        )
        u, v = pts2d[:, 0], pts2d[:, 1]
        in_bounds = (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)
        use = valid & in_bounds
        if not np.any(use):
            return None
        x0 = int(max(0, np.floor(u[use].min())))
        y0 = int(max(0, np.floor(v[use].min())))
        x1 = int(min(w_img - 1, np.ceil(u[use].max())))
        y1 = int(min(h_img - 1, np.ceil(v[use].max())))
        if not (x1 > x0 and y1 > y0):
            return None
        box = (x0, y0, x1, y1)

    bx0, by0, bx1, by1 = map(int, box)

    # 3D dims from your class
    Hh, Wd, Ld = float(res["H"]), float(res["W"]), float(res["L"])

    # 3D location and yaw in camera frame (AirSim: X forward, Y right, Z down)
    R_cam = H.quaternion_to_rotation_matrix(cam_pose.orientation)
    cam_p = np.array([cam_pose.position.x_val,
                      cam_pose.position.y_val,
                      cam_pose.position.z_val], dtype=float)

    adj = res["adjusted_pose"]
    obj_c = np.array([adj.position.x_val,
                      adj.position.y_val,
                      adj.position.z_val], dtype=float)
    p_cam = R_cam.T @ (obj_c - cam_p)

    R_obj = H.quaternion_to_rotation_matrix(adj.orientation)
    heading_cam = R_cam.T @ R_obj @ np.array([1, 0, 0], dtype=float)
    rot_yaw = math.atan2(heading_cam[1], heading_cam[0])

    # Pedestrian vs Car (robust)
    lbl = str(res.get("label", "")).lower()
    if ("human" in lbl) or ("pedestrian" in lbl):
        label_type = "Pedestrian"
    else:
        label_type = "Pedestrian" if H.infer_object_type_from_label(res.get("label","")) == "human" else "Car"

    return {
        "type": label_type,
        "occluded_state": 0,
        "truncated_state": 0,
        "alpha": float(rot_yaw),
        "2d_box": {"xmin": float(bx0), "ymin": float(by0), "xmax": float(bx1), "ymax": float(by1)},
        "3d_dimensions": {"h": float(Hh), "w": float(Wd), "l": float(Ld)},
        "3d_location": {"x": float(p_cam[0]), "y": float(p_cam[1]), "z": float(p_cam[2])},
        "rotation": float(rot_yaw)
    }



# ------------------ MAIN ------------------
def main():
    # Clients
    drone_client = airsim.MultirotorClient(port=DRONE_PORT)
    car_client   = airsim.CarClient(port=HUSKY_PORT)
    print("[STARTUP] connecting…")
    drone_client.confirmConnection()
    car_client.confirmConnection()
    for n in SIDE_INF: drone_client.enableApiControl(True, vehicle_name=n)
    for n in SIDE_VEH: car_client.enableApiControl(True, vehicle_name=n)
    print("[STARTUP] connected.")

    # Pause and set small detection radii if you use simGetDetections elsewhere (optional)
    drone_client.simPause(True)

    # Calibs (use class method for K)
    veh_calib = build_calib_json(car_client,   SIDE_VEH[0])
    inf_calib = build_calib_json(drone_client, SIDE_INF[0])
    write_calib_json(os.path.join(PATHS["veh"]["calib"], "calib.json"), veh_calib)
    write_calib_json(os.path.join(PATHS["inf"]["calib"], "calib.json"), inf_calib)

    # Actor map
    id_to_label = H.load_actor_map(H.CSV_PATH)   # from your class config

    veh_ts_list, inf_ts_list = [], []

    num_ticks = int(round(DURATION_S / DT))
    for tick in range(1, num_ticks+1):
        # Advance one fixed slice WHILE PAUSED → all sensors & poses are from the same time
        try:
            drone_client.simContinueForTime(DT)  # one step; sim remains paused after
        except Exception as e:
            print(f"[ERROR] sim step failed at tick {tick}: {e}")
            continue

        t_ms = int(round(tick * DT * 1000.0))
        if tick % 100 == 0:
            print(f"[COLLECT] tick {tick}/{num_ticks}")

        # === VEHICLE SIDE ===
        veh_img = veh_depth = veh_seg = None
        veh_pts = None

        if tick % CAM_EVERY == 0:
            veh_img, veh_depth, veh_seg = get_images(car_client, SIDE_VEH[0])
            cv2.imwrite(os.path.join(PATHS["veh"]["img"],   f"{t_ms}{IMG_EXT}"), veh_img)
            np.save    (os.path.join(PATHS["veh"]["depth"], f"{t_ms}.npy"),      veh_depth)
            cv2.imwrite(os.path.join(PATHS["veh"]["seg"],   f"{t_ms}{IMG_EXT}"), veh_seg)

        if tick % LIDAR_EVERY == 0:
            veh_pts = get_lidar_points(car_client, SIDE_VEH[0])
            if veh_pts is not None:
                save_lidar_bin(os.path.join(PATHS["veh"]["lidar"], f"{t_ms}.bin"), veh_pts)
                veh_ts_list.append(t_ms)

        # === INFRA SIDE ===
        inf_img = inf_depth = inf_seg = None
        inf_pts = None

        if tick % CAM_EVERY == 0:
            inf_img, inf_depth, inf_seg = get_images(drone_client, SIDE_INF[0])
            cv2.imwrite(os.path.join(PATHS["inf"]["img"],   f"{t_ms}{IMG_EXT}"), inf_img)
            np.save    (os.path.join(PATHS["inf"]["depth"], f"{t_ms}.npy"),      inf_depth)
            cv2.imwrite(os.path.join(PATHS["inf"]["seg"],   f"{t_ms}{IMG_EXT}"), inf_seg)

        if tick % LIDAR_EVERY == 0:
            inf_pts = get_lidar_points(drone_client, SIDE_INF[0])
            if inf_pts is not None:
                save_lidar_bin(os.path.join(PATHS["inf"]["lidar"], f"{t_ms}.bin"), inf_pts)
                inf_ts_list.append(t_ms)

        # === Label both sides on the SAME paused step ===
        if (tick % CAM_EVERY == 0):
            # Camera infos/poses at this same paused step
            veh_info = car_client.simGetCameraInfo(CAM_NAME,   vehicle_name=SIDE_VEH[0])
            inf_info = drone_client.simGetCameraInfo(CAM_NAME, vehicle_name=SIDE_INF[0])
            cam_pose_veh = veh_info.pose
            cam_pose_inf = inf_info.pose

            # Pull P & sizes from our earlier calibs
            P_veh = veh_calib["P"]; W_veh, H_veh = veh_calib["image_size"]
            P_inf = inf_calib["P"]; W_inf, H_inf = inf_calib["image_size"]

            # Build targets via your class (CSV + scene + FOV/range gating)
            veh_targets = H.build_targets_from_csv_scene(car_client,   id_to_label, cam_pose_veh, P_veh, W_veh, H_veh)
            inf_targets = H.build_targets_from_csv_scene(drone_client, id_to_label, cam_pose_inf, P_inf, W_inf, H_inf)

            # Process each target using your per-target pipeline
            veh_labels = []
            if veh_img is not None:
                for tgt in veh_targets:
                    res = H.process_target(tgt, car_client, veh_info, cam_pose_veh, veh_img, veh_seg, veh_depth, P_veh)
                    kj = kitti_json_from_result(res, cam_pose_veh, P_veh, (W_veh, H_veh))
                    if kj is not None:
                        veh_labels.append(kj)

            inf_labels = []
            if inf_img is not None:
                for tgt in inf_targets:
                    res = H.process_target(tgt, drone_client, inf_info, cam_pose_inf, inf_img, inf_seg, inf_depth, P_inf)
                    kj = kitti_json_from_result(res, cam_pose_inf, P_inf, (W_inf, H_inf))
                    if kj is not None:
                        inf_labels.append(kj)

            # Write per-timestamp label JSONs (mirrored per side, and “label/{veh,inf}” convenience)
            with open(os.path.join(PATHS["veh"]["kitti_label"], f"{t_ms:06d}.json"), "w") as f:
                json.dump(veh_labels, f)
            with open(os.path.join(PATHS["inf"]["kitti_label"], f"{t_ms:06d}.json"), "w") as f:
                json.dump(inf_labels, f)
            with open(os.path.join(PATHS["label"]["veh"], f"{t_ms:06d}.json"), "w") as f:
                json.dump(veh_labels, f)
            with open(os.path.join(PATHS["label"]["inf"], f"{t_ms:06d}.json"), "w") as f:
                json.dump(inf_labels, f)

    # timestamps
    with open(os.path.join(PATHS["veh"]["ts"], "timestamp.txt"), "w") as f:
        for t in veh_ts_list: f.write(f"{t}\n")
    with open(os.path.join(PATHS["inf"]["ts"], "timestamp.txt"), "w") as f:
        for t in inf_ts_list: f.write(f"{t}\n")

    # Unpause
    drone_client.simPause(False)
    print("[DONE] Collection complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
