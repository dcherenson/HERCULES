#!/usr/bin/env python3.10
"""
hercules_multi_vehicle_data_collector.py

Pauses the Cosys-AirSim sim globally via the multirotor client, steps it at a fixed dt,
then collects synchronized IMU, odometry, camera, and LiDAR data from multiple multirotor
drones and multiple Husky UGVs running on separate API ports.
"""

import setup_path
import os
import numpy as np
import cv2
import cosysairsim as airsim

# Configuration
# DURATION        = 1200.0        # seconds
DURATION        = 700.0        # seconds
DT_RATE         = 200.0         # IMU rate (Hz)
# DT_RATE         = 20.0         # setting this since we are using synthetic imu data anyway
DT              = 1.0 / DT_RATE
# OUTDIR          = "/media/sgarimella34/hercules-collect/raw_data_hercules/forest_stereo_lodfix_centerseq_2ugvuav_752x480"
OUTDIR = "/media/sgarimella34/SSD2/raw_data_hercules/ausenv_stereo_lidarfix_perimeterseq_2ugvuav_752x480"
SAVE_DEPTH_PNG  = True          # if True, also write a visual 8-bit PNG

DRONE_NAMES   = ["Drone1", "Drone2"]
HUSKY_NAMES   = ["Husky1", "Husky2"]

# DRONE_NAMES   = ["Drone1"]
# HUSKY_NAMES   = []

# Use front_center only for depth/seg; stereo_* for RGB
CAMERA_NAME          = "front_center"
STEREO_CAMERA_NAMES  = ["stereo_left", "stereo_right"]
LIDAR_NAME           = "LidarSensor1"

DRONE_PORT    = 41451
HUSKY_PORT    = 41452

# --- Setup clients ---
drone_client = airsim.MultirotorClient(port=DRONE_PORT)
husky_client = airsim.CarClient(port=HUSKY_PORT)
drone_client.confirmConnection()
husky_client.confirmConnection()

for name in DRONE_NAMES:
    drone_client.enableApiControl(True, vehicle_name=name)
for name in HUSKY_NAMES:
    husky_client.enableApiControl(True, vehicle_name=name)

# Pause simulation globally via the drone client
drone_client.simPause(True)

# Prepare output dirs & files
os.makedirs(OUTDIR, exist_ok=True)
files = {}
all_vehicles = DRONE_NAMES + HUSKY_NAMES

for v in all_vehicles:
    base = os.path.join(OUTDIR, v)
    os.makedirs(base, exist_ok=True)
    files[v] = {
        "imu":  open(os.path.join(base, "imu.txt"),  "w"),
        "odom": open(os.path.join(base, "odom.txt"), "w"),
        "rgb": os.path.join(base, "rgb"),
        "rgb_stereo_left":  os.path.join(base, "rgb_stereo_left"),
        "rgb_stereo_right": os.path.join(base, "rgb_stereo_right"),
        "depth": os.path.join(base, "depth"),
        "seg":   os.path.join(base, "seg"),
        "lidar": os.path.join(base, "lidar"),
    }
    for sub in ("rgb", "rgb_stereo_left", "rgb_stereo_right", "depth", "seg", "lidar"):
        os.makedirs(files[v][sub], exist_ok=True)

# Sampling rates
odom_step  = int(round(DT_RATE / 20.0))  # 20 Hz
cam_step   = odom_step                   # 20 Hz
lidar_step = int(round(DT_RATE / 10.0))  # 10 Hz

total_steps = int(round(DURATION / DT))
print(f"Collecting {total_steps} steps @ {DT_RATE:.0f} Hz…")

def get_nonempty_images(client, vehicle_name, camera_name):
    """Retry simGetImages until we get valid Scene, DepthPlanar, Segmentation."""
    reqs = [
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene,        False, False),
        airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar,  True,  False),
        airsim.ImageRequest(camera_name, airsim.ImageType.Segmentation, False, False),
    ]
    while True:
        imgs = client.simGetImages(reqs, vehicle_name=vehicle_name)
        if all(
            img.width > 0 and img.height > 0 and
            (img.pixels_as_float or len(img.image_data_uint8) > 0)
            for img in imgs
        ):
            return imgs

def get_nonempty_lidar(client, vehicle_name, lidar_name):
    """Retry getLidarData until we get nonempty point_cloud."""
    while True:
        ld = client.getLidarData(lidar_name=lidar_name, vehicle_name=vehicle_name)
        if ld.point_cloud:
            pts = np.array(ld.point_cloud, dtype=np.float32).reshape(-1, 3)
            if pts.size:
                return pts

# Main loop
for step in range(1, total_steps + 1):
    # 1) step sim forward
    drone_client.simContinueForTime(DT)
    t = step * DT

    # 2) multirotors
    for name in DRONE_NAMES:
        c = drone_client

        # IMU
        imu = c.getImuData(vehicle_name=name)
        la, av = imu.linear_acceleration, imu.angular_velocity
        files[name]["imu"].write(
            f"{t:.6f} {la.x_val:.6f} {la.y_val:.6f} {la.z_val:.6f} "
            f"{av.x_val:.6f} {av.y_val:.6f} {av.z_val:.6f}\n"
        )

        # Odometry @ 20 Hz
        if step % odom_step == 0:
            st = c.getMultirotorState(vehicle_name=name)
            p = st.kinematics_estimated.position
            o = st.kinematics_estimated.orientation
            files[name]["odom"].write(
                f"{t:.6f} {p.x_val:.6f} {p.y_val:.6f} {p.z_val:.6f} "
                f"{o.w_val:.6f} {o.x_val:.6f} {o.y_val:.6f} {o.z_val:.6f}\n"
            )

        # Cameras @ 20 Hz
        if step % cam_step == 0:
            imgs = get_nonempty_images(c, name, CAMERA_NAME)
            _, depth, seg = imgs

            # Depth
            depth_arr = np.array(depth.image_data_float, dtype=np.float32)\
                            .reshape(depth.height, depth.width)
            np.save(os.path.join(files[name]["depth"], f"{t:.6f}.npy"), depth_arr)

            if SAVE_DEPTH_PNG:
                depth_vis = np.clip(depth_arr, 0.0, 100.0) / 100.0
                depth_vis = (depth_vis * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(files[name]["depth"], f"{t:.6f}.png"), depth_vis)

            # Segmentation
            seg_img = np.frombuffer(seg.image_data_uint8, dtype=np.uint8)\
                        .reshape(seg.height, seg.width, 3)
            seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(files[name]["seg"], f"{t:.6f}.png"), seg_img)

            # Stereo RGB
            for stereo_cam in STEREO_CAMERA_NAMES:
                stereo_imgs = get_nonempty_images(c, name, stereo_cam)
                stereo_scene = stereo_imgs[0]
                stereo_rgb = np.frombuffer(stereo_scene.image_data_uint8, dtype=np.uint8)\
                                .reshape(stereo_scene.height, stereo_scene.width, 3)
                stereo_rgb = cv2.cvtColor(stereo_rgb, cv2.COLOR_RGB2BGR)

                out_dir = (
                    files[name]["rgb_stereo_left"]
                    if stereo_cam == "stereo_left"
                    else files[name]["rgb_stereo_right"]
                )
                cv2.imwrite(os.path.join(out_dir, f"{t:.6f}.png"), stereo_rgb)

        # LiDAR @ 10 Hz
        if step % lidar_step == 0:
            pts = get_nonempty_lidar(c, name, LIDAR_NAME)
            np.save(os.path.join(files[name]["lidar"], f"{t:.6f}.npy"), pts)

    # 3) huskies
    for name in HUSKY_NAMES:
        c = husky_client

        # IMU
        imu = c.getImuData(vehicle_name=name)
        la, av = imu.linear_acceleration, imu.angular_velocity
        files[name]["imu"].write(
            f"{t:.6f} {la.x_val:.6f} {la.y_val:.6f} {la.z_val:.6f} "
            f"{av.x_val:.6f} {av.y_val:.6f} {av.z_val:.6f}\n"
        )

        # Odometry @ 20 Hz
        if step % odom_step == 0:
            st = c.getCarState(vehicle_name=name)
            p = st.kinematics_estimated.position
            o = st.kinematics_estimated.orientation
            files[name]["odom"].write(
                f"{t:.6f} {p.x_val:.6f} {p.y_val:.6f} {p.z_val:.6f} "
                f"{o.w_val:.6f} {o.x_val:.6f} {o.y_val:.6f} {o.z_val:.6f}\n"
            )

        # Cameras @ 20 Hz
        if step % cam_step == 0:
            imgs = get_nonempty_images(c, name, CAMERA_NAME)
            _, depth, seg = imgs

            depth_arr = np.array(depth.image_data_float, dtype=np.float32)\
                            .reshape(depth.height, depth.width)
            np.save(os.path.join(files[name]["depth"], f"{t:.6f}.npy"), depth_arr)

            if SAVE_DEPTH_PNG:
                depth_vis = np.clip(depth_arr, 0.0, 100.0) / 100.0
                depth_vis = (depth_vis * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(files[name]["depth"], f"{t:.6f}.png"), depth_vis)

            seg_img = np.frombuffer(seg.image_data_uint8, dtype=np.uint8)\
                        .reshape(seg.height, seg.width, 3)
            seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(files[name]["seg"], f"{t:.6f}.png"), seg_img)

            for stereo_cam in STEREO_CAMERA_NAMES:
                stereo_imgs = get_nonempty_images(c, name, stereo_cam)
                stereo_scene = stereo_imgs[0]
                stereo_rgb = np.frombuffer(stereo_scene.image_data_uint8, dtype=np.uint8)\
                                .reshape(stereo_scene.height, stereo_scene.width, 3)
                stereo_rgb = cv2.cvtColor(stereo_rgb, cv2.COLOR_RGB2BGR)

                out_dir = (
                    files[name]["rgb_stereo_left"]
                    if stereo_cam == "stereo_left"
                    else files[name]["rgb_stereo_right"]
                )
                cv2.imwrite(os.path.join(out_dir, f"{t:.6f}.png"), stereo_rgb)

        if step % lidar_step == 0:
            pts = get_nonempty_lidar(c, name, LIDAR_NAME)
            np.save(os.path.join(files[name]["lidar"], f"{t:.6f}.npy"), pts)

# finalize
drone_client.simPause(False)
for v in all_vehicles:
    files[v]["imu"].close()
    files[v]["odom"].close()

print("Done. Data saved under:", OUTDIR)
