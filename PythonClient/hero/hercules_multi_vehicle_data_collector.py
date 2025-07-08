#!/usr/bin/env python3.10
"""
hercules_multi_vehicle_data_collector.py

Pauses the Cosys-AirSim sim globally via the multirotor client, steps it at a fixed dt,
then collects synchronized IMU, odometry, camera, and LiDAR data from multiple multirotor
drones and multiple Husky UGVs running on separate API ports.

Drone APIs (MultirotorClient) run on port 41451 and control sim stepping.
Husky UGV APIs (CarClient) run on port 41452 for data queries only.
"""

import setup_path
import os
import numpy as np
import cv2
import cosysairsim as airsim

# Configuration
DURATION      = 660.0        # seconds
dt_rate       = 200.0       # IMU rate (Hz)
DT            = 1.0 / dt_rate
OUTDIR        = "/media/sgarimella34/hercules-collect/raw_data_hercules/test1_2uav2ugv"

# Define vehicle names
# DRONE_NAMES   = ["Drone1", "Drone2", "Drone3"]
# HUSKY_NAMES   = ["Husky1", "Husky2", "Husky3"]

DRONE_NAMES   = ["Drone1", "Drone2"]
HUSKY_NAMES   = ["Husky1", "Husky2"]

CAMERA_NAME   = "front_center"
LIDAR_NAME    = "LidarSensor1"

# Ports
DRONE_PORT    = 41451
HUSKY_PORT    = 41452

# Instantiate clients
drone_client = airsim.MultirotorClient(port=DRONE_PORT)
husky_client = airsim.CarClient      (port=HUSKY_PORT)

drone_client.confirmConnection()
husky_client.confirmConnection()

# Enable API control for each vehicle
for name in DRONE_NAMES:
    drone_client.enableApiControl(True, vehicle_name=name)
for name in HUSKY_NAMES:
    husky_client.enableApiControl(True, vehicle_name=name)

# Pause the world (global)
drone_client.simPause(True)

# Prepare output directories and file handles
os.makedirs(OUTDIR, exist_ok=True)
files = {}
all_vehicles = DRONE_NAMES + HUSKY_NAMES
for name in all_vehicles:
    base = os.path.join(OUTDIR, name)
    os.makedirs(base, exist_ok=True)
    # open text files
    dict_handles = {
        'imu':  open(os.path.join(base, 'imu.txt'),  'w'),
        'odom': open(os.path.join(base, 'odom.txt'), 'w'),
    }
    # create subdirectories for images and LiDAR
    for folder in ('rgb', 'depth', 'seg', 'lidar'):
        path = os.path.join(base, folder)
        os.makedirs(path, exist_ok=True)
        dict_handles[folder] = path
    files[name] = dict_handles

# Subsampling parameters
odom_step  = int(round(dt_rate / 20.0))   # 20 Hz
cam_step   = odom_step                   # 20 Hz
lidar_step = int(round(dt_rate / 10.0))  # 10 Hz

total_steps = int(round(DURATION / DT))
print(f"Collecting {total_steps} steps @ {dt_rate:.0f}Hz (dt={DT:.4f}s) for {len(all_vehicles)} vehicles…")

# Main loop
for step in range(1, total_steps+1):
    # Step the sim once (drone client controls)
    drone_client.simContinueForTime(DT)
    t = step * DT

    # -- Drone data
    for name in DRONE_NAMES:
        client = drone_client
        # IMU
        imu = client.getImuData(vehicle_name=name)
        la, av = imu.linear_acceleration, imu.angular_velocity
        files[name]['imu'].write(
            f"{t:.6f} {la.x_val:.6f} {la.y_val:.6f} {la.z_val:.6f} "
            f"{av.x_val:.6f} {av.y_val:.6f} {av.z_val:.6f}\n"
        )
        # Odometry @ 20 Hz
        if step % odom_step == 0:
            st = client.getMultirotorState(vehicle_name=name)
            pos = st.kinematics_estimated.position
            ori = st.kinematics_estimated.orientation
            files[name]['odom'].write(
                f"{t:.6f} {pos.x_val:.6f} {pos.y_val:.6f} {pos.z_val:.6f} "
                f"{ori.w_val:.6f} {ori.x_val:.6f} {ori.y_val:.6f} {ori.z_val:.6f}\n"
            )
        # Cameras @ 20 Hz
        if step % cam_step == 0:
            reqs = [
                airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene,       False, False),
                airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.DepthPlanar, True,  False),
                airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Segmentation,False, False),
            ]
            imgs = client.simGetImages(reqs, vehicle_name=name)
            for img, key in zip(imgs, ('rgb', 'depth', 'seg')):
                if img.pixels_as_float:
                    arr = np.array(img.image_data_float, dtype=np.float32).reshape(img.height, img.width)
                    norm = ((arr - arr.min())/(arr.max()-arr.min()) * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(files[name][key], f"{t:.6f}.png"), norm)
                else:
                    arr = np.frombuffer(img.image_data_uint8, dtype=np.uint8).reshape(img.height, img.width, 3)
                    cv2.imwrite(os.path.join(files[name][key], f"{t:.6f}.png"), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        # LiDAR @ 10 Hz
        if step % lidar_step == 0:
            ld = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=name)
            pts = np.array(ld.point_cloud, dtype=np.float32).reshape(-1, 3)
            np.save(os.path.join(files[name]['lidar'], f"{t:.6f}.npy"), pts)

    # -- Husky data
    for name in HUSKY_NAMES:
        client = husky_client
        # IMU
        imu = client.getImuData(vehicle_name=name)
        la, av = imu.linear_acceleration, imu.angular_velocity
        files[name]['imu'].write(
            f"{t:.6f} {la.x_val:.6f} {la.y_val:.6f} {la.z_val:.6f} "
            f"{av.x_val:.6f} {av.y_val:.6f} {av.z_val:.6f}\n"
        )
        # Odometry @ 20 Hz
        if step % odom_step == 0:
            st = client.getCarState(vehicle_name=name)
            pos = st.kinematics_estimated.position
            ori = st.kinematics_estimated.orientation
            files[name]['odom'].write(
                f"{t:.6f} {pos.x_val:.6f} {pos.y_val:.6f} {pos.z_val:.6f} "
                f"{ori.w_val:.6f} {ori.x_val:.6f} {ori.y_val:.6f} {ori.z_val:.6f}\n"
            )
        # Cameras @ 20 Hz
        if step % cam_step == 0:
            reqs = [
                airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene,       False, False),
                airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.DepthPlanar, True,  False),
                airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Segmentation,False, False),
            ]
            imgs = client.simGetImages(reqs, vehicle_name=name)
            for img, key in zip(imgs, ('rgb', 'depth', 'seg')):
                if img.pixels_as_float:
                    arr = np.array(img.image_data_float, dtype=np.float32).reshape(img.height, img.width)
                    norm = ((arr - arr.min())/(arr.max()-arr.min()) * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(files[name][key], f"{t:.6f}.png"), norm)
                else:
                    arr = np.frombuffer(img.image_data_uint8, dtype=np.uint8).reshape(img.height, img.width, 3)
                    cv2.imwrite(os.path.join(files[name][key], f"{t:.6f}.png"), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        # LiDAR @ 10 Hz
        if step % lidar_step == 0:
            ld = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=name)
            pts = np.array(ld.point_cloud, dtype=np.float32).reshape(-1, 3)
            np.save(os.path.join(files[name]['lidar'], f"{t:.6f}.npy"), pts)

# Cleanup: resume sim and close files
drone_client.simPause(False)
for handles in files.values():
    handles['imu'].close()
    handles['odom'].close()

print("Done. Data saved under:", OUTDIR)
