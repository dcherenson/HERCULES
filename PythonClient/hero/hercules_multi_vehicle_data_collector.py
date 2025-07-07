#!/usr/bin/env python3.10
"""
hercules_multi_vehicle_data_collector.py

Pauses the Cosys-AirSim sim globally via the multirotor client, steps it at a fixed dt,
then collects synchronized IMU, odometry, camera, and LiDAR data from both a multirotor
and a Husky UGV running on separate API ports.

Drone APIs (MultirotorClient) run on port 41451 and control sim stepping.
Husky UGV APIs (CarClient) run on port 41452 for data queries only.
"""

import setup_path
import os
import numpy as np
import cv2
import cosysairsim as airsim

# Configuration
DURATION      = 60.0        # seconds
OUTDIR        = "/media/sgarimella34/hercules-collect/raw_data_hercules/test2_2uav2ugv"   # root output directory
DT            = 1.0 / 200.0 # 200 Hz
DRONE_NAME    = "Drone1"
HUSKY_NAME    = "Husky1"
CAMERA_NAME   = "front_center"
LIDAR_NAME    = "LidarSensor1"
DRONE_PORT    = 41451
HUSKY_PORT    = 41452

# instantiate clients
drone_client = airsim.MultirotorClient(port=DRONE_PORT)
husky_client = airsim.CarClient      (port=HUSKY_PORT)

# connect & enable
drone_client.confirmConnection()
husky_client. confirmConnection()

for client, name in [(drone_client, DRONE_NAME), (husky_client, HUSKY_NAME)]:
    client.enableApiControl(True, vehicle_name=name)

# pause the world (global)
drone_client.simPause(True)

# prepare output structure
os.makedirs(OUTDIR, exist_ok=True)
files = {}
for name in (DRONE_NAME, HUSKY_NAME):
    sub = os.path.join(OUTDIR, name)
    os.makedirs(sub, exist_ok=True)
    files[name] = {
        'imu':  open(os.path.join(sub, 'imu.txt'),  'w'),
        'odom': open(os.path.join(sub, 'odom.txt'), 'w'),
        'rgb':  os.path.join(sub, 'rgb'),
        'depth':os.path.join(sub, 'depth'),
        'seg':  os.path.join(sub, 'seg'),
        'lidar':os.path.join(sub, 'lidar')
    }
    for d in ('rgb','depth','seg','lidar'):
        os.makedirs(files[name][d], exist_ok=True)

print(f"Starting data collection for {DURATION}s at 200Hz (dt={DT:.4f})…")
total_steps = int(round(DURATION / DT))

for step in range(1, total_steps+1):
    # step the sim once (drone client controls)
    drone_client.simContinueForTime(DT)
    t = step * DT

    # collect from each vehicle
    for client, name in [(drone_client, DRONE_NAME), (husky_client, HUSKY_NAME)]:
        # IMU
        imu = client.getImuData(vehicle_name=name)
        la, av = imu.linear_acceleration, imu.angular_velocity
        files[name]['imu'].write(
            f"{t:.6f} {la.x_val:.6f} {la.y_val:.6f} {la.z_val:.6f} "
            f"{av.x_val:.6f} {av.y_val:.6f} {av.z_val:.6f}\n"
        )

        # Odometry @ 20Hz
        if step % int(200/20) == 0:
            if isinstance(client, airsim.CarClient):
                st = client.getCarState(vehicle_name=name)
            else:
                st = client.getMultirotorState(vehicle_name=name)
            pos = st.kinematics_estimated.position
            ori = st.kinematics_estimated.orientation
            files[name]['odom'].write(
                f"{t:.6f} {pos.x_val:.6f} {pos.y_val:.6f} {pos.z_val:.6f} "
                f"{ori.w_val:.6f} {ori.x_val:.6f} {ori.y_val:.6f} {ori.z_val:.6f}\n"
            )

        # Cameras @ 20Hz
        if step % int(200/20) == 0:
            reqs = [
                airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene,       False, False),
                airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.DepthPlanar, True,  False),
                airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Segmentation,False, False),
            ]
            imgs = client.simGetImages(reqs, vehicle_name=name)
            for img, kind in zip(imgs, ('rgb','depth','seg')):
                # simple save helper
                arr = (np.array(img.image_data_float, dtype=np.float32).reshape(img.height, img.width)
                       if img.pixels_as_float
                       else np.frombuffer(img.image_data_uint8, dtype=np.uint8).reshape(img.height, img.width, 3))
                ext = '.png'
                path = os.path.join(files[name][kind], f"{t:.6f}{ext}")
                if img.pixels_as_float:
                    norm = ((arr - arr.min())/(arr.max()-arr.min()) * 255).astype(np.uint8)
                    cv2.imwrite(path, norm)
                else:
                    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

        # LiDAR @ 10Hz
        if step % int(200/10) == 0:
            ld = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=name)
            pts = np.array(ld.point_cloud, dtype=np.float32).reshape(-1, 3)
            np.save(os.path.join(files[name]['lidar'], f"{t:.6f}.npy"), pts)

# cleanup
drone_client.simPause(False)
for fdict in files.values():
    fdict['imu'].close()
    fdict['odom'].close()

print("Done. Data saved under:", OUTDIR)
