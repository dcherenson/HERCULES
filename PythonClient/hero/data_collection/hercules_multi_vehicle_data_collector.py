#!/usr/bin/env python3.10
"""
hercules_multi_vehicle_data_collector.py

Pauses the Cosys-AirSim sim globally via the multirotor client, steps it at a fixed dt,
then collects synchronized IMU, odometry, camera, and LiDAR data from multiple multirotor
drones and multiple Husky UGVs running on separate API ports.

All settings are CLI flags whose defaults match the previous hardcoded values, so
running with no arguments behaves exactly as before:

  python3 hercules_multi_vehicle_data_collector.py \
      --outdir /media/.../raw_data_hercules/smalltown_test1 \
      --duration 1300 --drones Drone1 --huskies Husky1

On SIGINT/SIGTERM the collector finishes the current step, unpauses the sim and
closes all files cleanly, so an orchestrator may stop it early once trajectory
replay has finished.
"""

import setup_path
import argparse
import os
import signal
import numpy as np
import cv2
import hercules_cosysairsim as airsim

stop_requested = False


def _request_stop(signum, frame):
    global stop_requested
    stop_requested = True


def parse_args():
    p = argparse.ArgumentParser(description="Synchronized multi-vehicle HERCULES data collector")
    p.add_argument("--duration", type=float, default=1300.0, help="Sim seconds to record")
    p.add_argument("--imu-rate", type=float, default=200.0, help="Sim step / IMU rate (Hz)")
    p.add_argument("--outdir",
                   default="/media/sgarimella34/T74/Hercules_Datasets/raw_data_hercules/smalltown_test1",
                   help="Root output folder")
    p.add_argument("--drones", nargs="*", default=["Drone1"], help="Multirotor vehicle names")
    p.add_argument("--huskies", nargs="*", default=["Husky1"], help="UGV vehicle names")
    p.add_argument("--camera", default="front_center", help="Camera used for depth/seg")
    p.add_argument("--stereo-cameras", nargs="*", default=["stereo_left", "stereo_right"],
                   help="Stereo camera names for RGB")
    p.add_argument("--lidar", default="LidarSensor1", help="LiDAR sensor name")
    p.add_argument("--drone-port", type=int, default=41451)
    p.add_argument("--husky-port", type=int, default=41452)
    p.add_argument("--odom-hz", type=float, default=20.0)
    p.add_argument("--cam-hz", type=float, default=20.0)
    p.add_argument("--lidar-hz", type=float, default=20.0)
    p.add_argument("--save-depth-png", action="store_true",
                   help="Also write a visual 8-bit depth PNG")
    return p.parse_args()


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


def collect_vehicle(client, name, files, args, step, t, odom_step, cam_step, lidar_step, is_drone):
    # IMU
    imu = client.getImuData(vehicle_name=name)
    la, av = imu.linear_acceleration, imu.angular_velocity
    files[name]["imu"].write(
        f"{t:.6f} {la.x_val:.6f} {la.y_val:.6f} {la.z_val:.6f} "
        f"{av.x_val:.6f} {av.y_val:.6f} {av.z_val:.6f}\n"
    )

    # Odometry
    if step % odom_step == 0:
        if is_drone:
            st = client.getMultirotorState(vehicle_name=name)
        else:
            st = client.getCarState(vehicle_name=name)
        p = st.kinematics_estimated.position
        o = st.kinematics_estimated.orientation
        files[name]["odom"].write(
            f"{t:.6f} {p.x_val:.6f} {p.y_val:.6f} {p.z_val:.6f} "
            f"{o.w_val:.6f} {o.x_val:.6f} {o.y_val:.6f} {o.z_val:.6f}\n"
        )

    # Cameras
    if step % cam_step == 0:
        imgs = get_nonempty_images(client, name, args.camera)
        _, depth, seg = imgs

        depth_arr = np.array(depth.image_data_float, dtype=np.float32)\
                        .reshape(depth.height, depth.width)
        np.save(os.path.join(files[name]["depth"], f"{t:.6f}.npy"), depth_arr)

        if args.save_depth_png:
            depth_vis = np.clip(depth_arr, 0.0, 100.0) / 100.0
            depth_vis = (depth_vis * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(files[name]["depth"], f"{t:.6f}.png"), depth_vis)

        seg_img = np.frombuffer(seg.image_data_uint8, dtype=np.uint8)\
                    .reshape(seg.height, seg.width, 3)
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(files[name]["seg"], f"{t:.6f}.png"), seg_img)

        for stereo_cam in args.stereo_cameras:
            stereo_imgs = get_nonempty_images(client, name, stereo_cam)
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

    # LiDAR
    if step % lidar_step == 0:
        pts = get_nonempty_lidar(client, name, args.lidar)
        np.save(os.path.join(files[name]["lidar"], f"{t:.6f}.npy"), pts)


def main():
    args = parse_args()
    dt = 1.0 / args.imu_rate

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    # --- Setup clients ---
    drone_client = airsim.MultirotorClient(port=args.drone_port) if args.drones else None
    husky_client = airsim.CarClient(port=args.husky_port) if args.huskies else None
    if drone_client:
        drone_client.confirmConnection()
    if husky_client:
        husky_client.confirmConnection()

    for name in args.drones:
        drone_client.enableApiControl(True, vehicle_name=name)
    for name in args.huskies:
        husky_client.enableApiControl(True, vehicle_name=name)

    # Pause simulation globally (prefer drone client, as before)
    pause_client = drone_client or husky_client
    pause_client.simPause(True)

    # Prepare output dirs & files
    os.makedirs(args.outdir, exist_ok=True)
    files = {}
    all_vehicles = list(args.drones) + list(args.huskies)

    for v in all_vehicles:
        base = os.path.join(args.outdir, v)
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

    odom_step  = int(round(args.imu_rate / args.odom_hz))
    cam_step   = int(round(args.imu_rate / args.cam_hz))
    lidar_step = int(round(args.imu_rate / args.lidar_hz))

    total_steps = int(round(args.duration / dt))
    print(f"Collecting {total_steps} steps @ {args.imu_rate:.0f} Hz…", flush=True)

    steps_done = 0
    try:
        for step in range(1, total_steps + 1):
            if stop_requested:
                print("Stop requested — finalizing early.", flush=True)
                break

            pause_client.simContinueForTime(dt)
            t = step * dt

            for name in args.drones:
                collect_vehicle(drone_client, name, files, args, step, t,
                                odom_step, cam_step, lidar_step, is_drone=True)
            for name in args.huskies:
                collect_vehicle(husky_client, name, files, args, step, t,
                                odom_step, cam_step, lidar_step, is_drone=False)
            steps_done = step
    finally:
        pause_client.simPause(False)
        for v in all_vehicles:
            files[v]["imu"].close()
            files[v]["odom"].close()
        print(f"Done ({steps_done}/{total_steps} steps). Data saved under: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
