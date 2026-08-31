#!/usr/bin/env python3
"""Launch AirSim and run the original heterogeneous formation demo.

The launch flags intentionally remain compatible with the previous script,
while the actual startup and RPC ownership now live in the shared mission
runtime.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PythonClient"))
from distributed_mission.simulation.airsim_runtime import AirSimFacade, AirSimLaunchConfig, AirSimLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-mode", choices=("headless", "visible", "existing"), default="visible")
    parser.add_argument("--map", dest="map_name", choices=("flyingcpp", "rural_australia"), default="rural_australia")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; zero waits for q on an interactive terminal")
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--unreal-editor-path", default=AirSimLaunchConfig().unreal_editor_path)
    parser.add_argument("--uproject-path", default=None)
    parser.add_argument("--settings-path", default=None)
    parser.add_argument("--resx", type=int, default=800)
    parser.add_argument("--resy", type=int, default=600)
    parser.add_argument("--camera-height", type=float, default=30.0)
    parser.add_argument("--camera-x", type=float, default=6.0)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--no-top-down-camera", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AirSimLaunchConfig(
        launch_mode=args.launch_mode,
        map_name=args.map_name,
        unreal_editor_path=args.unreal_editor_path,
        uproject_path=args.uproject_path,
        resolution=(args.resx, args.resy),
        startup_timeout=args.startup_timeout,
        settings_path=args.settings_path,
        camera_director_position=None if args.no_top_down_camera else (args.camera_x, args.camera_y, -abs(args.camera_height)),
    )
    launcher = AirSimLauncher(config)
    facade = AirSimFacade(config)
    drones = ["Drone1", "Drone2", "SimpleFlight", "Drone4", "Drone5"]
    huskies = ["Husky1", "Husky2", "Husky3"]
    names = [(name, "drone") for name in drones] + [(name, "ugv") for name in huskies]
    try:
        launcher.launch()
        facade.connect()
        facade.multirotor.simRunConsoleCommand("DisableAllScreenMessages")
        if args.launch_mode == "existing" and not args.no_top_down_camera:
            print("Warning: --launch-mode existing cannot apply the top-down CameraDirector override; "
                  "configure it before launching Unreal.", flush=True)
        poses = {
            "Drone1": (-2, -2, -0.5), "Drone2": (2, -2, -0.5), "SimpleFlight": (0, 0, -0.5),
            "Drone4": (-2, 2, -0.5), "Drone5": (2, 2, -0.5),
            "Husky1": (0, 0, -1),
            "Husky2": (-2 * np.sqrt(3), -2, -1),
            "Husky3": (-2 * np.sqrt(3), 2, -1),
        }
        for name, vehicle_type in names:
            position = poses[name]
            pose = facade.airsim.Pose(facade.airsim.Vector3r(*position), facade.airsim.to_quaternion(0, 0, 0))
            facade.spawn_vehicle(name, vehicle_type, pose)
            facade.enable(name, vehicle_type, True)

        takeoff_futures = [facade.multirotor.takeoffAsync(vehicle_name=name) for name in drones]
        for future in takeoff_futures:
            future.join()

        altitude_futures = []
        for name in drones:
            altitude_futures.append(
                facade.multirotor.moveToPositionAsync(
                    poses[name][0], poses[name][1], -5, 5.0, vehicle_name=name
                )
            )
        for future in altitude_futures:
            future.join()

        started = time.monotonic()
        tick = 0.0
        while True:
            vx, vy = 2.0 * np.cos(tick), 2.0 * np.sin(tick)
            for name in drones:
                facade.multirotor.moveByVelocityZAsync(vx, vy, -5, 0.2, vehicle_name=name)
            for name in huskies:
                facade.command_ugv(name, 2.0, 0.5)
            tick += 0.1
            if args.duration > 0 and time.monotonic() - started >= args.duration:
                break
            if args.duration == 0 and sys.stdin.isatty():
                import select
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready and sys.stdin.readline().strip().lower() == "q":
                    break
            elif args.duration == 0:
                time.sleep(0.1)
    finally:
        facade.close(names)
        launcher.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
