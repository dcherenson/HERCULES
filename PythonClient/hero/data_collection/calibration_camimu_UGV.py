#!/usr/bin/env python3

import setup_path
import hercules_cosysairsim as airsim
import argparse
import time
import numpy as np
from multiprocessing import Process

# Default UGV names from settings.json (override with --ugvs)
UGV_NAMES = ["Husky1"]
UGV_PORT = 41452

# Speed regulation parameters
TARGET_SPEED = 0.3    # m/s magnitude
KP           = 1.0    # proportional gain
DEADBAND     = 0.02   # m/s deadband around target
MAX_THROTTLE = 0.5    # throttle limit
DT           = 0.05   # control loop interval

def drive_distance(client, vehicle_name, distance, steering=0.0):
    """
    Drives 'distance' meters (positive=forward, negative=reverse) at ~TARGET_SPEED,
    using manual_gear for direction and a P-controller on signed speed.
    """
    controls = airsim.CarControls()
    controls.is_manual_gear = True
    gear = 1 if distance >= 0 else -1
    controls.manual_gear = gear
    controls.steering    = steering
    controls.brake       = 0.0  # never use brake in this controller

    # record start
    start = client.simGetVehiclePose(vehicle_name=vehicle_name).position
    start_xy = np.array([start.x_val, start.y_val])
    target = abs(distance)

    while True:
        state = client.getCarState(vehicle_name=vehicle_name)
        speed = state.speed                         # always ≥ 0
        signed_speed = speed * gear                  # positive in commanded direction

        error = TARGET_SPEED - signed_speed          # how far we are below target

        # compute throttle only
        if error > DEADBAND:
            throttle_cmd = min(KP * error, MAX_THROTTLE)
        else:
            throttle_cmd = 0.0

        controls.throttle = throttle_cmd
        # no brake at all
        client.setCarControls(controls, vehicle_name=vehicle_name)

        time.sleep(DT)

        # check travel
        pos = client.simGetVehiclePose(vehicle_name=vehicle_name).position
        traveled = np.linalg.norm([pos.x_val - start_xy[0],
                                   pos.y_val - start_xy[1]])
        if traveled >= target:
            break

    # clean stop
    controls.throttle = 0.0
    controls.brake    = 1.0
    client.setCarControls(controls, vehicle_name=vehicle_name)
    time.sleep(0.5)
    controls.brake = 0.0
    client.setCarControls(controls, vehicle_name=vehicle_name)
    time.sleep(0.2)

def run_ugv_calibration_motion(vehicle_name,
                               segment_count=4,
                               total_dist=5.0,
                               steer_amp=0.4,
                               port=UGV_PORT):
    client = airsim.CarClient(port=port)
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name=vehicle_name)
    try:
        client.armDisarm(True, vehicle_name=vehicle_name)
    except:
        pass
    time.sleep(1.0)

    # 1) forward straight
    drive_distance(client, vehicle_name, total_dist, steering=0.0)
    # 2) reverse straight
    drive_distance(client, vehicle_name, -total_dist, steering=0.0)

    # 3) forward oscillating
    seg = total_dist / segment_count
    for i in range(segment_count):
        angle = steer_amp if (i % 2 == 0) else -steer_amp
        drive_distance(client, vehicle_name, seg, steering=angle)

    # 4) reverse oscillating
    for i in range(segment_count):
        angle = steer_amp if (i % 2 == 1) else -steer_amp
        drive_distance(client, vehicle_name, -seg, steering=angle)

    print(f"[{vehicle_name}] calibration complete.")
    try:
        client.armDisarm(False, vehicle_name=vehicle_name)
    except:
        pass
    client.enableApiControl(False, vehicle_name=vehicle_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cam-IMU calibration maneuvers for UGVs")
    parser.add_argument("--ugvs", nargs="*", default=UGV_NAMES,
                        help="UGV vehicle names (default: %(default)s)")
    parser.add_argument("--port", type=int, default=UGV_PORT)
    cli = parser.parse_args()

    procs = []
    for name in cli.ugvs:
        p = Process(target=run_ugv_calibration_motion, args=(name,),
                    kwargs={"port": cli.port})
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("All UGVs calibrated.")
