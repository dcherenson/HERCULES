#!/usr/bin/env python3

import setup_path
import cosysairsim as airsim
import time
import numpy as np
from multiprocessing import Process

# List your UGV names from settings.json
UGV_NAMES = ["Husky1", "Husky2"]  # add more as needed

def drive_distance(client, vehicle_name, distance, throttle=0.05, steering=0.0):
    """
    Drives one UGV forward/backward by 'distance' meters.
    """
    controls = airsim.CarControls()
    controls.is_manual_gear = True
    controls.manual_gear = 1 if distance >= 0 else -1
    controls.throttle = abs(throttle)
    controls.steering = steering
    controls.brake = 0.0

    start = client.simGetVehiclePose(vehicle_name=vehicle_name).position
    start_xy = np.array([start.x_val, start.y_val])

    while True:
        client.setCarControls(controls, vehicle_name=vehicle_name)
        time.sleep(0.05)
        pos = client.simGetVehiclePose(vehicle_name=vehicle_name).position
        delta = np.linalg.norm([pos.x_val - start_xy[0],
                                pos.y_val - start_xy[1]])
        if delta >= abs(distance):
            break

    controls.throttle = 0.0
    controls.brake = 1.0
    client.setCarControls(controls, vehicle_name=vehicle_name)
    time.sleep(0.5)
    controls.brake = 0.0
    client.setCarControls(controls, vehicle_name=vehicle_name)
    time.sleep(0.2)

def run_ugv_calibration_motion(vehicle_name,
                               segment_count=4,
                               total_dist=5.0,
                               steer_amp=0.4):
    """
    1) 5 m forward
    2) 5 m backward
    3) 5 m forward with alternating steering
    4) 5 m backward with alternating steering
    """
    client = airsim.CarClient(port=41452)
    client.confirmConnection()

    # Enable control for this UGV
    client.enableApiControl(True, vehicle_name=vehicle_name)
    client.armDisarm(True, vehicle_name=vehicle_name)
    time.sleep(1.0)

    # 1) Straight forward
    drive_distance(client, vehicle_name,  total_dist, throttle=0.05, steering=0.0)

    # 2) Straight backward
    drive_distance(client, vehicle_name, -total_dist, throttle=0.05, steering=0.0)

    # 3) Forward oscillating
    seg_len = total_dist / segment_count
    for i in range(segment_count):
        angle = steer_amp if (i % 2 == 0) else -steer_amp
        drive_distance(client, vehicle_name, seg_len, throttle=0.05, steering=angle)

    # 4) Backward oscillating
    for i in range(segment_count):
        angle = steer_amp if (i % 2 == 1) else -steer_amp
        drive_distance(client, vehicle_name, -seg_len, throttle=0.05, steering=angle)

    print(f"[{vehicle_name}] calibration complete.")

    # Teardown
    client.armDisarm(False, vehicle_name=vehicle_name)
    client.enableApiControl(False, vehicle_name=vehicle_name)

if __name__ == "__main__":
    procs = []
    for name in UGV_NAMES:
        p = Process(target=run_ugv_calibration_motion, args=(name,))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("All UGVs calibrated.")
