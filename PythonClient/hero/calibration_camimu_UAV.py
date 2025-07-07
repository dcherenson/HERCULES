#!/usr/bin/env python3

import setup_path
import cosysairsim as airsim
import numpy as np
import time

# List here all of your multirotor names as defined in your settings.json
DRONE_NAMES = [
    "Drone1",
    "Drone2"
]

def parellel_join(futures):
    """ Helper to wait for a list of AirSim futures. """
    for f in futures:
        f.join()

def run_calibration_on_all(client, drone_names):
    # 1) Move to (0.5,0,-2) at speed 2m/s
    futures = [
        client.moveToPositionAsync(0.5, 0, -2, 2, vehicle_name=name)
        for name in drone_names
    ]
    parellel_join(futures)
    time.sleep(2)

    # 2) Angular and velocity maneuvers
    seq = [
        ("moveByAngleZAsync",   ( np.pi/16, 0, -2,    0, 1 )),
        ("moveByAngleZAsync",   (-np.pi/16, 0, -2,    0, 1 )),
        ("moveByAngleZAsync",   ( 0, -np.pi/20, -2,   0, 1 )),
        ("moveByAngleZAsync",   ( 0,  np.pi/20, -2,   0, 1 )),
        ("moveByAngleZAsync",   ( 0, 0, -2,   -np.pi/10, 1 )),
        ("moveByAngleZAsync",   ( 0, 0, -2,    np.pi/10, 2 )),
        ("moveByAngleZAsync",   ( 0, 0, -2,   -np.pi/20, 1 )),
    ]
    for method, args in seq:
        futures = [
            getattr(client, method)(*args, vehicle_name=name)
            for name in drone_names
        ]
        parellel_join(futures)
        # zero‐velocity pause
        zeros = [
            client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=name)
            for name in drone_names
        ]
        parellel_join(zeros)

    print("CHECKPOINT1")

    # 3) Velocity maneuvers
    vel_seq = [
        ( 0,  0.5,  0, 1),
        ( 0, -0.5,  0, 2),
        ( 0,  0.5, -0.2, 2),
        ( 0, -0.5,  0.2, 2),
        (-0.5, 0,   0, 1),
        ( 0.5, 0,   0, 2),
        (-0.5, 0,   0, 1),
        ( 0,  0.3, 0, 1),
        ( 0,  0,  -1, 0.5),
        ( 0,  0,   1, 1),
    ]
    for vx, vy, vz, dur in vel_seq:
        futures = [
            client.moveByVelocityAsync(vx, vy, vz, dur, vehicle_name=name)
            for name in drone_names
        ]
        parellel_join(futures)
        # brief stop between each
        stops = [
            client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=name)
            for name in drone_names
        ]
        parellel_join(stops)

    print("CHECKPOINT2")
    print("Calibration motion finished on all drones.")

if __name__ == "__main__":
    client = airsim.MultirotorClient(port=41451)
    client.confirmConnection()

    # takeoff and arm every drone
    for name in DRONE_NAMES:
        client.enableApiControl(True, vehicle_name=name)
        client.armDisarm(True, vehicle_name=name)

    # give them a moment
    time.sleep(1.0)

    run_calibration_on_all(client, DRONE_NAMES)

    # disarm & release control
    for name in DRONE_NAMES:
        client.armDisarm(False, vehicle_name=name)
        client.enableApiControl(False, vehicle_name=name)
