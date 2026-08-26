#!/usr/bin/env python3

import os
import sys
import time
import termios
import tty

# Use below in settings.json with Blocks environment
"""
{
    "SettingsVersion": 1.2,
    "SimMode": "Hero",

    "Vehicles": {}
}

"""

sys.path.append(os.path.join(os.path.dirname(__file__), "PythonClient"))
import hercules_cosysairsim as airsim

print("Connecting to existing AirSim server...")
client = airsim.MultirotorClient(port=41451)

# Retry connection until available
connected = False
while not connected:
    try:
        client.confirmConnection()
        connected = True
    except Exception:
        print("Waiting for AirSim connection on port 41451...")
        time.sleep(2)

print("AirSim Connected! Spawning vehicles via API...")

drones = [("Drone1", 0, -4, -0.5), ("Drone2", 0, -2, -0.5), ("SimpleFlight", 0, 0, -0.5), ("Drone4", 0, 2, -0.5), ("Drone5", 0, 4, -0.5)]
huskies = [("Husky1", -3, -3, -1), ("Husky2", -3, 0, -1), ("Husky3", -3, 3, -1)]

existing_vehicles = client.listVehicles()

for name, x, y, z in drones:
    if name in existing_vehicles:
        print(f"Vehicle {name} already exists, skipping spawn.")
        continue
    pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, 0))
    print(f"Adding {name}...")
    try:
        client.simAddVehicle(name, "simpleflight", pose)
    except Exception as e:
        print(f"Note on {name}: {e}")

for name, x, y, z in huskies:
    if name in existing_vehicles:
        print(f"Vehicle {name} already exists, skipping spawn.")
        continue
    pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, 0))
    print(f"Adding {name}...")
    try:
        client.simAddVehicle(name, "cphusky", pose)
    except Exception as e:
        print(f"Note on {name}: {e}")

print("Vehicles spawned! Waiting for physics initialization...")
time.sleep(2.0)

print("Taking off drones...")

# Need to enable API control and arm
for name, _, _, _ in drones:
    client.enableApiControl(True, name)
    client.armDisarm(True, name)

futures = []
for name, _, _, _ in drones:
    futures.append(client.takeoffAsync(vehicle_name=name))

for f in futures:
    f.join()

print("Moving to hover positions...")
futures = []
for name, x, y, _ in drones:
    # Fly to z = -5
    futures.append(client.moveToPositionAsync(x, y, -5, 5.0, vehicle_name=name))

for f in futures:
    f.join()

import threading
import math

stop_thread = False

def pattern_control_loop():
    car_client = airsim.CarClient(port=41452)
    
    # Enable API control and wake up physics for huskies
    for name, _, _, _ in huskies:
        car_client.enableApiControl(True, name)
        
        # Unreal Engine physics bodies go to sleep if they sit still. 
        # If this script is restarted, the Huskies might be asleep and ignore throttle.
        # We wake them up by teleporting them slightly in place.
        state = car_client.simGetVehiclePose(vehicle_name=name)
        state.position.z_val -= 0.1 # bump it up 10cm
        car_client.simSetVehiclePose(state, True, vehicle_name=name)
        
    t = 0.0
    dt = 0.1
    while not stop_thread:
        # UAV pattern: Fly in a circle using velocity commands
        vx = 2.0 * math.cos(t)
        vy = 2.0 * math.sin(t)
        for name, _, _, _ in drones:
            client.moveByVelocityZAsync(vx, vy, -5, dt * 2, vehicle_name=name)
            
        # UGV pattern: Drive in a circle
        controls = airsim.CarControls()
        controls.throttle = 1.0
        controls.steering = 0.5  # Constant steering makes it drive in circles
        controls.is_manual_gear = True
        controls.manual_gear = 1
        for name, _, _, _ in huskies:
            car_client.setCarControls(controls, vehicle_name=name)
            
        t += dt
        time.sleep(dt)

    # Stop UGVs
    controls = airsim.CarControls()
    for name, _, _, _ in huskies:
        car_client.setCarControls(controls, vehicle_name=name)
        car_client.enableApiControl(False, name)

print("Starting pattern control loop...")
control_thread = threading.Thread(target=pattern_control_loop)
control_thread.start()

print("All vehicles are now moving in a pattern!")
print("Press 'q' to exit.")

def get_char():
    if sys.stdin.isatty():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    else:
        try:
            return sys.stdin.read(1)
        except Exception:
            time.sleep(1)
            return ''


while True:
    ch = get_char()
    if ch.lower() == 'q':
        print("\nExiting...")
        stop_thread = True
        control_thread.join()
        break

# Clean up API control
for name, _, _, _ in drones:
    try:
        client.hoverAsync(vehicle_name=name).join()
        client.enableApiControl(False, name)
    except Exception:
        pass

print("Done.")

