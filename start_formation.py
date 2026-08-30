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
import subprocess
import atexit

# LAUNCH_MODE options:
# "headless" : Automatically launch Unreal Engine hidden in the background (no UI)
# "visible"  : Automatically launch Unreal Engine with a visible window
# "existing" : Do not launch Unreal Engine; connect to an already running instance
LAUNCH_MODE = "visible" 
UNREAL_EDITOR_PATH = "/Users/Shared/Epic Games/UE_5.2/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
UPROJECT_PATH = os.path.join(os.path.dirname(__file__), "Unreal/Environments/Blocks/Blocks.uproject")

unreal_process = None
if LAUNCH_MODE in ["headless", "visible"]:
    print(f"Launching Unreal Engine ({LAUNCH_MODE} mode)...")
    cmd = [
        UNREAL_EDITOR_PATH,
        UPROJECT_PATH,
        "/Game/RuralAustralia/Maps/RuralAustralia_Example_01?game=/Script/AirSim.AirSimGameMode",
        "-game",
        "-windowed",
        "-resx=800",
        "-resy=600"
    ]
    if LAUNCH_MODE == "headless":
        cmd.append("-RenderOffscreen")
        
    # Launch in the background, suppressing spammy Unreal Engine logs
    unreal_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    def cleanup_unreal():
        if unreal_process and unreal_process.poll() is None:
            print("\nShutting down Unreal Engine...")
            unreal_process.terminate()
            unreal_process.wait()
            
    atexit.register(cleanup_unreal)

import socket

print("Waiting for AirSim RPC server to boot up on port 41451 (this can take up to 20 seconds)...")
server_up = False
while not server_up:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect(('127.0.0.1', 41451))
            server_up = True
    except Exception:
        time.sleep(1)

print("RPC Server is up! Connecting AirSim client...")
client = airsim.MultirotorClient(port=41451)
client.confirmConnection()

# Automatically wipe the Unreal Engine HUD and screen messages
client.simRunConsoleCommand("DisableAllScreenMessages")

print("AirSim Connected! Spawning vehicles via API...")

drones = [("Drone1", 0, -4, -0.5), ("Drone2", 0, -2, -0.5), ("SimpleFlight", 0, 0, -0.5), ("Drone4", 0, 2, -0.5), ("Drone5", 0, 4, -0.5)]
huskies = [("Husky1", -3, -3, -1), ("Husky2", -3, 0, -1), ("Husky3", -3, 3, -1)]

all_vehicle_names = [d[0] for d in drones] + [h[0] for h in huskies]
trajectory_data = {name: {'x': [], 'y': [], 'z': []} for name in all_vehicle_names}

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
            
        # Record trajectory
        for name in all_vehicle_names:
            try:
                if name.startswith("Husky"):
                    pos = car_client.simGetVehiclePose(vehicle_name=name).position
                else:
                    pos = client.simGetVehiclePose(vehicle_name=name).position
                trajectory_data[name]['x'].append(pos.x_val)
                trajectory_data[name]['y'].append(pos.y_val)
                trajectory_data[name]['z'].append(pos.z_val)
            except Exception:
                pass
            
        # Log sensor data every 10 ticks (1 second)
        # Log sensor data every 1 second
        if t % 1.0 < dt:
            try:
                # Log Drone1 sensors
                imu_drone = client.getImuData(vehicle_name="Drone1")
                dist_drone = client.getDistanceSensorData(distance_sensor_name="Distance", vehicle_name="Drone1")
                thermal_req = airsim.ImageRequest("front_center", airsim.ImageType.ThermalIR, False, False)
                img_res = client.simGetImages([thermal_req], vehicle_name="Drone1")
                
                sys.stdout.write(f"\r[SENSOR LOG] Drone1 - Accel Z: {imu_drone.linear_acceleration.z_val:.2f}, "
                                 f"Distance: {dist_drone.distance:.2f}m, "
                                 f"ThermalIR Acquired: {len(img_res) > 0}\r\n")
                      
                # Log Husky1 sensors
                imu_husky = car_client.getImuData(vehicle_name="Husky1")
                lidar_husky = car_client.getLidarData(lidar_name="Lidar1", vehicle_name="Husky1")
                
                sys.stdout.write(f"\r[SENSOR LOG] Husky1 - Accel Z: {imu_husky.linear_acceleration.z_val:.2f}, "
                                 f"Lidar points: {len(lidar_husky.point_cloud) // 3}\r\n")
                sys.stdout.flush()
                      
            except Exception as e:
                sys.stdout.write(f"\r[SENSOR LOG] Error reading sensors: {e}\r\n")
                sys.stdout.flush()
                
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

print("Done. Generating trajectory plot...")

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.tab10.colors
for idx, (name, data) in enumerate(trajectory_data.items()):
    if len(data['x']) > 0:
        c = colors[idx % len(colors)]
        ax.plot(data['x'], data['y'], data['z'], label=name, color=c, linewidth=2)
        ax.scatter(data['x'][0], data['y'][0], data['z'][0], color=c, marker='o') # Start point
        ax.scatter(data['x'][-1], data['y'][-1], data['z'][-1], color=c, marker='x') # End point

ax.set_title("Heterogeneous Swarm Trajectory (UAVs & UGVs)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

# Invert Z axis because Unreal/AirSim uses NED (Z down is positive, up is negative)
# Drones fly at -5, UGVs drive at ~0. Inverting makes visually up be altitude.
ax.invert_zaxis() 
ax.legend()
plt.tight_layout()
plt.savefig("swarm_trajectory.png")
print("Plot saved as 'swarm_trajectory.png' in the current directory!")
plt.show()
