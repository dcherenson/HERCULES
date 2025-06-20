import setup_path 
import cosysairsim as airsim
import numpy as np
import time

import keyboard
import random

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
time.sleep(1)
client.moveToPositionAsync(0.5, 0, -2, 2).join()
time.sleep(2)

client.moveByAngleZAsync(np.pi / 16, 0, -2, 0, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByAngleZAsync(-np.pi / 16, 0, -2, 0, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByAngleZAsync(0, -np.pi / 20, -2, 0, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByAngleZAsync(0, np.pi / 20, -2, 0, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByAngleZAsync(0, 0, -2, -np.pi / 10, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByAngleZAsync(0, 0, -2, np.pi / 10, 2).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByAngleZAsync(0, 0, -2, -np.pi / 20, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()

client.moveByVelocityAsync(0, 0.5, 0, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByVelocityAsync(0, -0.5, 0, 2).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByVelocityAsync(0, 0.5, -0.2, 2).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByVelocityAsync(0, -0.5, 0.2, 2).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByVelocityAsync(-0.5, 0, 0, 1).join()
client.moveByVelocityAsync(0.5, 0, 0, 2).join()
client.moveByVelocityAsync(-0.5, 0, 0, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByVelocityAsync(0, 0.3, 0, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByVelocityAsync(0, 0, -1, 0.5).join()
client.moveByVelocityAsync(0, 0, 1, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()

client.moveByAngleZAsync(np.pi / 16, 0, -2, 0, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByAngleZAsync(-np.pi / 16, 0, -2, 0, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByAngleZAsync(0, -np.pi / 20, -2, 0, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByAngleZAsync(0, np.pi / 20, -2, 0, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByAngleZAsync(0, 0, -2, -np.pi / 10, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByAngleZAsync(0, 0, -2, np.pi / 10, 2).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByAngleZAsync(0, 0, -2, -np.pi / 20, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()

client.moveByVelocityAsync(0, 0.5, 0, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByVelocityAsync(0, -0.5, 0, 2).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByVelocityAsync(0, 0.5, -0.2, 2).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByVelocityAsync(0, -0.5, 0.2, 2).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByVelocityAsync(-0.5, 0, 0, 1).join()
client.moveByVelocityAsync(0.5, 0, 0, 2).join()
client.moveByVelocityAsync(-0.5, 0, 0, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByVelocityAsync(0, 0.3, 0, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()
client.moveByVelocityAsync(0, 0, -1, 0.5).join()
client.moveByVelocityAsync(0, 0, 1, 1).join()
client.moveByVelocityAsync(0, 0, 0, 1).join()


# --- TELEOP PARAMETERS ---
speed = 1.5         # m/s for linear motion
yaw_rate = 30       # degrees/s for yaw
duration = 0.1      # seconds per command
print("Teleop mode: WASD=XY, RF=Z, QE=Yaw, X=exit")

try:
    while True:
        vx = vy = vz = 0.0
        yr = 0.0

        # Horizontal translation
        if keyboard.is_pressed('w'): vx += speed         # forward
        if keyboard.is_pressed('s'): vx -= speed         # back
        if keyboard.is_pressed('d'): vy += speed         # right
        if keyboard.is_pressed('a'): vy -= speed         # left

        # Vertical translation
        if keyboard.is_pressed('r'): vz += speed         # up
        if keyboard.is_pressed('f'): vz -= speed         # down

        # Yaw rotation
        if keyboard.is_pressed('q'): yr -= yaw_rate      # yaw left
        if keyboard.is_pressed('e'): yr += yaw_rate      # yaw right

        # Exit teleop
        if keyboard.is_pressed('x'):
            print("Exiting teleop...")
            break

        # Send velocity + yaw-rate command
        client.moveByVelocityAsync(
            vx, vy, vz, duration,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yr)
        ).join()
except KeyboardInterrupt:
    pass

print("Teleop ended.")

print("Random path complete. Disarming and releasing control…")
client.armDisarm(False)                 # disarm the rotors
client.enableApiControl(False)          # give control back to the simulator
print("Client disconnected and API control released.")