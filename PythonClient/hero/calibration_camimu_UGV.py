import setup_path
import cosysairsim as airsim
import time
import numpy as np

def drive_distance(client, distance, throttle=0.3, steering=0.0):
    """
    Drives the UGV forward (distance > 0) or backward (distance < 0)
    by the specified meters. Uses manual gear for reverse.
    """
    controls = airsim.CarControls()
    controls.is_manual_gear = True
    controls.manual_gear = 1 if distance >= 0 else -1
    controls.steering = steering
    controls.throttle = abs(throttle)
    controls.brake = 0.0

    start = client.simGetVehiclePose().position
    start_xy = np.array([start.x_val, start.y_val])

    while True:
        client.setCarControls(controls)
        time.sleep(0.05)
        pos = client.simGetVehiclePose().position
        delta = np.linalg.norm(np.array([pos.x_val, pos.y_val]) - start_xy)
        if delta >= abs(distance):
            break

    # stop
    controls.throttle = 0.0
    controls.brake = 1.0
    client.setCarControls(controls)
    time.sleep(0.5)
    controls.brake = 0.0
    client.setCarControls(controls)
    time.sleep(0.2)

def run_ugv_calibration_motion(client, segment_count=4, total_dist=5.0, steer_amp=0.4):
    """
    1) 5 m straight forward
    2) 5 m straight backward
    3) 5 m forward with alternating left/right steering
    4) 5 m backward with alternating left/right steering
    """
    # 1) Straight forward
    drive_distance(client,  total_dist, throttle=0.3, steering=0.0)

    # 2) Straight backward
    drive_distance(client, -total_dist, throttle=0.3, steering=0.0)

    # 3) Forward with steering oscillation
    seg_len = total_dist / segment_count
    for i in range(segment_count):
        angle = steer_amp if (i % 2 == 0) else -steer_amp
        drive_distance(client, seg_len, throttle=0.3, steering=angle)

    # 4) Backward with steering oscillation
    for i in range(segment_count):
        angle = steer_amp if (i % 2 == 1) else -steer_amp
        drive_distance(client, -seg_len, throttle=0.3, steering=angle)

    print("UGV calibration motion complete; ended near start spot.")

if __name__ == "__main__":
    client = airsim.CarClient(port=41452)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)  # no-op for cars
    time.sleep(1.0)

    run_ugv_calibration_motion(client,
                               segment_count=4,
                               total_dist=5.0,
                               steer_amp=0.4)

    client.enableApiControl(False)
