#!/usr/bin/env python3
"""
Script to retrieve scene image, print camera pose in NED world frame,
print NED pose of object with mesh ID name StaticMeshActor_0 in world frame,
and print the pose of the object in the camera frame (NED coordinates).
"""
import setup_path                   
import cosysairsim as airsim
import numpy as np
from PIL import Image

def to_numpy_vector(v):
    return np.array([v.x_val, v.y_val, v.z_val])

def quat_to_rot_matrix(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)]
    ])

def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # 1. Get scene image
    req = airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
    res = client.simGetImages([req], vehicle_name="Drone1")[0]
    img1d = np.frombuffer(res.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(res.height, res.width, 3)
    Image.fromarray(img_rgb).save("/home/sgarimella34/multi-robot-coordination/Cosys-AirSim/PythonClient/hero/scene.png")
    print("Scene image saved to scene.png")

    # 2. Print camera pose in NED world frame
    cam_info = client.simGetCameraInfo("front_center", vehicle_name="Drone1")
    cam_p = to_numpy_vector(cam_info.pose.position)
    cam_q = np.array([
        cam_info.pose.orientation.w_val,
        cam_info.pose.orientation.x_val,
        cam_info.pose.orientation.y_val,
        cam_info.pose.orientation.z_val
    ])
    print("Camera pose in NED world frame:")
    print("  Position (NED):", cam_p)
    print("  Orientation (w, x, y, z):", cam_q)

    # 3. Print object pose StaticMeshActor_0 in world frame
    obj_name = "StaticMeshActor_0"
    obj_pose = client.simGetObjectPose(obj_name)
    obj_p = to_numpy_vector(obj_pose.position)
    obj_q = np.array([
        obj_pose.orientation.w_val,
        obj_pose.orientation.x_val,
        obj_pose.orientation.y_val,
        obj_pose.orientation.z_val
    ])
    print(f"Object '{obj_name}' pose in NED world frame:")
    print("  Position (NED):", obj_p)
    print("  Orientation (w, x, y, z):", obj_q)

    # 4. Print object pose in camera frame (NED)
    R_cam = quat_to_rot_matrix(cam_q)
    obj_p_cam = R_cam.T.dot(obj_p - cam_p)
    cam_q_conj = quat_conj(cam_q)
    q_rel = quat_mul(cam_q_conj, obj_q)
    print(f"Object '{obj_name}' pose in camera frame (NED coordinates):")
    print("  Position (NED):", obj_p_cam)
    print("  Orientation (w, x, y, z):", q_rel)

if __name__ == "__main__":
    main()
