#!/usr/bin/env python3
"""
Script to retrieve a scene image and project a 3D bounding box onto it using both:
  1) AirSim's built-in projection matrix P
  2) Manual intrinsics K (horizontal FOV + aspect ratio)
Compares both projections side-by-side and prints both matrices.
"""
import setup_path
import cosysairsim as airsim
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def to_numpy_vector(v):
    return np.array([v.x_val, v.y_val, v.z_val])

def quat_to_rot_matrix(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)]
    ])

def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim multirotor client.")

    # 1. Get scene image
    resp = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
    ], vehicle_name="Drone1")[0]
    w_img, h_img = resp.width, resp.height
    img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
    scene = Image.fromarray(img1d.reshape(h_img, w_img, 3))
    scene.save("scene.png")
    print("Saved scene.png")

    # 2. Camera info
    cam_info = client.simGetCameraInfo("front_center", vehicle_name="Drone1")
    cam_p = to_numpy_vector(cam_info.pose.position)
    cam_q = np.array([
        cam_info.pose.orientation.w_val,
        cam_info.pose.orientation.x_val,
        cam_info.pose.orientation.y_val,
        cam_info.pose.orientation.z_val
    ])

    # 3. Build 8 bounding-box corners in world
    dx, dy, dz = 1.0, 1.0, 1.0
    half = np.array([dx/2, dy/2, dz/2])
    local = np.array([
        [-half[0], +half[1], +half[2]],
        [-half[0], +half[1], -half[2]],
        [-half[0], -half[1], +half[2]],
        [-half[0], -half[1], -half[2]],
        [+half[0], +half[1], +half[2]],
        [+half[0], +half[1], -half[2]],
        [+half[0], -half[1], +half[2]],
        [+half[0], -half[1], -half[2]],
    ])
    obj_pose = client.simGetObjectPose("StaticMeshActor_0")
    obj_p = to_numpy_vector(obj_pose.position)
    obj_q = np.array([
        obj_pose.orientation.w_val,
        obj_pose.orientation.x_val,
        obj_pose.orientation.y_val,
        obj_pose.orientation.z_val
    ])
    R_obj = quat_to_rot_matrix(obj_q)
    world_corners = (R_obj @ local.T).T + obj_p

    # 4. Transform to camera frame
    R_cam = quat_to_rot_matrix(cam_q)
    cam_corners = (R_cam.T @ (world_corners - cam_p).T).T
    print("Corners in camera frame:")
    for i, c in enumerate(cam_corners): print(f"  {i}: {c}")

    # 5. AirSim projection (P-matrix)
    P = np.array(cam_info.proj_mat.matrix, dtype=float).reshape((4, 4))
    print("AirSim projection matrix P:")
    print(P)
    pts_h = np.hstack([cam_corners, np.ones((8, 1))])  # Nx4
    clip = (P @ pts_h.T).T                             # Nx4
    ndc = clip[:, :3] / clip[:, 3:4]                   # Nx3
    u_p = (1 - (ndc[:, 0] * 0.5 + 0.5)) * w_img  # flip X
    v_p = (ndc[:, 1] * 0.5 + 0.5) * h_img         # direct Y
    projP = np.stack([u_p, v_p], axis=1)

    # 6. Manual intrinsics projection (K)
    hFOV = np.deg2rad(cam_info.fov)  # horizontal FOV
    fx = w_img / (2 * np.tan(hFOV / 2))
    vFOV = 2 * np.arctan((h_img / w_img) * np.tan(hFOV / 2))
    fy = h_img / (2 * np.tan(vFOV / 2))
    cx, cy = w_img / 2, h_img / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    print("Manual intrinsic matrix K:")
    print(K)

    # 6b. Build 4x4 manual projection matrix using standard OpenGL conventions
    near, far = 1.0, 10.0  # match AirSim's near/far clip planes
    # OpenGL-style projection (RH, clip-space Z in [-1,1])
    P_manual = np.zeros((4,4))
    P_manual[0,0] = 2*near/w_img
    P_manual[1,1] = 2*near/h_img
    P_manual[0,2] = (w_img-2*cx)/w_img
    P_manual[1,2] = (2*cy-h_img)/h_img
    P_manual[2,2] = -(far+near)/(far-near)
    P_manual[2,3] = -2*far*near/(far-near)
    P_manual[3,2] = -1
    print("Manual OpenGL-style projection matrix P_manual:")
    print(P_manual)
    print(K)

    # 6b. Build a 4x4 manual projection matrix for comparison
    P_manual = np.array([
        [fx,  0,  cx, 0],
        [0,  fy,  cy, 0],
        [0,   0,   1, 0],
        [0,   0,   0, 1]
    ], dtype=float)
    print("Manual projection matrix P_manual (4x4):")
    print(P_manual)

    hFOV = np.deg2rad(cam_info.fov)  # horizontal FOV
    fx = w_img / (2 * np.tan(hFOV / 2))
    vFOV = 2 * np.arctan((h_img / w_img) * np.tan(hFOV / 2))
    fy = h_img / (2 * np.tan(vFOV / 2))
    cx, cy = w_img / 2, h_img / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    print("Manual intrinsic matrix K:")
    print(K)
    Xc = cam_corners[:, 1]  # right
    Yc = cam_corners[:, 2]  # down
    Zc = cam_corners[:, 0]  # forward
    u_k = fx * (Xc / Zc) + cx
    v_k = fy * (Yc / Zc) + cy
    projK = np.stack([u_k, v_k], axis=1)

    # 7. Draw comparison
    img = scene.copy(); draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for i, j in edges:
        # AirSim P in red
        rp1 = tuple(projP[i].astype(int)); rp2 = tuple(projP[j].astype(int))
        draw.line([rp1, rp2], fill="red", width=2)
        # Manual K in blue
        bp1 = tuple(projK[i].astype(int)); bp2 = tuple(projK[j].astype(int))
        draw.line([bp1, bp2], fill="blue", width=1)
    # label
    for i, (xp, yp) in enumerate(projP): draw.text((int(xp)+3, int(yp)-10), f"P{i}", fill="red", font=font)
    for i, (xk, yk) in enumerate(projK): draw.text((int(xk)+3, int(yk)+3), f"K{i}", fill="blue", font=font)
    img.save("scene_compare.png")
    print("Saved scene_compare.png: red=P, blue=K projections.")

if __name__ == '__main__':
    main()
