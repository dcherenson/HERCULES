#!/usr/bin/env python3
"""
Minimal script to:
  1. Find an object/actor by name/regex in the world.
  2. Print that actor's 6DOF pose and the camera's 6DOF pose in the world frame (NED).
  3. Compute & print the camera intrinsics matrix, assuming AirSim/Unreal returns horizontal FOV.
     Vertical FOV is derived from the aspect ratio.
  4. Query and (optionally) disable any lens distortion parameters.
  5. Define a virtual 3D bounding box (L*W*H) in the actor's local frame, transform it into world
     coordinates using its pose, print the 8 world-frame corners, and visualize.
  6. Project those 3D corners into the image using a classic pinhole model and draw
     the resulting 3D bounding box on the image — or, if toggled, draw the 2D axis-aligned box
     around those projected corners.
"""

import math
import numpy as np
import setup_path                    # ensure cosysairsim is on PYTHONPATH
import cosysairsim as airsim
import cv2  # required for decoding compressed image

# optional visualization
try:
    import open3d as o3d
except ImportError:
    o3d = None

# === CONFIGURATION ===
# ACTOR_PATTERN      = "StaticMeshActor_UAID_6C6E07.*"   # regex to match actor name(s)
ACTOR_PATTERN      = "BP_SplineHuman_Type10.*"   # regex to match actor name(s)

CAMERA_NAME        = "front_center"              # camera to query
CLIENT_CLASS       = airsim.MultirotorClient     # or airsim.CarClient
PORT               = 41451                       # adjust if your AirSim uses a different port

# BOX_LENGTH         = 1.0   # meters (local forward)
# BOX_WIDTH          = 1.0   # meters (local right)
# BOX_HEIGHT         = 1.0  # meters (local up)

BOX_LENGTH         = 0.5   # meters (local forward)
BOX_WIDTH          = 0.5  # meters (local right)
BOX_HEIGHT         = 1.6  # meters (local up)


PROJECTION_ENABLED = True  # toggle 3D→2D projection

# NEW TOGGLE: if True, draws a 2D axis-aligned bounding rectangle
# around the projected 3D corners. If False, draws the full 3D wireframe.
DRAW_2D_BBOX       = True
# =====================


def quaternion_to_euler(q):
    w,x,y,z = q.w_val, q.x_val, q.y_val, q.z_val
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return 0.0, 0.0, 0.0
    w,x,y,z = w/norm, x/norm, y/norm, z/norm
    sinr = 2*(w*x + y*z)
    cosr = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr, cosr)
    sinp = 2*(w*y - z*x)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp)>=1 else math.asin(sinp)
    siny = 2*(w*z + x*y)
    cosy = 1 - 2*(y*y + z*z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def quaternion_to_rotation_matrix(q):
    w,x,y,z = q.w_val, q.x_val, q.y_val, q.z_val
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return np.eye(3)
    w,x,y,z = w/norm, x/norm, y/norm, z/norm
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),  2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),  1-2*(x*x+y*y)]
    ], dtype=float)


def print_pose(label, pose):
    if pose is None:
        print(f"{label}: <no pose>")
        return
    p,o = pose.position, pose.orientation
    roll,pitch,yaw = quaternion_to_euler(o)
    print(f"=== {label} ===")
    print(f" Position (NED): x={p.x_val:.6f}, y={p.y_val:.6f}, z={p.z_val:.6f}")
    print(f" Quaternion (w,x,y,z): "
          f"({o.w_val:.6f}, {o.x_val:.6f}, {o.y_val:.6f}, {o.z_val:.6f})")
    print(f" Euler   (deg): "
          f"roll={math.degrees(roll):.2f}, pitch={math.degrees(pitch):.2f}, yaw={math.degrees(yaw):.2f}")
    print()


def compute_intrinsics_from_horizontal_fov(hfov_deg, width, height):
    hfov = math.radians(hfov_deg)
    fx = (width/2.0) / math.tan(hfov/2.0)
    fy = fx
    cx, cy = width/2.0, height/2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=float)
    vfov = 2 * math.degrees(math.atan((height/2.0)/fy))
    return K, vfov


def compute_bounding_box_corners_world(pose, L, W, H):
    hl, hw, hh = L/2.0, W/2.0, H/2.0
    corners_local = np.array([
        [ hl,  hw,  hh], [ hl,  hw, -hh],
        [ hl, -hw,  hh], [ hl, -hw, -hh],
        [-hl,  hw,  hh], [-hl,  hw, -hh],
        [-hl, -hw,  hh], [-hl, -hw, -hh],
    ], dtype=float)
    R = quaternion_to_rotation_matrix(pose.orientation)
    t = np.array([pose.position.x_val,
                  pose.position.y_val,
                  pose.position.z_val], dtype=float)
    return (R @ corners_local.T).T + t


def project_world_points_to_image(world_pts, cam_pose, K):
    """
    Manual pinhole projection into pixel coords.
    """
    R_wc = quaternion_to_rotation_matrix(cam_pose.orientation)  # cam→world
    R_cw = R_wc.T                                              # world→cam
    C = np.array([cam_pose.position.x_val,
                  cam_pose.position.y_val,
                  cam_pose.position.z_val], dtype=float)

    Xc = (R_cw @ ((world_pts - C).T)).T
    x_fwd, y_right, z_down = Xc[:,0], Xc[:,1], Xc[:,2]

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    valid = x_fwd > 1e-6
    u = np.zeros_like(x_fwd)
    v = np.zeros_like(x_fwd)
    u[valid] = (y_right[valid] / x_fwd[valid]) * fx + cx
    v[valid] = (z_down[valid]  / x_fwd[valid]) * fy + cy

    pts2d = np.stack([u, v], axis=1)
    return pts2d, x_fwd, valid


def draw_projected_box(img, pts2d, depth, valid):
    h, w = img.shape[:2]

    if DRAW_2D_BBOX:
        # draw 2D axis-aligned bbox
        us = pts2d[valid, 0]
        vs = pts2d[valid, 1]
        if us.size and vs.size:
            x0, x1 = int(us.min()), int(us.max())
            y0, y1 = int(vs.min()), int(vs.max())
            cv2.rectangle(img, (x0,y0), (x1,y1), (0,255,0), 2)
    else:
        # draw full 3D wireframe
        edges = [
            [0,1],[1,3],[3,2],[2,0],
            [4,5],[5,7],[7,6],[6,4],
            [0,4],[1,5],[2,6],[3,7]
        ]
        for i,j in edges:
            if valid[i] and valid[j] and depth[i]>0 and depth[j]>0:
                p1 = tuple(map(int, pts2d[i]))
                p2 = tuple(map(int, pts2d[j]))
                if 0<=p1[0]<w and 0<=p1[1]<h and 0<=p2[0]<w and 0<=p2[1]<h:
                    cv2.line(img, p1, p2, (0,255,0), 2)

        for idx in range(8):
            if valid[idx] and depth[idx]>0:
                u,v = map(int, pts2d[idx])
                if 0<=u<w and 0<=v<h:
                    cv2.circle(img, (u,v), 4, (0,0,255), -1)
                    cv2.putText(img, str(idx), (u+3, v-3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)


def main():
    np.set_printoptions(precision=4, suppress=True)
    client = CLIENT_CLASS(port=PORT)
    client.confirmConnection()

    # --- 1) zero lens distortion if any ---
    dparams = client.simGetDistortionParams(CAMERA_NAME)
    print("Distortion params:", dparams)
    if any(abs(d)>1e-9 for d in dparams):
        print(" Zeroing distortion.")
        client.simSetDistortionParams(CAMERA_NAME,
            {"K1":0.0, "K2":0.0, "K3":0.0, "P1":0.0, "P2":0.0})
    else:
        print(" No distortion active.")
    print()

    # --- 2) find actor ---
    objs = client.simListSceneObjects(ACTOR_PATTERN)
    if not objs:
        print(f"No actor matches '{ACTOR_PATTERN}'")
        return
    actor = objs[0]
    print("Target actor:", actor)

    # --- 3) pause + grab everything synchronously ---
    client.simPause(True)
    try:
        try:
            actor_pose = client.simGetObjectPose(actor, True)
        except TypeError:
            actor_pose = client.simGetObjectPose(actor)
        cam_info = client.simGetCameraInfo(CAMERA_NAME)
        cam_pose = cam_info.pose if cam_info else None

        img_resp = client.simGetImages([
            airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, True)
        ])[0]
        img = cv2.imdecode(np.frombuffer(img_resp.image_data_uint8, np.uint8),
                           cv2.IMREAD_COLOR)
    finally:
        client.simPause(False)

    # --- 4) print poses ---
    print_pose(f"Actor ({actor})", actor_pose)
    print_pose(f"Camera ({CAMERA_NAME})", cam_pose)

    # --- 5) intrinsics ---
    h,w = img.shape[:2]
    K, vfov = compute_intrinsics_from_horizontal_fov(cam_info.fov, w, h)
    print(f"Resolution: {w}×{h}, HFOV: {cam_info.fov:.4f}°, VFOV: {vfov:.4f}°")
    print("K =\n", K, "\n")

    # --- 6) compute world-frame corners & print ---
    corners_w = compute_bounding_box_corners_world(
        actor_pose, BOX_LENGTH, BOX_WIDTH, BOX_HEIGHT)
    print(f"Box L×W×H = {BOX_LENGTH}×{BOX_WIDTH}×{BOX_HEIGHT} m")
    for i, c in enumerate(corners_w):
        print(f" [{i}] x={c[0]:.4f}, y={c[1]:.4f}, z={c[2]:.4f}")
    print()

    # --- 7) project & draw ---
    disp = img.copy()
    if PROJECTION_ENABLED and cam_pose:
        pts2d, depth, valid = project_world_points_to_image(corners_w, cam_pose, K)
        print("Using manual pinhole projection:")
        for i, ((u,v), d, ok) in enumerate(zip(pts2d, depth, valid)):
            status = "vis" if ok and d>0 else "out"
            print(f"[{i}] u={u:.1f}, v={v:.1f}, depth={d:.3f} ({status})")
        print()
        draw_projected_box(disp, pts2d, depth, valid)
    else:
        print("Skipping projection.\n")

    # --- 8) annotate & show ---
    cv2.putText(disp, f"Actor: {actor}", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.namedWindow("Projected 3D Bounding Box", cv2.WINDOW_NORMAL)
    cv2.imshow("Projected 3D Bounding Box", disp)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- 9) optional Open3D viz ---
    if o3d:
        try:
            frames = []
            Ta = np.eye(4)
            Ta[:3,:3] = quaternion_to_rotation_matrix(actor_pose.orientation)
            Ta[:3,3] = [actor_pose.position.x_val,
                        actor_pose.position.y_val,
                        actor_pose.position.z_val]
            fa = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            fa.transform(Ta)
            frames.append(fa)

            if cam_pose:
                Tc = np.eye(4)
                Tc[:3,:3] = quaternion_to_rotation_matrix(cam_pose.orientation)
                Tc[:3,3] = [cam_pose.position.x_val,
                            cam_pose.position.y_val,
                            cam_pose.position.z_val]
                fc = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
                fc.transform(Tc)
                frames.append(fc)

            edges = [[0,1],[1,3],[3,2],[2,0],
                     [4,5],[5,7],[7,6],[6,4],
                     [0,4],[1,5],[2,6],[3,7]]
            ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(corners_w),
                lines=o3d.utility.Vector2iVector(edges))
            ls.colors = o3d.utility.Vector3dVector([[1,0,0]]*len(edges))
            frames.append(ls)

            print("Showing 3D viz in Open3D.")
            o3d.visualization.draw_geometries(frames)
        except Exception as e:
            print("Open3D error:", e)
    else:
        print("open3d not installed; skipping 3D visualization.")


if __name__ == "__main__":
    main()
