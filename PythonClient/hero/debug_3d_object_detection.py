#!/usr/bin/env python3
"""
Minimal script to:
  1) Find an object/actor by name/regex in the world.
  2) Print actor & camera poses (NED).
  3) Define a box (L,W,H) in actor frame (with Z-offset in NED), transform to world.
  4) Project 3D corners using AirSim's 4x4 projection matrix; draw amodal 2D box.
  5) ALSO show Segmentation ROI + Depth ROI inside that 2D box in separate windows.
  6) (NEW) Optionally depth-clip the segmentation ROI to keep only pixels with depth ≤ threshold.

Keeps car vs human profile selection.
"""

import math
import numpy as np
import setup_path                    # ensure cosysairsim is on PYTHONPATH
import cosysairsim as airsim
import cv2

# optional visualization
try:
    import open3d as o3d
except ImportError:
    o3d = None

# === CONFIGURATION ===
# Set the actor search pattern for your scene
# ACTOR_PATTERN      = "SkeletalMeshActor_UAID.*"
ACTOR_PATTERN      = "StaticMeshActor_UAID_E08F4CF5208AA07502_2022041209.*"

CAMERA_NAME        = "front_center"
CLIENT_CLASS       = airsim.MultirotorClient
PORT               = 41451

PROJECTION_ENABLED = True
DRAW_2D_BBOX       = True

# --- choose object profile: "human" or "car"
OBJECT_TYPE = "car"  # change to "human" for human-sized box

# --- per-object profiles (dimensions in meters; Z is NED +Z down)
PROFILES = {
    "human": {"L": 0.5, "W": 0.75, "H": 1.9,  "Z": -0.90},
    "car":   {"L": 4.2, "W": 1.90, "H": 1.60, "Z": -0.55},
}
if OBJECT_TYPE not in PROFILES:
    raise ValueError(f"Unknown OBJECT_TYPE '{OBJECT_TYPE}'. Choose from {list(PROFILES.keys())}.")

# derive active dims/offset from the selected profile
BOX_LENGTH    = PROFILES[OBJECT_TYPE]["L"]
BOX_WIDTH     = PROFILES[OBJECT_TYPE]["W"]
BOX_HEIGHT    = PROFILES[OBJECT_TYPE]["H"]
Z_OFFSET_NED  = PROFILES[OBJECT_TYPE]["Z"]

# show cropped segmentation/depth views?
SHOW_ROI_WINDOWS   = True

# --- Depth-clip settings (NEW) ---
DEPTH_CLIP_ENABLE   = True     # if True, mask seg ROI by depth threshold
DEPTH_CLIP_MAX_M    = 35.0     # keep pixels with depth <= this (meters)
SHOW_ORIGINAL_SEG_ROI = True   # also show the raw (unclipped) seg ROI for comparison

# =====================

def quaternion_to_euler(q):
    w,x,y,z = q.w_val, q.x_val, q.y_val, q.z_val
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0: return 0.0, 0.0, 0.0
    w,x,y,z = w/norm, x/norm, y/norm, z/norm
    sinr = 2*(w*x + y*z);  cosr = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr, cosr)
    sinp = 2*(w*y - z*x)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp)>=1 else math.asin(sinp)
    siny = 2*(w*z + x*y);  cosy = 1 - 2*(y*y + z*z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw

def quaternion_to_rotation_matrix(q):
    w,x,y,z = q.w_val, q.x_val, q.y_val, q.z_val
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0: return np.eye(3)
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
    r,pit,y = quaternion_to_euler(o)
    print(f"=== {label} ===")
    print(f" Position (NED): x={p.x_val:.6f}, y={p.y_val:.6f}, z={p.z_val:.6f}")
    print(f" Quaternion (w,x,y,z): ({o.w_val:.6f}, {o.x_val:.6f}, {o.y_val:.6f}, {o.z_val:.6f})")
    print(f" Euler (deg): roll={math.degrees(r):.2f}, pitch={math.degrees(pit):.2f}, yaw={math.degrees(y):.2f}\n")

def compute_intrinsics_from_horizontal_fov(hfov_deg, width, height):
    hfov = math.radians(hfov_deg)
    fx = (width/2.0) / math.tan(hfov/2.0)
    fy = fx
    cx, cy = width/2.0, height/2.0
    K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=float)
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
    t = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val], dtype=float)
    return (R @ corners_local.T).T + t

def project_world_points_to_image(world_pts, cam_pose, P, width, height):
    R_cam = quaternion_to_rotation_matrix(cam_pose.orientation)
    cam_p = np.array([cam_pose.position.x_val, cam_pose.position.y_val, cam_pose.position.z_val], dtype=float)
    cam_pts = (R_cam.T @ (world_pts - cam_p).T).T  # Nx3

    pts_h = np.hstack([cam_pts, np.ones((cam_pts.shape[0], 1), dtype=float)])  # Nx4
    clip  = (P @ pts_h.T).T
    w_comp = clip[:, 3:4]
    ndc   = clip[:, :3] / w_comp

    u = (1.0 - (ndc[:, 0] * 0.5 + 0.5)) * width
    v = (ndc[:, 1] * 0.5 + 0.5) * height

    pts2d = np.stack([u, v], axis=1)
    depth_forward = cam_pts[:, 0]  # X-forward in camera frame
    valid = depth_forward > 1e-6
    return pts2d, depth_forward, valid

def draw_2d_bbox_and_get_rect(pts2d, valid, w, h, img_to_draw=None, color=(0,255,0), thickness=2):
    us = pts2d[valid, 0]; vs = pts2d[valid, 1]
    if us.size == 0 or vs.size == 0:
        return None
    x0 = int(max(0, math.floor(us.min())))
    x1 = int(min(w-1, math.ceil(us.max())))
    y0 = int(max(0, math.floor(vs.min())))
    y1 = int(min(h-1, math.ceil(vs.max())))
    if img_to_draw is not None:
        cv2.rectangle(img_to_draw, (x0,y0), (x1,y1), color, thickness)
    return (x0, y0, x1, y1)

def resize_to(img, target_w, target_h, is_depth=False):
    """Resize to (target_h, target_w). Use NEAREST for depth/labels."""
    if img.shape[1] == target_w and img.shape[0] == target_h:
        return img
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

def depth_roi_to_vis(depth_roi):
    """8-bit colormap visualization from float meters; robust to outliers."""
    d = depth_roi.copy()
    finite = np.isfinite(d) & (d > 0)
    if np.any(finite):
        lo = np.percentile(d[finite], 2.0)
        hi = np.percentile(d[finite], 98.0)
        if hi <= lo: hi = lo + 1e-3
        d = np.clip(d, lo, hi)
        d[~finite] = hi
        vis = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        vis = np.zeros_like(d, dtype=np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)

def main():
    np.set_printoptions(precision=4, suppress=True)
    client = CLIENT_CLASS(port=PORT)
    client.confirmConnection()
    print("Connected!\n")

    # Zero lens distortion if any
    dparams = client.simGetDistortionParams(CAMERA_NAME)
    print("Distortion params:", dparams)
    if any(abs(d)>1e-9 for d in dparams):
        print(" Zeroing distortion.")
        client.simSetDistortionParams(CAMERA_NAME, {"K1":0.0, "K2":0.0, "K3":0.0, "P1":0.0, "P2":0.0})
    else:
        print(" No distortion active.")
    print()

    # Find target actor
    objs = client.simListSceneObjects(ACTOR_PATTERN)
    if not objs:
        print(f"No actor matches '{ACTOR_PATTERN}'"); return
    actor = objs[0]
    print("Target actor:", actor)

    # Pause and capture synchronously
    client.simPause(True)
    try:
        try:
            actor_pose = client.simGetObjectPose(actor, True)
        except TypeError:
            actor_pose = client.simGetObjectPose(actor)

        cam_info = client.simGetCameraInfo(CAMERA_NAME)
        cam_pose = cam_info.pose if cam_info else None

        # Capture Scene, Segmentation, DepthPerspective together
        reqs = [
            airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene,         False, True),
            airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Segmentation,  False, True),
            airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.DepthPerspective, True,  False),
        ]
        scene_resp, seg_resp, depth_resp = client.simGetImages(reqs)

        # Decode Scene
        img = cv2.imdecode(np.frombuffer(scene_resp.image_data_uint8, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to decode Scene image"); return
        h, w = img.shape[:2]

        # Decode Segmentation
        seg_img = cv2.imdecode(np.frombuffer(seg_resp.image_data_uint8, np.uint8), cv2.IMREAD_COLOR)
        if seg_img is None:
            print("Failed to decode Segmentation image")
            seg_img = np.zeros((h, w, 3), np.uint8)

        # Decode DepthPerspective (float meters)
        if depth_resp.height == 0 or depth_resp.width == 0:
            print("DepthPerspective invalid size; creating zeros.")
            depth_img = np.zeros((h, w), np.float32)
        else:
            depth_flat = np.array(depth_resp.image_data_float, dtype=np.float32)
            depth_img  = depth_flat.reshape(depth_resp.height, depth_resp.width)

        # Align seg/depth to Scene resolution if needed
        if seg_img.shape[:2] != (h, w):
            seg_img = resize_to(seg_img, w, h, is_depth=False)
        if depth_img.shape[:2] != (h, w):
            depth_img = resize_to(depth_img, w, h, is_depth=True)

    finally:
        client.simPause(False)

    # Log poses
    print_pose(f"Actor ({actor})", actor_pose)
    print_pose(f"Camera ({CAMERA_NAME})", cam_pose)

    # Apply Z-offset to actor centroid in NED from selected profile
    adjusted_actor_pose = airsim.Pose(
        position_val=airsim.Vector3r(
            actor_pose.position.x_val,
            actor_pose.position.y_val,
            actor_pose.position.z_val + Z_OFFSET_NED
        ),
        orientation_val=actor_pose.orientation
    )
    print(f"Profile: {OBJECT_TYPE} | Applied Z offset (NED): {Z_OFFSET_NED:+.4f} m")
    print_pose(f"Actor+Zoffset ({actor})", adjusted_actor_pose)

    # Intrinsics (log only)
    K, vfov = compute_intrinsics_from_horizontal_fov(cam_info.fov, w, h)
    print(f"Resolution: {w}×{h}, HFOV: {cam_info.fov:.4f}°, VFOV: {vfov:.4f}°")
    print("K =\n", K, "\n")

    # AirSim projection matrix (4x4 row-major)
    P = np.array(cam_info.proj_mat.matrix, dtype=np.float64).reshape((4,4))
    print("AirSim projection matrix P=\n", P, "\n")

    # Compute box corners (world)
    corners_w = compute_bounding_box_corners_world(
        adjusted_actor_pose, BOX_LENGTH, BOX_WIDTH, BOX_HEIGHT)
    print(f"[{OBJECT_TYPE}] Box L×W×H = {BOX_LENGTH}×{BOX_WIDTH}×{BOX_HEIGHT} m (Z-offset applied)")
    for i, c in enumerate(corners_w):
        print(f" [{i}] x={c[0]:.4f}, y={c[1]:.4f}, z={c[2]:.4f}")
    print()

    # Project & draw amodal 2D box; show ROI crops for seg/depth
    disp = img.copy()
    amodal_bbox = None
    if PROJECTION_ENABLED and cam_pose is not None:
        pts2d, depth_forward, valid = project_world_points_to_image(
            corners_w, cam_pose, P, w, h)

        # draw amodal 2D rectangle and get the rect tuple
        amodal_bbox = draw_2d_bbox_and_get_rect(
            pts2d, valid, w, h, img_to_draw=disp, color=(0,255,0), thickness=2)
    else:
        print("Skipping projection.\n")

    # ROI windows (segmentation & depth) inside the same 2D box
    if SHOW_ROI_WINDOWS and amodal_bbox is not None:
        x0, y0, x1, y1 = amodal_bbox
        # safety clip
        x0 = max(0, min(w-1, x0)); x1 = max(0, min(w-1, x1))
        y0 = max(0, min(h-1, y0)); y1 = max(0, min(h-1, y1))
        if x1 > x0 and y1 > y0:
            seg_roi   = seg_img[y0:y1+1, x0:x1+1, :]
            depth_roi = depth_img[y0:y1+1, x0:x1+1]
            depth_vis = depth_roi_to_vis(depth_roi)

            # --- NEW: apply depth clip to segmentation ROI ---
            if DEPTH_CLIP_ENABLE:
                # keep only pixels with finite depth in (0, DEPTH_CLIP_MAX_M]
                valid_depth_mask = np.isfinite(depth_roi) & (depth_roi > 0) & (depth_roi <= DEPTH_CLIP_MAX_M)
                seg_roi_clipped = np.zeros_like(seg_roi)
                seg_roi_clipped[valid_depth_mask] = seg_roi[valid_depth_mask]

                # --- NEW: list unique colors (RGB) in clipped seg ROI ---
                flat = seg_roi_clipped.reshape(-1, 3)
                # treat pure black as "masked-out"; skip it
                nonzero = np.any(flat != 0, axis=1)
                flat_nz = flat[nonzero]
                if flat_nz.size > 0:
                    # unique in BGR (OpenCV), then convert to RGB for printing
                    colors_bgr, counts = np.unique(flat_nz, axis=0, return_counts=True)
                    colors_rgb = colors_bgr[:, ::-1]  # BGR -> RGB
                    # pretty print
                    rgb_list = [tuple(int(c) for c in row) for row in colors_rgb]
                    print(f"[Depth-clip <= {DEPTH_CLIP_MAX_M:.2f} m] Unique instance colors in seg ROI: {len(rgb_list)}")
                    print(" RGB colors:", rgb_list)
                else:
                    print(f"[Depth-clip <= {DEPTH_CLIP_MAX_M:.2f} m] Unique instance colors in seg ROI: 0")

                # show windows
                if SHOW_ORIGINAL_SEG_ROI:
                    cv2.namedWindow("Seg ROI (raw)", cv2.WINDOW_NORMAL)
                    cv2.imshow("Seg ROI (raw)", seg_roi)
                cv2.namedWindow(f"Seg ROI (depth <= {DEPTH_CLIP_MAX_M:.1f} m)", cv2.WINDOW_NORMAL)
                cv2.imshow(f"Seg ROI (depth <= {DEPTH_CLIP_MAX_M:.1f} m)", seg_roi_clipped)
            else:
                # show only raw seg ROI
                cv2.namedWindow("Seg ROI", cv2.WINDOW_NORMAL)
                cv2.imshow("Seg ROI", seg_roi)

            cv2.namedWindow("Depth ROI", cv2.WINDOW_NORMAL)
            cv2.imshow("Depth ROI", depth_vis)
        else:
            print("Amodal bbox collapsed after clipping; no ROI to show.")

    # annotate & show main window
    cv2.putText(disp, f"Actor: {actor} | Profile: {OBJECT_TYPE} | Zoff(NED): {Z_OFFSET_NED:+.2f}m",
                (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.namedWindow("Projected 3D Bounding Box", cv2.WINDOW_NORMAL)
    cv2.imshow("Projected 3D Bounding Box", disp)

    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optional Open3D viz (unchanged)
    if o3d:
        try:
            frames = []
            Ta = np.eye(4); Ta[:3,:3] = quaternion_to_rotation_matrix(actor_pose.orientation)
            Ta[:3,3]  = [actor_pose.position.x_val, actor_pose.position.y_val, actor_pose.position.z_val]
            fa = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5); fa.transform(Ta)
            frames.append(fa)

            if cam_pose:
                Tc = np.eye(4); Tc[:3,:3] = quaternion_to_rotation_matrix(cam_pose.orientation)
                Tc[:3,3]  = [cam_pose.position.x_val, cam_pose.position.y_val, cam_pose.position.z_val]
                fc = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5); fc.transform(Tc)
                frames.append(fc)

            edges = [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]]
            ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(corners_w),
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
