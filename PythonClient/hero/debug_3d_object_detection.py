#!/usr/bin/env python3
"""
Visible-only 2D bounding box via fusion of:
 - projected 3D cuboid (amodal ROI),
 - DepthPerspective image (meters),
 - instance-segmentation color image (unique color per object).
Robust to mixed resolutions: depth/seg are resized to Scene resolution if needed.
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
# ACTOR_PATTERN      = "BP_VehicleAI_sportscar_.*"
ACTOR_PATTERN      = "StaticMeshActor_UAID_E08F4CF5208AA07502_2022041209"

CAMERA_NAME        = "front_center"
CLIENT_CLASS       = airsim.MultirotorClient     # or airsim.CarClient
PORT               = 41451

PROJECTION_ENABLED = True
DRAW_2D_BBOX       = True

# object profile: "human" or "car"
OBJECT_TYPE = "car"

PROFILES = {
    "human": {"L": 0.5, "W": 0.75, "H": 1.9, "Z": -0.9},
    "car":   {"L": 4.2, "W": 1.9,  "H": 1.6, "Z": -0.55},
}
if OBJECT_TYPE not in PROFILES:
    raise ValueError(f"Unknown OBJECT_TYPE '{OBJECT_TYPE}'. Choose from {list(PROFILES.keys())}.")

BOX_LENGTH    = PROFILES[OBJECT_TYPE]["L"]
BOX_WIDTH     = PROFILES[OBJECT_TYPE]["W"]
BOX_HEIGHT    = PROFILES[OBJECT_TYPE]["H"]
Z_OFFSET_NED  = PROFILES[OBJECT_TYPE]["Z"]

# 2D bbox mode
BBOX_2D_MODE = "modal_fusion"  # "amodal" or "modal_fusion"

# Depth / fusion tuning
SAMPLES_PER_EDGE   = 60
DEPTH_EPS_METERS   = 0.08
COLOR_MIN_PIX      = 50
COLOR_MATCH_RATIO  = 0.15

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

# -------- projection using AirSim's 4x4 projection matrix P ----------
def project_world_points_to_image(world_pts, cam_pose, P, width, height):
    R_cam = quaternion_to_rotation_matrix(cam_pose.orientation)
    cam_p = np.array([cam_pose.position.x_val,
                      cam_pose.position.y_val,
                      cam_pose.position.z_val], dtype=float)
    cam_pts = (R_cam.T @ (world_pts - cam_p).T).T  # Nx3
    pts_h = np.hstack([cam_pts, np.ones((cam_pts.shape[0], 1), dtype=float)])  # Nx4
    clip = (P @ pts_h.T).T
    w_comp = clip[:, 3:4]
    ndc = clip[:, :3] / w_comp
    u = (1.0 - (ndc[:, 0] * 0.5 + 0.5)) * width
    v = (ndc[:, 1] * 0.5 + 0.5) * height
    pts2d = np.stack([u, v], axis=1)
    depth_forward = cam_pts[:, 0]  # X-forward
    valid = depth_forward > 1e-6
    return pts2d, depth_forward, valid, cam_pts
# ---------------------------------------------------------------------

def draw_projected_box(img, pts2d, depth, valid):
    h, w = img.shape[:2]
    if DRAW_2D_BBOX:
        us = pts2d[valid, 0]
        vs = pts2d[valid, 1]
        if us.size and vs.size:
            x0 = int(max(0, np.floor(us.min())))
            x1 = int(min(w-1, np.ceil(us.max())))
            y0 = int(max(0, np.floor(vs.min())))
            y1 = int(min(h-1, np.ceil(vs.max())))
            cv2.rectangle(img, (x0,y0), (x1,y1), (0,255,0), 2)
    else:
        edges = [
            [0,1],[1,3],[3,2],[2,0],
            [4,5],[5,7],[7,6],[6,4],
            [0,4],[1,5],[2,6],[3,7]
        ]
        for i,j in edges:
            if valid[i] and valid[j] and depth[i]>0 and depth[j]>0:
                p1 = tuple(map(int, pts2d[i])); p2 = tuple(map(int, pts2d[j]))
                if 0<=p1[0]<w and 0<=p1[1]<h and 0<=p2[0]<w and 0<=p2[1]<h:
                    cv2.line(img, p1, p2, (0,255,0), 2)
        for idx in range(8):
            if valid[idx] and depth[idx]>0:
                u,v = map(int, pts2d[idx])
                if 0<=u<w and 0<=v<h:
                    cv2.circle(img, (u,v), 4, (0,0,255), -1)
                    cv2.putText(img, str(idx), (u+3, v-3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

# ---------- Fusion utilities (depth + instance-seg color) ----------
def get_depth_perspective(client, camera_name):
    """DepthPerspective as float32 meters (H,W)."""
    resp = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True, False)
    ])[0]
    depth = np.array(resp.image_data_float, dtype=np.float32)
    if resp.height == 0 or resp.width == 0 or depth.size != resp.height * resp.width:
        raise RuntimeError("Depth image invalid or empty")
    return depth.reshape(resp.height, resp.width)  # meters

def get_segmentation_png(client, camera_name):
    """Segmentation image (unique color per instance) as BGR uint8 (H,W,3)."""
    resp = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.Segmentation, False, True)
    ])[0]
    seg = cv2.imdecode(np.frombuffer(resp.image_data_uint8, np.uint8), cv2.IMREAD_COLOR)
    if seg is None:
        raise RuntimeError("Failed to decode Segmentation image")
    return seg

def box_faces_from_corners(c):
    return [
        np.vstack([c[0], c[1], c[3], c[2]]),  # +X
        np.vstack([c[4], c[5], c[7], c[6]]),  # -X
        np.vstack([c[0], c[1], c[5], c[4]]),  # +Y
        np.vstack([c[2], c[3], c[7], c[6]]),  # -Y
        np.vstack([c[0], c[2], c[6], c[4]]),  # +Z
        np.vstack([c[1], c[3], c[7], c[5]]),  # -Z
    ]

# (fixed) bilinear sampler with proper broadcasting
def bilinear_face_samples(quad_pts, n):
    v00, v10, v11, v01 = [np.asarray(v, dtype=float) for v in quad_pts]
    s = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 1.0, n)
    S, T = np.meshgrid(s, t, indexing='xy')   # (n,n)
    S = S[..., None]                          # (n,n,1)
    T = T[..., None]                          # (n,n,1)
    pts = ((1 - S) * (1 - T) * v00 +
           S       * (1 - T) * v10 +
           S       * T       * v11 +
           (1 - S) * T       * v01)          # -> (n,n,3)
    return pts.reshape(-1, 3)

def build_predicted_depth_map_from_box(corners_w, cam_pose, P, width, height,
                                       samples_per_edge=SAMPLES_PER_EDGE):
    """
    Project dense samples from all 6 faces. For each pixel, store the minimum
    Euclidean range (meters) among samples mapping to that pixel.
    """
    faces = box_faces_from_corners(corners_w)
    pts_world = np.vstack([bilinear_face_samples(f, samples_per_edge) for f in faces])
    # Project
    R_cam = quaternion_to_rotation_matrix(cam_pose.orientation)
    cam_p = np.array([cam_pose.position.x_val, cam_pose.position.y_val, cam_pose.position.z_val], dtype=float)
    cam_pts = (R_cam.T @ (pts_world - cam_p).T).T  # (N,3)
    pts_h  = np.hstack([cam_pts, np.ones((cam_pts.shape[0],1), dtype=float)])
    clip   = (P @ pts_h.T).T
    ndc    = clip[:, :3] / clip[:, 3:4]
    u_pix  = np.round((1.0 - (ndc[:,0]*0.5 + 0.5)) * width).astype(int)
    v_pix  = np.round((ndc[:,1]*0.5 + 0.5) * height).astype(int)
    in_front = cam_pts[:,0] > 1e-6
    on_img   = (u_pix >= 0) & (u_pix < width) & (v_pix >= 0) & (v_pix < height)
    keep = in_front & on_img
    if not np.any(keep):
        return None
    u_pix, v_pix = u_pix[keep], v_pix[keep]
    rng = np.linalg.norm(cam_pts[keep], axis=1).astype(np.float32)  # meters along ray
    pred = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(pred, (v_pix, u_pix), rng)
    return pred

def unique_colors_in_roi(seg_img, x0, y0, x1, y1):
    roi = seg_img[max(0,y0):min(seg_img.shape[0], y1+1),
                  max(0,x0):min(seg_img.shape[1], x1+1)]
    if roi.size == 0:
        return []
    colors, counts = np.unique(roi.reshape(-1,3), axis=0, return_counts=True)
    order = np.argsort(-counts)
    out = []
    for idx in order:
        color = tuple(int(v) for v in colors[idx])
        if color == (0,0,0):
            continue
        if counts[idx] < COLOR_MIN_PIX:
            continue
        out.append(color)
    return out

def modal_box_via_fusion(seg_img, depth_img, pred_depth, amodal_bbox,
                         depth_eps=DEPTH_EPS_METERS):
    """
    Resolve which instance-seg color inside amodal ROI corresponds to the target box
    by checking depth consistency against predicted box depth. Return visible-only box.
    """
    h, w = seg_img.shape[:2]
    x0a, y0a, x1a, y1a = amodal_bbox
    x0a = max(0, min(w-1, x0a)); x1a = max(0, min(w-1, x1a))
    y0a = max(0, min(h-1, y0a)); y1a = max(0, min(h-1, y1a))
    if x1a <= x0a or y1a <= y0a:
        return None, None, None

    # candidate colors inside ROI
    cand_colors = unique_colors_in_roi(seg_img, x0a, y0a, x1a, y1a)
    if len(cand_colors) == 0:
        return None, None, None

    # crop arrays to ROI for efficiency
    seg_roi   = seg_img[y0a:y1a+1, x0a:x1a+1, :]
    depth_roi = depth_img[y0a:y1a+1, x0a:x1a+1]
    pred_roi  = pred_depth[y0a:y1a+1, x0a:x1a+1]

    best_color = None
    best_agree = -1
    best_mask  = None

    finite_pred = np.isfinite(pred_roi)
    valid_depth = (depth_roi > 0)

    for color in cand_colors:
        mask = np.all(seg_roi == np.array(color, dtype=np.uint8), axis=2)
        if mask.sum() < COLOR_MIN_PIX:
            continue
        agree = mask & finite_pred & valid_depth & (np.abs(depth_roi - pred_roi) <= depth_eps)
        agree_count = int(agree.sum())
        agree_ratio = agree_count / float(max(1, mask.sum()))
        if agree_ratio >= COLOR_MATCH_RATIO and agree_count > best_agree:
            best_agree = agree_count
            best_color = color
            best_mask  = agree

    if best_mask is None:
        return None, None, None

    ys, xs = np.where(best_mask)
    if xs.size == 0:
        return None, None, None

    # modal (visible) bbox in full-image coords
    x0 = x0a + int(xs.min())
    x1 = x0a + int(xs.max())
    y0 = y0a + int(ys.min())
    y1 = y0a + int(ys.max())
    modal_bbox = (x0, y0, x1, y1)

    # occlusion estimate
    vis_area   = (x1 - x0 + 1) * (y1 - y0 + 1)
    amodal_area= (x1a - x0a + 1) * (y1a - y0a + 1)
    vis_ratio  = vis_area / float(max(1, amodal_area))
    if   vis_ratio > 0.95: occ_tag = 0
    elif vis_ratio > 0.5:  occ_tag = 1
    else:                  occ_tag = 2
    ignore_flag = (vis_ratio < 0.2)

    return modal_bbox, occ_tag, ignore_flag

# ---------- RESOLUTION ALIGNMENT ----------
def resize_to(img, target_w, target_h, is_depth=False):
    """Resize image to (target_h, target_w). Depth uses nearest to preserve values."""
    if img.shape[1] == target_w and img.shape[0] == target_h:
        return img
    interp = cv2.INTER_NEAREST if is_depth or img.ndim == 3 else cv2.INTER_NEAREST
    return cv2.resize(img, (target_w, target_h), interpolation=interp)

def main():
    np.set_printoptions(precision=4, suppress=True)
    client = CLIENT_CLASS(port=PORT)
    client.confirmConnection()

    print("Connected!")

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

    # --- 3) pause + grab synchronously ---
    client.simPause(True)
    try:
        try:
            actor_pose = client.simGetObjectPose(actor, True)
        except TypeError:
            actor_pose = client.simGetObjectPose(actor)
        cam_info = client.simGetCameraInfo(CAMERA_NAME)
        cam_pose = cam_info.pose if cam_info else None

        # Scene RGB
        img_resp = client.simGetImages([
            airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, True)
        ])[0]
        img = cv2.imdecode(np.frombuffer(img_resp.image_data_uint8, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Scene image decode failed")
        h, w = img.shape[:2]

        # Depth & Segmentation
        if PROJECTION_ENABLED and cam_pose is not None and BBOX_2D_MODE == "modal_fusion":
            depth_img_raw = get_depth_perspective(client, CAMERA_NAME)  # (Hd,Wd)
            seg_img_raw   = get_segmentation_png(client, CAMERA_NAME)   # (Hs,Ws,3)

            # shape logs
            print(f"Scene shape: {img.shape}  Seg shape: {seg_img_raw.shape}  Depth shape: {depth_img_raw.shape}")

            # align to Scene resolution if needed
            depth_img = resize_to(depth_img_raw, w, h, is_depth=True)
            seg_img   = resize_to(seg_img_raw,   w, h, is_depth=False)

            if depth_img.shape[:2] != (h, w) or seg_img.shape[:2] != (h, w):
                raise RuntimeError("Failed to align depth/seg to Scene resolution")
            if depth_img.dtype != np.float32:
                depth_img = depth_img.astype(np.float32)
        else:
            depth_img = None
            seg_img   = None
    finally:
        client.simPause(False)

    # --- 4) print poses ---
    print_pose(f"Actor ({actor})", actor_pose)
    print_pose(f"Camera ({CAMERA_NAME})", cam_pose)

    # --- 4a) apply Z-offset to actor centroid in NED ---
    adjusted_actor_pose = airsim.Pose(
        position_val=airsim.Vector3r(
            actor_pose.position.x_val,
            actor_pose.position.y_val,
            actor_pose.position.z_val + Z_OFFSET_NED  # +Z is DOWN in NED
        ),
        orientation_val=actor_pose.orientation
    )
    print(f"Profile: {OBJECT_TYPE} | Applied Z offset (NED): {Z_OFFSET_NED:+.4f} m")
    print_pose(f"Actor+Zoffset ({actor})", adjusted_actor_pose)

    # --- 5) intrinsics (for logging) ---
    K, vfov = compute_intrinsics_from_horizontal_fov(cam_info.fov, w, h)
    print(f"Resolution: {w}×{h}, HFOV: {cam_info.fov:.4f}°, VFOV: {vfov:.4f}°")
    print("K =\n", K, "\n")

    # --- AirSim projection matrix (4×4 row-major) ---
    P = np.array(cam_info.proj_mat.matrix, dtype=np.float64).reshape((4,4))
    print("AirSim projection matrix P=\n", P, "\n")

    # --- 6) box corners (world) ---
    corners_w = compute_bounding_box_corners_world(
        adjusted_actor_pose, BOX_LENGTH, BOX_WIDTH, BOX_HEIGHT)
    print(f"[{OBJECT_TYPE}] Box L×W×H = {BOX_LENGTH}×{BOX_WIDTH}×{BOX_HEIGHT} m (Z-offset applied)")
    for i, c in enumerate(corners_w):
        print(f" [{i}] x={c[0]:.4f}, y={c[1]:.4f}, z={c[2]:.4f}")
    print()

    # --- 7) project & draw ---
    disp = img.copy()
    if PROJECTION_ENABLED and cam_pose is not None:
        pts2d, depth_forward, valid, _ = project_world_points_to_image(
            corners_w, cam_pose, P, w, h)
        print("Using AirSim P-based projection:")
        for i, ((u,v), d, ok) in enumerate(zip(pts2d, depth_forward, valid)):
            status = "vis" if ok and d>0 else "out"
            print(f"[{i}] u={u:.1f}, v={v:.1f}, depthX={d:.3f} ({status})")
        print()

        # Amodal 2D rectangle from valid projected corners (clipped)
        us = pts2d[valid, 0]; vs = pts2d[valid, 1]
        amodal_bbox = None
        if us.size and vs.size:
            x0a = int(max(0, math.floor(us.min())))
            x1a = int(min(w-1, math.ceil(us.max())))
            y0a = int(max(0, math.floor(vs.min())))
            y1a = int(min(h-1, math.ceil(vs.max())))
            amodal_bbox = (x0a, y0a, x1a, y1a)

        drew_modal = False
        if DRAW_2D_BBOX and BBOX_2D_MODE == "modal_fusion" and amodal_bbox is not None and depth_img is not None and seg_img is not None:
            # 1) predicted object depth map from the 3D box (same resolution as Scene)
            pred_depth = build_predicted_depth_map_from_box(
                corners_w, cam_pose, P, w, h, samples_per_edge=SAMPLES_PER_EDGE)

            if pred_depth is not None:
                # shape log for safety
                print(f"pred_depth shape: {pred_depth.shape}  depth_img shape: {depth_img.shape}  seg shape: {seg_img.shape}")

                # 2) choose instance color that best matches object's predicted depth in ROI
                modal_bbox, occ_tag, ignore_flag = modal_box_via_fusion(
                    seg_img, depth_img, pred_depth, amodal_bbox, depth_eps=DEPTH_EPS_METERS)
                if modal_bbox is not None:
                    x0,y0,x1,y1 = modal_bbox
                    cv2.rectangle(disp, (x0,y0), (x1,y1), (0,200,255), 2)  # modal visible box
                    # faint amodal for debugging
                    xa,ya,xb,yb = amodal_bbox
                    cv2.rectangle(disp, (xa,ya), (xb,yb), (100,100,100), 1)
                    tag_txt = f"occ={occ_tag}{' IGNORE' if ignore_flag else ''}"
                    cv2.putText(disp, tag_txt, (x0, max(0,y0-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)
                    drew_modal = True

        if DRAW_2D_BBOX and not drew_modal:
            # fallback: draw amodal rectangle
            if amodal_bbox is not None:
                xa,ya,xb,yb = amodal_bbox
                cv2.rectangle(disp, (xa,ya), (xb,yb), (0,255,0), 2)
        if not DRAW_2D_BBOX:
            # draw 3D wireframe instead
            draw_projected_box(disp, pts2d, depth_forward, valid)
    else:
        print("Skipping projection.\n")

    # --- 8) annotate & show ---
    cv2.putText(disp, f"Actor: {actor} | Profile: {OBJECT_TYPE} | Zoff(NED): {Z_OFFSET_NED:+.2f}m",
                (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
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
            fa = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5); fa.transform(Ta)
            frames.append(fa)

            if cam_pose:
                Tc = np.eye(4)
                Tc[:3,:3] = quaternion_to_rotation_matrix(cam_pose.orientation)
                Tc[:3,3] = [cam_pose.position.x_val,
                            cam_pose.position.y_val,
                            cam_pose.position.z_val]
                fc = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5); fc.transform(Tc)
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
