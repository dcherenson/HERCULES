#!/usr/bin/env python3

"""
Visualize DAIR-V2X-style outputs:
- 2D image with labeled 2D boxes and reprojected 3D wireframes (from labels)
- 3D view (Open3D, if available) with LiDAR point cloud and 3D boxes in LiDAR frame
  (requires T_cam_lidar in calib.json). Falls back to BEV with Matplotlib if Open3D
  is not installed; if extrinsics are missing, shows boxes without fusing LiDAR.

Usage examples:
  python3 visualize_dairv2x.py --root /path/to/_dair_v2x_out --side veh --step 5
  python3 visualize_dairv2x.py --root /path/to/_dair_v2x_out --side inf --limit 100 --show_3d
"""

import os
import json
import argparse
import math
import glob
import warnings
import numpy as np
import cv2

# Optional 3D backends
try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ---------------------------- Paths helpers ----------------------------
def dair_paths(root):
    return {
        "inf": {
            "img":         f"{root}/cooperative/infrastructure-side/image",
            "lidar":       f"{root}/cooperative/infrastructure-side/lidar",
            "calib":       f"{root}/cooperative/infrastructure-side/calib",
            "kitti_label": f"{root}/cooperative/infrastructure-side/kitti_label",
            "kitti_label_pp": f"{root}/cooperative/infrastructure-side/kitti_label_pp",
            "ts":          f"{root}/cooperative/infrastructure-side/timestamp",
        },
        "veh": {
            "img":         f"{root}/cooperative/vehicle-side/image",
            "lidar":       f"{root}/cooperative/vehicle-side/lidar",
            "calib":       f"{root}/cooperative/vehicle-side/calib",
            "kitti_label": f"{root}/cooperative/vehicle-side/kitti_label",
            "kitti_label_pp": f"{root}/cooperative/vehicle-side/kitti_label_pp",
            "ts":          f"{root}/cooperative/vehicle-side/timestamp",
        },
    }


# ---------------------------- Geometry utils ----------------------------
_EDGES = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]

def rotz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[ c,-s,0],
                     [ s, c,0],
                     [ 0, 0,1]], dtype=float)

def box3d_corners_cam_airshim(length, width, height, center_cam, yaw_cam):
    """
    Build 8 corners in the AirSim CAMERA frame:
      X forward, Y right, Z down. yaw_cam is rotation about +Z (down) axis.
    """
    l, w, h = float(length), float(width), float(height)
    # local box corners (X forward, Y right, Z down)
    x = l/2.0; y = w/2.0; z = h/2.0
    corners_local = np.array([
        [ x,  y,  z], [ x,  y, -z],
        [ x, -y,  z], [ x, -y, -z],
        [-x,  y,  z], [-x,  y, -z],
        [-x, -y,  z], [-x, -y, -z],
    ], dtype=float)
    R = rotz(yaw_cam)
    corners_cam = (R @ corners_local.T).T + np.asarray(center_cam, dtype=float)
    return corners_cam  # (8,3) in camera coords

def project_cam_airshim_to_image(K, pts_cam, img_w, img_h):
    """
    Project AirSim CAMERA-frame points to pixels with pinhole intrinsics K.
    AirSim camera axes: X forward, Y right, Z down
    Image pinhole expects: X_img right, Y_img down, Z_img forward
        => X_img = Y_cam,  Y_img = Z_cam,  Z_img = X_cam
    Pixel: u = fx * X_img/Z_img + cx; v = fy * Y_img/Z_img + cy
    Returns (uv Nx2), valid mask (in front and inside image).
    """
    Xf = pts_cam[:,0]; Yr = pts_cam[:,1]; Zd = pts_cam[:,2]
    Zi = Xf
    valid_z = Zi > 1e-6
    u = K[0,0]*(Yr/Zi) + K[0,2]
    v = K[1,1]*(Zd/Zi) + K[1,2]
    uv = np.stack([u, v], axis=1)
    in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    valid = valid_z & np.isfinite(u) & np.isfinite(v) & in_img
    return uv, valid

def rect_from_points(uv, mask, w, h):
    if not np.any(mask):
        return None
    u = uv[mask,0]; v = uv[mask,1]
    x0 = int(max(0, np.floor(u.min())))
    y0 = int(max(0, np.floor(v.min())))
    x1 = int(min(w-1, np.ceil(u.max())))
    y1 = int(min(h-1, np.ceil(v.max())))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)

def iou(a, b):
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter + 1e-9
    return inter / union


# ---------------------------- LiDAR utils ----------------------------
def load_lidar_bin(bin_path):
    """KITTI-like .bin: float32 [x, y, z, reflectance]; we keep Nx3."""
    pts = np.fromfile(bin_path, dtype=np.float32)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    pts = pts.reshape(-1, 4)[:, :3]
    return pts

def transform_points(T, pts):
    """Apply 4x4 homogeneous transform to Nx3 points."""
    if pts.size == 0:
        return pts.reshape(0, 3)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=float)])
    out = (T @ pts_h.T).T[:, :3]
    return out


# ---------------------------- 3D viz helpers ----------------------------
def corners3d_from_label_cam(obj):
    """Return (8,3) box corners in CAMERA frame using AirSim camera axes."""
    dims = obj["3d_dimensions"]
    Hh, Wd, Ld = float(dims["h"]), float(dims["w"]), float(dims["l"])
    loc = obj["3d_location"]
    Xf, Yr, Zd = float(loc["x"]), float(loc["y"]), float(loc["z"])
    yaw = float(obj["rotation"])
    return box3d_corners_cam_airshim(Ld, Wd, Hh, (Xf, Yr, Zd), yaw)

def lineset_from_corners_o3d(corners_xyz, color=(1.0, 0.0, 0.0)):
    """Open3D LineSet for one box (corners in same frame as point cloud)."""
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners_xyz.astype(float))
    ls.lines  = o3d.utility.Vector2iVector(_EDGES)
    col = np.array([color], dtype=float).repeat(len(_EDGES), axis=0)
    ls.colors = o3d.utility.Vector3dVector(col)
    return ls

def plot_bev(ax, pts_xy, boxes_xy_list, pts_stride=2):
    """Matplotlib BEV plot (X forward, Y right)."""
    ax.cla()
    if pts_xy.size:
        ax.scatter(pts_xy[::pts_stride, 0], pts_xy[::pts_stride, 1], s=0.2, c="k", alpha=0.5)
    for B in boxes_xy_list:
        for a, b in _EDGES:
            ax.plot([B[a, 0], B[b, 0]], [B[a, 1], B[b, 1]], linewidth=1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X forward (m)")
    ax.set_ylabel("Y right (m)")
    ax.grid(True)


# ---------------------------- Main ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="OUT_ROOT used in run_dairv2x_oop.py")
    ap.add_argument("--side", required=True, choices=["veh", "inf"])
    ap.add_argument("--limit", type=int, default=100, help="max number of images to visualize")
    ap.add_argument("--step", type=int, default=1, help="sample every Nth image")
    ap.add_argument("--show_3d", action="store_true", help="Open 3D/BEV viewer (Open3D preferred, else Matplotlib)")
    ap.add_argument("--max_lidar_pts", type=int, default=250000, help="downsample LiDAR if more than this many points")
    ap.add_argument("--pp", action="store_true",
                    help="Read LiDAR-frame labels from kitti_label_pp instead of camera-frame kitti_label")
    ap.add_argument("--delay_ms", type=int, default=0,
                    help="Per-frame delay for the 2D window (0 = wait for key press)")
    ap.add_argument("--hold_last", action="store_true",
                    help="After processing, keep the last 2D/3D window open until you close it")
    ap.add_argument("--overlay_3d_on_2d", action="store_true",
                    help="(Optional) also draw 3D wireframes on the 2D image; OFF by default")
    args = ap.parse_args()

    P = dair_paths(args.root)[args.side]
    label_dir_key = "kitti_label_pp" if args.pp else "kitti_label"
    calib_path = os.path.join(P["calib"], "calib.json")
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"Missing calib: {calib_path}")

    calib = json.load(open(calib_path, "r"))
    # Prefer K derived from saved 4x4 projection if present (matches labeling-time math)
    P_cal = np.asarray(calib.get("P", np.eye(4)), dtype=float)
    if P_cal.shape == (4, 4) and np.isfinite(P_cal).all() and not np.allclose(P_cal, 0):
        K = P_cal[:3, :3].copy()
    else:
        K = np.asarray(calib["K"], dtype=float)

    img_size = calib["image_size"]
    # support [w,h] or [h,w]
    if len(img_size) == 2:
        # Most of your scripts store as (w, h)
        img_w, img_h = int(img_size[0]), int(img_size[1])
    else:
        raise ValueError("Unexpected image_size format in calib.json")

    # Extrinsics: lidar -> camera (already in Virtual-LiDAR basis in your collector)
    T_cam_lidar = None
    if "T_cam_lidar" in calib:
        T_cam_lidar = np.asarray(calib["T_cam_lidar"], dtype=float)
        if T_cam_lidar.shape != (4, 4):
            warnings.warn("T_cam_lidar found but not 4x4; ignoring.")
            T_cam_lidar = None

    # If we have lidar->camera, we can make camera->lidar transform
    T_lidar_cam = None
    if T_cam_lidar is not None:
        try:
            T_lidar_cam = np.linalg.inv(T_cam_lidar)
        except np.linalg.LinAlgError:
            warnings.warn("T_cam_lidar not invertible; 3D fusion disabled.")
            T_lidar_cam = None

    imgs = sorted(glob.glob(os.path.join(P["img"], "*.png")))
    if not imgs:
        print(f"[INFO] No images found in {P['img']}")
        return
    print(f"[INFO] Found {len(imgs)} images.")

    # 3D/BEV viewers (created lazily)
    o3d_vis = None
    bev_fig = bev_ax = None
    warned_no_extrinsics = False
    last_lidar = np.empty((0, 3), dtype=np.float32)
    last_boxes_lidar = []

    bad_iou = 0
    total_iou = 0

    # Iterate frames
    for impath in imgs[::max(1, args.step)][:args.limit]:
        base = os.path.splitext(os.path.basename(impath))[0]
        # Labels: prefer zero-padded numeric filename if applicable
        if base.isdigit():
            lbl_path = os.path.join(P[label_dir_key], f"{int(base):06d}.json")
            lidar_path = os.path.join(P["lidar"], f"{int(base):06d}.bin")
        else:
            lbl_path = os.path.join(P[label_dir_key], f"{base}.json")
            lidar_path = os.path.join(P["lidar"], f"{base}.bin")

        if not os.path.isfile(lbl_path):
            # try mirror name (non-padded)
            alt = os.path.join(P[label_dir_key], f"{base}.json")            
            if os.path.isfile(alt):
                lbl_path = alt
            else:
                print(f"[WARN] Missing label for {base}")
                continue

        # Make LiDAR path robust to your non-padded naming (e.g., 100.bin)
        if args.show_3d and not os.path.isfile(lidar_path):
            alt_lidar = os.path.join(P["lidar"], f"{base}.bin")
            if os.path.isfile(alt_lidar):
                lidar_path = alt_lidar
            else:
                print(f"[WARN] Missing LiDAR for {base} (looked for {os.path.basename(lidar_path)} and {os.path.basename(alt_lidar)})")


        # Load image + labels
        img = cv2.imread(impath, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read image {impath}")
            continue
        h, w = img.shape[:2]

        labels = json.load(open(lbl_path, "r"))
        if not isinstance(labels, list):
            print(f"[WARN] Label file not a list: {lbl_path}")
            labels = []

        # Optional: load LiDAR
        lidar = np.empty((0, 3), dtype=np.float32)
        if args.show_3d and os.path.isfile(lidar_path):
            lidar = load_lidar_bin(lidar_path)
            if lidar.shape[0] > args.max_lidar_pts:
                # simple voxel-ish downsample: stride based on count
                stride = int(math.ceil(lidar.shape[0] / args.max_lidar_pts))
                lidar = lidar[::max(1, stride)]

        # Prepare per-frame 3D box list in LiDAR frame (for 3D viewer)
        boxes_lidar_this_frame = []

        # 2D draw loop
        for obj in labels:
            typ = obj.get("type", "Object")
            bx = obj.get("2d_box", {})
            box2d = (
                int(bx.get("xmin", 0)),
                int(bx.get("ymin", 0)),
                int(bx.get("xmax", 0)),
                int(bx.get("ymax", 0)),
            )
            color2d = (0, 255, 0) if typ == "Car" else (0, 128, 255)
            cv2.rectangle(img, (box2d[0], box2d[1]), (box2d[2], box2d[3]), color2d, 2)
            cv2.putText(img, typ, (box2d[0], max(0, box2d[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color2d, 2, cv2.LINE_AA)

            # Reproject 3D box (from label 3D in camera frame) for IoU check only
            try:
                corners_cam = corners3d_from_label_cam(obj)
            except Exception as e:
                print(f"[WARN] Bad 3D label in {lbl_path}: {e}")
                continue

            uv, valid = project_cam_airshim_to_image(K, corners_cam, w, h)
            # (No 3D overlay on 2D by default; enable with --overlay_3d_on_2d)
            if args.overlay_3d_on_2d:
                for a, b in _EDGES:
                    if valid[a] and valid[b]:
                        p0 = (int(uv[a, 0]), int(uv[a, 1]))
                        p1 = (int(uv[b, 0]), int(uv[b, 1]))
                        cv2.line(img, p0, p1, (255, 0, 0), 2)

            # IoU between labeled 2D rect and projected 3D rect hull
            proj_rect = rect_from_points(uv, valid, w, h)
            if proj_rect is not None:
                i = iou(box2d, proj_rect)
                total_iou += 1
                if i < 0.25:
                    bad_iou += 1
                    cv2.putText(img, f"IoU:{i:.2f}", (box2d[0], min(h - 5, box2d[1] + 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # For 3D viewer: transform camera-frame box corners into LiDAR frame
            if args.show_3d and lidar.size and (T_lidar_cam is not None):
                corners_cam_h = np.hstack([corners_cam, np.ones((8, 1))])
                corners_lid_h = (T_lidar_cam @ corners_cam_h.T).T
                boxes_lidar_this_frame.append(corners_lid_h[:, :3])

        # Show 2D overlay (blocking by default unless delay>0)
        cv2.imshow(f"{args.side} image overlay", img)
        delay = args.delay_ms
        key = cv2.waitKey(0 if delay <= 0 else delay) & 0xFF
        if key in (27, ord('q')):
            break

        # 3D / BEV visualization
        if args.show_3d:
            if lidar.size and (T_lidar_cam is not None):
                # Fused 3D: LiDAR frame + boxes transformed into LiDAR frame
                if HAS_O3D:
                    if o3d_vis is None:
                        o3d_vis = o3d.visualization.Visualizer()
                        o3d_vis.create_window(window_name=f"{args.side} LiDAR 3D", width=960, height=720)

                    # Rebuild simple scene each frame (keeps code straightforward)
                    o3d_vis.clear_geometries()
                    o3d_vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

                    # Point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(lidar.astype(float))
                    # neutral gray
                    if len(lidar):
                        col = np.full((lidar.shape[0], 3), 0.5, dtype=float)
                        pcd.colors = o3d.utility.Vector3dVector(col)
                    o3d_vis.add_geometry(pcd)

                    # Boxes
                    for B in boxes_lidar_this_frame:
                        ls = lineset_from_corners_o3d(B, color=(1.0, 0.0, 0.0))
                        o3d_vis.add_geometry(ls)

                    o3d_vis.poll_events()
                    o3d_vis.update_renderer()

                    # remember last visuals (for optional hold)
                    last_lidar = lidar.copy()
                    last_boxes_lidar = [B.copy() for B in boxes_lidar_this_frame]

                elif HAS_MPL:
                    if bev_fig is None:
                        bev_fig, bev_ax = plt.subplots(figsize=(6, 6))
                        plt.ion()
                        plt.show()
                    pts_xy = lidar[:, :2]
                    plot_bev(bev_ax, pts_xy, boxes_lidar_this_frame, pts_stride=2)
                    plt.pause(0.001)
                else:
                    if not warned_no_extrinsics:
                        print("[INFO] Open3D/Matplotlib not available; skipping 3D/BEV.")
                        warned_no_extrinsics = True

            else:
                # Missing LiDAR or extrinsics: we cannot fuse; warn once
                if not warned_no_extrinsics:
                    if not lidar.size:
                        print("[WARN] No LiDAR for this frame; 3D fusion disabled for now.")
                    if T_lidar_cam is None and T_cam_lidar is None:
                        print("[WARN] calib.json missing T_cam_lidar → cannot fuse camera/ LiDAR. "
                              "Add T_cam_lidar to calib.json to enable.")
                    warned_no_extrinsics = True

    cv2.destroyAllWindows()

    if HAS_O3D and o3d_vis is not None:
        o3d_vis.destroy_window()
    if HAS_MPL and plt.get_fignums():
        plt.close('all')

    # Optionally hold last frame(s) open (blocking) until user closes
    if args.hold_last:
        # 2D window: block until key press (if still open)
        cv2.imshow(f"{args.side} image overlay", np.zeros((1,1,3), dtype=np.uint8))  # ensure exists
        cv2.waitKey(0)
        # 3D window: rebuild a static Open3D scene and block
        if args.show_3d and HAS_O3D and last_lidar.size:
            geoms = []
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(last_lidar.astype(float))
            col = np.full((last_lidar.shape[0], 3), 0.5, dtype=float)
            pcd.colors = o3d.utility.Vector3dVector(col)
            geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
            geoms.append(pcd)
            for B in last_boxes_lidar:
                geoms.append(lineset_from_corners_o3d(B, color=(1.0, 0.0, 0.0)))
            o3d.visualization.draw_geometries(geoms, window_name=f"{args.side} LiDAR 3D (hold)")

    if total_iou > 0:
        frac = 100.0 * bad_iou / total_iou
        print(f"[CHECK] Projected-vs-labeled IoU: bad {bad_iou}/{total_iou} ({frac:.1f}% with IoU<0.25)")
    else:
        print("[CHECK] No comparable boxes encountered.")


if __name__ == "__main__":
    main()
