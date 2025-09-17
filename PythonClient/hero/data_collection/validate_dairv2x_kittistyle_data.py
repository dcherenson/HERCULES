#!/usr/bin/env python3
"""
DAIR-V2X-C viewer (infra/veh) for 2D GT (OpenCV) and 3D GT (Open3D)

What it does
------------
- Shows the camera image with KITTI-format 2D GT boxes from label_2/*.txt.
- Shows the LiDAR point cloud (.bin) with 3D GT boxes drawn in the LiDAR frame.
- Works for either 'infra' (infrastructure-side) or 'veh' (vehicle-side).
- No CLI args; set variables below.

Controls
--------
- Press any key in the OpenCV window to advance to the next frame.
- Open3D window opens per frame; close it to continue (or set SHOW_OPEN3D=False).
- Set SAVE_OUTPUTS=True to also save image overlays and exported Open3D snapshots.

Notes
-----
- For infrastructure-side, calibrations may use "virtuallidar_to_camera" internally.
  This script automatically picks Tr_* line from the calib txt file:
  one of {'Tr_velo_to_cam', 'Tr_lidar_to_cam', 'Tr_lidar_to_camera',
          'Tr_virtuallidar_to_camera'} plus R0_rect and P2.
- The 3D KITTI box is defined in CAMERA coordinates (rect), so we:
    1) build 3D box corners in camera-rect coords using (h,w,l), center (x,y,z), ry
    2) transform to LiDAR coords via inv(Tr_velo_to_cam) * inv(R0_rect)
- If you are running headless (no X), set SHOW_OPENCV=False, SHOW_OPEN3D=False and use SAVE_OUTPUTS=True.
"""

import os
import glob
import math
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path

# ===================== User-configurable variables =====================

# Path to your DAIR-V2X-C cooperative-vehicle-infrastructure folder
# DATA_ROOT = "/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets/DAIR-V2X-C/cooperative-vehicle-infrastructure"
DATA_ROOT = "/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets/dair_v2x_synth_kitti/"
# Choose side: 'infra' or 'veh'
SIDE = "infra"     # "infra" -> infrastructure-side, "veh" -> vehicle-side
# SIDE = "veh"

# Split to visualize
SPLIT = "training"  # "training" or "testing" (2D labels exist typically for training)

# Start index and how many frames to show
START_IDX = 0
MAX_FRAMES = 20

# Visualization toggles
SHOW_OPENCV  = True
SHOW_OPEN3D  = True
SAVE_OUTPUTS = False
OUTPUT_DIR = "/home/sgarimella34/vis_custom_viewer"

# Point size in Open3D and box line width
O3D_POINT_SIZE = 1.0

# =======================================================================
def side_dir_name(side: str) -> str:
    if side.lower() in ("infra", "infrastructure", "infrastructure-side"):
        return "infrastructure-side"
    elif side.lower() in ("veh", "vehicle", "vehicle-side"):
        return "vehicle-side"
    else:
        raise ValueError("SIDE must be 'infra' or 'veh'")


def get_split_dirs(root: str, side: str, split: str):
    sd = side_dir_name(side)
    base = Path(root) / sd / split
    img_dir   = base / "image_2"
    lbl2_dir  = base / "label_2"
    velo_dir  = base / "velodyne"
    calib_dir = base / "calib"
    for p in [img_dir, lbl2_dir, velo_dir, calib_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Expected path missing: {p}")
    return img_dir, lbl2_dir, velo_dir, calib_dir


def read_calib(calib_file: str):
    """
    Reads KITTI-like calib file. Returns:
      P2 (3x4), R0_rect (3x3), Tr_velo_to_cam (4x4)
    Works for DAIR-V2X-C by accepting several possible line keys for the lidar->cam extrinsic.
    """
    P2 = None
    R0_rect = None
    Tr = None

    # possible keys in DAIR / KITTI-style files
    lidar_to_cam_keys = [
        "Tr_velo_to_cam", "Tr_velo_to_camera",
        "Tr_lidar_to_cam", "Tr_lidar_to_camera",
        "Tr_virtuallidar_to_camera"
    ]

    with open(calib_file, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or ":" not in line:
                continue
            key, val = line.split(":", 1)
            val = val.strip()
            nums = np.array([float(x) for x in val.split()], dtype=np.float64)

            if key == "P2":
                P2 = nums.reshape(3, 4)
            elif key in ("R0_rect", "R_rect"):
                # Sometimes listed as 9 nums
                if nums.size == 9:
                    R0_rect = nums.reshape(3, 3)
                elif nums.size == 12:
                    # edge case: 3x4, take first 3x3
                    R0_rect = nums.reshape(3, 4)[:, :3]
                else:
                    raise ValueError("Unexpected R0_rect size")
            elif key in lidar_to_cam_keys:
                if nums.size == 12:
                    Tr = nums.reshape(3, 4)
                    # convert to 4x4
                    Tr = np.vstack([Tr, np.array([0, 0, 0, 1.0])])
                elif nums.size == 16:
                    Tr = nums.reshape(4, 4)
                else:
                    raise ValueError("Unexpected Tr_* size")

    if P2 is None:
        raise ValueError(f"P2 not found in {calib_file}")
    if R0_rect is None:
        # If absent, assume identity (rare)
        R0_rect = np.eye(3, dtype=np.float64)
    if Tr is None:
        raise ValueError(f"No lidar->cam extrinsic found in {calib_file}")

    # Build 4x4 rectified rotation
    R_rect_4x4 = np.eye(4, dtype=np.float64)
    R_rect_4x4[:3, :3] = R0_rect

    return P2, R0_rect, R_rect_4x4, Tr


def load_point_cloud(bin_file: str) -> np.ndarray:
    pts = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)  # x,y,z,intensity
    return pts


def parse_label2(label_file: str):
    """
    Parse KITTI label_2 file.
    Returns a list of dict:
      {
        'type': str, 'truncated': float, 'occluded': int, 'alpha': float,
        'bbox': [l,t,r,b],
        'dims': [h,w,l],
        'loc':  [x,y,z],  (in camera rect coords)
        'ry':   float      (rotation around Y in camera coords)
      }
    """
    objs = []
    if not os.path.exists(label_file):
        return objs  # might be test split
    with open(label_file, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            # KITTI format has at least 15 fields
            if len(parts) < 15:
                continue
            obj_type = parts[0]
            truncated = float(parts[1])
            occluded = int(float(parts[2]))
            alpha = float(parts[3])
            bbox = [float(parts[4]), float(parts[5]),
                    float(parts[6]), float(parts[7])]
            h = float(parts[8]); w = float(parts[9]); l = float(parts[10])
            x = float(parts[11]); y = float(parts[12]); z = float(parts[13])
            ry = float(parts[14])
            objs.append({
                "type": obj_type,
                "truncated": truncated,
                "occluded": occluded,
                "alpha": alpha,
                "bbox": bbox,
                "dims": [h, w, l],
                "loc":  [x, y, z],
                "ry":   ry
            })
    return objs


def kitti_3d_box_corners_in_cam(dims, loc, ry):
    """
    Computes 8 corners of a KITTI 3D box in CAMERA (rect) coords.
    dims = [h,w,l], loc = [x,y,z] (bottom-centered in KITTI), ry: rotation around +Y.
    Returns (8,3) array.
    """
    h, w, l = dims
    x, y, z = loc

    # in the object's local coord system (camera coords)
    # KITTI uses bottom center as (x,y,z), y is down in camera coords
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2 ]
    y_corners = [   0,    0,    0,    0, -h,  -h,   -h,   -h  ]  # top = -h
    z_corners = [ w/2, -w/2, -w/2,  w/2, w/2, -w/2, -w/2,  w/2 ]

    corners = np.vstack([x_corners, y_corners, z_corners])  # (3,8)

    # rotation around Y
    c = math.cos(ry)
    s = math.sin(ry)
    R = np.array([[ c, 0, s],
                  [ 0, 1, 0],
                  [-s, 0, c]], dtype=np.float64)

    corners_rot = R @ corners  # (3,8)
    corners_trans = corners_rot + np.array([[x],[y],[z]])
    return corners_trans.T  # (8,3)


def cam_to_lidar_points(X_cam: np.ndarray, R_rect_4x4: np.ndarray, Tr_velo_to_cam_4x4: np.ndarray):
    """
    Transform Nx3 (camera rect) -> Nx3 (LiDAR) using:
      X_cam_h = [X_cam, 1]
      X_lidar = inv(Tr_velo_to_cam) * inv(R_rect) * X_cam_h
    """
    N = X_cam.shape[0]
    X_cam_h = np.hstack([X_cam, np.ones((N,1), dtype=np.float64)])
    M = np.linalg.inv(Tr_velo_to_cam_4x4) @ np.linalg.inv(R_rect_4x4)
    X_lidar_h = (M @ X_cam_h.T).T
    return X_lidar_h[:, :3]


def make_open3d_box(corners_lidar: np.ndarray, color=(1.0, 0.0, 0.0)):
    """
    corners_lidar: (8,3) in a standard corner order (same as we built).
    Build Open3D LineSet to render as edges.
    """
    # 12 edges by index pairs for a cuboid with our corner ordering:
    # (0-1-2-3) top face, (4-5-6-7) bottom face, and verticals (0-4,1-5,2-6,3-7)
    lines = [
        [0,1],[1,2],[2,3],[3,0],   # top
        [4,5],[5,6],[6,7],[7,4],   # bottom
        [0,4],[1,5],[2,6],[3,7]    # sides
    ]
    # Some viewers prefer (0..3) as top, but our y axis has top at indices 4..7.
    # We'll keep this consistent with kitti_3d_box_corners_in_cam definition.
    # Reorder to put the actual "top" at indices 4..7 for prettier edges:
    # Swap top/bottom if needed:
    # Here we assume indices 0..3 were y=0, 4..7 were y=-h; treat 0..3 as "roof".
    # This is fine visually.

    colors = [color for _ in lines]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners_lidar),
        lines=o3d.utility.Vector2iVector(lines)
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def visualize_frame(img_path: Path, lbl_path: Path, velo_path: Path, calib_path: Path,
                    side: str, save_prefix: Path = None):
    # -------- 2D IMAGE + 2D BOXES --------
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    objs = parse_label2(str(lbl_path))
    # draw 2D GT boxes
    img_vis = img.copy()
    for o in objs:
        l,t,r,b = [int(round(v)) for v in o["bbox"]]
        color = (0, 255, 0)  # green
        cv2.rectangle(img_vis, (l,t), (r,b), color, 2)
        cv2.putText(img_vis, o["type"], (l, max(0, t-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # -------- 3D LIDAR + 3D BOXES --------
    P2, R0_rect, R_rect_4x4, Tr = read_calib(str(calib_path))

    # Load points (N,4) -> (N,3)
    pts = load_point_cloud(str(velo_path))[:, :3]

    # build Open3D point cloud
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    # optional: set uniform color
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(pts) * 0.5)

    # Build Open3D geometries for all boxes
    boxes = []
    for o in objs:
        corners_cam = kitti_3d_box_corners_in_cam(o["dims"], o["loc"], o["ry"])
        corners_lidar = cam_to_lidar_points(corners_cam, R_rect_4x4, Tr)
        boxes.append(make_open3d_box(corners_lidar, color=(1.0, 0.0, 0.0)))

    # -------- Display / Save --------
    # OpenCV image
    if SAVE_OUTPUTS:
        save_prefix.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_prefix.with_suffix(".jpg")), img_vis)

    if SHOW_OPENCV:
        cv2.imshow(f"{side.upper()} 2D (GT boxes)", img_vis)
        cv2.waitKey(1)  # a brief pause so window updates

    # Open3D window
    if SHOW_OPEN3D:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"{side.upper()} LiDAR (GT 3D boxes)", width=1280, height=720, visible=True)
        opt = vis.get_render_option()
        if opt:
            opt.point_size = O3D_POINT_SIZE
            opt.background_color = np.array([0,0,0])

        vis.add_geometry(pcd)
        for b in boxes:
            vis.add_geometry(b)
        vis.poll_events()
        vis.update_renderer()

        if SAVE_OUTPUTS:
            # Save a snapshot of the Open3D view
            vis.capture_screen_image(str(save_prefix.with_suffix(".o3d.png")), do_render=True)

        print("Close the Open3D window to proceed to next frame...")
        vis.run()
        vis.destroy_window()


def main():
    side = SIDE
    img_dir, lbl2_dir, velo_dir, calib_dir = get_split_dirs(DATA_ROOT, side, SPLIT)

    # Gather frame IDs from images (assumes KITTI-style naming *.png or *.jpg)
    img_list = sorted(glob.glob(str(img_dir / "*.png")) + glob.glob(str(img_dir / "*.jpg")))
    if len(img_list) == 0:
        raise RuntimeError(f"No images found in {img_dir}")

    end_idx = min(len(img_list), START_IDX + MAX_FRAMES)
    print(f"[INFO] Showing frames {START_IDX}..{end_idx-1} ({end_idx-START_IDX} total) for side='{side}', split='{SPLIT}'")

    if SAVE_OUTPUTS:
        out_img_dir = Path(OUTPUT_DIR) / f"{side}_{SPLIT}_2d"
        out_3d_dir  = Path(OUTPUT_DIR) / f"{side}_{SPLIT}_3d"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_3d_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(START_IDX, end_idx):
        img_path = Path(img_list[idx])
        stem = img_path.stem  # KITTI id (e.g., 000123)
        lbl_path  = lbl2_dir / f"{stem}.txt"
        velo_path = velo_dir / f"{stem}.bin"
        calib_path = calib_dir / f"{stem}.txt"

        if not lbl_path.exists():
            print(f"[WARN] Missing label_2 for {stem}; skipping 2D/3D boxes.")
        if not velo_path.exists():
            print(f"[WARN] Missing velodyne for {stem}; skipping point cloud.")
            continue
        if not calib_path.exists():
            print(f"[WARN] Missing calib for {stem}; cannot draw 3D boxes.")
            continue

        print(f"[INFO] Frame {idx}  id={stem}")
        save_prefix = None
        if SAVE_OUTPUTS:
            save_prefix = Path(OUTPUT_DIR) / ("{}_{}".format(side, stem))

        try:
            visualize_frame(img_path, lbl_path, velo_path, calib_path, side, save_prefix)
        except Exception as e:
            print(f"[ERROR] Frame {stem} failed: {e}")

    if SHOW_OPENCV:
        print("[INFO] Done. Press a key to close the OpenCV window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
