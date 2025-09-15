#!/usr/bin/env python3
"""
Custom DAIR-V2X-C-like viewer for your JSON layout.

Layout expected (what you showed):

<DATA_ROOT>/
  infrastructure-side/
    image/000000.png
    label/camera/000000.json
    label/lidar/000000.json
    velodyne/000000.bin
    calib/...
  vehicle-side/
    image/000000.png
    label/camera/000000.json
    label/lidar/000000.json
    velodyne/000000.bin
    calib/...

What this script does
- Shows image with *2D GT boxes from label/camera/*.json* (no projection).
- Shows LiDAR point cloud with *3D GT boxes from label/lidar/*.json* in LiDAR coords.

No CLI args—edit the variables below.

Notes
- We DO NOT need calibration to draw 3D boxes in the LiDAR view (boxes are in LiDAR/virtual-LiDAR frame).
- JSON keys vary; robust parsers try multiple common names and warn if something can't be parsed.
"""

import os
import glob
import json
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

# ===================== User-configurable variables =====================

DATA_ROOT = "/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets/dair_v2x_synth/cooperative-vehicle-infrastructure"

# Choose side: 'infra' (infrastructure-side) or 'veh' (vehicle-side)
SIDE = "infra"  # "infra" or "veh"

# Start index and how many frames to show
START_IDX = 0
MAX_FRAMES = 20

# Visualization toggles
SHOW_OPENCV  = True
SHOW_OPEN3D  = True
SAVE_OUTPUTS = False
OUTPUT_DIR   = "/home/sgarimella34/vis_custom_viewer"

# Open3D look
O3D_POINT_SIZE = 1.0

# --- LiDAR coord-system fixes ---
LIDAR_FLIP_X = False
LIDAR_FLIP_Y = True    # your synth has y right; KITTI uses y left
LIDAR_FLIP_Z = False
FLIP_YAW_WHEN_Y_FLIPPED = True  # mirror about Y implies yaw sign flip

# =======================================================================

def side_dir_name(side: str) -> str:
    if side.lower() in ("infra", "infrastructure", "infrastructure-side"):
        return "infrastructure-side"
    elif side.lower() in ("veh", "vehicle", "vehicle-side"):
        return "vehicle-side"
    else:
        raise ValueError("SIDE must be 'infra' or 'veh'")

def get_dirs(root: str, side: str):
    """Match your JSON layout (no 'training' split, no KITTI folders)."""
    sd = side_dir_name(side)
    base = Path(root) / sd
    img_dir    = base / "image"
    lidar_lbl  = base / "label" / "lidar"
    cam_lbl    = base / "label" / "camera"
    velo_dir   = base / "velodyne"
    # calib_dir optional for this script
    for p in [img_dir, lidar_lbl, cam_lbl, velo_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Expected path missing: {p}")
    return img_dir, cam_lbl, lidar_lbl, velo_dir

# ------------------------------- Parsers --------------------------------

def _read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _num(x):
    # robust float conversion for ints, floats, or numeric strings
    try:
        return float(x)
    except Exception:
        return None


def parse_2d_label_json(path: Path):
    if not path.exists():
        return []
    data = _read_json(path)

    # tolerate list, dict-with-array, or single-object dict
    if isinstance(data, dict) and "annotations" in data and isinstance(data["annotations"], list):
        records = data["annotations"]
    elif isinstance(data, list):
        records = data
    elif isinstance(data, dict) and "2d_box" in data:
        records = [data]  # single object file
    else:
        print(f"[WARN] 2D label file has unknown format: {path.name}")
        return []

    out = []
    for obj in records:
        cls = obj.get("type") or obj.get("category") or obj.get("name") or "Obj"

        bbox = None
        if "2d_box" in obj and isinstance(obj["2d_box"], dict):
            b = obj["2d_box"]
            l = _num(b.get("xmin")); t = _num(b.get("ymin"))
            r = _num(b.get("xmax")); btm = _num(b.get("ymax"))
            if None not in (l,t,r,btm):
                bbox = [l,t,r,btm]

        # fallbacks (optional)
        if bbox is None and "box2d" in obj and isinstance(obj["box2d"], (list,tuple)) and len(obj["box2d"])==4:
            l,t,r,btm = obj["box2d"]
            bbox = [_num(l), _num(t), _num(r), _num(btm)]
        if bbox is None and "bbox" in obj:
            b = obj["bbox"]
            if isinstance(b, (list,tuple)) and len(b)==4:  # COCO [x,y,w,h]
                x,y,w,h = [_num(v) for v in b]
                bbox = [x, y, x+w, y+h]
            elif isinstance(b, dict):
                l = _num(b.get("x1") or b.get("left"))
                t = _num(b.get("y1") or b.get("top"))
                r = _num(b.get("x2") or b.get("right"))
                btm = _num(b.get("y2") or b.get("bottom"))
                if None not in (l,t,r,btm):
                    bbox = [l,t,r,btm]

        if bbox is None:
            print(f"[WARN] Could not parse 2D bbox in {path.name}; skipping an object.")
            continue

        # skip sentinel -1 boxes
        if min(bbox) is not None and min(bbox) < 0:
            continue

        out.append({"bbox": [float(v) for v in bbox], "type": cls})
    return out


def parse_3d_label_lidar_json(path: Path):
    """
    Parse your 3D JSON (LiDAR frame):
      top-level: list[ obj ]
      obj: {
        "type": "...",
        "3d_dimensions": {"h":..., "w":..., "l":...},
        "3d_location":  {"x":..., "y":..., "z":...},
        "rotation": <yaw in radians (float or numeric string)>
      }
    Returns list of {"center":[x,y,z], "size":[l,w,h], "yaw": float, "type": str}
    """
    if not path.exists():
        return []
    data = _read_json(path)
    records = data["labels"] if isinstance(data, dict) and "labels" in data else data
    if not isinstance(records, list):
        print(f"[WARN] 3D label file has unknown format: {path.name}")
        return []

    out = []
    for obj in records:
        typ = obj.get("type") or obj.get("category") or obj.get("name") or "Obj"

        # Your keys
        dims = obj.get("3d_dimensions") or obj.get("dimensions") or obj.get("size")
        loc  = obj.get("3d_location")  or obj.get("location")   or obj.get("center")
        yaw  = obj.get("rotation")     or obj.get("yaw")        or obj.get("rotation_y") or obj.get("rz")

        if isinstance(dims, dict):
            h = _num(dims.get("h")); w = _num(dims.get("w")); l = _num(dims.get("l"))
            if None in (h,w,l):
                dims_lwh = None
            else:
                dims_lwh = [float(l), float(w), float(h)]  # convert to [l,w,h]
        elif isinstance(dims, (list,tuple)) and len(dims)==3:
            # Try to detect [h,w,l] versus [l,w,h]
            a,b,c = [_num(v) for v in dims]
            if None in (a,b,c):
                dims_lwh = None
            else:
                # If first is clearly the smallest, treat as h,w,l -> l,w,h
                dims_lwh = [float(c), float(b), float(a)] if (a < b and a < c) else [float(a), float(b), float(c)]
        else:
            dims_lwh = None

        if isinstance(loc, dict):
            x = _num(loc.get("x")); y = _num(loc.get("y")); z = _num(loc.get("z"))
            center = None if None in (x,y,z) else [float(x), float(y), float(z)]
        elif isinstance(loc, (list,tuple)) and len(loc)==3:
            x,y,z = [_num(v) for v in loc]
            center = None if None in (x,y,z) else [float(x), float(y), float(z)]
        else:
            center = None

        if isinstance(yaw, str) and yaw.strip():
            try:
                yaw = float(yaw)
            except Exception:
                yaw = None
        elif isinstance(yaw, (int,float)):
            yaw = float(yaw)

        if yaw is not None and abs(yaw) > 2*np.pi:
            yaw = math.radians(yaw)

        if center is None or dims_lwh is None or yaw is None:
            print(f"[WARN] Could not parse 3D box fields in {path.name}; skipping an object.")
            continue
        
        # ---- apply coord-system flips (if any) ----
        if LIDAR_FLIP_X:
            center[0] = -center[0]
        if LIDAR_FLIP_Y:
            center[1] = -center[1]
            if FLIP_YAW_WHEN_Y_FLIPPED and yaw is not None:
                yaw = -yaw
        if LIDAR_FLIP_Z:
            center[2] = -center[2]

        out.append({"center": center, "size": dims_lwh, "yaw": yaw, "type": typ})
    return out


# ----------------------------- Geometry ---------------------------------

def make_open3d_box_from_lwh(center, size_lwh, yaw, color=(1.0, 0.0, 0.0)):
    """
    Build an Open3D LineSet for a box in LiDAR coords:
      center = [x,y,z], size_lwh = [l,w,h], yaw around +Z.
    """
    l, w, h = size_lwh
    x, y, z = center
    # 8 corners around origin
    # top (z + h/2) and bottom (z - h/2)
    corners = np.array([
        [ l/2,  w/2,  h/2],
        [ l/2, -w/2,  h/2],
        [-l/2, -w/2,  h/2],
        [-l/2,  w/2,  h/2],
        [ l/2,  w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2],
        [-l/2,  w/2, -h/2],
    ], dtype=np.float64)
    # rotation around Z
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[ c, -s, 0],
                   [ s,  c, 0],
                   [ 0,  0, 1]], dtype=np.float64)
    corners = (Rz @ corners.T).T + np.array([x, y, z])

    lines = [
        [0,1],[1,2],[2,3],[3,0],   # top
        [4,5],[5,6],[6,7],[7,4],   # bottom
        [0,4],[1,5],[2,6],[3,7]    # sides
    ]
    ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(corners),
                              lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector([color]*len(lines))
    return ls

# --------------------------- Visualization ------------------------------

def visualize_frame(img_path: Path, cam_lbl_path: Path, lidar_lbl_path: Path,
                    velo_path: Path, save_prefix: Path = None, side_tag=""):
    # 2D image + 2D boxes
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    objs2d = parse_2d_label_json(cam_lbl_path)
    img_vis = img.copy()
    for o in objs2d:
        l,t,r,b = [int(round(v)) for v in o["bbox"]]
        color = (0, 255, 0)
        cv2.rectangle(img_vis, (l,t), (r,b), color, 2)
        cv2.putText(img_vis, o["type"], (l, max(0, t-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if SAVE_OUTPUTS:
        save_prefix.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_prefix.with_suffix(".jpg")), img_vis)
    if SHOW_OPENCV:
        cv2.imshow(f"{side_tag} 2D (GT)", img_vis)
        cv2.waitKey(1)

    # LiDAR + 3D boxes in LiDAR coords
    pts = np.fromfile(str(velo_path), dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(pts) * 0.6)

    objs3d = parse_3d_label_lidar_json(lidar_lbl_path)
    geoms = [pcd]
    for o in objs3d:
        try:
            geoms.append(make_open3d_box_from_lwh(o["center"], o["size"], o["yaw"],
                                                  color=(1.0, 0.0, 0.0)))
        except Exception as e:
            print(f"[WARN] Could not build 3D box for an object: {e}")

    if SHOW_OPEN3D:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"{side_tag} LiDAR (GT 3D)", width=1280, height=720, visible=True)
        opt = vis.get_render_option()
        if opt:
            opt.point_size = O3D_POINT_SIZE
            opt.background_color = np.array([0,0,0])
        for g in geoms:
            vis.add_geometry(g)
        vis.poll_events()
        vis.update_renderer()
        if SAVE_OUTPUTS:
            vis.capture_screen_image(str(save_prefix.with_suffix(".o3d.png")), do_render=True)
        print("Close the Open3D window to continue...")
        vis.run()
        vis.destroy_window()
    elif SAVE_OUTPUTS:
        # Offscreen snapshot requires EGL/OSMesa; skip here and rely on saved 2D image only.
        pass

def main():
    side = SIDE
    img_dir, cam_lbl_dir, lidar_lbl_dir, velo_dir = get_dirs(DATA_ROOT, side)

    # Collect image ids from image/*.png or *.jpg
    img_list = sorted(glob.glob(str(img_dir / "*.png")) + glob.glob(str(img_dir / "*.jpg")))
    if len(img_list) == 0:
        raise RuntimeError(f"No images found in {img_dir}")

    end_idx = min(len(img_list), START_IDX + MAX_FRAMES)
    side_tag = side_dir_name(side).upper()
    print(f"[INFO] Showing frames {START_IDX}..{end_idx-1} ({end_idx-START_IDX} total) for side='{side_tag}'")

    if SAVE_OUTPUTS:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for i in range(START_IDX, end_idx):
        img_path = Path(img_list[i])
        stem = img_path.stem
        cam_lbl_path   = cam_lbl_dir  / f"{stem}.json"
        lidar_lbl_path = lidar_lbl_dir/ f"{stem}.json"
        velo_path      = velo_dir     / f"{stem}.bin"

        if not cam_lbl_path.exists():
            print(f"[WARN] Missing 2D label for {stem}; 2D overlay will have none.")
        if not lidar_lbl_path.exists():
            print(f"[WARN] Missing 3D label for {stem}; LiDAR view will be points only.")
        if not velo_path.exists():
            print(f"[WARN] Missing velodyne for {stem}; skipping LiDAR view.")

        print(f"[INFO] Frame {i} id={stem}")
        save_prefix = None
        if SAVE_OUTPUTS:
            save_prefix = Path(OUTPUT_DIR) / f"{side}_{stem}"

        try:
            visualize_frame(img_path, cam_lbl_path, lidar_lbl_path, velo_path,
                            save_prefix, side_tag=side_tag)
        except Exception as e:
            print(f"[ERROR] Frame {stem} failed: {e}")

    if SHOW_OPENCV:
        print("[INFO] Done. Press a key to close the OpenCV window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
