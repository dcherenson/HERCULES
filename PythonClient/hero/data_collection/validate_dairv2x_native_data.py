#!/usr/bin/env python3
"""
DAIR-V2X (native / non-KITTI) viewer for 2D & 3D labels.

What it shows (very similar to validate_dairv2x_kittistyle_data.py):
- Camera image with 2D GT boxes from <side>/label/camera/<id>.json.
- LiDAR point cloud (.bin) with 3D GT boxes drawn in the LiDAR frame from
  <side>/label/lidar/<id>.json.

Directory assumptions (matching run_dairv2x_oop_skeleton.py and dataset_kitti_converter.py):
- Images:                  <ROOT>/<side>/image/<id>.png|jpg
- LiDAR (x,y,z,intensity): <ROOT>/<side>/velodyne/<id>.bin
- 2D labels:               <ROOT>/<side>/label/camera/<id>.json    (may be list, or {"annotations":[...]})
- 3D labels:               <ROOT>/<side>/label/lidar/<id>.json     (may be list, or {"labels":[...]})
- Intrinsics:              <ROOT>/<side>/calib/camera_intrinsic/<id>.json
- Lidar->Camera extrinsic: vehicle-side:        <ROOT>/vehicle-side/calib/lidar_to_camera/<id>.json
                           infrastructure-side: <ROOT>/infrastructure-side/calib/virtuallidar_to_camera/<id>.json

Notes
-----
- Keys accepted for 2D boxes: {"2d_box":{"xmin","ymin","xmax","ymax"}} or COCO-style {"bbox":[x,y,w,h]} or {"bbox":{"x1"/"left","y1"/"top","x2"/"right","y2"/"bottom"}}.
- Keys accepted for 3D boxes (LiDAR frame): center from one of {"3d_location":{"x","y","z"} | "center":[x,y,z]}
  size from one of {"3d_dimensions":{"h","w","l"} | "dimensions":{"h","w","l"} | "size":[l,w,h]}
  yaw from one of {"rotation": rad | "yaw": rad}
- If intrinsics/extrinsic are present, the script can optionally project 3D box corners into the image for visual cross-check.
  Toggle PROJECT_3D_ON_IMAGE to enable.

Controls
--------
- Close the Open3D window to proceed to next frame.
- Press any key in the OpenCV window to advance (if SHOW_OPENCV=True).

"""

import os
import glob
import json
import math
from pathlib import Path
from typing import Tuple, List, Dict, Any

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

# ===================== User-configurable variables =====================

# Base path to your cooperative-vehicle-infrastructure folder (native format)
# DATA_ROOT = "/media/sgarimella34/hercules-collect/collaborative-perception-BEVP/datasets/dair_v2x_synth/cooperative-vehicle-infrastructure"
DATA_ROOT = "/media/sgarimella34/hercules-collect/collaborative-perception-BEVP/datasets/dair_v2x_synth/cooperative-vehicle-infrastructure/"
# Choose side: 'veh' or 'infra' (or 'vehicle-side' / 'infrastructure-side')
# SIDE = "veh"
SIDE = "infra"


# How many frames to show
START_IDX = 0
MAX_FRAMES = 20

# Visualization toggles
SHOW_OPENCV  = True
SHOW_OPEN3D  = True
SAVE_OUTPUTS = False
OUTPUT_DIR   = "/home/sgarimella34/vis_native_viewer"

# Optional: also project 3D box corners into the image (needs K and lidar->cam)
PROJECT_3D_ON_IMAGE = False

# Open3D point size
O3D_POINT_SIZE = 1.0


# ===================== Path helpers =====================

def side_dir_name(side: str) -> str:
    if side.lower() in ("infra", "infrastructure", "infrastructure-side"):
        return "infrastructure-side"
    elif side.lower() in ("veh", "vehicle", "vehicle-side"):
        return "vehicle-side"
    else:
        raise ValueError("SIDE must be 'infra' or 'veh'")


def get_dirs_native(root: str, side: str):
    sd = side_dir_name(side)
    base = Path(root) / sd
    img_dir   = base / "image"
    lbl2_dir  = base / "label" / "camera"
    lbl3_dir  = base / "label" / "lidar"
    velo_dir  = base / "velodyne"
    calib_dir = base / "calib"
    # optional subdirs under calib
    K_dir     = calib_dir / "camera_intrinsic"
    if sd == "vehicle-side":
        T_dir = calib_dir / "lidar_to_camera"
    else:
        T_dir = calib_dir / "virtuallidar_to_camera"
    # Return existing-only T_dir (may be missing)
    return img_dir, lbl2_dir, lbl3_dir, velo_dir, K_dir, T_dir


# ===================== JSON readers (robust to variants) =====================

def read_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)


def load_intrinsic(K_json: Path) -> np.ndarray:
    """
    Flexible intrinsic loader that accepts matrices under various keys or fx/fy/cx/cy.
    Returns a 3x3 K matrix.
    """
    J = read_json(K_json)

    def try_matrix(obj):
        v = np.array(obj, dtype=float)
        if v.size == 9:
            return v.reshape(3,3)
        if v.shape == (3,3):
            return v
        return None

    # common keys
    for key in ("cam_K", "K", "intrinsic", "camera_matrix", "matrix"):
        if key in J:
            v = J[key]
            if isinstance(v, dict):
                for kk in ("matrix","data","values"):
                    if kk in v:
                        K = try_matrix(v[kk])
                        if K is not None:
                            return K
            else:
                K = try_matrix(v)
                if K is not None:
                    return K

    # fx/fy/cx/cy either at root or under "intrinsic"
    def _try_fx_fy_cx_cy(d: Dict[str, Any]):
        fx, fy, cx, cy = (d.get("fx"), d.get("fy"), d.get("cx"), d.get("cy"))
        if None not in (fx, fy, cx, cy):
            fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
            return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.0]], dtype=float)
        return None

    K = _try_fx_fy_cx_cy(J)
    if K is not None:
        return K
    if "intrinsic" in J and isinstance(J["intrinsic"], dict):
        K = _try_fx_fy_cx_cy(J["intrinsic"])
        if K is not None:
            return K

    raise ValueError(f"Unrecognized intrinsic JSON: {K_json}")


def load_extrinsic_to_cam(T_json: Path) -> np.ndarray:
    """
    Returns 4x4 lidar->camera transform. Accepts many layouts.
    If filename suggests camera->lidar, auto-invert.
    """
    J = read_json(T_json)

    def as44(arr):
        arr = np.array(arr, dtype=float)
        if arr.size == 16:
            return arr.reshape(4,4)
        if arr.size == 12:
            M = np.eye(4, dtype=float); M[:3,:] = arr.reshape(3,4); return M
        return None

    def build_from_rt(rot, trans):
        R = None
        if isinstance(rot, (list, tuple)):
            a = np.array(rot, dtype=float).reshape(-1)
            if a.size == 9:
                R = a.reshape(3,3)
            elif a.size == 4:  # quaternion (w,x,y,z) or (x,y,z,w)
                q = a
                if abs(q[0]) >= 0.5:
                    w,x,y,z = q
                else:
                    x,y,z,w = q
                n = math.sqrt(w*w + x*x + y*y + z*z) or 1.0
                w,x,y,z = w/n, x/n, y/n, z/n
                R = np.array([
                    [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                    [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
                    [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
                ], dtype=float)
            elif a.size == 3:  # euler (roll,pitch,yaw) in rad (deg tolerated)
                r,p,y = a.tolist()
                deg = max(abs(r),abs(p),abs(y)) > 2*math.pi
                def Rx(a): 
                    ca,sa = math.cos(a), math.sin(a)
                    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], float)
                def Ry(a):
                    ca,sa = math.cos(a), math.sin(a)
                    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], float)
                def Rz(a):
                    ca,sa = math.cos(a), math.sin(a)
                    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], float)
                if deg:
                    r,p,y = math.radians(r), math.radians(p), math.radians(y)
                R = Rz(y) @ Ry(p) @ Rx(r)
        elif isinstance(rot, dict):
            if all(k in rot for k in ("w","x","y","z")):
                w,x,y,z = float(rot["w"]), float(rot["x"]), float(rot["y"]), float(rot["z"])
                n = math.sqrt(w*w + x*x + y*y + z*z) or 1.0
                w,x,y,z = w/n, x/n, y/n, z/n
                R = np.array([
                    [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                    [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
                    [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
                ], dtype=float)
            elif all(k in rot for k in ("roll","pitch","yaw")):
                r,p,y = float(rot["roll"]), float(rot["pitch"]), float(rot["yaw"])
                def Rx(a): 
                    ca,sa = math.cos(a), math.sin(a)
                    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], float)
                def Ry(a):
                    ca,sa = math.cos(a), math.sin(a)
                    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], float)
                def Rz(a):
                    ca,sa = math.cos(a), math.sin(a)
                    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], float)
                R = Rz(y) @ Ry(p) @ Rx(r)
            elif "matrix" in rot:
                R = np.array(rot["matrix"], dtype=float).reshape(3,3)
        t = None
        if isinstance(trans, dict) and all(k in trans for k in ("x","y","z")):
            t = np.array([trans["x"], trans["y"], trans["z"]], dtype=float)
        elif isinstance(trans, (list,tuple)) and len(trans)==3:
            t = np.array(trans, dtype=float)
        if R is not None and t is not None:
            M = np.eye(4, dtype=float)
            M[:3,:3] = R
            M[:3, 3] = t
            return M
        return None

    M = None
    for key in ("matrix","transform","Tr","T","extrinsic","pose","Mat","M"):
        if key in J:
            v = J[key]
            if isinstance(v, dict):
                for kk in ("matrix","data","values"):
                    if kk in v:
                        M = as44(v[kk])
                        if M is not None:
                            break
                if M is None and ("rotation" in v and "translation" in v):
                    M = build_from_rt(v["rotation"], v["translation"])
            else:
                M = as44(v)
            if M is not None:
                break

    if M is None:
        if "R" in J and "t" in J:
            R = np.array(J["R"], dtype=float).reshape(3,3)
            t = np.array(J["t"], dtype=float).reshape(3)
            M = np.eye(4, dtype=float); M[:3,:3]=R; M[:3,3]=t
        elif "rotation" in J and "translation" in J:
            M = build_from_rt(J["rotation"], J["translation"])

    if M is None:
        raise ValueError(f"Unrecognized extrinsic JSON: {T_json}")

    lower = T_json.as_posix().lower()
    if "camera_to_lidar" in lower or "cam_to_lidar" in lower or "camera2lidar" in lower:
        M = np.linalg.inv(M)
    return M


# ===================== Label parsing helpers =====================

def parse_2d_json(path_json: Path) -> List[Dict[str, Any]]:
    if not path_json.exists():
        return []
    data = read_json(path_json)
    records = data if isinstance(data, list) else data.get("annotations", [data])
    out = []
    for o in (records or []):
        typ = o.get("type", "Car")
        if "2d_box" in o and isinstance(o["2d_box"], dict):
            bb = o["2d_box"]
            xmin = float(bb.get("xmin", bb.get("x1", bb.get("left", 0))))
            ymin = float(bb.get("ymin", bb.get("y1", bb.get("top", 0))))
            xmax = float(bb.get("xmax", bb.get("x2", bb.get("right", 0))))
            ymax = float(bb.get("ymax", bb.get("y2", bb.get("bottom", 0))))
        elif "bbox" in o and isinstance(o["bbox"], dict):
            bb = o["bbox"]
            xmin = float(bb.get("x1", bb.get("left", 0)))
            ymin = float(bb.get("y1", bb.get("top", 0)))
            xmax = float(bb.get("x2", bb.get("right", 0)))
            ymax = float(bb.get("y2", bb.get("bottom", 0)))
        elif "bbox" in o and isinstance(o["bbox"], (list,tuple)) and len(o["bbox"])==4:
            x,y,w,h = [float(v) for v in o["bbox"]]
            xmin, ymin, xmax, ymax = x, y, x+w, y+h
        else:
            continue
        out.append({"type": typ, "bbox": [xmin, ymin, xmax, ymax]})
    return out


def _extract_center(g: Dict[str, Any]) -> np.ndarray:
    if "3d_location" in g and isinstance(g["3d_location"], dict):
        x = float(g["3d_location"].get("x", 0.0))
        y = float(g["3d_location"].get("y", 0.0))
        z = float(g["3d_location"].get("z", 0.0))
        return np.array([x,y,z], dtype=float)
    if "center" in g and isinstance(g["center"], (list,tuple)) and len(g["center"])==3:
        return np.array([float(g["center"][0]), float(g["center"][1]), float(g["center"][2])], dtype=float)
    return np.array([0.0,0.0,0.0], dtype=float)


def _extract_dims_hwl(g: Dict[str, Any]) -> Tuple[float,float,float]:
    if "3d_dimensions" in g and isinstance(g["3d_dimensions"], dict):
        h = float(g["3d_dimensions"].get("h", 0.0))
        w = float(g["3d_dimensions"].get("w", 0.0))
        l = float(g["3d_dimensions"].get("l", 0.0))
        return h,w,l
    if "dimensions" in g and isinstance(g["dimensions"], dict):
        h = float(g["dimensions"].get("h", 0.0))
        w = float(g["dimensions"].get("w", 0.0))
        l = float(g["dimensions"].get("l", 0.0))
        return h,w,l
    if "size" in g and isinstance(g["size"], (list,tuple)) and len(g["size"])==3:
        l,w,h = [float(v) for v in g["size"]]
        return h,w,l
    return 0.0,0.0,0.0


def _extract_yaw(g: Dict[str, Any]) -> float:
    if "rotation" in g:
        return float(g["rotation"])
    if "yaw" in g:
        return float(g["yaw"])
    if "rotation_y" in g:
        # If someone stored camera-ry by mistake, accept it but treat as LiDAR yaw
        return float(g["rotation_y"])
    return 0.0


def parse_3d_json(path_json: Path) -> List[Dict[str, Any]]:
    if not path_json.exists():
        return []
    data = read_json(path_json)
    recs = data if isinstance(data, list) else data.get("labels", [data])
    out = []
    for g in (recs or []):
        typ = g.get("type", "Car")
        c = _extract_center(g)
        h,w,l = _extract_dims_hwl(g)
        yaw = _extract_yaw(g)  # radians, LiDAR frame (z-up, y-left)
        out.append({"type": typ, "center": c, "dims_hwl": (h,w,l), "yaw": yaw})
    return out


# ===================== Geometry =====================

def lidar_box_corners(center: np.ndarray, h: float, w: float, l: float, yaw: float) -> np.ndarray:
    """
    Build 8 corners (N=8,3) of a LiDAR-frame cuboid with DAIR/KITTI convention:
    - LiDAR frame: x forward, y left, z up
    - yaw about +Z
    - Center is geometric center of the cuboid
    Corner order similar to KITTI camera box for consistency:
        (top face)    0:l/2,w/2, h/2; 1:l/2,-w/2, h/2; 2:-l/2,-w/2, h/2; 3:-l/2,w/2, h/2
        (bottom face) 4:l/2,w/2,-h/2; 5:l/2,-w/2,-h/2; 6:-l/2,-w/2,-h/2; 7:-l/2,w/2,-h/2
    """
    x_c = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2 ]
    y_c = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2 ]
    z_c = [ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2 ]
    corners = np.vstack([x_c, y_c, z_c])  # (3,8)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[ c,-s, 0],
                  [ s, c, 0],
                  [ 0, 0, 1]], dtype=float)
    corners = (R @ corners).T + center.reshape(1,3)
    return corners  # (8,3)


def make_open3d_box(corners_lidar: np.ndarray, color=(1.0, 0.0, 0.0)):
    """
    corners_lidar: (8,3) in the order produced by lidar_box_corners().
    """
    if o3d is None:
        return None
    lines = [
        [0,1],[1,2],[2,3],[3,0],  # top
        [4,5],[5,6],[6,7],[7,4],  # bottom
        [0,4],[1,5],[2,6],[3,7],  # verticals
    ]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners_lidar),
        lines=o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    )
    cols = np.tile(np.array(color, dtype=float).reshape(1,3), (len(lines),1))
    ls.colors = o3d.utility.Vector3dVector(cols)
    return ls


def load_point_cloud(bin_file: str) -> np.ndarray:
    if not os.path.exists(bin_file):
        return np.zeros((0,4), dtype=np.float32)
    pts = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)  # x,y,z,intensity
    return pts


# ===================== Optional projection for cross-check =====================

def project_lidar_points_to_image(pts_lidar: np.ndarray, K: np.ndarray, T_cam_lidar: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (uv, valid) from LiDAR (x,y,z) -> image space using K [3x3] and T_cam_lidar [4x4].
    """
    if pts_lidar.shape[0] == 0:
        return np.zeros((0,2)), np.zeros((0,), dtype=bool)
    pts = pts_lidar[:, :3]
    N = pts.shape[0]
    pts_h = np.hstack([pts, np.ones((N,1), dtype=float)])
    cam = (T_cam_lidar @ pts_h.T).T[:, :3]
    z = cam[:, 2]
    uv = (K @ cam.T).T
    uv = uv[:, :2] / np.maximum(z.reshape(-1,1), 1e-6)
    valid = (z > 1e-6) & (uv[:,0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:,1] < h)
    return uv, valid


def draw_projected_box_on_image(img: np.ndarray, corners_lidar: np.ndarray, K: np.ndarray, T_cam_lidar: np.ndarray, color=(0,255,255), thickness=2):
    h, w = img.shape[:2]
    N = corners_lidar.shape[0]
    pts_h = np.hstack([corners_lidar, np.ones((N,1), dtype=float)])
    cam = (T_cam_lidar @ pts_h.T).T[:, :3]
    z = cam[:,2]
    uv = (K @ cam.T).T
    uv = uv[:, :2] / np.maximum(z.reshape(-1,1), 1e-6)
    valid = (z > 1e-6)
    # edges like make_open3d_box
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i,j in edges:
        if valid[i] and valid[j]:
            p1 = (int(round(uv[i,0])), int(round(uv[i,1])))
            p2 = (int(round(uv[j,0])), int(round(uv[j,1])))
            cv2.line(img, p1, p2, color, thickness)


# ===================== Main visualization per-frame =====================

def visualize_frame(img_path: Path, lbl2_path: Path, lbl3_path: Path, velo_path: Path, K_path: Path, T_path: Path, side: str, save_prefix: Path | None):
    # --- Image & 2D ---
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img_vis = img.copy()
    objs2d = parse_2d_json(lbl2_path) if lbl2_path and lbl2_path.exists() else []
    for o in objs2d:
        x1,y1,x2,y2 = [int(round(v)) for v in o["bbox"]]
        cv2.rectangle(img_vis, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img_vis, o.get("type",""), (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    # --- 3D + point cloud ---
    pts = load_point_cloud(str(velo_path))[:, :3]
    boxes3d = parse_3d_json(lbl3_path) if lbl3_path and lbl3_path.exists() else []

    o3d_boxes = []
    if o3d is not None:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.colors = o3d.utility.Vector3dVector(np.ones_like(pts) * 0.5)
    else:
        pcd = None

    # optional projection setup
    K = None; T_cl = None
    if PROJECT_3D_ON_IMAGE and K_path.exists() and T_path.exists():
        try:
            K = load_intrinsic(K_path)
            T_cl = load_extrinsic_to_cam(T_path)
        except Exception as e:
            print(f"[WARN] Could not load K/T for projection: {e}")

    for o in boxes3d:
        h,w,l = o["dims_hwl"]
        c = o["center"]
        yaw = o["yaw"]
        corners = lidar_box_corners(c, h, w, l, yaw)
        if PROJECT_3D_ON_IMAGE and (K is not None) and (T_cl is not None):
            draw_projected_box_on_image(img_vis, corners, K, T_cl, color=(0,255,255), thickness=2)
        if o3d is not None:
            o3d_boxes.append(make_open3d_box(corners, color=(1.0, 0.0, 0.0)))

    # -------- Display / Save --------
    if SAVE_OUTPUTS:
        save_prefix.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_prefix.with_suffix(".jpg")), img_vis)

    if SHOW_OPENCV:
        cv2.imshow(f"{side.upper()} 2D (native GT boxes)", img_vis)
        cv2.waitKey(1)

    if SHOW_OPEN3D and o3d is not None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"{side.upper()} LiDAR (native 3D GT)", width=1280, height=720, visible=True)
        opt = vis.get_render_option()
        if opt:
            opt.point_size = O3D_POINT_SIZE
            opt.background_color = np.array([0,0,0])
        if pcd is not None:
            vis.add_geometry(pcd)
        for b in o3d_boxes:
            if b is not None:
                vis.add_geometry(b)
        vis.poll_events()
        vis.update_renderer()
        if SAVE_OUTPUTS:
            vis.capture_screen_image(str(save_prefix.with_suffix(".o3d.png")), do_render=True)
        print("Close the Open3D window to proceed to next frame...")
        vis.run()
        vis.destroy_window()


# ===================== Driver =====================

def main():
    side = SIDE
    img_dir, lbl2_dir, lbl3_dir, velo_dir, K_dir, T_dir = get_dirs_native(DATA_ROOT, side)

    img_list = sorted(glob.glob(str(img_dir / "*.png")) + glob.glob(str(img_dir / "*.jpg")))
    if len(img_list) == 0:
        raise RuntimeError(f"No images found in {img_dir}")

    end_idx = min(len(img_list), START_IDX + MAX_FRAMES)
    print(f"[INFO] Showing frames {START_IDX}..{end_idx-1} ({end_idx-START_IDX} total) for side='{side_dir_name(side)}'")

    if SAVE_OUTPUTS:
        out_img_dir = Path(OUTPUT_DIR) / f"{side_dir_name(side)}_2d"
        out_3d_dir  = Path(OUTPUT_DIR) / f"{side_dir_name(side)}_3d"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_3d_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(START_IDX, end_idx):
        img_path = Path(img_list[idx])
        stem = img_path.stem

        lbl2_path = lbl2_dir / f"{stem}.json"
        lbl3_path = lbl3_dir / f"{stem}.json"
        velo_path = velo_dir / f"{stem}.bin"
        K_path    = K_dir / f"{stem}.json"
        T_path    = T_dir / f"{stem}.json" if T_dir.exists() else Path("")

        if not velo_path.exists():
            print(f"[WARN] Missing velodyne for {stem}; skipping point cloud.")
            continue

        save_prefix = None
        if SAVE_OUTPUTS:
            save_prefix = Path(OUTPUT_DIR) / ("{}_{}".format(side_dir_name(side), stem))

        try:
            visualize_frame(img_path, lbl2_path, lbl3_path, velo_path, K_path, T_path, side, save_prefix)
        except Exception as e:
            print(f"[ERROR] Frame {stem} failed: {e}")

    if SHOW_OPENCV:
        print("[INFO] Done. Press a key to close the OpenCV window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
