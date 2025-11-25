#!/usr/bin/env python3
"""
DAIR-V2X (native / non-KITTI) viewer for 2D & 3D labels + cooperative world view.

Existing modes:
- 'veh'  or 'infrastructure': original side-specific viewers (2D image + 3D LiDAR-frame boxes).
New mode:
- 'cooperative_world': draw 3D GT boxes from cooperative/label_world/<id>.json (world frame),
  and show BOTH vehicle-side and infrastructure-side LiDAR point clouds transformed into the
  same world frame using calib transforms.

Directory assumptions (native format):
ROOT/
  vehicle-side/
    image/, velodyne/, label/camera/, label/lidar/, calib/...
  infrastructure-side/
    image/, velodyne/, label/camera/, label/lidar/, calib/...
  cooperative/
    label_world/                   <-- used in cooperative_world mode
    calib/                         <-- optional/world transforms may live here in some datasets
"""

import os
import glob
import json
import math
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

# ===================== User-configurable variables =====================

# Base path to your cooperative-vehicle-infrastructure folder (native format)
# DATA_ROOT = "/media/sgarimella34/hercules-collect/collaborative-perception-BEVP/dair_v2x_synth_FULL/cooperative-vehicle-infrastructure/"
DATA_ROOT = "/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets/DAIR-V2X-C-SUBSET1/cooperative-vehicle-infrastructure/"

# View mode: 'veh', 'infra', or 'cooperative_world'
# The SIDE value is ignored in cooperative world mode
VIEW_MODE = "infra"   # 'veh' | 'infra' | 'cooperative_world'

# For side-only modes:
SIDE = "infra"   # 'veh' or 'infra' (ignored for cooperative_world)

# How many frames to show
START_IDX = 0
MAX_FRAMES = 20

# Visualization toggles
SHOW_OPENCV  = True          # used only by side-only modes
SHOW_OPEN3D  = True
SAVE_OUTPUTS = False
OUTPUT_DIR   = "/home/sgarimella34/vis_native_viewer"

# Optional: also project 3D box corners into the image (needs K and lidar->cam)
PROJECT_3D_ON_IMAGE = False

# Open3D point size
O3D_POINT_SIZE = 1.0

# ===== New format toggles =====
# If True, prefer *.jpg (still falls back to *.png if jpg missing)
USE_JPG_IMAGES = True
# If True, load velodyne from *.pcd (otherwise from *.bin). It will try the other if missing.
USE_PCD_LIDAR = True

# ===== Cooperative world label toggles =====
# Draw parametric boxes reconstructed from (center, dims, yaw) in world:
DRAW_PARAM_BOXES_IN_WORLD = True
# ALSO draw world_8_points if label has it (as yellow wireframe in world):
VIS_WORLD_8_POINTS = True

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
    return img_dir, lbl2_dir, lbl3_dir, velo_dir, K_dir, T_dir

def get_dirs_cooperative(root: str):
    coop_dir = Path(root) / "cooperative"
    label_world_dir = coop_dir / "label_world"
    coop_calib_dir = coop_dir / "calib"  # may or may not exist

    # both possible lidar dirs for each side
    veh_velo_dirs = [
        Path(root) / "vehicle-side" / "velodyne",
        Path(root) / "vehicle-side" / "velodyne_bin",
    ]
    infra_velo_dirs = [
        Path(root) / "infrastructure-side" / "velodyne",
        Path(root) / "infrastructure-side" / "velodyne_bin",
    ]
    return label_world_dir, coop_calib_dir, veh_velo_dirs, infra_velo_dirs

# add this small helper once (near your other I/O helpers)
def load_point_cloud_any_from_dirs(velodyne_dirs: List[Path], stem: str) -> Tuple[np.ndarray, Path]:
    """Try multiple velodyne dirs (e.g., velodyne/ and velodyne_bin/)."""
    last_attempt = None
    for d in velodyne_dirs:
        pts, used = load_point_cloud_any(d, stem)
        last_attempt = used
        # if file exists (even if empty), or we actually loaded points, stop here
        if used.exists() or pts.shape[0] > 0:
            return pts, used
    # nothing found
    if last_attempt is None:
        last_attempt = velodyne_dirs[-1] / (stem + (".pcd" if USE_PCD_LIDAR else ".bin"))
    return np.zeros((0,4), dtype=np.float32), last_attempt

# ===================== JSON readers (robust to variants) =====================
def _to_float_ndarray(x) -> Optional[np.ndarray]:
    """
    Coerce arbitrarily nested lists/tuples (possibly with dict-wrapped scalars)
    into a float ndarray. Returns None if x isn't array-like.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        obj = np.array(x, dtype=object)
        vec = np.vectorize(lambda v: _as_float(v, default=np.nan))
        arr = vec(obj).astype(float)
        return arr
    return None


def _as_float(v, default=0.0):
    """Coerce numbers that may be nested in dicts or strings into float."""
    try:
        if v is None:
            return default
        # direct numeric
        if isinstance(v, (int, float, np.number)):
            return float(v)
        # string -> float
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return default
            return float(v)
        # dict wrappers like {"value": ...}, {"data": ...}, {"v": ...}
        if isinstance(v, dict):
            for k in ("value", "val", "data", "v", "num"):
                if k in v:
                    return _as_float(v[k], default)
            # sometimes {"x":..,"y":..,"z":..} appears where a scalar was expected; fall back
            # (not ideal, but avoids crashes)
            for k in ("x", "y", "z"):
                if k in v and isinstance(v[k], (int, float, str, np.number)):
                    return _as_float(v[k], default)
        # singletons like [3.14]
        if isinstance(v, (list, tuple)) and len(v) == 1:
            return _as_float(v[0], default)
    except Exception:
        pass
    return default


def read_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def load_intrinsic(K_json: Path) -> np.ndarray:
    J = read_json(K_json)

    def try_matrix(obj):
        v = np.array(obj, dtype=float)
        if v.size == 9:
            return v.reshape(3,3)
        if v.shape == (3,3):
            return v
        return None

    for key in ("cam_K", "K", "intrinsic", "camera_matrix", "matrix"):
        if key in J:
            v = J[key]
            if isinstance(v, dict):
                for kk in ("matrix","data","values"):
                    if kk in v:
                        K = try_matrix(v[kk]);  # noqa
                        if K is not None:
                            return K
            else:
                K = try_matrix(v)
                if K is not None:
                    return K

    def _try_fx_fy_cx_cy(d: Dict[str, Any]):
        fx, fy, cx, cy = (d.get("fx"), d.get("fy"), d.get("cx"), d.get("cy"))
        if None not in (fx, fy, cx, cy):
            fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
            return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.0]], dtype=float)
        return None

    K = _try_fx_fy_cx_cy(J)
    if K is not None: return K
    if "intrinsic" in J and isinstance(J["intrinsic"], dict):
        K = _try_fx_fy_cx_cy(J["intrinsic"])
        if K is not None: return K

    raise ValueError(f"Unrecognized intrinsic JSON: {K_json}")

def load_extrinsic_to_cam(T_json: Path) -> np.ndarray:
    J = read_json(T_json)

    def as44(arr):
        arr = np.array(arr, dtype=float)
        if arr.size == 16: return arr.reshape(4,4)
        if arr.size == 12:
            M = np.eye(4, dtype=float); M[:3,:] = arr.reshape(3,4); return M
        return None

    def build_from_rt(rot, trans):
        R = None
        if isinstance(rot, (list, tuple)):
            a = np.array(rot, dtype=float).reshape(-1)
            if a.size == 9:
                R = a.reshape(3,3)
            elif a.size == 4:  # quaternion
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
            elif a.size == 3:  # euler (roll,pitch,yaw)
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
                        M = as44(v[kk]);  # noqa
                        if M is not None: break
                if M is None and ("rotation" in v and "translation" in v):
                    M = build_from_rt(v["rotation"], v["translation"])
            else:
                M = as44(v)
            if M is not None: break

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

# --------- Robust world transform loader (for cooperative mode) ---------
def _load_44_json(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    J = read_json(path)

    def arr_to_44(arr: np.ndarray) -> Optional[np.ndarray]:
        if arr is None:
            return None
        flat = arr.reshape(-1)
        if flat.size == 16:
            return flat.reshape(4, 4)
        if flat.size == 12:
            M = np.eye(4, dtype=float)
            M[:3, :] = flat.reshape(3, 4)
            return M
        return None

    # 1) Look for common top-level keys with either raw arrays or nested dicts
    for k in ("matrix", "transform", "Tr", "T", "pose", "Mat", "M"):
        if k in J:
            v = J[k]
            if isinstance(v, dict):
                # nested: try matrix/data/values first
                for kk in ("matrix", "data", "values"):
                    if kk in v:
                        A = arr_to_44(_to_float_ndarray(v[kk]))
                        if A is not None:
                            return A
                # also allow rotation+translation nested under this key
                if "rotation" in v and "translation" in v:
                    try:
                        return load_extrinsic_to_world_like(v)
                    except Exception:
                        pass
            else:
                A = arr_to_44(_to_float_ndarray(v))
                if A is not None:
                    return A

    # 2) rotation + translation at the root
    if "rotation" in J and "translation" in J:
        try:
            return load_extrinsic_to_world_like(J)
        except Exception:
            pass

    # 3) Flat list at the root (ONLY if array-like, not a dict)
    A = _to_float_ndarray(J)
    M = arr_to_44(A)
    if M is not None:
        return M

    return None


def load_extrinsic_to_world_like(J: Dict[str, Any]) -> np.ndarray:
    # helper used above
    def build_from_rt(rot, trans):
        # same as in load_extrinsic_to_cam but without filename heuristics
        R = None
        if isinstance(rot, (list,tuple)):
            a = np.array(rot, dtype=float).reshape(-1)
            if a.size == 9:
                R = a.reshape(3,3)
            elif a.size == 4:
                q = a
                if abs(q[0]) >= 0.5: w,x,y,z = q
                else: x,y,z,w = q
                n = math.sqrt(w*w+x*x+y*y+z*z) or 1.0
                w,x,y,z = w/n, x/n, y/n, z/n
                R = np.array([
                    [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                    [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
                    [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
                ], float)
            elif a.size == 3:
                r,p,y = a.tolist()
                deg = max(abs(r),abs(p),abs(y)) > 2*math.pi
                if deg: r,p,y = map(math.radians,[r,p,y])
                def Rx(a): ca,sa = math.cos(a), math.sin(a); return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], float)
                def Ry(a): ca,sa = math.cos(a), math.sin(a); return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], float)
                def Rz(a): ca,sa = math.cos(a), math.sin(a); return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], float)
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
                ], float)
            elif all(k in rot for k in ("roll","pitch","yaw")):
                r,p,y = float(rot["roll"]), float(rot["pitch"]), float(rot["yaw"])
                def Rx(a): ca,sa = math.cos(a), math.sin(a); return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], float)
                def Ry(a): ca,sa = math.cos(a), math.sin(a); return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], float)
                def Rz(a): ca,sa = math.cos(a), math.sin(a); return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], float)
                R = Rz(y) @ Ry(p) @ Rx(r)
            elif "matrix" in rot:
                R = np.array(rot["matrix"], dtype=float).reshape(3,3)
        t = None
        if isinstance(trans, dict) and all(k in trans for k in ("x","y","z")):
            t = np.array([trans["x"], trans["y"], trans["z"]], dtype=float)
        elif isinstance(trans, (list,tuple)) and len(trans)==3:
            t = np.array(trans, dtype=float)
        if R is None or t is None:
            raise ValueError("Invalid rotation/translation structure")
        M = np.eye(4); M[:3,:3]=R; M[:3,3]=t; return M
    return build_from_rt(J["rotation"], J["translation"])

def find_world_T_lidar(root: str, stem: str, side: str) -> Optional[np.ndarray]:
    """
    Returns 4x4 T_world_lidar for 'veh' or 'infra', using your dataset's layout.
    """
    sd = side_dir_name(side)

    # 1) Primary paths for YOUR structure
    if sd == "vehicle-side":
        # Chain: novatel_to_world ∘ lidar_to_novatel
        p_lidar_to_novatel = Path(root)/"vehicle-side"/"calib"/"lidar_to_novatel"/f"{stem}.json"
        p_novatel_to_world = Path(root)/"vehicle-side"/"calib"/"novatel_to_world"/f"{stem}.json"
        M1 = _load_44_json(p_lidar_to_novatel)
        M2 = _load_44_json(p_novatel_to_world)
        if M1 is not None and M2 is not None:
            return M2 @ M1
        # Optional direct fallbacks if they exist in other variants
        direct_candidates = [
            Path(root)/"vehicle-side"/"calib"/"lidar_to_world"/f"{stem}.json",
            Path(root)/"vehicle-side"/"calib"/"LIDAR_to_world"/f"{stem}.json",
            Path(root)/"vehicle-side"/"calib"/"lidar2world"/f"{stem}.json",
        ]
    else:
        # Infra has direct virtuallidar_to_world in your tree
        direct_candidates = [
            Path(root)/"infrastructure-side"/"calib"/"virtuallidar_to_world"/f"{stem}.json",
            Path(root)/"infrastructure-side"/"calib"/"virtual_lidar_to_world"/f"{stem}.json",
            Path(root)/"infrastructure-side"/"calib"/"lidar_to_world"/f"{stem}.json",  # fallback variant
        ]

    # 2) Try direct candidates
    for p in direct_candidates:
        M = _load_44_json(p)
        if M is not None:
            return M

    # 3) Generic chained fallbacks from earlier version (kept just in case)
    chain_options = []
    if sd == "vehicle-side":
        chain_options += [
            (Path(root)/"vehicle-side"/"calib"/"lidar_to_vehicle"/f"{stem}.json",
             Path(root)/"vehicle-side"/"calib"/"vehicle_to_world"/f"{stem}.json"),
            (Path(root)/"vehicle-side"/"calib"/"lidar_to_ego"/f"{stem}.json",
             Path(root)/"vehicle-side"/"calib"/"ego_to_world"/f"{stem}.json"),
        ]
    else:
        chain_options += [
            (Path(root)/"infrastructure-side"/"calib"/"virtuallidar_to_base"/f"{stem}.json",
             Path(root)/"infrastructure-side"/"calib"/"base_to_world"/f"{stem}.json"),
        ]
    for p1, p2 in chain_options:
        M1 = _load_44_json(p1)
        M2 = _load_44_json(p2)
        if M1 is not None and M2 is not None:
            return M2 @ M1

    print(f"[WARN] Could not find T_world_lidar for {sd} '{stem}'. Checked novatel/base chains and direct *_to_world.")
    return None

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
    # DAIR variants: "3d_location": {"x":..., "y":..., "z":...} and each may be nested (e.g., {"value": ...})
    if "3d_location" in g and isinstance(g["3d_location"], dict):
        loc = g["3d_location"]
        x = _as_float(loc.get("x"), 0.0)
        y = _as_float(loc.get("y"), 0.0)
        z = _as_float(loc.get("z"), 0.0)
        return np.array([x, y, z], dtype=float)

    # Alternate: "center": [x,y,z] or {"x":..}
    if "center" in g:
        c = g["center"]
        if isinstance(c, (list, tuple)) and len(c) == 3:
            return np.array([_as_float(c[0]), _as_float(c[1]), _as_float(c[2])], dtype=float)
        if isinstance(c, dict):
            x = _as_float(c.get("x"), 0.0)
            y = _as_float(c.get("y"), 0.0)
            z = _as_float(c.get("z"), 0.0)
            return np.array([x, y, z], dtype=float)

    return np.array([0.0, 0.0, 0.0], dtype=float)


def _extract_dims_hwl(g: Dict[str, Any]) -> Tuple[float, float, float]:
    # Standard DAIR keys
    if "3d_dimensions" in g and isinstance(g["3d_dimensions"], dict):
        d = g["3d_dimensions"]
        h = _as_float(d.get("h"), 0.0)
        w = _as_float(d.get("w"), 0.0)
        l = _as_float(d.get("l"), 0.0)
        return h, w, l
    if "dimensions" in g and isinstance(g["dimensions"], dict):
        d = g["dimensions"]
        h = _as_float(d.get("h"), 0.0)
        w = _as_float(d.get("w"), 0.0)
        l = _as_float(d.get("l"), 0.0)
        return h, w, l

    # Some formats use "size": [l, w, h] possibly wrapped
    if "size" in g:
        s = g["size"]
        if isinstance(s, (list, tuple)) and len(s) == 3:
            l = _as_float(s[0], 0.0)
            w = _as_float(s[1], 0.0)
            h = _as_float(s[2], 0.0)
            return h, w, l
        if isinstance(s, dict):  # e.g., {"l": {...}, "w": {...}, "h": {...}}
            h = _as_float(s.get("h"), 0.0)
            w = _as_float(s.get("w"), 0.0)
            l = _as_float(s.get("l"), 0.0)
            return h, w, l

    return 0.0, 0.0, 0.0


def _extract_yaw(g: Dict[str, Any]) -> float:
    # Accept "rotation", "yaw", or "rotation_y" in any wrapped form
    for key in ("rotation", "yaw", "rotation_y", "ry"):
        if key in g:
            return _as_float(g[key], 0.0)
    # Sometimes orientation is a dict with degrees/radians flag; if so, try to unwrap "value"
    if "orientation" in g:
        return _as_float(g["orientation"], 0.0)
    return 0.0


def _extract_world8(g: Dict[str, Any]) -> Optional[np.ndarray]:
    pts = g.get("world_8_points", None)
    if pts is None:
        return None
    # case 1: already numeric [[x,y,z], ...]
    try:
        arr = np.array(pts, dtype=float)
        if arr.shape == (8, 3):
            return arr
        if arr.size == 24:
            return arr.reshape(8, 3)
    except Exception:
        pass

    # case 2: list of dicts [{"x":..,"y":..,"z":..}, ...]
    if isinstance(pts, list) and len(pts) == 8 and isinstance(pts[0], dict):
        out = np.zeros((8, 3), dtype=float)
        for i, p in enumerate(pts):
            out[i, 0] = _as_float(p.get("x"), 0.0)
            out[i, 1] = _as_float(p.get("y"), 0.0)
            out[i, 2] = _as_float(p.get("z"), 0.0)
        return out

    # case 3: dict with a nested field like {"points": [...]} or {"data": [...]}
    if isinstance(pts, dict):
        for k in ("points", "data", "value", "vals"):
            if k in pts:
                return _extract_world8({"world_8_points": pts[k]})
    return None

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
        yaw = _extract_yaw(g)
        w8 = _extract_world8(g)
        out.append({"type": typ, "center": c, "dims_hwl": (h,w,l), "yaw": yaw, "world8": w8})
    return out

# ===================== Geometry & I/O =====================

def lidar_box_corners(center: np.ndarray, h: float, w: float, l: float, yaw: float) -> np.ndarray:
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

def make_open3d_box(corners: np.ndarray, color=(1.0, 0.0, 0.0)):
    if o3d is None: return None
    lines = [
        [0,1],[1,2],[2,3],[3,0],  # top
        [4,5],[5,6],[6,7],[7,4],  # bottom
        [0,4],[1,5],[2,6],[3,7],  # verticals
    ]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    )
    cols = np.tile(np.array(color, dtype=float).reshape(1,3), (len(lines),1))
    ls.colors = o3d.utility.Vector3dVector(cols)
    return ls

def _load_bin(bin_file: str) -> np.ndarray:
    if not os.path.exists(bin_file):
        return np.zeros((0,4), dtype=np.float32)
    pts = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)  # x,y,z,intensity
    return pts

def _load_pcd(pcd_file: str) -> np.ndarray:
    if not os.path.exists(pcd_file):
        return np.zeros((0,4), dtype=np.float32)
    if o3d is not None:
        pcd = o3d.io.read_point_cloud(pcd_file)
        if pcd.is_empty():
            return np.zeros((0,4), dtype=np.float32)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.shape[1] == 3:
            intens = np.zeros((pts.shape[0], 1), dtype=np.float32)
            return np.hstack([pts, intens])
        elif pts.shape[1] >= 4:
            return pts[:, :4]
        else:
            intens = np.zeros((pts.shape[0], 1), dtype=np.float32)
            return np.hstack([pts[:, :3], intens])
    try:
        from pypcd import pypcd
        pc = pypcd.PointCloud.from_path(pcd_file)
        N = pc.points
        x = np.asarray(pc.pc_data["x"], dtype=np.float32).reshape(N,1)
        y = np.asarray(pc.pc_data["y"], dtype=np.float32).reshape(N,1)
        z = np.asarray(pc.pc_data["z"], dtype=np.float32).reshape(N,1)
        if "intensity" in pc.pc_data:
            i = np.asarray(pc.pc_data["intensity"], dtype=np.float32).reshape(N,1)
        else:
            i = np.zeros((N,1), dtype=np.float32)
        return np.hstack([x,y,z,i])
    except Exception:
        raise RuntimeError(f"Cannot read PCD: {pcd_file} (install open3d or pypcd)")

def load_point_cloud_any(velodyne_dir: Path, stem: str) -> Tuple[np.ndarray, Path]:
    first_ext = ".pcd" if USE_PCD_LIDAR else ".bin"
    second_ext = ".bin" if USE_PCD_LIDAR else ".pcd"
    p_first = velodyne_dir / f"{stem}{first_ext}"
    p_second = velodyne_dir / f"{stem}{second_ext}"
    if p_first.exists():
        pts = _load_pcd(str(p_first)) if first_ext == ".pcd" else _load_bin(str(p_first))
        return pts, p_first
    if p_second.exists():
        pts = _load_bin(str(p_second)) if second_ext == ".bin" else _load_pcd(str(p_second))
        return pts, p_second
    return np.zeros((0,4), dtype=np.float32), p_first

# ===================== Optional projection (unchanged) =====================

def draw_projected_box_on_image(img: np.ndarray, corners_lidar: np.ndarray, K: np.ndarray, T_cam_lidar: np.ndarray, color=(0,255,255), thickness=2):
    h, w = img.shape[:2]
    N = corners_lidar.shape[0]
    pts_h = np.hstack([corners_lidar, np.ones((N,1), dtype=float)])
    cam = (T_cam_lidar @ pts_h.T).T[:, :3]
    z = cam[:,2]
    uv = (K @ cam.T).T
    uv = uv[:, :2] / np.maximum(z.reshape(-1,1), 1e-6)
    valid = (z > 1e-6)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i,j in edges:
        if valid[i] and valid[j]:
            p1 = (int(round(uv[i,0])), int(round(uv[i,1])))
            p2 = (int(round(uv[j,0])), int(round(uv[j,1])))
            cv2.line(img, p1, p2, color, thickness)

# ===================== Visualization (side-only, unchanged) =====================

def visualize_frame_side(img_path: Path, lbl2_path: Path, lbl3_path: Path, pts: np.ndarray, K_path: Path, T_path: Path, side: str, save_prefix: Path | None):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img_vis = img.copy()
    objs2d = parse_2d_json(lbl2_path) if lbl2_path and lbl2_path.exists() else []
    for o in objs2d:
        x1,y1,x2,y2 = [int(round(v)) for v in o["bbox"]]
        cv2.rectangle(img_vis, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img_vis, o.get("type",""), (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    boxes3d = parse_3d_json(lbl3_path) if lbl3_path and lbl3_path.exists() else []

    o3d_boxes = []
    if o3d is not None:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[:, :3]))
        pcd.colors = o3d.utility.Vector3dVector(np.ones_like(pts[:, :3]) * 0.5)
    else:
        pcd = None

    K = None; T_cl = None
    if PROJECT_3D_ON_IMAGE and K_path.exists() and T_path.exists():
        try:
            K = load_intrinsic(K_path)
            T_cl = load_extrinsic_to_cam(T_path)
        except Exception as e:
            print(f"[WARN] Could not load K/T for projection: {e}")

    for o in boxes3d:
        h,w,l = o["dims_hwl"]; c = o["center"]; yaw = o["yaw"]
        corners = lidar_box_corners(c, h, w, l, yaw)
        if PROJECT_3D_ON_IMAGE and (K is not None) and (T_cl is not None):
            draw_projected_box_on_image(img_vis, corners, K, T_cl, color=(0,255,255), thickness=2)
        if o3d is not None:
            o3d_boxes.append(make_open3d_box(corners, color=(1.0, 0.0, 0.0)))

    if SAVE_OUTPUTS and save_prefix is not None:
        save_prefix.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_prefix.with_suffix(".jpg")), img_vis)

    if SHOW_OPENCV:
        cv2.imshow(f"{side_dir_name(side).upper()} 2D (native GT boxes)", img_vis)
        cv2.waitKey(1)

    if SHOW_OPEN3D and o3d is not None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"{side_dir_name(side).upper()} LiDAR (native 3D GT)", width=1280, height=720, visible=True)
        opt = vis.get_render_option()
        if opt:
            opt.point_size = O3D_POINT_SIZE
            opt.background_color = np.array([0,0,0])
        if pcd is not None:
            vis.add_geometry(pcd)
        for b in o3d_boxes:
            if b is not None:
                vis.add_geometry(b)
        vis.poll_events(); vis.update_renderer()
        if SAVE_OUTPUTS and save_prefix is not None:
            vis.capture_screen_image(str(save_prefix.with_suffix(".o3d.png")), do_render=True)
        print("Close the Open3D window to proceed to next frame...")
        vis.run()
        vis.destroy_window()


def _as_dir_list(dirs_or_dir) -> List[Path]:
    """Normalize a Path or List[Path] into a List[Path]."""
    if isinstance(dirs_or_dir, (list, tuple)):
        return [Path(d) for d in dirs_or_dir]
    return [Path(dirs_or_dir)]

def load_point_cloud_any_flex(dirs_or_dir, stem: str) -> Tuple[np.ndarray, Path]:
    """Try *.pcd/bin across one or many velodyne dirs."""
    dirs = _as_dir_list(dirs_or_dir)
    last_attempt: Optional[Path] = None
    for d in dirs:
        pts, used = load_point_cloud_any(d, stem)
        last_attempt = used
        if used.exists() or pts.shape[0] > 0:
            return pts, used
    if last_attempt is None:
        last_attempt = dirs[-1] / (stem + (".pcd" if USE_PCD_LIDAR else ".bin"))
    return np.zeros((0,4), dtype=np.float32), last_attempt

def collect_stems_flex(dirs_or_dir, exts: List[str]) -> set:
    """Return stems from one or many dirs for given extensions."""
    dirs = _as_dir_list(dirs_or_dir)
    stems = set()
    for d in dirs:
        for ext in exts:
            for p in glob.glob(str(d / f"*{ext}")):
                stems.add(Path(p).stem)
    return stems

# ===================== Cooperative world visualization =====================

def visualize_frame_coop_world(root: str, stem: str):
    """
    Loads:
      - cooperative/label_world/<stem>.json  (3D boxes in world)
      - vehicle-side velodyne (pcd/bin)  -> T_world_veh_lidar (novatel chain)
      - infrastructure-side velodyne (pcd/bin) -> T_world_infra_lidar (direct)
    Draws fused clouds + boxes in one Open3D window.
    """
    label_world_dir, _coop_calib_dir, veh_velo_dirs, infra_velo_dirs = get_dirs_cooperative(root)

    # --- world-frame labels
    lbl_world_path = label_world_dir / f"{stem}.json"
    if not lbl_world_path.exists():
        raise FileNotFoundError(f"World label not found: {lbl_world_path}")
    boxes_world = parse_3d_json(lbl_world_path)

    # --- point clouds (try velodyne/ then velodyne_bin/)
    veh_pts,   veh_used   = load_point_cloud_any_flex(veh_velo_dirs, stem)
    infra_pts, infra_used = load_point_cloud_any_flex(infra_velo_dirs, stem)

    if veh_pts.shape[0] == 0:
        print(f"[WARN] Vehicle cloud missing/empty for '{stem}' (tried {veh_used})")
    if infra_pts.shape[0] == 0:
        print(f"[WARN] Infra cloud missing/empty for '{stem}' (tried {infra_used})")

    # --- world transforms
    T_w_veh = find_world_T_lidar(root, stem, "veh")
    T_w_inf = find_world_T_lidar(root, stem, "infra")

    def apply_T(pts: np.ndarray, T: Optional[np.ndarray]) -> np.ndarray:
        if pts.shape[0] == 0:
            return pts
        if T is None:
            print("[WARN] Missing world transform; leaving points in original lidar frame.")
            return pts[:, :4]
        P = np.hstack([pts[:, :3], np.ones((pts.shape[0], 1), dtype=float)])  # Nx4
        Pw = (T @ P.T).T[:, :3]
        return np.hstack([Pw, pts[:, 3:4]])  # keep intensity

    veh_world   = apply_T(veh_pts,   T_w_veh)
    infra_world = apply_T(infra_pts, T_w_inf)

    # --- Open3D viz
    if o3d is None:
        raise RuntimeError("open3d is required for cooperative world visualization")

    geoms = []
    if veh_world.shape[0] > 0:
        pcd_v = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(veh_world[:, :3]))
        pcd_v.colors = o3d.utility.Vector3dVector(np.ones_like(veh_world[:, :3]) * 0.6)  # gray
        geoms.append(pcd_v)

    if infra_world.shape[0] > 0:
        pcd_i = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(infra_world[:, :3]))
        pcd_i.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.5, 0.6, 1.0]]), (infra_world.shape[0], 1))
        )  # bluish
        geoms.append(pcd_i)

    # world-frame boxes from params (red)
    if DRAW_PARAM_BOXES_IN_WORLD:
        for o in boxes_world:
            h, w, l = o["dims_hwl"]; c = o["center"]; yaw = o["yaw"]
            corners_w = lidar_box_corners(c, h, w, l, yaw)
            b = make_open3d_box(corners_w, color=(1.0, 0.0, 0.0))
            if b is not None:
                geoms.append(b)

    # optional world_8_points overlays (yellow)
    if VIS_WORLD_8_POINTS:
        for o in boxes_world:
            w8 = o.get("world8", None)
            if w8 is not None and w8.shape == (8,3):
                b8 = make_open3d_box(w8, color=(1.0, 1.0, 0.0))  # yellow
                if b8 is not None:
                    geoms.append(b8)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"COOPERATIVE WORLD — {stem}", width=1440, height=900, visible=True)
    opt = vis.get_render_option()
    if opt:
        opt.point_size = O3D_POINT_SIZE
        opt.background_color = np.array([0, 0, 0])
    for g in geoms:
        vis.add_geometry(g)
    vis.poll_events(); vis.update_renderer()
    print("Close the Open3D window to proceed...")
    vis.run()
    vis.destroy_window()

# ===================== Driver =====================

def main():
    root = DATA_ROOT
    mode = VIEW_MODE.lower()

    if mode in ("veh","infra","vehicle-side","infrastructure-side"):
        side = "veh" if mode.startswith("veh") else "infra"
        img_dir, lbl2_dir, lbl3_dir, velo_dir, K_dir, T_dir = get_dirs_native(root, side)

        # Image search with preference
        img_globs = [str(img_dir / "*.jpg"), str(img_dir / "*.png")] if USE_JPG_IMAGES else [str(img_dir / "*.png"), str(img_dir / "*.jpg")]
        img_list = []
        for pat in img_globs:
            img_list.extend(glob.glob(pat))
        img_list = sorted(set(img_list))
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
            K_path    = K_dir / f"{stem}.json"
            T_path    = T_dir / f"{stem}.json" if T_dir.exists() else Path("")

            pts, used_velo = load_point_cloud_any(velo_dir, stem)
            if pts.shape[0] == 0:
                print(f"[WARN] Missing/empty velodyne for {stem} (looked for {used_velo.with_suffix('.pcd')} or .bin); skipping point cloud.")

            save_prefix = None
            if SAVE_OUTPUTS:
                save_prefix = Path(OUTPUT_DIR) / ("{}_{}".format(side_dir_name(side), stem))

            try:
                visualize_frame_side(img_path, lbl2_path, lbl3_path, pts, K_path, T_path, side, save_prefix)
            except Exception as e:
                print(f"[ERROR] Frame {stem} failed: {e}")

        if SHOW_OPENCV:
            print("[INFO] Done. Press a key to close the OpenCV window.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif mode == "cooperative_world":
        label_world_dir, _, veh_velo_dir, infra_velo_dir = get_dirs_cooperative(root)
        # Build the id list from whichever source is available:
        # Prefer world labels directory as the master list.
        ids = sorted([Path(p).stem for p in glob.glob(str(label_world_dir / "*.json"))])
        if len(ids) == 0:
            # Fall back: intersect veh/infra velodyne stems
            veh_ids = collect_stems_flex(veh_velo_dir, [".pcd", ".bin"])
            inf_ids = collect_stems_flex(infra_velo_dir, [".pcd", ".bin"])
            ids = sorted(veh_ids & inf_ids)
        if len(ids) == 0:
            raise RuntimeError("No frames found for cooperative_world mode (no label_world/*.json or velodyne files).")

        end_idx = min(len(ids), START_IDX + MAX_FRAMES)
        print(f"[INFO] Cooperative world frames {START_IDX}..{end_idx-1} ({end_idx-START_IDX} total)")

        for k in range(START_IDX, end_idx):
            stem = ids[k]
            try:
                visualize_frame_coop_world(root, stem)
            except Exception as e:
                print(f"[ERROR] Coop world frame {stem} failed: {e}")

    else:
        raise ValueError("VIEW_MODE must be 'veh', 'infra', or 'cooperative_world'")

if __name__ == "__main__":
    main()
