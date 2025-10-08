#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
from math import atan2, cos, sin, pi

# ==================== EDIT THIS PATH ====================
ROOT = Path("/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets/dair_v2x_synth_TEST1/cooperative-vehicle-infrastructure")
# ========================================================

VEH_LABEL_DIR = ROOT / "vehicle-side/label/lidar"
VEH_CAL_L2NOV = ROOT / "vehicle-side/calib/lidar_to_novatel"
VEH_CAL_NOV2W = ROOT / "vehicle-side/calib/novatel_to_world"
OUT_DIR       = ROOT / "cooperative/label_world_vehicle_only"

CLASS_MAP = {
    "Car": 0, "Truck": 1, "Bus": 2, "Van": 3,
    "Pedestrian": 4, "Cyclist": 5, "Motorcyclist": 6,
    "Trafficcone": 7, "Tricycle": 8, "Forklift": 9,
}

def jload(p: Path):
    return json.loads(p.read_text())

def jdump(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))

def _as3x3(R):
    return np.array(R, dtype=float).reshape(3,3)

def _as3x1(t):
    arr = np.array(t, dtype=float).reshape(-1)
    return arr[:3].reshape(3,1)

def load_lidar_to_novatel(p: Path):
    if not p.exists():
        return np.eye(3), np.zeros((3,1))
    d = jload(p)
    if "transform" in d:
        d = d["transform"]
    R = _as3x3(d["rotation"])
    T = _as3x1(d["translation"])
    return R, T

def load_novatel_to_world(p: Path):
    if not p.exists():
        return np.eye(3), np.zeros((3,1))
    d = jload(p)
    R = _as3x3(d["rotation"])
    T = _as3x1(d["translation"])
    return R, T

def chain(R_ba, T_ba, R_cb, T_cb):
    """A->B then B->C gives A->C (compose rigid transforms)."""
    R = R_cb @ R_ba
    T = R_cb @ T_ba + T_cb
    return R, T

def wrap_angle(a):
    # wrap to [-pi, pi)
    return (a + pi) % (2*pi) - pi

def yaw_from_Rz(R):
    # assuming Z-up; extract yaw from rotation matrix
    return atan2(R[1,0], R[0,0])

def box_corners_local(l, w, h):
    """
    Returns 8 corners in the box *local* frame (centered at origin),
    order: 4 bottom then 4 top. Axes: l (x), w (y), h (z).
    """
    x = l / 2.0; y = w / 2.0; z = h / 2.0
    # bottom (z = -z), then top (z = +z)
    corners = np.array([
        [ +x, +y, -z ],
        [ +x, -y, -z ],
        [ -x, -y, -z ],
        [ -x, +y, -z ],
        [ +x, +y, +z ],
        [ +x, -y, +z ],
        [ -x, -y, +z ],
        [ -x, +y, +z ],
    ], dtype=float)  # shape (8,3)
    return corners

def rotate_z(points, yaw):
    c, s = cos(yaw), sin(yaw)
    Rz = np.array([[c, -s, 0],
                   [s,  c, 0],
                   [0,  0, 1]], dtype=float)
    return (Rz @ points.T).T

def transform_points(points, R, T):
    # points: (N,3), R: 3x3, T: (3,1)
    return (R @ points.T + T).T

def ensure_field(obj, key, default):
    if key not in obj:
        obj[key] = default
    return obj[key]

def parse_objects_lidar(file_path: Path):
    """
    Reads the *list of objects* (GT or detections) in LiDAR frame.
    Each item ideally has keys:
      type, 3d_dimensions{h,w,l}, 3d_location{x,y,z}, rotation
    Returns the raw dicts (so we can carry extra fields through).
    """
    data = jload(file_path)
    if isinstance(data, list):
        return data
    # if file is {"boxes_3d":[...]} style (older), adapt minimally
    objs = []
    if isinstance(data, dict) and "boxes_3d" in data:
        for arr in data["boxes_3d"]:
            x,y,z,l,w,h,yaw = arr
            objs.append({
                "type": "car",
                "occluded_state": 0,
                "truncated_state": 0,
                "alpha": 0.0,
                "2d_box": {"xmin":0,"ymin":0,"xmax":0,"ymax":0},
                "3d_dimensions": {"h": float(h), "w": float(w), "l": float(l)},
                "3d_location": {"x": float(x), "y": float(y), "z": float(z)},
                "rotation": float(yaw)
            })
    return objs

def obj_lwh_xyz_yaw(obj):
    dims = obj.get("3d_dimensions", {})
    loc  = obj.get("3d_location", {})
    h = float(dims.get("h", 1.5))
    w = float(dims.get("w", 1.8))
    l = float(dims.get("l", 4.0))
    x = float(loc.get("x", 0.0))
    y = float(loc.get("y", 0.0))
    z = float(loc.get("z", 0.0))
    yaw = float(obj.get("rotation", 0.0))
    return l, w, h, x, y, z, yaw

def to_required_schema_world(obj_in, center_w, yaw_w, corners_w):
    """
    Build one output dict in the required schema, in WORLD frame.
    - center_w: (3,) world center
    - yaw_w: float
    - corners_w: (8,3) in world
    """
    # carry over or default fields
    typ = obj_in.get("type", "car")

    occluded_state  = obj_in.get("occluded_state", 0)
    truncated_state = obj_in.get("truncated_state", 0)
    alpha           = obj_in.get("alpha", 0.0)

    # keep given 2D box if present; else zeros
    box2d = obj_in.get("2d_box", {
        "xmin": 0.0, "ymin": 0.0, "xmax": 0.0, "ymax": 0.0
    })

    dims = obj_in.get("3d_dimensions", {})
    h = float(dims.get("h", 1.5))
    w = float(dims.get("w", 1.8))
    l = float(dims.get("l", 4.0))

    out = {
        "type": typ,
        "occluded_state": int(occluded_state),
        "truncated_state": int(truncated_state),
        "alpha": float(alpha),
        "2d_box": {
            "xmin": float(box2d.get("xmin", 0.0)),
            "ymin": float(box2d.get("ymin", 0.0)),
            "xmax": float(box2d.get("xmax", 0.0)),
            "ymax": float(box2d.get("ymax", 0.0)),
        },
        "3d_dimensions": {"h": float(h), "w": float(w), "l": float(l)},
        "3d_location": {
            "x": float(center_w[0]),
            "y": float(center_w[1]),
            "z": float(center_w[2]),
        },
        "rotation": float(yaw_w),
        "world_8_points": corners_w.tolist()
    }
    return out

def main():
    if not VEH_LABEL_DIR.exists():
        print(f"ERROR: {VEH_LABEL_DIR} not found")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(VEH_LABEL_DIR.glob("*.json"))
    written = 0

    for f in files:
        vid = f.stem  # e.g. 000000

        # Load objects in LiDAR frame
        objs_lidar = parse_objects_lidar(f)

        # Load transforms: LiDAR->Novatel, Novatel->World, compose to LiDAR->World
        R_l2n, T_l2n = load_lidar_to_novatel(VEH_CAL_L2NOV / f"{vid}.json")
        R_n2w, T_n2w = load_novatel_to_world(VEH_CAL_NOV2W / f"{vid}.json")
        R_l2w, T_l2w = chain(R_l2n, T_l2n, R_n2w, T_n2w)

        # Extract yaw component of LiDAR->World to rotate headings
        yaw_l2w = yaw_from_Rz(R_l2w)

        out_list = []
        for obj in objs_lidar:
            l, w, h, x, y, z, yaw = obj_lwh_xyz_yaw(obj)

            # center LiDAR -> World
            center_l = np.array([x, y, z]).reshape(3,1)
            center_w = (R_l2w @ center_l + T_l2w).reshape(3)

            # yaw LiDAR -> World (add the transform heading around Z)
            yaw_w = wrap_angle(yaw + yaw_l2w)

            # compute world 8 corners
            corners_local = box_corners_local(l, w, h)          # (8,3)
            corners_lidar = rotate_z(corners_local, yaw)        # orient in LiDAR
            corners_lidar = corners_lidar + np.array([x, y, z]) # shift to LiDAR center
            corners_world = transform_points(corners_lidar, R_l2w, T_l2w)  # (8,3)

            out_obj = to_required_schema_world(obj, center_w, yaw_w, corners_world)
            out_list.append(out_obj)

        out_p = OUT_DIR / f"{vid}.json"
        jdump(out_list, out_p)
        written += 1

    print(f"[done] wrote {written} WORLD-frame vehicle-only label files to: {OUT_DIR}")

if __name__ == "__main__":
    main()
