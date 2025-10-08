#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vehicle LiDAR -> WORLD rich labels (list of dicts).

- Reads cooperative/data_info.json to get vehicle IDs
- Loads vehicle-side labels from vehicle-side/label/lidar/<veh_id>.json
  Accepts either:
    * list of dicts (already rich schema)
    * dict with "boxes_3d" + "labels"
- Loads calibrations:
    vehicle-side/calib/lidar_to_novatel/<veh_id>.json   (wrapped or flat)
    vehicle-side/calib/novatel_to_world/<veh_id>.json   (flat)
- Transforms center and all 8 corners to WORLD
- Writes cooperative/label_world/<veh_id>.json as a LIST of dicts
  (fields: type, occluded_state, truncated_state, alpha, 2d_box, 3d_dimensions,
   3d_location, rotation, world_8_points)

Only stdlib + numpy.
"""

import os
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

# ================= USER CONFIG =================
DATA_ROOT = Path("/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets/dair_v2x_synth_TEST1/cooperative-vehicle-infrastructure/")
# If your input labels are ints, map them to strings here:
INT_TO_NAME = {0:"car", 1:"truck", 2:"bus", 3:"pedestrian", 4:"cyclist"}
# ===============================================

def info(m): print(f"[INFO] {m}")
def warn(m): print(f"[WARN] {m}", file=sys.stderr)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def load_json(p: Path) -> Any:
    with open(p, "r") as f: return json.load(f)
def save_json(p: Path, obj: Any):
    with open(p, "w") as f: json.dump(obj, f, indent=2)

def stem_from_path_like(s: str) -> str:
    b = os.path.basename(s)
    return os.path.splitext(b)[0]

# ---------- calib utils ----------
def to_vec3(val) -> np.ndarray:
    if val is None: return np.zeros(3, dtype=float)
    if isinstance(val, dict):
        return np.array([float(val.get("x", 0.0)),
                         float(val.get("y", 0.0)),
                         float(val.get("z", 0.0))], dtype=float)
    arr = np.array(val, dtype=float)
    if arr.shape == (3,): return arr
    if arr.shape == (3,1): return arr[:,0]
    if arr.shape == (1,3): return arr[0]
    arr = arr.reshape(-1)
    if arr.size == 3: return arr
    warn("Unrecognized translation; using zeros.")
    return np.zeros(3, dtype=float)

def to_rot3x3(val) -> np.ndarray:
    if val is None: return np.eye(3, dtype=float)
    arr = np.array(val, dtype=float)
    if arr.size == 9: return arr.reshape(3,3)
    warn("Unrecognized rotation; using identity.")
    return np.eye(3, dtype=float)

def parse_lidar_to_novatel(J: Dict) -> Tuple[np.ndarray, np.ndarray]:
    # Supports {"transform":{"rotation":...,"translation":...}} or flat
    if isinstance(J.get("transform"), dict):
        T = J["transform"]; return to_rot3x3(T.get("rotation")), to_vec3(T.get("translation"))
    return to_rot3x3(J.get("rotation")), to_vec3(J.get("translation"))

def parse_rt_flat(J: Dict) -> Tuple[np.ndarray, np.ndarray]:
    return to_rot3x3(J.get("rotation")), to_vec3(J.get("translation"))

def compose(R2, t2, R1, t1):
    # Apply 2 after 1
    return R2 @ R1, R2 @ t1 + t2

# ---------- geometry ----------
def rotz(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def corners_3d_local(l: float, w: float, h: float) -> np.ndarray:
    """Eight corners around origin before yaw: 4 bottom then 4 top."""
    lx, wy, hz = 0.5*l, 0.5*w, 0.5*h
    return np.array([
        [+lx, -wy, -hz],
        [-lx, -wy, -hz],
        [-lx, +wy, -hz],
        [+lx, +wy, -hz],
        [+lx, -wy, +hz],
        [-lx, -wy, +hz],
        [-lx, +wy, +hz],
        [+lx, +wy, +hz],
    ], dtype=float)

def corners_3d(center: np.ndarray, l: float, w: float, h: float, yaw: float) -> np.ndarray:
    local = corners_3d_local(l,w,h)
    Rz = rotz(yaw)
    return (local @ Rz.T) + center.reshape(1,3)

# ---------- label parsing ----------
def _get(d: Dict, keys: List[str], default=None):
    for k in keys:
        if k in d: return d[k]
    return default

def _parse_center(obj: Dict) -> np.ndarray:
    return to_vec3(_get(obj, ["3d_location","center","location"], None))

def _parse_dims_hwl(obj: Dict) -> Tuple[float,float,float]:
    dims = _get(obj, ["3d_dimensions","dimensions","size"], None)
    if isinstance(dims, dict):
        h = float(dims.get("h", dims.get("height", 0.0)))
        w = float(dims.get("w", dims.get("width",  0.0)))
        l = float(dims.get("l", dims.get("length", 0.0)))
        return h,w,l
    if isinstance(dims, (list,tuple)) and len(dims)==3:
        # assume [l,w,h] (most compact arrays are x,y,z,l,w,h,yaw with l,w,h order)
        l,w,h = float(dims[0]), float(dims[1]), float(dims[2])
        return h,w,l
    return 0.0,0.0,0.0

def _parse_yaw(obj: Dict) -> float:
    v = _get(obj, ["rotation","yaw","rotation_y","ry"], 0.0)
    try: return float(v)
    except: return 0.0

def _parse_type(obj: Dict):
    return _get(obj, ["type","class","label","category"], "car")

def normalize_to_rich_list(J: Any) -> List[Dict[str,Any]]:
    """
    If labels already a list of dicts -> return as-is.
    If dict with boxes_3d + labels -> convert to list-of-dicts with minimal fields.
    """
    if isinstance(J, list):
        return [x for x in J if isinstance(x, dict)]

    if isinstance(J, dict):
        boxes = J.get("boxes_3d") or J.get("boxes") or []
        labels = J.get("labels") or J.get("classes") or []
        if len(labels) < len(boxes):
            labels += [0]*(len(boxes)-len(labels))
        elif len(labels) > len(boxes):
            labels = labels[:len(boxes)]
        out = []
        for i, b in enumerate(boxes):
            arr = np.array(b, dtype=float).reshape(-1)
            if arr.size < 7: continue
            x,y,z,l,w,h,yaw = arr[:7].tolist()
            t = labels[i]
            if isinstance(t, int):
                typ = INT_TO_NAME.get(t, "car")
            elif isinstance(t, str):
                typ = t
            else:
                typ = "car"
            out.append({
                "type": typ,
                "occluded_state": 0,
                "truncated_state": 0,
                "alpha": 0.0,
                "2d_box": {"xmin":0.0,"ymin":0.0,"xmax":0.0,"ymax":0.0},
                "3d_dimensions": {"h": float(h), "w": float(w), "l": float(l)},
                "3d_location":  {"x": float(x), "y": float(y), "z": float(z)},
                "rotation": float(yaw)
            })
        return out

    warn("Unrecognized vehicle label schema; returning empty.")
    return []

# ---------- main ----------
def main():
    root = DATA_ROOT.resolve()
    coop_dir = root / "cooperative"
    data_info = coop_dir / "data_info.json"
    out_dir = coop_dir / "label_world"
    ensure_dir(out_dir)

    veh_lbl_dir = root / "vehicle-side" / "label" / "lidar"
    veh_l2n_dir = root / "vehicle-side" / "calib" / "lidar_to_novatel"
    veh_n2w_dir = root / "vehicle-side" / "calib" / "novatel_to_world"

    # load pairs
    try:
        pairs = load_json(data_info)
        if not isinstance(pairs, list):
            pairs = pairs.get("pairs", [])
    except Exception as e:
        warn(f"Failed to read {data_info}: {e}")
        return

    info(f"Found {len(pairs)} pairs")
    written = 0
    total_objs = 0

    for entry in pairs:
        veh_id = stem_from_path_like(entry.get("vehicle_image_path", "") or entry.get("vehicle_image",""))
        if not veh_id:
            warn("Skipping missing veh_id")
            continue

        # labels in LiDAR frame
        lbl_path = veh_lbl_dir / f"{veh_id}.json"
        if not lbl_path.exists():
            warn(f"Vehicle labels missing: {lbl_path}")
            save_json(out_dir / f"{veh_id}.json", [])
            continue

        try:
            raw = load_json(lbl_path)
        except Exception as e:
            warn(f"Failed to read labels {lbl_path}: {e}")
            save_json(out_dir / f"{veh_id}.json", [])
            continue

        items = normalize_to_rich_list(raw)

        # vehicle transforms
        try:
            R_l2n, t_l2n = parse_lidar_to_novatel(load_json(veh_l2n_dir / f"{veh_id}.json"))
        except Exception:
            warn(f"Missing/bad lidar_to_novatel for {veh_id}; using identity")
            R_l2n, t_l2n = np.eye(3), np.zeros(3)
        try:
            R_n2w, t_n2w = parse_rt_flat(load_json(veh_n2w_dir / f"{veh_id}.json"))
        except Exception:
            warn(f"Missing/bad novatel_to_world for {veh_id}; using identity")
            R_n2w, t_n2w = np.eye(3), np.zeros(3)
        R_v, t_v = compose(R_n2w, t_n2w, R_l2n, t_l2n)  # LiDAR -> WORLD

        out_items: List[Dict[str,Any]] = []
        for obj in items:
            dims = obj.get("3d_dimensions", {})
            loc = obj.get("3d_location", {})
            l = float(dims.get("l", 0.0))
            w = float(dims.get("w", 0.0))
            h = float(dims.get("h", 0.0))
            x = float(loc.get("x", 0.0))
            y = float(loc.get("y", 0.0))
            z = float(loc.get("z", 0.0))
            yaw = float(obj.get("rotation", 0.0))

            center_l = np.array([x,y,z], dtype=float)
            corners_l = corners_3d(center_l, l, w, h, yaw)

            center_w = (R_v @ center_l) + t_v
            corners_w = (corners_l @ R_v.T) + t_v

            new_obj = dict(obj)  # preserve keys like type/2d_box/alpha/etc.
            new_obj["3d_location"] = {"x": float(center_w[0]),
                                      "y": float(center_w[1]),
                                      "z": float(center_w[2])}
            new_obj["world_8_points"] = [[float(p[0]), float(p[1]), float(p[2])]
                                         for p in corners_w]
            # rotation left as original LiDAR yaw (acceptable per your earlier note)
            out_items.append(new_obj)

        out_path = out_dir / f"{veh_id}.json"
        save_json(out_path, out_items)
        info(f"{veh_id}: wrote {len(out_items)} objects -> {out_path.name}")
        written += 1
        total_objs += len(out_items)

    print("\n===== SUMMARY =====")
    print(f"Files written   : {written}")
    print(f"Objects written : {total_objs}")
    print(f"Output dir      : {out_dir}")
    print("===================\n")

if __name__ == "__main__":
    main()
