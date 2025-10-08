#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np

# ==================== EDIT THIS PATH ====================
ROOT = Path("/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets/dair_v2x_synth_TEST1/cooperative-vehicle-infrastructure")
# ========================================================

# Inputs
VEH_LABEL_DIR     = ROOT / "vehicle-side/label/lidar"
VEH_CAL_L2NOV     = ROOT / "vehicle-side/calib/lidar_to_novatel"
VEH_CAL_NOV2W     = ROOT / "vehicle-side/calib/novatel_to_world"

INF_LABEL_DIR     = ROOT / "infrastructure-side/label/lidar"
INF_CAL_VL2W      = ROOT / "infrastructure-side/calib/virtuallidar_to_world"
INF_CAL_VL2BASE   = ROOT / "infrastructure-side/calib/virtuallidar_to_base"
INF_CAL_BASE2W    = ROOT / "infrastructure-side/calib/base_to_world"

# Output (merged vehicle + infrastructure, in WORLD frame)
OUT_DIR           = ROOT / "cooperative/label_world"

# (Optional map, currently unused but kept for completeness)
CLASS_MAP = {
    "Car": 0, "Truck": 1, "Bus": 2, "Van": 3,
    "Pedestrian": 4, "Cyclist": 5, "Motorcyclist": 6,
    "Trafficcone": 7, "Tricycle": 8, "Forklift": 9,
}

# ---------- I/O helpers ----------

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

# ---------- Vehicle transforms (lidar -> novatel -> world) ----------

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

# ---------- Infrastructure transforms (prefer direct virtuallidar_to_world) ----------

def _load_rt_json(p: Path):
    """Load a JSON with {rotation: 3x3 or list(9), translation: list(3)}. Returns (R,T) or (I,0) if missing."""
    if not p.exists():
        return None, None
    d = jload(p)
    src = d.get("transform", d)
    if "rotation" in src and "translation" in src:
        try:
            R = _as3x3(src["rotation"])
            T = _as3x1(src["translation"])
            return R, T
        except Exception:
            pass
    # Minimal other shapes not handled here; your dataset uses rotation/translation.
    return None, None

def load_infra_lidar_to_world(stem: str):
    """
    Try infrastructure-side direct virtuallidar_to_world/<id>.json
    else chain: base_to_world @ virtuallidar_to_base
    Returns (R_i2w, T_i2w). If nothing found, returns (I,0).
    """
    # Direct first
    R, T = _load_rt_json(INF_CAL_VL2W / f"{stem}.json")
    if (R is not None) and (T is not None):
        return R, T

    # Chain fallback
    R_v2b, T_v2b = _load_rt_json(INF_CAL_VL2BASE / f"{stem}.json")
    R_b2w, T_b2w = _load_rt_json(INF_CAL_BASE2W / f"{stem}.json")
    if (R_v2b is not None) and (R_b2w is not None):
        return chain(R_v2b, T_v2b, R_b2w, T_b2w)  # virtuallidar->base then base->world

    # Nothing found
    print(f"[WARN] No infra world transform for '{stem}' (looked in '{INF_CAL_VL2W}', or chain via base). Using identity.")
    return np.eye(3), np.zeros((3,1))

# ---------- Rigid composition ----------

def chain(R_ba, T_ba, R_cb, T_cb):
    """A->B then B->C gives A->C (compose rigid transforms)."""
    R = R_cb @ R_ba
    T = R_cb @ T_ba + T_cb
    return R, T

# ---------- Box geometry ----------

def box_corners_local(l, w, h):
    """
    8 corners in the box local frame (centered), order: 4 bottom then 4 top.
    Axes: l (x), w (y), h (z).
    """
    x = l / 2.0; y = w / 2.0; z = h / 2.0
    return np.array([
        [ +x, +y, -z ],
        [ +x, -y, -z ],
        [ -x, -y, -z ],
        [ -x, +y, -z ],
        [ +x, +y, +z ],
        [ +x, -y, +z ],
        [ -x, -y, +z ],
        [ -x, +y, +z ],
    ], dtype=float)

def Rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)

# ---------- Label parsing ----------

def parse_objects_lidar(file_path: Path):
    """
    Supports:
      - list of rich dicts
      - legacy {"boxes_3d": [[x,y,z,l,w,h,yaw], ...]}
    """
    if not file_path.exists():
        return []
    data = jload(file_path)
    if isinstance(data, list):
        return data
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
    typ = obj_in.get("type", "car")
    occluded_state  = obj_in.get("occluded_state", 0)
    truncated_state = obj_in.get("truncated_state", 0)
    alpha           = obj_in.get("alpha", 0.0)
    box2d = obj_in.get("2d_box", {"xmin":0.0,"ymin":0.0,"xmax":0.0,"ymax":0.0})
    dims = obj_in.get("3d_dimensions", {})
    h = float(dims.get("h", 1.5))
    w = float(dims.get("w", 1.8))
    l = float(dims.get("l", 4.0))
    return {
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
        "3d_location": {"x": float(center_w[0]), "y": float(center_w[1]), "z": float(center_w[2])},
        "rotation": float(yaw_w),
        "world_8_points": corners_w.tolist()
    }

# ---------- Main ----------

def main():
    if not VEH_LABEL_DIR.exists():
        print(f"ERROR: {VEH_LABEL_DIR} not found")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(VEH_LABEL_DIR.glob("*.json"))
    written = 0

    for f in files:
        vid = f.stem

        # --- VEHICLE SIDE ---
        veh_objs_lidar = parse_objects_lidar(f)

        # LiDAR->World via (lidar->novatel) then (novatel->world)
        R_l2n, T_l2n = load_lidar_to_novatel(VEH_CAL_L2NOV / f"{vid}.json")
        R_n2w, T_n2w = load_novatel_to_world(VEH_CAL_NOV2W / f"{vid}.json")
        R_v_l2w, T_v_l2w = chain(R_l2n, T_l2n, R_n2w, T_n2w)

        merged_world_list = []

        for obj in veh_objs_lidar:
            l, w, h, x, y, z, yaw_l = obj_lwh_xyz_yaw(obj)

            center_l = np.array([x, y, z], dtype=float).reshape(3,1)
            center_w = (R_v_l2w @ center_l + T_v_l2w).reshape(3)

            R_obj_w = R_v_l2w @ Rz(yaw_l)
            yaw_w = float(np.arctan2(R_obj_w[1,0], R_obj_w[0,0]))

            corners_local = box_corners_local(l, w, h)
            corners_world = (R_obj_w @ corners_local.T).T + center_w.reshape(1,3)

            merged_world_list.append(to_required_schema_world(obj, center_w, yaw_w, corners_world))

        # --- INFRASTRUCTURE SIDE (if present) ---
        infra_label_path = INF_LABEL_DIR / f"{vid}.json"
        infra_objs_lidar = parse_objects_lidar(infra_label_path)
        if len(infra_objs_lidar) > 0:
            R_i_l2w, T_i_l2w = load_infra_lidar_to_world(vid)

            for obj in infra_objs_lidar:
                l, w, h, x, y, z, yaw_l = obj_lwh_xyz_yaw(obj)

                center_l = np.array([x, y, z], dtype=float).reshape(3,1)
                center_w = (R_i_l2w @ center_l + T_i_l2w).reshape(3)

                R_obj_w = R_i_l2w @ Rz(yaw_l)
                yaw_w = float(np.arctan2(R_obj_w[1,0], R_obj_w[0,0]))

                corners_local = box_corners_local(l, w, h)
                corners_world = (R_obj_w @ corners_local.T).T + center_w.reshape(1,3)

                merged_world_list.append(to_required_schema_world(obj, center_w, yaw_w, corners_world))
        else:
            # No infra labels for this id – that's fine; we still write vehicle-only
            pass

        # --- WRITE MERGED WORLD LABELS ---
        out_p = OUT_DIR / f"{vid}.json"
        jdump(merged_world_list, out_p)
        written += 1

    print(f"[done] wrote {written} WORLD-frame cooperative label files (veh + infra when available) to: {OUT_DIR}")

if __name__ == "__main__":
    main()
