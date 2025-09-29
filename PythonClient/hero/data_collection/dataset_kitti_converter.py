#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_kitti_converter.py (fixed)

Convert a DAIR-V2X-style dataset produced by your simulator into a KITTI-style
dataset (images, velodyne, calib, label_2) — one side at a time.

Key choices in this version:
- It keeps your original dimension mapping EXACTLY as you had it:
    given source dims {h,w,l}, we pass size_lwh = [h, l, w]
  because that's what matches your LiDAR geometry.
- It applies a constant yaw compensation BEFORE the LiDAR->camera conversion
  so the heading aligns with the remapped local box axes:
    yaw_corr = yaw + YAW_OFFSET_RAD
  Default is +90°. If headings are 90° the other way, run with --yaw_offset_deg -90.

Assumed per-side layout:
  <SRC>/<side>/image/<id>.png|jpg
  <SRC>/<side>/velodyne/<id>.bin
  <SRC>/<side>/label/camera/<id>.json
  <SRC>/<side>/label/lidar/<id>.json
  <SRC>/<side>/calib/camera_intrinsic/<id>.json
  <SRC>/<side>/calib/lidar_to_camera/<id>.json           (vehicle-side)
  <SRC>/<side>/calib/virtuallidar_to_camera/<id>.json     (infrastructure-side)

Output (KITTI):
  <OUT>/training/{image_2, velodyne, label_2, calib, ImageSets}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import cv2

# ===================== User configuration =====================
# You can override these with CLI flags.
SRC_ROOT = Path("/media/sgarimella34/hercules-collect/collaborative-perception-BEVP/datasets/dair_v2x_synth/cooperative-vehicle-infrastructure/")
OUT_ROOT = Path("/media/sgarimella34/hercules-collect/collaborative-perception-BEVP/datasets/dair_v2x_synth_kitti")
SIDES    = ["vehicle-side", "infrastructure-side"]
MAKE_VAL_SPLIT = False

# Keep your original dimension mapping AND rotate yaw accordingly.
# If headings are 90° the other way, change to -math.pi/2 or pass --yaw_offset_deg -90.
YAW_OFFSET_RAD = math.pi / 2.0
# YAW_OFFSET_RAD = 0.0
# ===============================================================


# ------------------------ IO helpers ------------------------
def read_json(p: Path) -> Any:
    with open(p, "r") as f:
        return json.load(f)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def get_ids(img_dir: Path) -> List[str]:
    ids = [p.stem for p in sorted(img_dir.glob("*.png"))] + \
          [p.stem for p in sorted(img_dir.glob("*.jpg"))]
    return sorted(set(ids))


# -------------------- Intrinsic parsing --------------------
def _try_matrix(obj: Any) -> np.ndarray | None:
    try:
        arr = np.array(obj, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size == 9:
        return arr.reshape(3, 3)
    return None

def parse_intrinsic(K_json: Path) -> np.ndarray:
    """
    Accepts any of:
      - {"K":[9] or [[3],[3],[3]]}
      - {"camera_matrix":{"data":[9]}} or {"intrinsic":{"matrix":[9]}}
      - {"fx","fy","cx","cy"} (optionally nested under "intrinsic")
    """
    J = read_json(K_json)

    # common containers
    for k in ("K", "camera_matrix", "intrinsic", "cam_K", "matrix"):
        if k in J:
            node = J[k]
            if isinstance(node, dict):
                for kk in ("K","matrix","data","values"):
                    if kk in node:
                        M = _try_matrix(node[kk])
                        if M is not None:
                            return M
            else:
                M = _try_matrix(node)
                if M is not None:
                    return M

    # fx,fy,cx,cy at top-level or under "intrinsic"
    def extract_fx_fy_cx_cy(d: Dict[str, Any]) -> Tuple[float,float,float,float] | None:
        fx, fy, cx, cy = d.get("fx"), d.get("fy"), d.get("cx"), d.get("cy")
        if None not in (fx, fy, cx, cy):
            fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
            return fx, fy, cx, cy
        return None

    tpl = extract_fx_fy_cx_cy(J)
    if tpl is None and "intrinsic" in J and isinstance(J["intrinsic"], dict):
        tpl = extract_fx_fy_cx_cy(J["intrinsic"])
    if tpl is not None:
        fx, fy, cx, cy = tpl
        return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)

    raise ValueError(f"Unrecognized intrinsic JSON format: {K_json}")


# -------------------- Extrinsic parsing --------------------
def _quat_to_R(q: Iterable[float]) -> np.ndarray:
    q = np.asarray(list(q), dtype=float).reshape(-1)
    if q.size != 4:
        raise ValueError("Quaternion must have 4 elements [w,x,y,z] or [x,y,z,w].")
    # accept [x,y,z,w] too
    if abs(q[0]) < 0.5 and abs(q[3]) > 0.5:
        q = np.array([q[3], q[0], q[1], q[2]], dtype=float)
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=float)
    return R

def _euler_to_R(roll: float, pitch: float, yaw: float, degrees: bool=False) -> np.ndarray:
    if degrees:
        roll, pitch, yaw = math.radians(roll), math.radians(pitch), math.radians(yaw)
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ]
    ], dtype=float)
    return R

def parse_extrinsic(T_json: Path) -> np.ndarray:
    """
    Returns 4x4 matrix for LiDAR->Camera.
    If the filename indicates camera->lidar, we auto-invert.
    Accepts variants with:
      - {"matrix":[4x4] or [16]} / {"T":[...]} / {"transform":[...]}
      - {"rotation":{matrix|quat|euler}, "translation":{x,y,z}}
    """
    J = read_json(T_json)

    def try_matrix_like(obj: Any) -> np.ndarray | None:
        try:
            arr = np.array(obj, dtype=float).reshape(-1)
        except Exception:
            return None
        if arr.size == 16:
            M = np.eye(4, dtype=float); M[:] = arr.reshape(4,4)
            return M
        if arr.size == 12:
            M = np.eye(4, dtype=float); M[:3,:] = arr.reshape(3,4)
            return M
        return None

    # direct matrix
    for k in ("matrix","T","transform","data","values"):
        if k in J:
            M = try_matrix_like(J[k])
            if M is not None:
                break
    else:
        # nested candidates
        M = None
        for k in ("matrix","T","transform"):
            if k in J and isinstance(J[k], dict):
                for kk in ("matrix","data","values"):
                    M = try_matrix_like(J[k].get(kk))
                    if M is not None:
                        break
            if M is not None:
                break
        if M is None and "extrinsic" in J and isinstance(J["extrinsic"], dict):
            node = J["extrinsic"]
            for kk in ("matrix","data","values"):
                M = try_matrix_like(node.get(kk))
                if M is not None:
                    break

    if M is None:
        # build from rotation + translation
        def build_from_rt(rot: Any, trans: Any) -> np.ndarray | None:
            R = None
            if isinstance(rot, dict):
                if all(k in rot for k in ("w","x","y","z")):
                    R = _quat_to_R([rot["w"],rot["x"],rot["y"],rot["z"]])
                elif all(k in rot for k in ("x","y","z","w")):
                    R = _quat_to_R([rot["w"],rot["x"],rot["y"],rot["z"]])
                elif all(k in rot for k in ("roll","pitch","yaw")):
                    R = _euler_to_R(rot["roll"],rot["pitch"],rot["yaw"])
                elif "matrix" in rot:
                    R = np.array(rot["matrix"], dtype=float).reshape(3,3)


            elif isinstance(rot, (list, tuple, np.ndarray)):
                # Flatten any nested lists/tuples (e.g., [[r],[p],[y]] or 3x3) to 1-D numeric
                arr = np.array(rot, dtype=float).reshape(-1)
                if arr.size == 3:
                    # Heuristic: if any angle has magnitude > 2π, treat as degrees
                    deg = (np.abs(arr).max() > 2*math.pi + 1e-6)
                    R = _euler_to_R(arr[0], arr[1], arr[2], degrees=deg)
                elif arr.size == 4:
                    R = _quat_to_R(arr)
                elif arr.size == 9:
                    R = arr.reshape(3, 3)

            t = None
            if isinstance(trans, dict) and all(k in trans for k in ("x","y","z")):
                t = np.array([trans["x"],trans["y"],trans["z"]], dtype=float)
            elif isinstance(trans, (list,tuple)) and len(trans) == 3:
                t = np.array([trans[0],trans[1],trans[2]], dtype=float)
            if R is not None and t is not None:
                M = np.eye(4, dtype=float); M[:3,:3] = R; M[:3,3] = t
                return M
            return None

        rot = J.get("rotation") or J.get("rot") or J.get("R")
        trans = J.get("translation") or J.get("t") or J.get("trans") or J.get("T")
        M = build_from_rt(rot, trans)

    if M is None:
        raise ValueError(f"Unrecognized extrinsic JSON: {T_json}")

    # auto invert if filename says camera_to_lidar
    lower = T_json.as_posix().lower()
    if "camera_to_lidar" in lower or "cam_to_lidar" in lower or "camera2lidar" in lower:
        M = np.linalg.inv(M)

    return M


# -------------------- Box conversions --------------------
def lidar_box_cam_fields(center_l: Iterable[float], dims_lwh: Iterable[float],
                         yaw_l: float, T_cam_l: np.ndarray) -> Tuple[List[float], List[float], float]:
    """
    Convert LiDAR box (center_l [x,y,z], size [l,w,h], yaw about +Z_l)
    to KITTI camera fields:
      dims_cam [h, w, l], loc_cam [x, y, z] bottom-centered, rotation_y (about +Y_cam).
    """
    center_l = np.array(center_l, dtype=float).reshape(3)
    l, w, h = map(float, dims_lwh)
    R = T_cam_l[:3,:3]; t = T_cam_l[:3,3]

    # center to camera
    Xc = R @ center_l + t

    # heading: LiDAR +Z yaw -> direction vector in LiDAR, then rotate to camera
    # v_l = np.array([math.cos(yaw_l), math.sin(yaw_l), 0.0], dtype=float)
    # v_c = R @ v_l
    # ry = math.atan2(v_c[0], v_c[2])  # KITTI yaw is about +Y in camera

    ry=yaw_l

    # bottom-center shift in camera coords (y down)
    loc_cam = Xc.copy()
    loc_cam[1] += h / 2.0

    dims_cam = [h, w, l]  # KITTI order
    return dims_cam, loc_cam.tolist(), ry

def kitti_corners_3d_in_cam(dims_cam: Iterable[float], loc_cam: Iterable[float], ry: float) -> np.ndarray:
    """
    Return (8,3) corners in camera coords for KITTI: dims [h,w,l], loc bottom-centered, ry about +Y.
    """
    h, w, l = map(float, dims_cam)
    x, y, z = map(float, loc_cam)

    x_c = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2 ]
    y_c = [   0,    0,    0,    0,  -h,   -h,   -h,   -h ]
    z_c = [ w/2, -w/2, -w/2,  w/2, w/2, -w/2, -w/2,  w/2 ]
    corners = np.vstack([x_c, y_c, z_c])  # (3,8)

    c, s = math.cos(ry), math.sin(ry)
    R = np.array([[ c, 0, s],
                  [ 0, 1, 0],
                  [-s, 0, c]], dtype=float)
    corners = (R @ corners).T + np.array([x, y, z], dtype=float)
    return corners  # (8,3)

def project_to_image(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    P: 3x4 projection matrix (we use [K|0]); X: (N,3) points in camera coords.
    Returns (N,2) image points. Points with z<=0 will be nonsense (caller can filter).
    """
    X_h = np.hstack([X, np.ones((X.shape[0], 1), dtype=float)])
    x = (P @ X_h.T).T
    z = x[:, 2:3]
    z[z == 0] = 1e-6
    uv = x[:, :2] / z
    return uv


# ------------------------- Writers -------------------------
def write_kitti_calib(out_txt: Path, K: np.ndarray, T_cam_l: np.ndarray) -> None:
    ensure_dir(out_txt.parent)
    P2 = np.zeros((3,4), dtype=float); P2[:3,:3] = K
    R0 = np.eye(3, dtype=float)
    Tr = T_cam_l[:3,:]  # 3x4

    def row(a: np.ndarray) -> str:
        return " ".join(f"{v:.12e}" for v in a.reshape(-1))

    with open(out_txt, "w") as f:
        for i in range(4):
            Pi = P2 if i == 2 else np.zeros((3,4))
            f.write(f"P{i}: {row(Pi)}\n")
        f.write(f"R0_rect: {row(R0)}\n")
        f.write(f"Tr_velo_to_cam: {row(Tr)}\n")

def write_kitti_label2(out_txt: Path, objs: List[Dict[str,Any]], K: np.ndarray,
                       T_cam_l: np.ndarray, im_w: int, im_h: int) -> None:
    ensure_dir(out_txt.parent)
    P2 = np.zeros((3,4), dtype=float); P2[:3,:3] = K

    def clamp_box(l: float,t: float,r: float,b: float) -> Tuple[float,float,float,float]:
        l = max(0.0, min(l, im_w-1.0))
        r = max(0.0, min(r, im_w-1.0))
        t = max(0.0, min(t, im_h-1.0))
        b = max(0.0, min(b, im_h-1.0))
        return l,t,r,b

    lines: List[str] = []
    for o in objs:
        typ = o.get("type","Car")

        # 3D fields from LiDAR box (native JSON)
        dims = o.get("3d_dimensions") or o.get("dimensions") or o.get("size")
        loc  = o.get("3d_location")  or o.get("location")   or o.get("center")
        yaw  = o.get("rotation") 
        if yaw is None:
            yaw = 0.0
        yaw = float(yaw)
        if abs(yaw) > 2*math.pi:
            yaw = math.radians(yaw)

        # ---- YOUR ORIGINAL MAPPING (unchanged) ----
        # Pass LiDAR size as [l, w, h] with no remapping
        if isinstance(dims, dict):
            h_o = float(dims["h"]); w_o = float(dims["w"]); l_o = float(dims["l"])
            # size_lwh = [h_o, l_o, w_o]   # L' = H,  W' = L,  H' = W
            size_lwh = [l_o, w_o, h_o] 
        else:
            h_o, w_o, l_o = [float(v) for v in dims]  # [h,w,l]
            # size_lwh = [h_o, l_o, w_o]
            size_lwh = [l_o, w_o, h_o] 
        # -------------------------------------------

        if isinstance(loc, dict):
            center_l = [float(loc["x"]), float(loc["y"]), float(loc["z"])]
        else:
            center_l = [float(loc[0]), float(loc[1]), float(loc[2])]

        # YAW COMPENSATION so that heading matches the remapped local axes
        yaw_corr = yaw+YAW_OFFSET_RAD

        dims_cam, loc_cam, ry = lidar_box_cam_fields(center_l, size_lwh, yaw_corr, T_cam_l)
        alpha = ry - math.atan2(loc_cam[0], loc_cam[2])

        # 2D bbox: prefer provided, else project corners
        bb = o.get("2d_box", {}) or o.get("bbox", {})
        have_2d = all(k in bb for k in ("xmin","ymin","xmax","ymax"))
        if have_2d:
            l = float(bb["xmin"]); t = float(bb["ymin"])
            r = float(bb["xmax"]); b = float(bb["ymax"])
            l,t,r,b = clamp_box(l,t,r,b)
        else:
            # project 3D corners and bound
            corners_cam = kitti_corners_3d_in_cam(dims_cam, loc_cam, ry)
            uv = project_to_image(P2, corners_cam)
            z = corners_cam[:,2]
            mask = z > 0.1
            if mask.any():
                uvm = uv[mask]
                l, t = uvm[:,0].min(), uvm[:,1].min()
                r, b = uvm[:,0].max(), uvm[:,1].max()
                l,t,r,b = clamp_box(l,t,r,b)
            else:
                cx = min(max(loc_cam[0]/max(loc_cam[2],1e-3)*K[0,0]+K[0,2],0), im_w-1)
                cy = min(max(loc_cam[1]/max(loc_cam[2],1e-3)*K[1,1]+K[1,2],0), im_h-1)
                l = max(0, cx-1); r = min(im_w-1, cx+1)
                t = max(0, cy-1); b = min(im_h-1, cy+1)

        truncated = 0.0
        occluded  = 0
        score     = -1.0

        line = f"{typ} {truncated:.2f} {occluded:d} {alpha:.6f} " \
               f"{l:.2f} {t:.2f} {r:.2f} {b:.2f} " \
               f"{dims_cam[0]:.6f} {dims_cam[1]:.6f} {dims_cam[2]:.6f} " \
               f"{loc_cam[0]:.6f} {loc_cam[1]:.6f} {loc_cam[2]:.6f} " \
               f"{ry:.6f} {score:.6f}"
        lines.append(line)

    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


# ---------------------- Side conversion ----------------------
def convert_side(side: str, src_root: Path, out_root: Path) -> None:
    print(f"[INFO] Converting side: {side}")
    in_base  = src_root / side
    img_dir  = in_base / "image"
    lid2d    = in_base / "label" / "camera"
    lid3d    = in_base / "label" / "lidar"
    calib    = in_base / "calib"

    # calibration dirs
    if side == "vehicle-side":
        T_dir = calib / "lidar_to_camera"
    else:
        T_dir = calib / "virtuallidar_to_camera"
    K_dir = calib / "camera_intrinsic"

    ids = get_ids(img_dir)
    print(f"[INFO] Found {len(ids)} frames")

    # out dirs
    train_root = out_root / side / "training"
    image_out  = train_root / "image_2"
    vel_out    = train_root / "velodyne"
    calib_out  = train_root / "calib"
    label_out  = train_root / "label_2"
    ensure_dir(image_out); ensure_dir(vel_out); ensure_dir(calib_out); ensure_dir(label_out)

    train_ids: List[str] = []
    for sid in ids:
        src_img = (img_dir / f"{sid}.png")
        if not src_img.exists():
            src_img = (img_dir / f"{sid}.jpg")
        src_bin = (in_base / "velodyne" / f"{sid}.bin")
        j2d     = (lid2d / f"{sid}.json")
        j3d     = (lid3d / f"{sid}.json")
        Kin     = (K_dir / f"{sid}.json")
        Tin     = (T_dir / f"{sid}.json")

        if not (src_img.exists() and src_bin.exists() and Kin.exists() and Tin.exists()):
            print(f"[WARN] Missing one or more files for {sid}; skipping.")
            continue

        # copy image & lidar
        im = cv2.imread(str(src_img), cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"[WARN] Failed to read image {src_img}; skipping.")
            continue
        im_h, im_w = int(im.shape[0]), int(im.shape[1])
        cv2.imwrite(str(image_out / f"{sid}.png"), im)
        shutil.copy2(src_bin, vel_out / f"{sid}.bin")

        # load intrinsics/extrinsics
        K = parse_intrinsic(Kin)
        T_cam_l = parse_extrinsic(Tin)

        # Normalize to KITTI camera basis: x right, y down, z forward
        # Mapping we want: LiDAR X->Cam Z, LiDAR Y->Cam -X, LiDAR Z->Cam -Y
        S = np.array([[ 0, -1,  0, 0],
                    [ 0,  0, -1, 0],
                    [ 1,  0,  0, 0],
                    [ 0,  0,  0, 1]], dtype=float)
        T_cam_l = S @ T_cam_l


        # write calib
        write_kitti_calib(calib_out / f"{sid}.txt", K, T_cam_l)

        # load 2D and 3D labels
        recs2 = read_json(j2d) if j2d.exists() else []
        recs3 = read_json(j3d) if j3d.exists() else []
        recs2 = recs2 if isinstance(recs2, list) else recs2.get("labels", [recs2])
        recs3 = recs3 if isinstance(recs3, list) else recs3.get("labels", [recs3])

        # normalize 2D list
        twod = []
        for o in (recs2 or []):
            if "2d_box" in o and isinstance(o["2d_box"], dict):
                twod.append({"2d_box": o["2d_box"], "type": o.get("type","Car")})
            elif "bbox" in o and isinstance(o["bbox"], dict):
                bb = o["bbox"]
                twod.append({"2d_box": {
                    "xmin": bb.get("xmin") or bb.get("x1") or bb.get("left"),
                    "ymin": bb.get("ymin") or bb.get("y1") or bb.get("top"),
                    "xmax": bb.get("xmax") or bb.get("x2") or bb.get("right"),
                    "ymax": bb.get("ymax") or bb.get("y2") or bb.get("bottom"),
                }, "type": o.get("type","Car")})
            else:
                b = o.get("bbox")
                if isinstance(b, (list,tuple)) and len(b) == 4:
                    x,y,w,h = b
                    twod.append({"2d_box": {"xmin":x,"ymin":y,"xmax":x+w,"ymax":y+h},
                                 "type": o.get("type","Car")})

        # normalize 3D list
        threed = []
        for o in (recs3 or []):
            threed.append(o)

        # merge: 3D list drives count; attach 2D if available by index
        merged = []
        for i, g in enumerate(threed):
            rec = {"type": g.get("type","Car")}
            if i < len(twod):
                rec["2d_box"] = twod[i]["2d_box"]
                rec["type"]   = twod[i].get("type", rec["type"])
            for k in ("3d_location","3d_dimensions","rotation","yaw","rotation_y","center","size","dimensions"):
                if k in g:
                    rec[k] = g[k]
            merged.append(rec)

        # write label_2
        write_kitti_label2(label_out / f"{sid}.txt", merged, K, T_cam_l, im_w, im_h)

        train_ids.append(sid)

    # split files (optional)
    is_dir = out_root / side / "training" / "ImageSets"
    ensure_dir(is_dir)
    if MAKE_VAL_SPLIT:
        n = len(train_ids)
        k = int(round(n*0.2))
        val_ids = set(train_ids[::max(1,n//max(1,k))])
        train_ids_sorted = [s for s in train_ids if s not in val_ids]
        val_ids_sorted   = sorted(val_ids)
    else:
        train_ids_sorted = sorted(train_ids)
        val_ids_sorted   = []

    with open(is_dir / "train.txt", "w") as f:
        for s in train_ids_sorted:
            f.write(s + "\n")
    with open(is_dir / "val.txt", "w") as f:
        for s in val_ids_sorted:
            f.write(s + "\n")
    with open(is_dir / "test.txt", "w") as f:
        pass  # empty

# --------------------------- main ---------------------------
def main():
    global YAW_OFFSET_RAD
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=str(SRC_ROOT),
                        help="Source dataset root.")
    parser.add_argument("--out", type=str, default=str(OUT_ROOT),
                        help="Output KITTI root.")
    parser.add_argument("--sides", type=str, nargs="+", default=SIDES,
                        help="Which sides to convert.")
    parser.add_argument("--val_split", action="store_true", default=MAKE_VAL_SPLIT,
                        help="Create a 20% val split in training/ImageSets.")
    parser.add_argument("--yaw_offset_deg", type=float, default=math.degrees(YAW_OFFSET_RAD),
                        help="Yaw offset (degrees) applied BEFORE LiDAR->camera conversion "
                             "to match axis mapping. Use +/-90 depending on your mapping.")
    args = parser.parse_args()

    src = Path(args.src); out = Path(args.out)
    ensure_dir(out)

    # update yaw offset if provided
    YAW_OFFSET_RAD = math.radians(args.yaw_offset_deg)

    for side in args.sides:
        convert_side(side, src, out)
    print(f"[OK] Wrote KITTI-style dataset to: {out}")

if __name__ == "__main__":
    main()
