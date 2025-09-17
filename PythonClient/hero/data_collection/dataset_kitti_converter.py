#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert your dair_v2x_synth to KITTI-style (for each side separately).

Assumptions
- Images: <SRC>/<side>/image/<id>.png|jpg
- LiDAR:  <SRC>/<side>/velodyne/<id>.bin  (x y z intensity as float32 per point)
- 2D GT:  <SRC>/<side>/label/camera/<id>.json with {"2d_box":{"xmin","ymin","xmax","ymax"}, "type":...}
- 3D GT:  <SRC>/<side>/label/lidar/<id>.json with
          {"3d_location":{"x","y","z"}, "3d_dimensions":{"h","w","l"}, "rotation": yaw_rad, "type":...}
- Calib (vehicle-side):        calib/{camera_intrinsic, lidar_to_camera}/<id>.json
- Calib (infrastructure-side): calib/{camera_intrinsic, virtuallidar_to_camera}/<id>.json

Notes
- We treat *_to_camera JSON as LiDAR->Camera (as named). If a path contains "camera_to_lidar",
  we will invert it automatically.
- We set R0_rect = I and P2 = [K|0], which is fine for MMDet3D browsing & most tooling.
"""

import json, math, os, shutil
from pathlib import Path
import numpy as np
import cv2

# -------- EDIT THESE --------
SRC_ROOT = Path("/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets/dair_v2x_synth/cooperative-vehicle-infrastructure/")
OUT_ROOT = Path("/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets/dair_v2x_synth_kitti")
SIDES = ["vehicle-side", "infrastructure-side"]  # or pick one
MAKE_VAL_SPLIT = False   # hold out 20% to val if True
# ---------------------------

# ---------- IO helpers ----------
def read_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def get_ids(img_dir: Path):
    ids = [p.stem for p in sorted(img_dir.glob("*.png"))] + [p.stem for p in sorted(img_dir.glob("*.jpg"))]
    return sorted(set(ids))

# ---------- Intrinsic parsing ----------
def load_intrinsic(K_json: Path):
    J = read_json(K_json)

    def try_matrix(obj):
        v = np.array(obj, dtype=float)
        if v.size == 9:  # flat
            return v.reshape(3,3)
        if v.shape == (3,3):
            return v
        return None

    # direct matrix under common keys
    for key in ("cam_K", "K", "intrinsic", "camera_matrix", "matrix"):
        if key in J:
            if isinstance(J[key], dict):
                for kk in ("matrix", "data", "values"):
                    if kk in J[key]:
                        K = try_matrix(J[key][kk])
                        if K is not None:
                            return K
            else:
                K = try_matrix(J[key])
                if K is not None:
                    return K

    # fx,fy,cx,cy at top-level
    fx, fy, cx, cy = (J.get("fx"), J.get("fy"), J.get("cx"), J.get("cy"))
    if None not in (fx, fy, cx, cy):
        fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
        return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)

    # nested dict
    if "intrinsic" in J and isinstance(J["intrinsic"], dict):
        ii = J["intrinsic"]
        fx, fy, cx, cy = (ii.get("fx"), ii.get("fy"), ii.get("cx"), ii.get("cy"))
        if None not in (fx, fy, cx, cy):
            fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
            return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)

    raise ValueError(f"Unrecognized intrinsic JSON: {K_json}")

# ---------- Extrinsic parsing ----------
def _quat_to_R(q):
    q = np.array([float(v) for v in q], dtype=float).reshape(-1)
    if q.size != 4:
        return None
    # guess ordering
    if abs(q[0]) >= 0.5:  # likely w first
        w, x, y, z = q
    else:                 # likely w last
        x, y, z, w = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-9: return None
    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=float)
    return R

def _euler_to_R(roll, pitch, yaw, degrees=False, order="xyz"):
    r, p, y = float(roll), float(pitch), float(yaw)
    if degrees:
        r = math.radians(r); p = math.radians(p); y = math.radians(y)
    Rx = np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]], dtype=float)
    Ry = np.array([[math.cos(p),0,math.sin(p)],[0,1,0],[-math.sin(p),0,math.cos(p)]], dtype=float)
    Rz = np.array([[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]], dtype=float)
    maps = {'x': Rx, 'y': Ry, 'z': Rz}
    R = np.eye(3, dtype=float)
    for ax in order.lower():
        R = R @ maps[ax]
    return R

def load_extrinsic_to_cam(T_json: Path):
    """
    Return 4x4 LiDAR->Camera transform (T_cam_lidar).
    Handles:
      - {'matrix':[16] or [[4],[4]]}, {'transform': {...}}
      - {'R':[9], 't':[3]} or nested {'rotation':..., 'translation':...}
      - Quaternion + translation
      - Euler angles + translation
      - Or your exact: {'rotation': [[3x3]], 'translation':[3]}  ✔️
    If the path name suggests camera->lidar, we invert the result.
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
        # rot: list (9,4,3) or dict; trans: list or dict
        R = None
        if isinstance(rot, (list, tuple)):
            if len(rot) == 9:
                R = np.array(rot, dtype=float).reshape(3,3)
            elif len(rot) == 4:
                R = _quat_to_R(rot)
            elif len(rot) == 3:
                R = _euler_to_R(rot[0], rot[1], rot[2], degrees=(max(map(abs,rot))>2*math.pi))
        elif isinstance(rot, dict):
            if all(k in rot for k in ("w","x","y","z")):
                R = _quat_to_R([rot["w"], rot["x"], rot["y"], rot["z"]])
            elif all(k in rot for k in ("x","y","z")) and "w" in rot:
                R = _quat_to_R([rot["x"], rot["y"], rot["z"], rot["w"]])
            elif all(k in rot for k in ("roll","pitch","yaw")):
                R = _euler_to_R(rot["roll"], rot["pitch"], rot["yaw"],
                                degrees=(max(abs(rot["roll"]),abs(rot["pitch"]),abs(rot["yaw"]))>2*math.pi))
            elif "matrix" in rot:
                R = np.array(rot["matrix"], dtype=float).reshape(3,3)
        t = None
        if isinstance(trans, dict) and all(k in trans for k in ("x","y","z")):
            t = np.array([trans["x"], trans["y"], trans["z"]], dtype=float)
        elif isinstance(trans, (list,tuple)) and len(trans)==3:
            t = np.array([trans[0], trans[1], trans[2]], dtype=float)
        if R is not None and t is not None:
            M = np.eye(4, dtype=float); M[:3,:3]=R; M[:3,3]=t
            return M
        return None

    # direct matrices
    for key in ("matrix","transform","Tr","T","extrinsic","pose","Mat","M"):
        if key in J:
            v = J[key]
            if isinstance(v, dict):
                for kk in ("matrix", "data", "values"):
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
    else:
        M = None

    # separate R / t at top-level or your exact "rotation"+"translation"
    if M is None:
        if "R" in J and "t" in J:
            R = np.array(J["R"], dtype=float).reshape(3,3)
            t = np.array(J["t"], dtype=float).reshape(3)
            M = np.eye(4, dtype=float); M[:3,:3]=R; M[:3,3]=t
        elif "rotation" in J and "translation" in J:
            rot = J["rotation"]; trans = J["translation"]
            # rot can be 3x3 matrix or list; trans is [x,y,z]
            if isinstance(rot, (list, tuple)) and np.array(rot).size == 9:
                R = np.array(rot, dtype=float).reshape(3,3)
                t = np.array(trans, dtype=float).reshape(3)
                M = np.eye(4, dtype=float); M[:3,:3]=R; M[:3,3]=t
            else:
                M = build_from_rt(rot, trans)

    if M is None:
        raise ValueError(f"Unrecognized extrinsic JSON: {T_json}")

    # auto-invert for camera->lidar filenames
    lower = T_json.as_posix().lower()
    if "camera_to_lidar" in lower or "cam_to_lidar" in lower or "camera2lidar" in lower:
        M = np.linalg.inv(M)

    return M

# ---------- Box conversions ----------
def lidar_box_cam_fields(center_l, dims_lwh, yaw_l, T_cam_l):
    """
    Convert LiDAR box (center_l [x,y,z], size [l,w,h], yaw around +Z_l) to KITTI camera fields:
      dims_cam[h,w,l], loc_cam[x,y,z] (bottom-centered), rotation_y (around +Y_cam).
    """
    center_l = np.array(center_l, dtype=float)
    l,w,h = map(float, dims_lwh)
    R = T_cam_l[:3,:3]; t = T_cam_l[:3,3]

    # center to cam
    Xc = R @ center_l + t  # (3,)

    # heading: LiDAR +Z yaw -> direction vector in LiDAR, then rotate to camera
    v_l = np.array([math.cos(yaw_l), math.sin(yaw_l), 0.0])
    v_c = R @ v_l
    ry = math.atan2(v_c[0], v_c[2])

    # shift to bottom-center in camera (camera y is down)
    loc_cam = Xc.copy()
    loc_cam[1] += h/2.0

    dims_cam = [h, w, l]  # KITTI order
    return dims_cam, loc_cam.tolist(), ry

def kitti_corners_3d_in_cam(dims_cam, loc_cam, ry):
    """
    corners (8,3) in camera coords for KITTI [h,w,l], bottom-centered loc_cam, ry about +Y.
    """
    h, w, l = dims_cam
    x, y, z = loc_cam
    # define in object frame (camera coords): y is down
    x_c = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2 ]
    y_c = [   0,    0,    0,    0, -h,  -h,   -h,   -h  ]
    z_c = [ w/2, -w/2, -w/2,  w/2, w/2, -w/2, -w/2,  w/2 ]
    corners = np.vstack([x_c, y_c, z_c])  # (3,8)

    c = math.cos(ry); s = math.sin(ry)
    R = np.array([[ c, 0, s],
                  [ 0, 1, 0],
                  [-s, 0, c]], dtype=float)
    corners = (R @ corners).T + np.array([x,y,z], dtype=float)
    return corners  # (8,3)

def project_to_image(P, X):
    """
    P: 3x4 camera matrix (here [K|0]); X: (N,3) in camera coords.
    Returns (N,2) image points; ignores points with z<=0 (they'll produce junk).
    """
    N = X.shape[0]
    Xh = np.hstack([X, np.ones((N,1), dtype=float)])  # (N,4)
    x = (P @ Xh.T).T  # (N,3)
    z = x[:,2:3]
    eps = 1e-6
    z = np.maximum(z, eps)
    uv = x[:, :2] / z
    return uv

# ---------- Writers ----------
def write_kitti_calib(out_txt: Path, K: np.ndarray, T_cam_l: np.ndarray):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    P2 = np.zeros((3,4), dtype=float)
    P2[:3,:3] = K
    R0 = np.eye(3, dtype=float)
    Tr = T_cam_l[:3,:]  # 3x4

    def row(a): return " ".join(f"{v:.12e}" for v in a.reshape(-1))
    with open(out_txt, "w") as f:
        for i in range(4):  # P0..P3; only P2 matters
            Pi = P2 if i == 2 else np.zeros((3,4))
            f.write(f"P{i}: {row(Pi)}\n")
        f.write(f"R0_rect: {row(R0)}\n")
        f.write(f"Tr_velo_to_cam: {row(Tr)}\n")
        # add a dummy IMU->velo so MMDet3D's parser has lines[6]
        f.write("Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n")

def write_kitti_label2(out_txt: Path, objs, K, T_cam_l, im_w, im_h):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    P2 = np.zeros((3,4), dtype=float); P2[:3,:3] = K

    def clamp_box(l,t,r,b):
        l = max(0.0, min(l, im_w-1.0))
        r = max(0.0, min(r, im_w-1.0))
        t = max(0.0, min(t, im_h-1.0))
        b = max(0.0, min(b, im_h-1.0))
        return l,t,r,b

    lines = []
    for o in objs:
        typ = o.get("type","Car")

        # 3D fields from LiDAR box
        dims = o.get("3d_dimensions") or o.get("dimensions") or o.get("size")
        loc  = o.get("3d_location")  or o.get("location")   or o.get("center")
        yaw  = o.get("rotation")     or o.get("yaw")        or o.get("rotation_y") or o.get("rz")
        if yaw is None: yaw = 0.0
        yaw = float(yaw)
        if abs(yaw) > 2*math.pi: yaw = math.radians(yaw)

        # --- remap dimensions: L' <- H_orig,  H' <- W_orig,  W' <- L_orig ---
        if isinstance(dims, dict):
            h_o = float(dims["h"])
            w_o = float(dims["w"])
            l_o = float(dims["l"])
            # size_lwh is expected as [length, width, height] in the LiDAR frame
            size_lwh = [h_o, l_o, w_o]   # L' = H,  W' = L,  H' = W
        else:
            # If an array is ever encountered, assume order [h, w, l] and apply the same remap.
            h_o, w_o, l_o = [float(v) for v in dims]
            size_lwh = [h_o, l_o, w_o]   # L' = H,  W' = L,  H' = W

        if isinstance(loc, dict):
            center_l = [float(loc["x"]), float(loc["y"]), float(loc["z"])]
        else:
            center_l = [float(loc[0]), float(loc[1]), float(loc[2])]

        dims_cam, loc_cam, ry = lidar_box_cam_fields(center_l, size_lwh, yaw, T_cam_l)
        alpha = ry - math.atan2(loc_cam[0], loc_cam[2])

        # 2D bbox: prefer provided, else project corners
        bb = o.get("2d_box", {}) or o.get("bbox", {})
        have_2d = all(k in bb for k in ("xmin","ymin","xmax","ymax"))
        if have_2d:
            l = float(bb["xmin"]); t = float(bb["ymin"])
            r = float(bb["xmax"]); b = float(bb["ymax"])
            l,t,r,b = clamp_box(l,t,r,b)
        else:
            # project 3D box corners
            corners_cam = kitti_corners_3d_in_cam(dims_cam, loc_cam, ry)
            uv = project_to_image(P2, corners_cam)
            # keep only points with positive depth in cam coords
            z = corners_cam[:,2]
            mask = z > 0.1
            if mask.any():
                uvm = uv[mask]
                l, t = uvm[:,0].min(), uvm[:,1].min()
                r, b = uvm[:,0].max(), uvm[:,1].max()
                l,t,r,b = clamp_box(l,t,r,b)
            else:
                # degenerate: set tiny box at center
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

# ---------- Side conversion ----------
def convert_side(side: str):
    print(f"[INFO] Converting side: {side}")
    in_base  = SRC_ROOT / side
    img_dir  = in_base / "image"
    lid2d    = in_base / "label" / "camera"
    lid3d    = in_base / "label" / "lidar"
    calib    = in_base / "calib"

    if side == "vehicle-side":
        T_dir = calib / "lidar_to_camera"
    else:
        T_dir = calib / "virtuallidar_to_camera"
    K_dir = calib / "camera_intrinsic"

    out_base = OUT_ROOT / side / "training"
    (out_base / "image_2").mkdir(parents=True, exist_ok=True)
    (out_base / "velodyne").mkdir(parents=True, exist_ok=True)
    (out_base / "label_2").mkdir(parents=True, exist_ok=True)
    (out_base / "calib").mkdir(parents=True, exist_ok=True)

    ids = get_ids(img_dir)
    imagesets = []
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
            print(f"[WARN] missing one or more files for {sid}; skipping")
            continue

        # copy image + lidar
        shutil.copy2(src_img, out_base / "image_2" / f"{sid}.png")
        shutil.copy2(src_bin, out_base / "velodyne" / f"{sid}.bin")

        # calib
        K = load_intrinsic(Kin)
        T = load_extrinsic_to_cam(Tin)
        write_kitti_calib(out_base / "calib" / f"{sid}.txt", K, T)

        # image size
        im = cv2.imread(str(src_img), cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"[WARN] could not read image for size: {src_img}; skipping")
            continue
        h_img, w_img = im.shape[:2]

        # labels
        if j2d.exists():
            data2 = read_json(j2d)
            recs2 = data2 if isinstance(data2, list) else data2.get("annotations", [data2])
            # normalize 2D list
            twod = []
            for o in (recs2 or []):
                if "2d_box" in o and isinstance(o["2d_box"], dict):
                    twod.append({"2d_box": o["2d_box"], "type": o.get("type","Car")})
                elif "bbox" in o and isinstance(o["bbox"], dict):
                    twod.append({"2d_box": {
                        "xmin": o["bbox"].get("x1") or o["bbox"].get("left"),
                        "ymin": o["bbox"].get("y1") or o["bbox"].get("top"),
                        "xmax": o["bbox"].get("x2") or o["bbox"].get("right"),
                        "ymax": o["bbox"].get("y2") or o["bbox"].get("bottom"),
                    }, "type": o.get("type","Car")})
                else:
                    # COCO-like [x,y,w,h]
                    b = o.get("bbox")
                    if isinstance(b, (list,tuple)) and len(b)==4:
                        x,y,w,h = b
                        twod.append({"2d_box": {"xmin":x,"ymin":y,"xmax":x+w,"ymax":y+h},
                                     "type": o.get("type","Car")})
        else:
            twod = []

        if j3d.exists():
            data3 = read_json(j3d)
            recs3 = data3 if isinstance(data3, list) else data3.get("labels", [data3])

            merged = []
            for i, g in enumerate(recs3):
                o = {"type": g.get("type","Car")}
                if i < len(twod):
                    o["2d_box"] = twod[i]["2d_box"]; o["type"] = twod[i]["type"]
                # pass through 3D fields
                for k in ("3d_location","3d_dimensions","rotation","yaw","rotation_y","center","size","dimensions"):
                    if k in g: o[k] = g[k]
                merged.append(o)

            write_kitti_label2(out_base / "label_2" / f"{sid}.txt", merged, K, T, w_img, h_img)
        else:
            # no 3D => write empty file
            open(out_base / "label_2" / f"{sid}.txt", "w").close()

        imagesets.append(sid)

    # ImageSets
    is_dir = OUT_ROOT / side / "ImageSets"
    is_dir.mkdir(parents=True, exist_ok=True)
    if MAKE_VAL_SPLIT and len(imagesets) >= 5:
        nval = max(1, int(0.2 * len(imagesets)))
        train_ids = imagesets[:-nval]
        val_ids   = imagesets[-nval:]
    else:
        train_ids = imagesets
        val_ids   = imagesets

    for name, lst in (("train.txt", train_ids), ("val.txt", val_ids), ("test.txt", [])):
        with open(is_dir / name, "w") as f:
            for s in lst:
                f.write(s + "\n")

# ---------- main ----------
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for side in SIDES:
        convert_side(side)
    print(f"[OK] Wrote KITTI-style dataset to: {OUT_ROOT}")

if __name__ == "__main__":
    main()
