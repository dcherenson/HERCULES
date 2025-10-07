#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_kitti_converter.py

Convert a DAIR-V2X-style dataset to KITTI and create splits in
<OUT>/<side>/ImageSets (sibling of 'training').

Defaults mimic DAIR-V2X-C usage with evaluation on val:
  - split = 85/15/0  -> empty test.txt, no testing/* needed

If you want an actual test set laid out under <side>/testing/*,
run with a non-zero test ratio and --testing-mode copy|symlink.
"""

from __future__ import annotations
import argparse, json, math, os, random, shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import cv2

# ===================== Defaults (override by CLI) =====================
SRC_ROOT = Path("/media/sgarimella34/hercules-collect/collaborative-perception-BEVP/dair_v2x_synth_TEST1/cooperative-vehicle-infrastructure/")
OUT_ROOT = Path("/media/sgarimella34/hercules-collect/collaborative-perception-BEVP/dair_v2x_synth_TEST1_kitti")
SIDES    = ["vehicle-side", "infrastructure-side"]
YAW_OFFSET_RAD = math.pi / 2.0  # change via --yaw_offset_deg
# =====================================================================

def read_json(p: Path) -> Any:
    with open(p, "r") as f:
        return json.load(f)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def get_ids(img_dir: Path) -> List[str]:
    ids = [p.stem for p in sorted(img_dir.glob("*.png"))] + \
          [p.stem for p in sorted(img_dir.glob("*.jpg"))]
    return sorted(set(ids))

def _try_matrix(obj: Any) -> np.ndarray | None:
    try:
        arr = np.array(obj, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size == 9:
        return arr.reshape(3, 3)
    return None

def parse_intrinsic(K_json: Path) -> np.ndarray:
    J = read_json(K_json)
    for k in ("K", "camera_matrix", "intrinsic", "cam_K", "matrix"):
        if k in J:
            node = J[k]
            if isinstance(node, dict):
                for kk in ("K","matrix","data","values"):
                    M = _try_matrix(node.get(kk))
                    if M is not None: return M
            else:
                M = _try_matrix(node)
                if M is not None: return M

    def extract_fx_fy_cx_cy(d: Dict[str, Any]):
        fx, fy, cx, cy = d.get("fx"), d.get("fy"), d.get("cx"), d.get("cy")
        if None not in (fx, fy, cx, cy):
            return float(fx), float(fy), float(cx), float(cy)

    tpl = extract_fx_fy_cx_cy(J)
    if tpl is None and isinstance(J.get("intrinsic"), dict):
        tpl = extract_fx_fy_cx_cy(J["intrinsic"])
    if tpl is not None:
        fx, fy, cx, cy = tpl
        return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)

    raise ValueError(f"Unrecognized intrinsic JSON format: {K_json}")

def _quat_to_R(q: Iterable[float]) -> np.ndarray:
    q = np.asarray(list(q), dtype=float).reshape(-1)
    if q.size != 4: raise ValueError("Quaternion must have 4 elements.")
    if abs(q[0]) < 0.5 and abs(q[3]) > 0.5:
        q = np.array([q[3], q[0], q[1], q[2]], dtype=float)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=float)

def _euler_to_R(roll: float, pitch: float, yaw: float, degrees: bool=False) -> np.ndarray:
    if degrees:
        roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ]
    ], dtype=float)

def parse_extrinsic(T_json: Path) -> np.ndarray:
    J = read_json(T_json)
    def try_matrix_like(obj: Any) -> np.ndarray | None:
        try: arr = np.array(obj, dtype=float).reshape(-1)
        except Exception: return None
        if arr.size == 16:
            M = np.eye(4); M[:] = arr.reshape(4,4); return M
        if arr.size == 12:
            M = np.eye(4); M[:3,:] = arr.reshape(3,4); return M
        return None

    M = None
    for k in ("matrix","T","transform","data","values"):
        if k in J:
            M = try_matrix_like(J[k]); 
            if M is not None: break
    if M is None:
        for k in ("matrix","T","transform"):
            node = J.get(k)
            if isinstance(node, dict):
                for kk in ("matrix","data","values"):
                    M = try_matrix_like(node.get(kk))
                    if M is not None: break
            if M is not None: break
    if M is None and isinstance(J.get("extrinsic"), dict):
        node = J["extrinsic"]
        for kk in ("matrix","data","values"):
            M = try_matrix_like(node.get(kk))
            if M is not None: break

    if M is None:
        rot = J.get("rotation") or J.get("rot") or J.get("R")
        trans = J.get("translation") or J.get("t") or J.get("trans") or J.get("T")
        def build_from_rt(rot, trans):
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
            elif isinstance(rot, (list,tuple,np.ndarray)):
                arr = np.array(rot, dtype=float).reshape(-1)
                if arr.size == 3:
                    deg = (np.abs(arr).max() > 2*math.pi + 1e-6)
                    R = _euler_to_R(arr[0],arr[1],arr[2], degrees=deg)
                elif arr.size == 4:
                    R = _quat_to_R(arr)
                elif arr.size == 9:
                    R = arr.reshape(3,3)
            t = None
            if isinstance(trans, dict) and all(k in trans for k in ("x","y","z")):
                t = np.array([trans["x"],trans["y"],trans["z"]], dtype=float)
            elif isinstance(trans, (list,tuple)) and len(trans) == 3:
                t = np.array([trans[0],trans[1],trans[2]], dtype=float)
            if R is not None and t is not None:
                M = np.eye(4); M[:3,:3] = R; M[:3,3] = t; return M
        M = build_from_rt(rot, trans)

    if M is None:
        raise ValueError(f"Unrecognized extrinsic JSON: {T_json}")

    lower = T_json.as_posix().lower()
    if any(s in lower for s in ("camera_to_lidar","cam_to_lidar","camera2lidar")):
        M = np.linalg.inv(M)
    return M

def lidar_box_cam_fields(center_l: Iterable[float], dims_lwh: Iterable[float],
                         yaw_l: float, T_cam_l: np.ndarray):
    center_l = np.array(center_l, dtype=float).reshape(3)
    l, w, h = map(float, dims_lwh)
    R = T_cam_l[:3,:3]; t = T_cam_l[:3,3]
    Xc = R @ center_l + t
    ry = yaw_l
    loc_cam = Xc.copy(); loc_cam[1] += h/2.0
    dims_cam = [h, w, l]
    return dims_cam, loc_cam.tolist(), ry

def kitti_corners_3d_in_cam(dims_cam: Iterable[float], loc_cam: Iterable[float], ry: float) -> np.ndarray:
    h, w, l = map(float, dims_cam); x, y, z = map(float, loc_cam)
    x_c = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2 ]
    y_c = [   0,    0,    0,    0,  -h,   -h,   -h,   -h ]
    z_c = [ w/2, -w/2, -w/2,  w/2, w/2, -w/2, -w/2,  w/2 ]
    corners = np.vstack([x_c, y_c, z_c])
    c, s = math.cos(ry), math.sin(ry)
    R = np.array([[ c, 0, s],[0,1,0],[-s,0,c]], dtype=float)
    return (R @ corners).T + np.array([x,y,z], dtype=float)

def project_to_image(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    X_h = np.hstack([X, np.ones((X.shape[0], 1), dtype=float)])
    x = (P @ X_h.T).T; z = x[:, 2:3]; z[z == 0] = 1e-6
    return x[:, :2] / z

def write_kitti_calib(out_txt: Path, K: np.ndarray, T_cam_l: np.ndarray) -> None:
    ensure_dir(out_txt.parent)
    P2 = np.zeros((3,4), dtype=float); P2[:3,:3] = K
    R0 = np.eye(3, dtype=float)
    Tr = T_cam_l[:3,:]
    def row(a: np.ndarray) -> str:
        return " ".join(f"{v:.12e}" for v in a.reshape(-1))
    with open(out_txt, "w") as f:
        for i in range(4):
            Pi = P2 if i == 2 else np.zeros((3,4))
            f.write(f"P{i}: {row(Pi)}\n")
        f.write(f"R0_rect: {row(R0)}\n")
        f.write(f"Tr_velo_to_cam: {row(Tr)}\n")
        # duplicate line to match your desired 7th line
        f.write(f"Tr_velo_to_cam: {row(Tr)}\n")

def write_kitti_label2(out_txt: Path, objs: List[Dict[str,Any]], K: np.ndarray,
                       T_cam_l: np.ndarray, im_w: int, im_h: int) -> None:
    ensure_dir(out_txt.parent)
    P2 = np.zeros((3,4), dtype=float); P2[:3,:3] = K
    def clamp_box(l,t,r,b):
        l = max(0.0, min(l, im_w-1.0)); r = max(0.0, min(r, im_w-1.0))
        t = max(0.0, min(t, im_h-1.0)); b = max(0.0, min(b, im_h-1.0))
        return l,t,r,b
    lines: List[str] = []
    for o in objs:
        typ = o.get("type","Car")
        dims = o.get("3d_dimensions") or o.get("dimensions") or o.get("size")
        loc  = o.get("3d_location")  or o.get("location")   or o.get("center")
        yaw  = float(o.get("rotation", 0.0))
        if abs(yaw) > 2*math.pi: yaw = math.radians(yaw)
        if isinstance(dims, dict):
            size_lwh = [float(dims["l"]), float(dims["w"]), float(dims["h"])]
        else:
            h_o, w_o, l_o = [float(v) for v in dims]; size_lwh = [l_o, w_o, h_o]
        if isinstance(loc, dict):
            center_l = [float(loc["x"]), float(loc["y"]), float(loc["z"])]
        else:
            center_l = [float(loc[0]), float(loc[1]), float(loc[2])]
        yaw_corr = yaw + YAW_OFFSET_RAD
        dims_cam, loc_cam, ry = lidar_box_cam_fields(center_l, size_lwh, yaw_corr, T_cam_l)
        alpha = ry - math.atan2(loc_cam[0], loc_cam[2])
        bb = o.get("2d_box", {}) or o.get("bbox", {}); have_2d = all(k in bb for k in ("xmin","ymin","xmax","ymax"))
        if have_2d:
            l,t,r,b = clamp_box(float(bb["xmin"]), float(bb["ymin"]), float(bb["xmax"]), float(bb["ymax"]))
        else:
            corners_cam = kitti_corners_3d_in_cam(dims_cam, loc_cam, ry)
            uv = project_to_image(P2, corners_cam); z = corners_cam[:,2]; mask = z > 0.1
            if mask.any():
                uvm = uv[mask]; l,t = uvm[:,0].min(), uvm[:,1].min(); r,b = uvm[:,0].max(), uvm[:,1].max()
                l,t,r,b = clamp_box(l,t,r,b)
            else:
                cx = min(max(loc_cam[0]/max(loc_cam[2],1e-3)*K[0,0]+K[0,2],0), im_w-1)
                cy = min(max(loc_cam[1]/max(loc_cam[2],1e-3)*K[1,1]+K[1,2],0), im_h-1)
                l = max(0, cx-1); r = min(im_w-1, cx+1); t = max(0, cy-1); b = min(im_h-1, cy+1)
        truncated = 0.0; occluded = 0; score = -1.0
        line = f"{typ} {truncated:.2f} {occluded:d} {alpha:.6f} " \
               f"{l:.2f} {t:.2f} {r:.2f} {b:.2f} " \
               f"{dims_cam[0]:.6f} {dims_cam[1]:.6f} {dims_cam[2]:.6f} " \
               f"{loc_cam[0]:.6f} {loc_cam[1]:.6f} {loc_cam[2]:.6f} " \
               f"{ry:.6f} {score:.6f}"
        lines.append(line)
    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

def _materialize_testing(side_root: Path, ids: List[str], mode: str):
    """Create <side>/testing/* from <side>/training/* if requested."""
    if not ids: return
    tst = side_root / "testing"
    img_out = tst / "image_2"; vel_out = tst / "velodyne"; calib_out = tst / "calib"
    for p in (img_out, vel_out, calib_out): ensure_dir(p)
    def cp_or_link(src: Path, dst_dir: Path):
        dst = dst_dir / src.name
        if mode == "copy":
            if src.exists(): shutil.copy2(src, dst)
        elif mode == "symlink":
            if dst.exists(): return
            if src.exists(): os.symlink(src, dst)
    tr = side_root / "training"
    for sid in ids:
        png = tr / "image_2" / f"{sid}.png"
        jpg = tr / "image_2" / f"{sid}.jpg"
        cp_or_link(png if png.exists() else jpg, img_out)
        cp_or_link(tr / "velodyne" / f"{sid}.bin", vel_out)
        cp_or_link(tr / "calib"    / f"{sid}.txt", calib_out)

def convert_side(side: str, src_root: Path, out_root: Path,
                 split: Tuple[float,float,float], split_seed: int,
                 testing_mode: str) -> None:
    print(f"[INFO] Converting side: {side}")
    in_base  = src_root / side
    img_dir  = in_base / "image"
    lid2d    = in_base / "label" / "camera"
    lid3d    = in_base / "label" / "lidar"
    calib    = in_base / "calib"
    T_dir = calib / ("lidar_to_camera" if side == "vehicle-side" else "virtuallidar_to_camera")
    K_dir = calib / "camera_intrinsic"

    ids = get_ids(img_dir)
    print(f"[INFO] Found {len(ids)} frames")

    train_root = out_root / side / "training"
    image_out  = train_root / "image_2"
    vel_out    = train_root / "velodyne"
    calib_out  = train_root / "calib"
    label_out  = train_root / "label_2"
    for p in (image_out, vel_out, calib_out, label_out): ensure_dir(p)

    all_ids: List[str] = []
    for sid in ids:
        src_img = (img_dir / f"{sid}.png")
        if not src_img.exists(): src_img = (img_dir / f"{sid}.jpg")
        src_bin = (in_base / "velodyne" / f"{sid}.bin")
        j2d     = (lid2d / f"{sid}.json")
        j3d     = (lid3d / f"{sid}.json")
        Kin     = (K_dir / f"{sid}.json")
        Tin     = (T_dir / f"{sid}.json")
        if not (src_img.exists() and src_bin.exists() and Kin.exists() and Tin.exists()):
            print(f"[WARN] Missing one or more files for {sid}; skipping.")
            continue

        im = cv2.imread(str(src_img), cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"[WARN] Failed to read image {src_img}; skipping."); continue
        im_h, im_w = int(im.shape[0]), int(im.shape[1])
        cv2.imwrite(str(image_out / f"{sid}.png"), im)
        shutil.copy2(src_bin, vel_out / f"{sid}.bin")

        K = parse_intrinsic(Kin)
        T_cam_l = parse_extrinsic(Tin)

        # Normalize to KITTI basis: x right, y down, z forward
        S = np.array([[ 0, -1,  0, 0],
                      [ 0,  0, -1, 0],
                      [ 1,  0,  0, 0],
                      [ 0,  0,  0, 1]], dtype=float)
        T_cam_l = S @ T_cam_l

        write_kitti_calib(calib_out / f"{sid}.txt", K, T_cam_l)

        recs2 = read_json(j2d) if j2d.exists() else []
        recs3 = read_json(j3d) if j3d.exists() else []
        recs2 = recs2 if isinstance(recs2, list) else recs2.get("labels", [recs2])
        recs3 = recs3 if isinstance(recs3, list) else recs3.get("labels", [recs3])

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

        threed = [o for o in (recs3 or [])]
        merged = []
        for i, g in enumerate(threed):
            rec = {"type": g.get("type","Car")}
            if i < len(twod):
                rec["2d_box"] = twod[i]["2d_box"]
                rec["type"]   = twod[i].get("type", rec["type"])
            for k in ("3d_location","3d_dimensions","rotation","yaw","rotation_y","center","size","dimensions"):
                if k in g: rec[k] = g[k]
            merged.append(rec)

        write_kitti_label2(label_out / f"{sid}.txt", merged, K, T_cam_l, im_w, im_h)
        all_ids.append(sid)

    # -------------- Splits --------------
    imagesets_dir = out_root / side / "ImageSets"
    ensure_dir(imagesets_dir)

    n = len(all_ids)
    if n == 0:
        print(f"[WARN] No frames converted for side '{side}'. Skipping split.")
        for name in ("train","val","test","trainval"):
            with open(imagesets_dir / f"{name}.txt", "w") as f: pass
        return

    r_train, r_val, r_test = split
    s = max(1e-9, (r_train + r_val + r_test)); r_train, r_val, r_test = r_train/s, r_val/s, r_test/s

    idxs = list(range(n)); rnd = random.Random(split_seed); rnd.shuffle(idxs)
    n_train = int(round(r_train * n)); n_val = int(round(r_val * n))
    if n_train + n_val > n: n_val = max(0, n - n_train)
    n_test  = max(0, n - n_train - n_val)

    train_idx = idxs[:n_train]
    val_idx   = idxs[n_train:n_train+n_val]
    test_idx  = idxs[n_train+n_val:n_train+n_val+n_test]

    train_ids_sorted = [all_ids[i] for i in sorted(train_idx)]
    val_ids_sorted   = [all_ids[i] for i in sorted(val_idx)]
    test_ids_sorted  = [all_ids[i] for i in sorted(test_idx)]

    for name, lst in (("train", train_ids_sorted), ("val", val_ids_sorted), ("test", test_ids_sorted)):
        with open(imagesets_dir / f"{name}.txt", "w") as f:
            for s_id in lst: f.write(s_id + "\n")

    with open(imagesets_dir / "trainval.txt", "w") as f:
        for s_id in train_ids_sorted + val_ids_sorted: f.write(s_id + "\n")

    # Optionally create <side>/testing/* if test split present and requested.
    if testing_mode in ("copy","symlink") and len(test_ids_sorted) > 0:
        _materialize_testing(out_root / side, test_ids_sorted, testing_mode)

    print(f"[INFO] Split {n} → {len(train_ids_sorted)} train / {len(val_ids_sorted)} val / {len(test_ids_sorted)} test")

def _parse_split_arg(s: str) -> Tuple[float,float,float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Split must be 'a,b,c'")
    vals = []
    for p in parts:
        v = float(p)
        if v > 1.0: v /= 100.0
        vals.append(v)
    if sum(vals) <= 0:
        raise argparse.ArgumentTypeError("Split ratios must be positive.")
    return (vals[0], vals[1], vals[2])

def main():
    global YAW_OFFSET_RAD
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=str(SRC_ROOT),
                        help="Source root (cooperative-vehicle-infrastructure).")
    parser.add_argument("--out", type=str, default=str(OUT_ROOT),
                        help="Output KITTI root.")
    parser.add_argument("--sides", type=str, nargs="+", default=SIDES,
                        help="Sides to convert.")
    parser.add_argument("--split", type=_parse_split_arg, default=(0.85, 0.15, 0.00),
                        help='Train/val/test split "85,15,0" or proportions "0.85,0.15,0.0".')
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--yaw_offset_deg", type=float, default=math.degrees(YAW_OFFSET_RAD))
    parser.add_argument("--testing-mode", type=str, choices=["none","copy","symlink"],
                        default="none",
                        help="If test>0, optionally create <side>/testing/* by copying or symlinking.")
    args = parser.parse_args()

    src = Path(args.src); out = Path(args.out); ensure_dir(out)
    YAW_OFFSET_RAD = math.radians(args.yaw_offset_deg)

    for side in args.sides:
        convert_side(side, src, out, split=args.split, split_seed=args.split_seed,
                     testing_mode=args.testing_mode)
    print(f"[OK] Wrote KITTI dataset (ImageSets next to training) to: {out}")

if __name__ == "__main__":
    main()
