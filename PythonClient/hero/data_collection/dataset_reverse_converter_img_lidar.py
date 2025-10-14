#!/usr/bin/env python3
"""
Inverse DAIR-V2X converter with compressed PCD support:
- image/*.jpg  -> move to image_jpg/ and write image/*.png
- velodyne/*.pcd -> move to velodyne_pcd/ and write velodyne/*.bin (KITTI: x,y,z,intensity float32)

Requires: open3d (for DATA binary_compressed). Falls back to a lightweight parser
for DATA ascii / DATA binary when open3d is unavailable.
"""

from pathlib import Path
import os, shutil, numpy as np
from PIL import Image

# ========= USER CONFIG =========
DATASET_ROOT = Path(
    "/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets/DAIR-V2X-C-SUBSET1"
)
# ===============================

# Try Open3D (handles binary_compressed)
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------- Images: JPG -> PNG ----------
def convert_jpgs_to_pngs(image_dir: Path):
    if not image_dir.exists():
        print(f"[image] skip (missing): {image_dir}")
        return
    dst_jpg_dir = image_dir.parent / "image_jpg"
    ensure_dir(dst_jpg_dir)

    jpg_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")))
    if not jpg_files:
        print(f"[image] no JPGs found in {image_dir}")
        return

    print(f"[image] processing {len(jpg_files)} JPGs under {image_dir}")
    moved = converted = 0
    for jpg_path in jpg_files:
        base = jpg_path.stem
        png_path = image_dir / f"{base}.png"
        archived_jpg_path = dst_jpg_dir / jpg_path.name

        if not archived_jpg_path.exists():
            shutil.move(str(jpg_path), str(archived_jpg_path)); moved += 1
        else:
            if jpg_path.exists(): jpg_path.unlink()

        if not png_path.exists():
            try:
                with Image.open(archived_jpg_path) as im:
                    # JPEG has no alpha; keep RGB
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    im.save(png_path, format="PNG", compress_level=3)
                converted += 1
            except Exception as e:
                print(f"[image][ERROR] {archived_jpg_path.name}: {e}")

    print(f"[image] moved JPGs: {moved} | wrote PNGs: {converted}")

# ---------- Lidar helpers ----------
def _parse_pcd_header(fp) -> dict:
    header = {}
    while True:
        line = fp.readline()
        if not line:
            break
        s = line.decode("utf-8", errors="ignore").strip()
        if s.startswith("DATA"):
            parts = s.split()
            header["DATA"] = (parts[1].lower() if len(parts) > 1 else "ascii")
            break
        if not s:
            continue
        key = s.split()[0].upper()
        header.setdefault(key, []).append(s)

    def get_tokens(key):
        if key not in header: return []
        return header[key][-1].split()[1:]

    fields = get_tokens("FIELDS")
    sizes  = [int(x) for x in get_tokens("SIZE")]  if "SIZE"  in header else [4]*len(fields)
    types  = get_tokens("TYPE")                    if "TYPE"  in header else ["F"]*len(fields)
    counts = [int(x) for x in get_tokens("COUNT")] if "COUNT" in header else [1]*len(fields)
    width  = int(get_tokens("WIDTH")[0])  if "WIDTH"  in header else None
    height = int(get_tokens("HEIGHT")[0]) if "HEIGHT" in header else 1
    points = int(get_tokens("POINTS")[0]) if "POINTS" in header else (width*height if width and height else None)
    mode   = header.get("DATA", "ascii")
    return dict(fields=fields, sizes=sizes, types=types, counts=counts,
                width=width, height=height, points=points, data=mode)

def _dtype_from_pcd_type(t: str, size: int) -> np.dtype:
    t = t.upper()
    if t == "F":
        return np.dtype("<f4") if size == 4 else np.dtype("<f8")
    if t == "I":
        return np.dtype("<i1") if size == 1 else np.dtype("<i2") if size == 2 else np.dtype("<i4")
    if t == "U":
        return np.dtype("<u1") if size == 1 else np.dtype("<u2") if size == 2 else np.dtype("<u4")
    return np.dtype("<f4")

def _pcd_names_dtype(fields, sizes, types, counts):
    names, dtypes = [], []
    for fld, sz, tp, cnt in zip(fields, sizes, types, counts):
        for i in range(cnt):
            name = fld if cnt == 1 else f"{fld}_{i}"
            names.append(name)
            dtypes.append(_dtype_from_pcd_type(tp, sz))
    return names, np.dtype(list(zip(names, dtypes)))

def _pcd_ascii_or_binary_to_xyzi(path: Path) -> np.ndarray:
    """Lightweight loader for DATA ascii / DATA binary only."""
    with open(path, "rb") as f:
        hdr = _parse_pcd_header(f)
        names, point_dtype = _pcd_names_dtype(hdr["fields"], hdr["sizes"], hdr["types"], hdr["counts"])
        npts = hdr["points"]
        mode = hdr["data"]

        if mode == "ascii":
            txt = f.read().decode("utf-8", errors="ignore").strip().splitlines()
            data = np.zeros((len(txt),), dtype=point_dtype)
            for i, line in enumerate(txt):
                vals = line.strip().split()
                if len(vals) != len(names): 
                    continue
                for j, name in enumerate(names):
                    data[name][i] = np.array(vals[j], dtype=point_dtype[name])
        elif mode == "binary":
            raw = f.read()
            data = np.frombuffer(raw, dtype=point_dtype, count=npts)
        else:
            raise ValueError(f"Unsupported PCD DATA mode here: {mode}")

    def pick(alts):
        for n in alts:
            if n in data.dtype.names:
                return n
        return None
    x = data[pick(["x","X"])].astype(np.float32)
    y = data[pick(["y","Y"])].astype(np.float32)
    z = data[pick(["z","Z"])].astype(np.float32)
    inten_name = pick(["intensity","Intensity","reflectance","Reflectance","intensity_0"])
    r = data[inten_name].astype(np.float32) if inten_name is not None else np.zeros_like(x, dtype=np.float32)
    return np.stack([x, y, z, r], axis=1)

def _pcd_any_to_xyzi_with_open3d(path: Path) -> np.ndarray:
    """
    Use Open3D to load any PCD (including DATA binary_compressed).
    Tries tensor API first (preserves extra fields), then legacy.
    """
    # Tensor API (>=0.17): keeps named point attributes
    try:
        pc = o3d.t.io.read_point_cloud(str(path))
        # Positions may be under "positions" or "points" depending on version
        if "positions" in pc.point:
            xyz = pc.point["positions"].numpy()
        elif "points" in pc.point:
            xyz = pc.point["points"].numpy()
        else:
            # fallback: convert to legacy
            leg = pc.to_legacy_pointcloud()
            xyz = np.asarray(leg.points, dtype=np.float32)
        inten = None
        for k in ("intensity","reflectance","Intensity","Reflectance"):
            if k in pc.point:
                inten = pc.point[k].numpy().reshape(-1).astype(np.float32)
                break
        if inten is None:
            # Sometimes intensity is stored as colors; reduce to grayscale
            if "colors" in pc.point:
                col = pc.point["colors"].numpy()
                inten = (0.299*col[:,0] + 0.587*col[:,1] + 0.114*col[:,2]).astype(np.float32)
            else:
                inten = np.zeros((xyz.shape[0],), dtype=np.float32)
        if xyz.dtype != np.float32:
            xyz = xyz.astype(np.float32)
        return np.column_stack([xyz, inten])
    except Exception:
        # Legacy API
        pcd = o3d.io.read_point_cloud(str(path))
        xyz = np.asarray(pcd.points, dtype=np.float32)
        if pcd.has_colors():
            col = np.asarray(pcd.colors, dtype=np.float32)
            inten = (0.299*col[:,0] + 0.587*col[:,1] + 0.114*col[:,2]).astype(np.float32)
        else:
            inten = np.zeros((xyz.shape[0],), dtype=np.float32)
        return np.column_stack([xyz, inten])

def write_kitti_bin(out_path: Path, xyzi: np.ndarray):
    xyzi.astype(np.float32).tofile(out_path)

def convert_pcds_to_bins(velodyne_dir: Path):
    if not velodyne_dir.exists():
        print(f"[velodyne] skip (missing): {velodyne_dir}")
        return

    dst_pcd_dir = velodyne_dir.parent / "velodyne_pcd"
    ensure_dir(dst_pcd_dir)

    pcd_files = sorted(velodyne_dir.glob("*.pcd"))
    if not pcd_files:
        print(f"[velodyne] no PCDs found in {velodyne_dir}")
        return

    print(f"[velodyne] processing {len(pcd_files)} PCDs under {velodyne_dir}")
    moved = converted = 0
    for pcd_path in pcd_files:
        base = pcd_path.stem
        bin_path = velodyne_dir / f"{base}.bin"
        archived_pcd_path = dst_pcd_dir / pcd_path.name

        if not archived_pcd_path.exists():
            shutil.move(str(pcd_path), str(archived_pcd_path)); moved += 1
        else:
            if pcd_path.exists(): pcd_path.unlink()

        if bin_path.exists():
            continue

        try:
            # Try Open3D for binary_compressed; fall back to lightweight loader
            if _HAS_O3D:
                xyzi = _pcd_any_to_xyzi_with_open3d(archived_pcd_path)
            else:
                xyzi = _pcd_ascii_or_binary_to_xyzi(archived_pcd_path)
            write_kitti_bin(bin_path, xyzi)
            converted += 1
        except Exception as e:
            mode_hint = ""
            try:
                with open(archived_pcd_path, "rb") as f:
                    mode_hint = _parse_pcd_header(f)["data"]
            except Exception:
                pass
            print(f"[velodyne][ERROR] {archived_pcd_path.name}: {e} (mode={mode_hint})")
            if not _HAS_O3D and mode_hint == "binary_compressed":
                print("         Hint: install Open3D:  pip install open3d")

    print(f"[velodyne] moved PCDs: {moved} | wrote BINs: {converted}")

# ---------- Process one side ----------
def process_side(side_root: Path):
    print(f"\n=== Processing: {side_root} ===")
    image_dir = side_root / "image"
    velodyne_dir = side_root / "velodyne"
    convert_jpgs_to_pngs(image_dir)
    convert_pcds_to_bins(velodyne_dir)

def main():
    cvi_root = DATASET_ROOT / "cooperative-vehicle-infrastructure"
    sides = [cvi_root / "infrastructure-side", cvi_root / "vehicle-side"]
    for side in sides:
        process_side(side)
    print("\nAll done!")

if __name__ == "__main__":
    main()
