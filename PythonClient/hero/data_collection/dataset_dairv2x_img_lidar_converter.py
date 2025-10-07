#!/usr/bin/env python3
"""
Convert DAIR-V2X dataset:
- image/*.png  -> move to image_png/ and write image/*.jpg
- velodyne/*.bin -> move to velodyne_bin/ and write velodyne/*.pcd (KITTI format: x,y,z,intensity)

Simply set DATASET_ROOT below and run:
    python convert_dairv2x.py
"""

import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image


# === USER CONFIGURATION ===
DATASET_ROOT = Path(
    "/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets/dair_v2x_synth_TEST1"
)
# ===========================


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def convert_pngs_to_jpgs(image_dir: Path):
    """Move all .png from image_dir to image_png/, then write .jpg (same basename) into image_dir."""
    if not image_dir.exists():
        print(f"[image] skip (missing): {image_dir}")
        return

    dst_png_dir = image_dir.parent / "image_png"
    ensure_dir(dst_png_dir)

    png_files = sorted(image_dir.glob("*.png"))
    if not png_files:
        print(f"[image] no PNGs found in {image_dir}")
        return

    print(f"[image] processing {len(png_files)} PNGs under {image_dir}")
    moved = 0
    converted = 0
    for png_path in png_files:
        base = png_path.stem
        jpg_path = image_dir / f"{base}.jpg"
        archived_png_path = dst_png_dir / png_path.name

        # Move PNG if not already moved
        if not archived_png_path.exists():
            shutil.move(str(png_path), str(archived_png_path))
            moved += 1
        else:
            if png_path.exists():
                png_path.unlink()

        # Convert to JPG only if missing
        if not jpg_path.exists():
            try:
                with Image.open(archived_png_path) as im:
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    im.save(jpg_path, format="JPEG", quality=95, optimize=True)
                converted += 1
            except Exception as e:
                print(f"[image][ERROR] {archived_png_path.name}: {e}")

    print(f"[image] moved PNGs: {moved} | wrote JPGs: {converted}")


def write_pcd_ascii(points_xyzi: np.ndarray, out_path: Path):
    """Write an ASCII PCD file with fields x y z intensity."""
    N = points_xyzi.shape[0]
    header = (
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {N}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {N}\n"
        "DATA ascii\n"
    )
    with open(out_path, "w") as f:
        f.write(header)
        for p in points_xyzi:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {p[3]:.6f}\n")


def convert_bin_to_pcds(velodyne_dir: Path):
    """Move all .bin from velodyne/ to velodyne_bin/, then write .pcd (same basename) into velodyne/."""
    if not velodyne_dir.exists():
        print(f"[velodyne] skip (missing): {velodyne_dir}")
        return

    dst_bin_dir = velodyne_dir.parent / "velodyne_bin"
    ensure_dir(dst_bin_dir)

    bin_files = sorted(velodyne_dir.glob("*.bin"))
    if not bin_files:
        print(f"[velodyne] no BINs found in {velodyne_dir}")
        return

    print(f"[velodyne] processing {len(bin_files)} BINs under {velodyne_dir}")
    moved = 0
    converted = 0
    for bin_path in bin_files:
        base = bin_path.stem
        pcd_path = velodyne_dir / f"{base}.pcd"
        archived_bin_path = dst_bin_dir / bin_path.name

        # Move BIN if not already moved
        if not archived_bin_path.exists():
            shutil.move(str(bin_path), str(archived_bin_path))
            moved += 1
        else:
            if bin_path.exists():
                bin_path.unlink()

        # Convert to PCD only if missing
        if not pcd_path.exists():
            try:
                data = np.fromfile(str(archived_bin_path), dtype=np.float32)
                if data.size % 4 != 0:
                    raise ValueError(f"Invalid BIN size: {archived_bin_path}")
                points = data.reshape(-1, 4)
                write_pcd_ascii(points, pcd_path)
                converted += 1
            except Exception as e:
                print(f"[velodyne][ERROR] {archived_bin_path.name}: {e}")

    print(f"[velodyne] moved BINs: {moved} | wrote PCDs: {converted}")


def process_side(side_root: Path):
    """Process one dataset side (infrastructure-side or vehicle-side)."""
    print(f"\n=== Processing: {side_root} ===")
    image_dir = side_root / "image"
    velodyne_dir = side_root / "velodyne"
    convert_pngs_to_jpgs(image_dir)
    convert_bin_to_pcds(velodyne_dir)


def main():
    cvi_root = DATASET_ROOT / "cooperative-vehicle-infrastructure"
    sides = [
        cvi_root / "infrastructure-side",
        cvi_root / "vehicle-side",
    ]

    for side in sides:
        process_side(side)

    print("\n All done!")


if __name__ == "__main__":
    main()
