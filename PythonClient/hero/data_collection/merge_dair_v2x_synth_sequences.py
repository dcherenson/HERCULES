#!/usr/bin/env python3
"""
Compose DAIR-V2X-style sequences into a single unique-indexed sequence.
- No CLI args: configure via CONFIG below.
- Copy only: NO hardlinks; sources are never modified.
- Validation: ensures required modalities AND per-frame calib exist per selected frame/side.
- COUNT can mean TOTAL across all sequences via MODE = "rand_total" or "first_total".

This version FIXES the missing-calib issue by copying/renaming per-frame calib JSONs
(e.g., camera_intrinsic/<id>.json and lidar_to_camera|virtuallidar_to_camera/<id>.json)
for each chosen frame, on each side.
"""

import sys
import json
import shutil
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# =====================
# ====== CONFIG =======
# =====================
CONFIG = {
    # Parent directory that contains dair_v2x_synth_testX folders
    "SOURCES_ROOT": "/media/sgarimella34/hercules-collect/collaborative-perception-BEVP/datasets",

    # Output location for the composed sequence
    "OUTPUT_ROOT": "/media/sgarimella34/hercules-collect/collaborative-perception-BEVP/datasets",
    "OUTPUT_NAME": "dair_v2x_synth_TEST1",

    # Selection mode (choose ONE):
    #   "first"       -> per-sequence FIRST N (COUNT per sequence)
    #   "rand"        -> per-sequence RANDOM N (COUNT per sequence)
    #   "perseq"      -> per-sequence counts via PERSEQ_COUNTS
    #
    #   NEW MODES:
    #   "rand_total"  -> RANDOM from ALL sequences combined; COUNT is TOTAL
    #   "first_total" -> FIRST across ALL sequences until TOTAL COUNT reached
    "MODE": "rand_total",     # "first" | "rand" | "perseq" | "rand_total" | "first_total"
    "COUNT": 100,            # For *_total modes: this is TOTAL across all sequences
    "SEED": 1337,
    "PERSEQ_COUNTS": {},      # used only if MODE == "perseq"

    # New filename numbering
    "START_INDEX": 0,
    "WIDTH": 6,               # zero-pad width (e.g., 6 -> 000123)

    # Bulk calib tree copy (not needed anymore; per-frame calib is copied/renamed):
    #   "first"  -> from first discovered sequence
    #   "<name>" -> from that specific sequence name
    #   "skip"   -> don't bulk-copy calib trees
    "COPY_CALIB_FROM": "skip",

    # Validation
    # Required modalities (per side) that must exist for a frame to be accepted.
    "REQUIRED_MODALITIES": [
        ("image", ["png"]),
        ("velodyne", ["bin", "pcd"]),
        ("label/camera", ["json"]),
        ("label/lidar", ["json"]),
    ],
    # Which sides to enforce the requirement on:
    "CHECK_SIDES": ["vehicle-side", "infrastructure-side"],

    # Behavior when required file(s) missing for per-sequence modes ("first", "rand", "perseq"):
    #   "skip"  -> skip the frame and continue
    #   "error" -> abort entire run
    "ON_MISSING": "skip",

    # If True, only print actions without writing files
    "DRY_RUN": False,
}

# Sides
SIDES = ["infrastructure-side", "vehicle-side"]

# Per-side REQUIRED calib subdirs (must exist for each frame)
CALIB_REQUIRED = {
    "vehicle-side": [
        "calib/camera_intrinsic",
        "calib/lidar_to_camera",
    ],
    "infrastructure-side": [
        "calib/camera_intrinsic",
        "calib/virtuallidar_to_camera",
    ],
}

# Optional extra per-frame calib subdirs to copy if present
CALIB_OPTIONAL = [
    "calib/virtuallidar_to_world",
    "calib/lidar_to_novatel",
    "calib/novatel_to_world",
]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, dry: bool = False):
    """Always copy (no hardlinks)."""
    ensure_dir(dst.parent)
    if dry:
        print(f"[DRY] COPY {src} -> {dst}")
        return
    shutil.copy2(src, dst)


def find_frames(seq_root: Path) -> List[str]:
    """
    Determine available frame basenames by listing vehicle-side/image/*.png.
    Assumes modalities share the same basename set.
    Returns sorted list of basenames like ["000000", "000001", ...].
    """
    vs_img = seq_root / "cooperative-vehicle-infrastructure" / "vehicle-side" / "image"
    if not vs_img.is_dir():
        raise FileNotFoundError(f"Missing path: {vs_img}")
    frames = [p.stem for p in vs_img.glob("*.png")]
    frames.sort()
    return frames


def first_existing(side_root: Path, sub: str, stem: str, exts: List[str]) -> Optional[Path]:
    """Return first existing path among exts for subdir/stem; else None."""
    base_dir = side_root / sub
    if not base_dir.exists():
        return None
    for ext in exts:
        cand = base_dir / f"{stem}.{ext}"
        if cand.exists():
            return cand
    return None


def validate_required_modalities(side_root: Path, stem: str, required) -> Optional[Dict[str, Path]]:
    """Validate required (image/velodyne/labels) exist; return mapping {subdir: Path} or None if any missing."""
    found: Dict[str, Path] = {}
    for sub, exts in required:
        p = first_existing(side_root, sub, stem, exts)
        if p is None:
            return None
        found[sub] = p
    return found


def validate_and_collect_calib(side_root: Path, side: str, stem: str) -> Optional[Dict[str, Path]]:
    """
    Ensure REQUIRED calib files exist for this side+frame and collect them;
    also collect OPTIONAL calib files if present.
    Returns { 'calib/<subdir>': Path(...) , ... } or None if a required calib is missing.
    """
    mapping: Dict[str, Path] = {}
    # required
    for sub in CALIB_REQUIRED.get(side, []):
        p = first_existing(side_root, sub, stem, ["json"])
        if p is None:
            return None
        mapping[sub] = p
    # optional
    for sub in CALIB_OPTIONAL:
        p = first_existing(side_root, sub, stem, ["json"])
        if p is not None:
            mapping[sub] = p
    return mapping


def copy_side(side_dst_root: Path, mapping: Dict[str, Path], new_base: str, dry: bool):
    """Copy all files in mapping into destination using new basename, keeping extensions."""
    for sub, src_file in mapping.items():
        dst_dir = side_dst_root / sub
        dst_file = dst_dir / f"{new_base}{src_file.suffix}"
        copy_file(src_file, dst_file, dry=dry)


def copy_calib_tree_from(seq_root: Path, out_seq_root: Path, dry: bool):
    """(Optional) Copy entire calib trees from a sequence (not needed with per-frame copying)."""
    coop = "cooperative-vehicle-infrastructure"
    for side in SIDES:
        src_calib = seq_root / coop / side / "calib"
        if src_calib.exists():
            dst_calib = out_seq_root / coop / side / "calib"
            if dry:
                print(f"[DRY] COPY TREE {src_calib} -> {dst_calib}")
            else:
                if not dst_calib.exists():
                    shutil.copytree(src_calib, dst_calib)


def build_valid_candidates(
    sources_root: Path,
    seq_names: List[str],
    required_modalities,
    check_sides: List[str],
) -> List[Dict]:
    """
    Build a list of VALID candidates across ALL sequences.
    Each candidate dict contains:
      {
        "sequence": seq_name,
        "old_basename": "000123",
        "per_side_maps": {
           "vehicle-side": { "image": Path(...), ..., "calib/camera_intrinsic": Path(...), ... },
           "infrastructure-side": { ... }
        },
      }
    Only frames that pass validation on all required sides are included.
    """
    candidates = []
    for seq in seq_names:
        seq_root = sources_root / seq
        coop_src = seq_root / "cooperative-vehicle-infrastructure"
        frames = find_frames(seq_root)
        for stem in frames:
            per_side_maps: Dict[str, Dict[str, Path]] = {}
            ok = True
            for side in check_sides:
                side_src_root = coop_src / side
                found_mods = validate_required_modalities(side_src_root, stem, required_modalities)
                if found_mods is None:
                    ok = False
                    break
                found_cal = validate_and_collect_calib(side_src_root, side, stem)
                if found_cal is None:
                    ok = False
                    break
                # merge both into one mapping for this side
                side_map = dict(found_mods)
                side_map.update(found_cal)
                per_side_maps[side] = side_map
            if ok:
                candidates.append({
                    "sequence": seq,
                    "old_basename": stem,
                    "per_side_maps": per_side_maps,
                })
    return candidates


def main():
    cfg = CONFIG
    random.seed(cfg["SEED"])

    sources_root = Path(cfg["SOURCES_ROOT"]).resolve()
    out_root = Path(cfg["OUTPUT_ROOT"]).resolve()
    out_seq_root = out_root / cfg["OUTPUT_NAME"]

    # Discover sequences
    seq_names = sorted([
        p.name for p in sources_root.iterdir()
        if p.is_dir() and p.name.startswith("dair_v2x_synth_")
    ])
    if not seq_names:
        print(f"No dair_v2x_synth_* folders found under {sources_root}", file=sys.stderr)
        sys.exit(1)

    # Prepare output root
    if not cfg["DRY_RUN"]:
        ensure_dir(out_seq_root / "cooperative-vehicle-infrastructure")

    # (Optional) bulk calib copy — not needed now but kept for compatibility
    calib_from = None
    if cfg["COPY_CALIB_FROM"] == "first":
        calib_from = seq_names[0]
    elif cfg["COPY_CALIB_FROM"] == "skip":
        calib_from = None
    else:
        if cfg["COPY_CALIB_FROM"] not in seq_names:
            print(f"COPY_CALIB_FROM '{cfg['COPY_CALIB_FROM']}' not found among {seq_names}", file=sys.stderr)
            sys.exit(1)
        calib_from = cfg["COPY_CALIB_FROM"]

    if calib_from:
        src_seq_root = sources_root / calib_from
        copy_calib_tree_from(src_seq_root, out_seq_root, dry=cfg["DRY_RUN"])

    # --- Selection logic ---
    global_idx = cfg["START_INDEX"]
    accepted = 0
    skipped = 0
    mapping = []

    mode = cfg["MODE"]
    total_target = cfg["COUNT"]

    if mode in ("rand_total", "first_total"):
        # Build a union of VALID candidates across all sequences (pre-validated incl. calib)
        candidates = build_valid_candidates(
            sources_root, seq_names,
            cfg["REQUIRED_MODALITIES"], cfg["CHECK_SIDES"]
        )

        total_available = len(candidates)
        if total_available == 0:
            print("No valid frames found across all sequences with the current validation settings.", file=sys.stderr)
            sys.exit(2)

        if total_target > total_available:
            print(f"Requested TOTAL COUNT={total_target}, but only {total_available} valid frames exist. Will take {total_available}.")
            total_target = total_available

        if mode == "rand_total":
            chosen = random.sample(candidates, total_target)
        else:  # first_total
            chosen = candidates[:total_target]

        # Copy the chosen frames (modalities + per-frame calib)
        coop_dst_root = out_seq_root / "cooperative-vehicle-infrastructure"
        for item in chosen:
            new_base = str(global_idx).zfill(cfg["WIDTH"])
            for side, mp in item["per_side_maps"].items():
                side_dst_root = coop_dst_root / side
                copy_side(side_dst_root, mp, new_base, cfg["DRY_RUN"])

            mapping.append({
                "sequence": item["sequence"],
                "old_basename": item["old_basename"],
                "new_basename": new_base
            })
            global_idx += 1
            accepted += 1

    elif mode in ("first", "rand", "perseq"):
        # Per-sequence behavior (COUNT per sequence), with ON_MISSING handling
        for seq in seq_names:
            seq_root = sources_root / seq
            frames = find_frames(seq_root)

            if mode == "first":
                take_n = min(cfg["COUNT"], len(frames))
                selected = frames[:take_n]
            elif mode == "rand":
                take_n = min(cfg["COUNT"], len(frames))
                selected = sorted(random.sample(frames, take_n))
            else:  # perseq
                want = int(cfg["PERSEQ_COUNTS"].get(seq, 0))
                take_n = min(want, len(frames))
                selected = frames[:take_n] if take_n > 0 else []

            coop_src = seq_root / "cooperative-vehicle-infrastructure"
            coop_dst = out_seq_root / "cooperative-vehicle-infrastructure"

            for stem in selected:
                per_side_maps: Dict[str, Dict[str, Path]] = {}
                missing = False
                for side in cfg["CHECK_SIDES"]:
                    side_src_root = coop_src / side
                    found_mods = validate_required_modalities(side_src_root, stem, cfg["REQUIRED_MODALITIES"])
                    found_cal  = validate_and_collect_calib(side_src_root, side, stem)
                    if (found_mods is None) or (found_cal is None):
                        missing = True
                        break
                    side_map = dict(found_mods); side_map.update(found_cal)
                    per_side_maps[side] = side_map

                if missing:
                    if cfg["ON_MISSING"] == "skip":
                        skipped += 1
                        continue
                    else:
                        print(f"Missing required files (modalities/calib) for frame {seq}:{stem}. Aborting.", file=sys.stderr)
                        sys.exit(2)

                new_base = str(global_idx).zfill(cfg["WIDTH"])
                for side, mp in per_side_maps.items():
                    side_dst_root = coop_dst / side
                    copy_side(side_dst_root, mp, new_base, cfg["DRY_RUN"])

                mapping.append({"sequence": seq, "old_basename": stem, "new_basename": new_base})
                global_idx += 1
                accepted += 1
    else:
        print(f"Invalid MODE: {mode}", file=sys.stderr)
        sys.exit(1)

    # Write manifest
    manifest_path = out_seq_root / "compose_manifest.json"
    manifest = {
        "sources_root": str(sources_root),
        "output_sequence": cfg["OUTPUT_NAME"],
        "mode": cfg["MODE"],
        "count": cfg["COUNT"],
        "perseq_counts": cfg["PERSEQ_COUNTS"],
        "start_index": cfg["START_INDEX"],
        "width": cfg["WIDTH"],
        "copied_calib_from": calib_from if calib_from else "skip",
        "required_modalities": cfg["REQUIRED_MODALITIES"],
        "check_sides": cfg["CHECK_SIDES"],
        "accepted": accepted,
        "skipped": skipped,
        "mapping": mapping,
    }
    if cfg["DRY_RUN"]:
        print(f"[DRY] Would write manifest with {len(mapping)} entries to {manifest_path}")
    else:
        ensure_dir(manifest_path.parent)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"Done. Accepted {accepted} frames; skipped {skipped}.")
    print(f"Output sequence at: {out_seq_root}")
    if not cfg["DRY_RUN"]:
        print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
