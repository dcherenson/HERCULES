#!/usr/bin/env python3
"""
Compose DAIR-V2X-style sequences into a single unique-indexed sequence.
- No CLI args: configure via CONFIG below.
- Copy only: NO hardlinks; sources are never modified.
- Validation: ensures required modalities exist per selected frame and side.
- NEW: COUNT can mean TOTAL across all sequences via MODE = "rand_total" or "first_total".

Usage:
  Edit CONFIG, then run:
      python3 merge_dair_v2x_synth_sequences.py
"""

import os
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
    "OUTPUT_NAME": "dair_v2x_synth_COMPOSED",

    # Selection mode (choose ONE):
    #   "first"       -> per-sequence FIRST N (old behavior: COUNT per sequence)
    #   "rand"        -> per-sequence RANDOM N (old behavior: COUNT per sequence)
    #   "perseq"      -> per-sequence counts via PERSEQ_COUNTS
    #
    #   NEW MODES:
    #   "rand_total"  -> RANDOM from ALL sequences combined; COUNT is TOTAL
    #   "first_total" -> FIRST across ALL sequences until TOTAL COUNT reached
    "MODE": "rand_total",     # "first" | "rand" | "perseq" | "rand_total" | "first_total"
    "COUNT": 5999,             # For *_total modes: this is TOTAL across all sequences
    "SEED": 1337,
    "PERSEQ_COUNTS": {},      # used only if MODE == "perseq"

    # New filename numbering
    "START_INDEX": 0,
    "WIDTH": 6,               # zero-pad width (e.g., 6 -> 000123)

    # Calibration: copy once from which sequence?
    #   "first"  -> from first discovered sequence
    #   "<name>" -> from that specific sequence name
    #   "skip"   -> don't copy calib
    "COPY_CALIB_FROM": "first",

    # Validation
    # Required modalities (per side) that must exist for a frame to be accepted.
    # If any missing, the frame is skipped (for *_total modes we pre-filter).
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

SIDES = ["infrastructure-side", "vehicle-side"]


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


def validate_required(side_root: Path, stem: str, required) -> Optional[Dict[str, Path]]:
    """Validate required modalities exist; return mapping {subdir: Path} or None if any missing."""
    found = {}
    for sub, exts in required:
        p = first_existing(side_root, sub, stem, exts)
        if p is None:
            return None
        found[sub] = p
    return found


def copy_side(side_dst_root: Path, mapping: Dict[str, Path], new_base: str, dry: bool):
    """Copy all files in mapping into destination using new basename, keeping extensions."""
    for sub, src_file in mapping.items():
        dst_dir = side_dst_root / sub
        dst_file = dst_dir / f"{new_base}{src_file.suffix}"
        copy_file(src_file, dst_file, dry=dry)


def copy_calib_from(seq_root: Path, out_seq_root: Path, dry: bool):
    """Copy calib trees (if present) from both sides in the selected sequence."""
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
           "vehicle-side": { "image": Path(...), "velodyne": Path(...), ... },
           "infrastructure-side": { ... }
        },
        "coop_src_root": Path to sequence's 'cooperative-vehicle-infrastructure'
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
                found = validate_required(side_src_root, stem, required_modalities)
                if found is None:
                    ok = False
                    break
                per_side_maps[side] = found
            if ok:
                candidates.append({
                    "sequence": seq,
                    "old_basename": stem,
                    "per_side_maps": per_side_maps,
                    "coop_src_root": coop_src,
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

    # Decide calib source
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
        copy_calib_from(src_seq_root, out_seq_root, dry=cfg["DRY_RUN"])

    # --- Selection logic ---
    global_idx = cfg["START_INDEX"]
    accepted = 0
    skipped = 0
    mapping = []

    mode = cfg["MODE"]
    total_target = cfg["COUNT"]

    if mode in ("rand_total", "first_total"):
        # Build a union of VALID candidates across all sequences (pre-validated)
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
            # Keep natural order: seq_names order, then frame order (as discovered by find_frames)
            # candidates were built in that order; just slice
            chosen = candidates[:total_target]

        # Copy the chosen frames
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
        # Original per-sequence behavior (COUNT is per-sequence), with ON_MISSING handling
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
                side_maps: Dict[str, Dict[str, Path]] = {}
                missing = False
                for side in cfg["CHECK_SIDES"]:
                    side_src_root = coop_src / side
                    found = validate_required(side_src_root, stem, cfg["REQUIRED_MODALITIES"])
                    if found is None:
                        missing = True
                        break
                    side_maps[side] = found

                if missing:
                    if cfg["ON_MISSING"] == "skip":
                        skipped += 1
                        continue
                    else:
                        print(f"Missing required files for frame {seq}:{stem}. Aborting.", file=sys.stderr)
                        sys.exit(2)

                new_base = str(global_idx).zfill(cfg["WIDTH"])
                for side, mp in side_maps.items():
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
