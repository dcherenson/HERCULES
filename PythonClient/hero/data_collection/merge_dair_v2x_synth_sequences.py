#!/usr/bin/env python3
"""
Compose DAIR-V2X-style sequences into a single unique-indexed sequence.
(… original header docstring unchanged …)
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
    "SOURCES_ROOT": "/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets/DAIR-V2X-C",
    # "SOURCES_ROOT": "/media/sgarimella34/hercules-collect/collaborative-perception-BEVP/datasets/",
    "OUTPUT_ROOT": "/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets",
    "OUTPUT_NAME": "DAIR-V2X-C-SUBSET1",
    "MODE": "rand_total",     # "first" | "rand" | "perseq" | "rand_total" | "first_total"
    "COUNT": 100,
    "SEED": 1337,
    "PERSEQ_COUNTS": {},

    "START_INDEX": 0,
    "WIDTH": 6,

    "COPY_CALIB_FROM": "skip",  # "first" | "skip" | "<exact sequence name>"

    "REQUIRED_MODALITIES": [
        ("image", ["png", "jpg"]),
        ("velodyne", ["bin", "pcd"]),
        ("label/camera", ["json"]),
        ("label/lidar", ["json"]),
    ],
    "CHECK_SIDES": ["vehicle-side", "infrastructure-side"],
    "ON_MISSING": "skip",
    "DRY_RUN": False,
}

SIDES = ["infrastructure-side", "vehicle-side"]

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

CALIB_OPTIONAL = [
    "calib/virtuallidar_to_world",
    "calib/lidar_to_novatel",
    "calib/novatel_to_world",
]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, dry: bool = False):
    ensure_dir(dst.parent)
    if dry:
        print(f"[DRY] COPY {src} -> {dst}")
        return
    shutil.copy2(src, dst)


def seq_has_coop_root(root: Path) -> bool:
    """Return True if 'cooperative-vehicle-infrastructure' exists under root."""
    return (root / "cooperative-vehicle-infrastructure").is_dir()


def discover_sequences(sources_root: Path) -> List[Tuple[str, Path]]:
    """
    Return a list of (sequence_name, sequence_path).
    Supports:
      1) Multiple subfolders named 'dair_v2x_synth_*' (classic).
      2) Any subfolder that contains 'cooperative-vehicle-infrastructure' (e.g., 'DAIR-V2X-C').
      3) Single-sequence-at-root where SOURCES_ROOT itself contains 'cooperative-vehicle-infrastructure'.
    """
    seqs: List[Tuple[str, Path]] = []

    # Case 1: classic multi-sequence layout
    for p in sorted(sources_root.iterdir()):
        if p.is_dir() and p.name.startswith("dair_v2x_synth_") and seq_has_coop_root(p):
            seqs.append((p.name, p))

    # Case 2: other subdirs that still contain a coop root (e.g., 'DAIR-V2X-C')
    named = {name for name, _ in seqs}
    for p in sorted(sources_root.iterdir()):
        if p.is_dir() and p.name not in named and seq_has_coop_root(p):
            seqs.append((p.name, p))

    # Case 3: SOURCES_ROOT itself is a single sequence
    if not seqs and seq_has_coop_root(sources_root):
        seqs.append((sources_root.name, sources_root))

    return seqs


def find_frames(seq_root: Path) -> List[str]:
    vs_img = seq_root / "cooperative-vehicle-infrastructure" / "vehicle-side" / "image"
    if not vs_img.is_dir():
        raise FileNotFoundError(f"Missing path: {vs_img}")
    frames = [p.stem for p in vs_img.glob("*.png")]
    frames.sort()
    return frames


def first_existing(side_root: Path, sub: str, stem: str, exts: List[str]) -> Optional[Path]:
    base_dir = side_root / sub
    if not base_dir.exists():
        return None
    for ext in exts:
        cand = base_dir / f"{stem}.{ext}"
        if cand.exists():
            return cand
    return None


def validate_required_modalities(side_root: Path, stem: str, required) -> Optional[Dict[str, Path]]:
    found: Dict[str, Path] = {}
    for sub, exts in required:
        p = first_existing(side_root, sub, stem, exts)
        if p is None:
            return None
        found[sub] = p
    return found


def validate_and_collect_calib(side_root: Path, side: str, stem: str) -> Optional[Dict[str, Path]]:
    mapping: Dict[str, Path] = {}
    for sub in CALIB_REQUIRED.get(side, []):
        p = first_existing(side_root, sub, stem, ["json"])
        if p is None:
            return None
        mapping[sub] = p
    for sub in CALIB_OPTIONAL:
        p = first_existing(side_root, sub, stem, ["json"])
        if p is not None:
            mapping[sub] = p
    return mapping


def copy_side(side_dst_root: Path, mapping: Dict[str, Path], new_base: str, dry: bool):
    for sub, src_file in mapping.items():
        dst_dir = side_dst_root / sub
        dst_file = dst_dir / f"{new_base}{src_file.suffix}"
        copy_file(src_file, dst_file, dry=dry)


def copy_calib_tree_from(seq_root: Path, out_seq_root: Path, dry: bool):
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
    sequences: List[Tuple[str, Path]],
    required_modalities,
    check_sides: List[str],
) -> List[Dict]:
    candidates = []
    for seq_name, seq_root in sequences:
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
                side_map = dict(found_mods)
                side_map.update(found_cal)
                per_side_maps[side] = side_map
            if ok:
                candidates.append({
                    "sequence": seq_name,
                    "sequence_root": str(seq_root),
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

    # Discover sequences (supports both layouts)
    sequences = discover_sequences(sources_root)
    if not sequences:
        print(
            f"No sequences found under {sources_root}.\n"
            f"Expected either subfolders named 'dair_v2x_synth_*' each containing "
            f"'cooperative-vehicle-infrastructure/', or the SOURCES_ROOT itself to contain it.",
            file=sys.stderr
        )
        sys.exit(1)

    # Prepare output root
    if not cfg["DRY_RUN"]:
        ensure_dir(out_seq_root / "cooperative-vehicle-infrastructure")

    # (Optional) bulk calib copy
    calib_from_tuple: Optional[Tuple[str, Path]] = None
    if cfg["COPY_CALIB_FROM"] == "first":
        calib_from_tuple = sequences[0]
    elif cfg["COPY_CALIB_FROM"] == "skip":
        calib_from_tuple = None
    else:
        # If a specific name is given, find it among discovered sequences
        wanted = cfg["COPY_CALIB_FROM"]
        for name, path in sequences:
            if name == wanted:
                calib_from_tuple = (name, path)
                break
        if cfg["COPY_CALIB_FROM"] not in ("first", "skip") and calib_from_tuple is None:
            print(
                f"COPY_CALIB_FROM '{cfg['COPY_CALIB_FROM']}' not found among {[n for n,_ in sequences]}",
                file=sys.stderr
            )
            sys.exit(1)

    if calib_from_tuple:
        _, src_seq_root = calib_from_tuple
        copy_calib_tree_from(src_seq_root, out_seq_root, dry=cfg["DRY_RUN"])

    # --- Selection logic ---
    global_idx = cfg["START_INDEX"]
    accepted = 0
    skipped = 0
    mapping = []

    mode = cfg["MODE"]
    total_target = cfg["COUNT"]

    if mode in ("rand_total", "first_total"):
        candidates = build_valid_candidates(
            sequences, cfg["REQUIRED_MODALITIES"], cfg["CHECK_SIDES"]
        )

        total_available = len(candidates)
        if total_available == 0:
            print("No valid frames found across all sequences with the current validation settings.", file=sys.stderr)
            sys.exit(2)

        if total_target > total_available:
            print(f"Requested TOTAL COUNT={total_target}, but only {total_available} valid frames exist. Will take {total_available}.")
            total_target = total_available

        chosen = random.sample(candidates, total_target) if mode == "rand_total" else candidates[:total_target]

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
        for seq_name, seq_root in sequences:
            frames = find_frames(seq_root)

            if mode == "first":
                take_n = min(cfg["COUNT"], len(frames))
                selected = frames[:take_n]
            elif mode == "rand":
                take_n = min(cfg["COUNT"], len(frames))
                selected = sorted(random.sample(frames, take_n))
            else:  # perseq
                want = int(cfg["PERSEQ_COUNTS"].get(seq_name, 0))
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
                        print(f"Missing required files (modalities/calib) for frame {seq_name}:{stem}. Aborting.", file=sys.stderr)
                        sys.exit(2)

                new_base = str(global_idx).zfill(cfg["WIDTH"])
                for side, mp in per_side_maps.items():
                    side_dst_root = coop_dst / side
                    copy_side(side_dst_root, mp, new_base, cfg["DRY_RUN"])

                mapping.append({"sequence": seq_name, "old_basename": stem, "new_basename": new_base})
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
        "copied_calib_from": calib_from_tuple[0] if calib_from_tuple else "skip",
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
