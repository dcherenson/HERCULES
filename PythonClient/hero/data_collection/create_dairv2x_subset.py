#!/usr/bin/env python3
# create_dairv2x_subset_nocli_pairs_altlabels.py
import sys, json, shutil, random, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Any, Union

# =====================
# ====== CONFIG  ======
# =====================
CONFIG = {
    # Root that contains your dataset folder
    "DATASETS_ROOT": "/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets",

    # The dataset folder name to sample from (must contain cooperative-vehicle-infrastructure/)
    "DATASET_NAME": "DAIR-V2X-C",

    # Output subset folder name (created under DATASETS_ROOT)
    "OUTPUT_NAME": "DAIR-V2X-C-SUBSET1",

    # Total pairs to copy at random
    "COUNT": 100,

    # Random seed
    "SEED": 1337,

    # Start index & dry run toggle
    "START_INDEX": 0,
    "DRY_RUN": False,
}

# ===== Constants =====
COOP = "cooperative-vehicle-infrastructure"
SIDES = ["vehicle-side", "infrastructure-side"]

# Required per-frame calib by side (DAIR-V2X-C per-frame JSONs)
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

# Optional calib (copy if present)
CALIB_OPTIONAL = [
    "calib/virtuallidar_to_world",
    "calib/lidar_to_novatel",
    "calib/novatel_to_world",
]

# === Side-specific required modalities
# Each entry can be:
#   ("subdir", ["ext1","ext2"])  -> required single path
#   [ ("subdirA", ["exts"]), ("subdirB", ["exts"]) ]  -> alternatives; first that exists is accepted
PER_SIDE_REQUIRED_MODALITIES: Dict[str, List[Union[Tuple[str, List[str]], List[Tuple[str, List[str]]]]]] = {
    "vehicle-side": [
        ("image", ["png", "jpg", "jpeg"]),
        ("velodyne", ["bin", "pcd"]),
        ("label/camera", ["json"]),
        ("label/lidar", ["json"]),
    ],
    "infrastructure-side": [
        ("image", ["png", "jpg", "jpeg"]),
        ("velodyne", ["bin", "pcd"]),
        ("label/camera", ["json"]),
        # accept either label/lidar or label/virtuallidar
        [("label/lidar", ["json"]), ("label/virtuallidar", ["json"])],
    ],
}

# =====================
# ===== Helpers   =====
# =====================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def first_existing(base: Path, rel: str, stem: str, exts: List[str]) -> Optional[Path]:
    d = base / rel
    if not d.exists():
        return None
    for ext in exts:
        f = d / f"{stem}.{ext}"
        if f.exists():
            return f
    return None

def copy_group(dst_side_root: Path, mapping: Dict[str, Path], new_base: str, dry: bool):
    for sub, src in mapping.items():
        out_dir = dst_side_root / sub
        ensure_dir(out_dir)
        out_file = out_dir / f"{new_base}{src.suffix}"
        if dry:
            print(f"[DRY] COPY {src} -> {out_file}")
        else:
            shutil.copy2(src, out_file)

def width_for(n: int) -> int:
    # at least 4 for nicer sorting, otherwise digits of (n-1)
    return max(4, len(str(max(0, n - 1))))


# ---------- Pairs from cooperative/data_info.json ----------
IMG_PATH_RE = re.compile(r"(vehicle-side|infrastructure-side)/image/([^/]+)\.(png|jpg|jpeg)$", re.IGNORECASE)

def _iter_strings(obj: Any) -> Iterable[str]:
    """Yield all string values recursively from an arbitrary JSON structure."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)

def extract_pair_from_record(rec: Any) -> Optional[Tuple[str, str]]:
    """
    Return (veh_stem, infra_stem) if both image paths are found anywhere in the record.
    """
    veh_stem = None
    infra_stem = None
    for s in _iter_strings(rec):
        m = IMG_PATH_RE.search(s.replace("\\", "/"))
        if not m:
            continue
        side, stem, _ext = m.groups()
        if side == "vehicle-side":
            veh_stem = veh_stem or stem
        elif side == "infrastructure-side":
            infra_stem = infra_stem or stem
        if veh_stem and infra_stem:
            return veh_stem, infra_stem
    return None

def load_pairs_from_data_info(coop_dir: Path) -> List[Tuple[str, str]]:
    """
    Parse cooperative/data_info.json and produce list of (veh_stem, infra_stem).
    """
    di_path = coop_dir / "cooperative" / "data_info.json"
    if not di_path.is_file():
        return []

    with open(di_path, "r") as f:
        data = json.load(f)

    candidates: List[Any] = []
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        for k in ("data_list", "items", "pairs", "annotations", "infos"):
            v = data.get(k)
            if isinstance(v, list):
                candidates = v
                break
        if not candidates:
            for v in data.values():
                if isinstance(v, list):
                    candidates.extend(v)

    pairs: List[Tuple[str, str]] = []
    for rec in candidates:
        p = extract_pair_from_record(rec)
        if p:
            pairs.append(p)

    # Deduplicate preserving order
    seen = set()
    uniq_pairs = []
    for v, i in pairs:
        if (v, i) not in seen:
            uniq_pairs.append((v, i))
            seen.add((v, i))
    return uniq_pairs


# ---------- Validation using per-side stems & alternatives ----------
def resolve_required(side_root: Path, requirement: Union[Tuple[str, List[str]], List[Tuple[str, List[str]]]], stem: str) -> Optional[Tuple[str, Path]]:
    """
    For a requirement that may be a single (sub,exts) or a list of alternatives,
    return (chosen_sub, existing_path) if satisfied, else None.
    """
    if isinstance(requirement, list):
        # Alternatives: return first that exists
        for sub, exts in requirement:
            p = first_existing(side_root, sub, stem, exts)
            if p is not None:
                return sub, p
        return None
    else:
        sub, exts = requirement
        p = first_existing(side_root, sub, stem, exts)
        if p is None:
            return None
        return sub, p

def validate_side_for_stem(side_root: Path, side: str, stem: str) -> Optional[Dict[str, Path]]:
    mapping: Dict[str, Path] = {}

    # required modalities (with alternatives supported)
    for req in PER_SIDE_REQUIRED_MODALITIES[side]:
        found = resolve_required(side_root, req, stem)
        if found is None:
            return None
        chosen_sub, p = found
        mapping[chosen_sub] = p

    # required calib
    for sub in CALIB_REQUIRED.get(side, []):
        p = first_existing(side_root, sub, stem, ["json"])
        if p is None:
            return None
        mapping[sub] = p

    # optional calib
    for sub in CALIB_OPTIONAL:
        p = first_existing(side_root, sub, stem, ["json"])
        if p is not None:
            mapping[sub] = p

    return mapping


def collect_valid_pairs(dataset_root: Path) -> List[Tuple[Tuple[str, str], Dict[str, Dict[str, Path]]]]:
    """
    Returns list of ((veh_stem, infra_stem), per_side_maps) for pairs that have all required files.
    """
    coop_root = dataset_root / COOP
    pairs = load_pairs_from_data_info(dataset_root / COOP)

    if not pairs:
        print(f"ERROR: Could not extract pairs from '{coop_root / 'cooperative' / 'data_info.json'}'.", file=sys.stderr)
        return []

    out: List[Tuple[Tuple[str, str], Dict[str, Dict[str, Path]]]] = []
    for veh_stem, infra_stem in pairs:
        ok = True
        per_side: Dict[str, Dict[str, Path]] = {}

        vs_root = coop_root / "vehicle-side"
        vs_map = validate_side_for_stem(vs_root, "vehicle-side", veh_stem)
        if vs_map is None:
            ok = False

        is_root = coop_root / "infrastructure-side"
        is_map = validate_side_for_stem(is_root, "infrastructure-side", infra_stem)
        if is_map is None:
            ok = False

        if ok:
            per_side["vehicle-side"] = vs_map
            per_side["infrastructure-side"] = is_map
            out.append(((veh_stem, infra_stem), per_side))

    return out


def diagnose_missing(dataset_root: Path, max_print: int = 12):
    """Show why some candidate pairs are failing."""
    coop_root = dataset_root / COOP
    pairs = load_pairs_from_data_info(dataset_root / COOP)
    if not pairs:
        print("[DIAG] No pairs could be parsed from cooperative/data_info.json")
        return
    printed = 0
    for veh_stem, infra_stem in pairs:
        issues = {}
        # vehicle-side
        vs_root = coop_root / "vehicle-side"
        miss_vs = []
        for req in PER_SIDE_REQUIRED_MODALITIES["vehicle-side"]:
            if isinstance(req, list):
                if all(first_existing(vs_root, sub, veh_stem, exts) is None for sub, exts in req):
                    alts = " OR ".join(f"{sub} ({'|'.join(exts)})" for sub, exts in req)
                    miss_vs.append(alts)
            else:
                sub, exts = req
                if first_existing(vs_root, sub, veh_stem, exts) is None:
                    miss_vs.append(f"{sub} ({'|'.join(exts)})")
        for sub in CALIB_REQUIRED.get("vehicle-side", []):
            if first_existing(vs_root, sub, veh_stem, ["json"]) is None:
                miss_vs.append(f"{sub} (.json)")
        if miss_vs:
            issues["vehicle-side"] = miss_vs

        # infrastructure-side
        is_root = coop_root / "infrastructure-side"
        miss_is = []
        for req in PER_SIDE_REQUIRED_MODALITIES["infrastructure-side"]:
            if isinstance(req, list):
                if all(first_existing(is_root, sub, infra_stem, exts) is None for sub, exts in req):
                    alts = " OR ".join(f"{sub} ({'|'.join(exts)})" for sub, exts in req)
                    miss_is.append(alts)
            else:
                sub, exts = req
                if first_existing(is_root, sub, infra_stem, exts) is None:
                    miss_is.append(f"{sub} ({'|'.join(exts)})")
        for sub in CALIB_REQUIRED.get("infrastructure-side", []):
            if first_existing(is_root, sub, infra_stem, ["json"]) is None:
                miss_is.append(f"{sub} (.json)")
        if miss_is:
            issues["infrastructure-side"] = miss_is

        if issues:
            print(f"[DIAG] veh={veh_stem}  infra={infra_stem}")
            for side, miss in issues.items():
                print(f"       {side}: " + "; ".join(miss))
            printed += 1
            if printed >= max_print:
                break


# =====================
# ====== Main    =====
# =====================
def main():
    cfg = CONFIG
    random.seed(cfg["SEED"])

    datasets_root = Path(cfg["DATASETS_ROOT"]).resolve()
    dataset_root = datasets_root / cfg["DATASET_NAME"]
    coop_root = dataset_root / COOP

    if not coop_root.is_dir():
        print(f"ERROR: '{coop_root}' not found. Make sure {cfg['DATASET_NAME']} contains '{COOP}/'.", file=sys.stderr)
        sys.exit(1)

    # Collect valid PAIRS
    valid_pairs = collect_valid_pairs(dataset_root)
    if not valid_pairs:
        print("No valid vehicle/infrastructure pairs with required modalities + per-frame calib.", file=sys.stderr)
        print("[Hint] Showing some diagnostics:")
        diagnose_missing(dataset_root, max_print=12)
        sys.exit(2)

    total_available = len(valid_pairs)
    take = min(int(cfg["COUNT"]), total_available)
    chosen = random.sample(valid_pairs, take)

    # Prepare output
    out_root = datasets_root / cfg["OUTPUT_NAME"]
    coop_out = out_root / COOP
    if not cfg["DRY_RUN"]:
        ensure_dir(coop_out / "vehicle-side")
        ensure_dir(coop_out / "infrastructure-side")

    # Auto width for filenames
    pad_width = width_for(int(cfg["START_INDEX"]) + take)

    # Copy
    global_idx = int(cfg["START_INDEX"])
    mapping = []
    for (veh_stem, infra_stem), per_side_maps in chosen:
        new_base = str(global_idx).zfill(pad_width)
        for side, mp in per_side_maps.items():
            copy_group(coop_out / side, mp, new_base, cfg["DRY_RUN"])
        mapping.append({
            "vehicle_old_basename": veh_stem,
            "infrastructure_old_basename": infra_stem,
            "new_basename": new_base
        })
        global_idx += 1

    # Manifest
    manifest = {
        "source_dataset": str(dataset_root),
        "output_dataset": str(out_root),
        "count_requested": int(cfg["COUNT"]),
        "count_copied": take,
        "start_index": int(cfg["START_INDEX"]),
        "width": pad_width,
        "seed": int(cfg["SEED"]),
        "mapping": mapping,
        "required_modalities_vehicle": PER_SIDE_REQUIRED_MODALITIES["vehicle-side"],
        "required_modalities_infra": PER_SIDE_REQUIRED_MODALITIES["infrastructure-side"],
        "required_calib": CALIB_REQUIRED,
        "optional_calib": CALIB_OPTIONAL,
        "paired_by": "cooperative/data_info.json",
    }
    if cfg["DRY_RUN"]:
        print(f"[DRY] Would write manifest to {out_root / 'compose_manifest.json'}")
    else:
        ensure_dir(out_root)
        with open(out_root / "compose_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"Done. Copied {take}/{total_available} valid pairs.")
    print(f"Output: {out_root}")
    if not cfg["DRY_RUN"]:
        print(f"Manifest: {out_root / 'compose_manifest.json'}")

if __name__ == "__main__":
    main()
