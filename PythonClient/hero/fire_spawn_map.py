#!/usr/bin/env python3
"""
Draw fire-spread boundaries on a PGM map and emit a paste-ready
UE5 Python snippet that spawns BP_FireCube actors ONLY inside those regions.

IMPROVEMENTS vs previous:
- Fills the polygon INTERIOR using a hexagonal lattice (best packing) with mild jitter.
- Orders points from centroid -> boundary to emulate a natural spread.
- Seeds are selected per polygon (first and every SEED_STRIDE thereafter).
- "Not over the top": density controlled via SPACING_M and optional MAX_CUBES_TOTAL.

HOW TO USE
----------
1) Adjust USER SETTINGS below (paths, map->world, density, UE params).
2) Run: python fire_spawn_map.py
3) Draw one or more polygons:
   - Left-click to add vertices
   - Press ENTER to finish a polygon
   - Draw more if needed
   - Close the window when done
4) Copy the printed UE5 snippet and paste it into the UE5 Python Console.
"""

import sys
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from PIL import Image

# =========================
# ===== USER SETTINGS =====
# =========================

# 1) Input (prints only; no file writes)
PGM_PATH = "/home/sgarimella34/multi-robot-coordination/trajectory_data/occupancy_grid_maps/customforest_0mAlt_OGM_0p5m.pgm"

# 2) Density / distribution
SPACING_M = 3.0         # target center-to-center spacing between FireCubes (meters)
JITTER_FRAC = 0.15      # random jitter as a fraction of spacing (0..0.3 recommended)
SEED_STRIDE = 10        # per polygon: first point is seed, then every Nth (distance order)
GLOBAL_SEED = 1337      # deterministic sampling/jitter and yaw
MAX_CUBES_TOTAL = 1200  # safety cap; set None to disable

# 3) Map->World parameters
RES_M = 0.5             # meters per pixel
ORIGIN_X_M = 0.0        # world X (m) at image center
ORIGIN_Y_M = 0.0        # world Y (m) at image center
FLIP_Y_WORLD = False    # set True if UE Y is inverted relative to map's up
WORLD_SCALE_CM = 100.0  # centimeters per meter for UE (usually 100)

# 4) UE spawn parameters
BP_REF = "/Game/Vefects/Free_Fire/Blueprints/BP_FireCube"
AUTO_ROTATE = True
ASSIGN_TAGS = True
TRACE_TOP_Z = 20000.0
TRACE_BOTTOM_Z = -20000.0
GROUND_Z_OFFSET_CM = 80.0

# =========================
# ====== CORE LOGIC =======
# =========================

@dataclass
class PolyStore:
    polys_imgpx: List[np.ndarray]  # list of (N, 2) arrays in IMAGE PIXELS (x,y)

class PolyCollector:
    """
    Cross-version PolygonSelector wrapper:
    - Matplotlib ≥3.8 uses 'props'/'handle_props'
    - Older versions use 'lineprops'/'markerprops'
    Recreates the selector after pressing Enter to avoid touching private attrs.
    """
    def __init__(self, ax):
        self.ax = ax
        self.store = PolyStore(polys_imgpx=[])
        self._current = None
        self.selector = self._make_selector()
        self.cid_key = ax.figure.canvas.mpl_connect('key_press_event', self.onkey)

    def _make_selector(self):
        try:
            # New API (mpl >= 3.8)
            return PolygonSelector(
                self.ax, self.onselect, useblit=True,
                props=dict(linewidth=2, alpha=0.9),
                handle_props=dict(markersize=4, alpha=0.9),
            )
        except TypeError:
            # Old API fallback
            return PolygonSelector(
                self.ax, self.onselect, useblit=True,
                lineprops=dict(linewidth=2, alpha=0.9),
                markerprops=dict(markersize=4, alpha=0.9),
            )

    def onselect(self, verts):
        self._current = np.array(verts, dtype=float)

    def onkey(self, event):
        if event.key == 'enter':
            if self._current is not None and len(self._current) >= 3:
                self.store.polys_imgpx.append(self._current.copy())
                print(f"[INFO] Stored polygon with {len(self._current)} points.")
                self._current = None
                try:
                    self.selector.disconnect_events()
                except Exception:
                    pass
                self.selector = self._make_selector()
                self.ax.figure.canvas.draw_idle()
            else:
                print("[WARN] Need at least 3 points; polygon not stored.")

    def disconnect(self):
        try:
            self.ax.figure.canvas.mpl_disconnect(self.cid_key)
        except Exception:
            pass
        try:
            self.selector.disconnect_events()
        except Exception:
            pass

def poly_area(path: Path) -> float:
    v = path.vertices
    x = v[:,0]; y = v[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def imgpx_to_world_cm(px_xy: np.ndarray, img_w: int, img_h: int, res_m: float,
                      origin_x_m: float, origin_y_m: float, flip_y: bool,
                      world_scale_cm: float) -> np.ndarray:
    """Convert image pixel coords (x right, y down) to UE world XY in centimeters."""
    cx = (img_w - 1) / 2.0
    cy = (img_h - 1) / 2.0
    dx = (px_xy[:,0] - cx) * res_m
    dy_img_up = (cy - px_xy[:,1]) * res_m  # image y down -> up

    if flip_y:
        wy_m = -dy_img_up + origin_y_m
    else:
        wy_m =  dy_img_up + origin_y_m
    wx_m = dx + origin_x_m

    wx_cm = wx_m * world_scale_cm
    wy_cm = wy_m * world_scale_cm
    return np.stack([wx_cm, wy_cm], axis=1)

def centroid_of_polygon(poly_px: np.ndarray) -> Tuple[float, float]:
    """Centroid in pixel coordinates (Shoelace formula)."""
    x = poly_px[:,0]; y = poly_px[:,1]
    a = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    if abs(a) < 1e-9:
        return float(x.mean()), float(y.mean())
    a *= 0.5
    cx = (np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)))/(6.0*a)
    cy = (np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)))/(6.0*a)
    return float(cx), float(cy)

def hex_fill_points_inside(path: Path, spacing_px: float) -> np.ndarray:
    """
    Generate a hexagonal lattice of points inside polygon 'path' with given spacing in pixels.
    Returns Nx2 array in pixel coords.
    """
    # Hex grid parameters
    dx = spacing_px
    dy = spacing_px * math.sqrt(3) / 2.0

    bbox = path.get_extents().get_points()
    xmin, ymin = bbox[0]
    xmax, ymax = bbox[1]

    # Build grid
    pts = []
    row = 0
    y = ymin
    while y <= ymax:
        # offset every other row by dx/2
        x_start = xmin + (dx * 0.5 if (row % 2) == 1 else 0.0)
        x = x_start
        while x <= xmax:
            if path.contains_point((x, y)):
                pts.append((x, y))
            x += dx
        y += dy
        row += 1

    if not pts:
        return np.zeros((0,2), dtype=float)
    return np.array(pts, dtype=float)

def apply_jitter(points_px: np.ndarray, spacing_px: float, frac: float, path: Path) -> np.ndarray:
    """
    Apply small random jitter to avoid perfect lattice look; keep jittered point
    inside the polygon by snapping back if needed.
    """
    if points_px.size == 0 or frac <= 0.0:
        return points_px
    mag = spacing_px * frac
    jitter = (np.random.rand(*points_px.shape) - 0.5) * 2.0 * mag
    jittered = points_px + jitter

    # If jittered falls outside, keep original
    mask_inside = np.array([path.contains_point(tuple(p)) for p in jittered], dtype=bool)
    out = points_px.copy()
    out[mask_inside] = jittered[mask_inside]
    return out

UE_TEMPLATE = r'''# === AUTO-GENERATED: Paste this into the UE5 Python Console ===
import unreal, random

# ----- SETTINGS YOU CAN TWEAK -----
BP_REF               = {bp_ref!r}
SEED                 = {seed}
AUTO_ROTATE          = {auto_rotate}
ASSIGN_TAGS          = {assign_tags}
TRACE_TOP_Z          = {trace_top_z}
TRACE_BOTTOM_Z       = {trace_bottom_z}
GROUND_Z_OFFSET_CM   = {ground_z_offset_cm}
# ----------------------------------

world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
lvl   = unreal.EditorLevelLibrary
random.seed(SEED)

def resolve_bp_class(bp_ref: str) -> unreal.Class:
    try:
        cls = unreal.EditorAssetLibrary.load_blueprint_class(bp_ref)
        if isinstance(cls, unreal.Class):
            return cls
    except Exception:
        pass
    def _try(path: str):
        obj = unreal.load_asset(path)
        if obj is None:
            return None
        if hasattr(obj, "generated_class"):
            return obj.generated_class
        if isinstance(obj, unreal.Class):
            return obj
        return None
    trials = []
    if bp_ref.endswith(".BP_FireCube") or bp_ref.endswith(".BP_FireCube_C"):
        trials.append(bp_ref)
    else:
        trials.extend([bp_ref + ".BP_FireCube", bp_ref + ".BP_FireCube_C"])
    for t in trials:
        cls = _try(t)
        if isinstance(cls, unreal.Class):
            return cls
    raise RuntimeError(f"Could not resolve a UClass from '{{bp_ref}}'.")

BP_CLASS = resolve_bp_class(BP_REF)

OBJECT_TYPES = [
    unreal.ObjectTypeQuery.OBJECT_TYPE_QUERY1,  # WorldStatic
    unreal.ObjectTypeQuery.OBJECT_TYPE_QUERY2   # WorldDynamic
]

def trace_down(start: unreal.Vector, end: unreal.Vector):
    res = unreal.SystemLibrary.line_trace_single_for_objects(
        world, start, end, OBJECT_TYPES, False, [], unreal.DrawDebugTrace.NONE, False
    )
    if isinstance(res, tuple):
        return res[0], res[1]
    hr = res
    hit = getattr(hr, "blocking_hit", False) or getattr(hr, "bBlockingHit", False)
    return hit, hr

def ground_project(x: float, y: float) -> unreal.Vector:
    start = unreal.Vector(x, y, TRACE_TOP_Z)
    end   = unreal.Vector(x, y, TRACE_BOTTOM_Z)
    hit, hr = trace_down(start, end)
    if hit:
        p = hr.location
        return unreal.Vector(p.x, p.y, p.z + GROUND_Z_OFFSET_CM)
    return unreal.Vector(x, y, 0.0 + GROUND_Z_OFFSET_CM)

def spawn_fire(loc: unreal.Vector, idx: int, is_seed: bool=False) -> unreal.Actor:
    yaw = random.uniform(0, 360) if AUTO_ROTATE else 0.0
    rot = unreal.Rotator(0.0, yaw, 0.0)
    actor = lvl.spawn_actor_from_class(BP_CLASS, loc, rot)
    actor.set_actor_label(f"BP_FireCube_{{idx+1}}", True)
    if ASSIGN_TAGS:
        tags = list(actor.tags)
        if "Flammable" not in tags: tags.append("Flammable")
        if is_seed and "StartFire" not in tags: tags.append("StartFire")
        actor.tags = tags
    return actor

# (x_cm, y_cm, is_seed) — ordered from centroid outward per polygon
FIRE_POINTS = [
{points_block}
]

idx = 0
for (x_cm, y_cm, is_seed) in FIRE_POINTS:
    loc = ground_project(x_cm, y_cm)
    spawn_fire(loc, idx, is_seed=bool(is_seed))
    idx += 1

print(f"Spawned {{idx}} BP_FireCube actors from filled polygon regions.")
# === END AUTO-GENERATED ===
'''

def emit_ue_snippet(points_world_cm_with_seed: np.ndarray,
                    bp_ref: str, seed: int, auto_rotate: bool, assign_tags: bool,
                    trace_top_z: float, trace_bottom_z: float, ground_z_offset_cm: float):
    # points_world_cm_with_seed: Nx3 [x_cm, y_cm, seed_flag]
    lines = [f"    ({x:.2f}, {y:.2f}, {int(s)})," for x, y, s in points_world_cm_with_seed]
    points_block = "\n".join(lines)

    txt = UE_TEMPLATE.format(
        bp_ref=bp_ref,
        seed=seed,
        auto_rotate="True" if auto_rotate else "False",
        assign_tags="True" if assign_tags else "False",
        trace_top_z=trace_top_z,
        trace_bottom_z=trace_bottom_z,
        ground_z_offset_cm=ground_z_offset_cm,
        points_block=points_block,
    )
    print("\n" + "="*72)
    print("PASTE THE FOLLOWING INTO UE5 PYTHON CONSOLE:\n")
    print(txt)
    print("="*72 + "\n")

def main():
    # Deterministic behavior
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)

    # Load PGM
    img = Image.open(PGM_PATH)
    img = np.array(img)
    if img.ndim == 3:
        img = img[...,0]
    H, W = img.shape

    # Draw polygons
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap="gray", origin="upper")
    ax.set_title("Draw polygons: click to add vertices, ENTER to finish one.\n"
                 "Draw multiple polygons, then close the window when done.")
    ax.set_xlim(0, W-1)
    ax.set_ylim(H-1, 0)

    collector = PolyCollector(ax)
    plt.show(block=True)
    collector.disconnect()

    polys_px = collector.store.polys_imgpx
    if len(polys_px) == 0:
        print("[ERROR] No polygons were drawn. Exiting.")
        sys.exit(1)

    print(f"[INFO] You drew {len(polys_px)} polygon(s).")

    # Convert to Paths
    paths = [Path(p) for p in polys_px]

    # Hex fill each polygon with requested spacing (in pixels)
    spacing_px = SPACING_M / RES_M
    all_points_cm_with_seed = []
    total_count = 0

    for poly_px, path in zip(polys_px, paths):
        # Generate hex lattice inside the polygon
        pts_px = hex_fill_points_inside(path, spacing_px)

        # Apply mild jitter to avoid a too-regular look
        pts_px = apply_jitter(pts_px, spacing_px, JITTER_FRAC, path)

        if pts_px.shape[0] == 0:
            # Fall back: at least drop one point at centroid
            cx, cy = centroid_of_polygon(poly_px)
            pts_px = np.array([[cx, cy]], dtype=float)

        # Order from centroid outward to emulate spread
        cx, cy = centroid_of_polygon(poly_px)
        dists = np.hypot(pts_px[:,0] - cx, pts_px[:,1] - cy)
        order = np.argsort(dists)
        pts_px_sorted = pts_px[order]

        # Convert to world centimeters
        pts_cm = imgpx_to_world_cm(
            pts_px_sorted, W, H,
            res_m=RES_M,
            origin_x_m=ORIGIN_X_M,
            origin_y_m=ORIGIN_Y_M,
            flip_y=FLIP_Y_WORLD,
            world_scale_cm=WORLD_SCALE_CM
        )

        # Seed pattern: first is a seed, then every SEED_STRIDE
        n = pts_cm.shape[0]
        seed_flags = np.zeros((n,), dtype=int)
        seed_flags[0] = 1
        if SEED_STRIDE > 1:
            seed_flags[SEED_STRIDE::SEED_STRIDE] = 1

        # Accumulate, respecting MAX_CUBES_TOTAL
        if MAX_CUBES_TOTAL is not None and total_count + n > MAX_CUBES_TOTAL:
            remaining = MAX_CUBES_TOTAL - total_count
            if remaining <= 0:
                break
            pts_cm = pts_cm[:remaining]
            seed_flags = seed_flags[:remaining]
            n = remaining

        block = np.concatenate([pts_cm, seed_flags[:,None]], axis=1)  # Nx3
        all_points_cm_with_seed.append(block)
        total_count += n

        if MAX_CUBES_TOTAL is not None and total_count >= MAX_CUBES_TOTAL:
            break

    if len(all_points_cm_with_seed) == 0:
        print("[ERROR] No points generated. Try lowering SPACING_M.")
        sys.exit(1)

    all_points_cm_with_seed = np.concatenate(all_points_cm_with_seed, axis=0)

    print(f"[INFO] Generated {all_points_cm_with_seed.shape[0]} spawn points at spacing ~{SPACING_M} m.")

    # Emit UE snippet (print only)
    emit_ue_snippet(
        points_world_cm_with_seed=all_points_cm_with_seed,
        bp_ref=BP_REF,
        seed=GLOBAL_SEED,
        auto_rotate=AUTO_ROTATE,
        assign_tags=ASSIGN_TAGS,
        trace_top_z=TRACE_TOP_Z,
        trace_bottom_z=TRACE_BOTTOM_Z,
        ground_z_offset_cm=GROUND_Z_OFFSET_CM
    )

if __name__ == "__main__":
    main()
