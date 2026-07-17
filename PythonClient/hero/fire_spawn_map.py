#!/usr/bin/env python3
"""
Draw fire-spread boundaries on a PGM map and emit a paste-ready
UE5 Python snippet that spawns BP_FireCube actors ONLY inside those regions.

Adds vertical layering:
- VERTICAL_LAYERS controls how many stacked layers to place (>=1).
- VERTICAL_LAYER_DZ_CM controls the per-layer vertical offset (in cm).
- REPLICATE_SEEDS_ACROSS_LAYERS controls whether seed flags are copied to all layers
  (True) or only the bottom layer gets seeds (False).

Other features:
- Fills polygon interiors with a hex lattice + mild jitter.
- Orders points from centroid -> boundary to emulate natural spread.
- Switchable seeding policy: exactly one seed per polygon OR centroid + stride.
- UE snippet hard-disables FX on non-seeds, only enables seeds at start,
  with optional timed spread waves.

Alignment helpers:
- WORLD_YAW_DEG: global rotation (degrees, +CCW) applied after translation from image px to meters.
  For "appears rotated 90° clockwise in UE", set WORLD_YAW_DEG = -90.0.
- ORIGIN_PX: optional pixel to use as the image origin (defaults to image center if None).
- MIRROR_X_UE / MIRROR_Y_UE: post-rotation reflections in UE XY (X forward, Y right).
  If your shape looks mirrored across the forward axis, set MIRROR_Y_UE=True.
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
from matplotlib.patches import Polygon as MplPolygon  # for filled shading
from PIL import Image

# =========================
# ===== USER SETTINGS =====
# =========================

# 1) Input (prints only; no file writes)
PGM_PATH = "/home/sgarimella34/multi-robot-coordination/trajectory_data/occupancy_grid_maps/customforest_0mAlt_OGM_0p5m.pgm"

# 2) Density / distribution
SPACING_M = 4.0         # target center-to-center spacing between FireCubes (meters)
JITTER_FRAC = 0.15      # random jitter as a fraction of spacing (0..0.3 recommended)
GLOBAL_SEED = 1337      # deterministic sampling/jitter and yaw
MAX_CUBES_TOTAL = 1200  # safety cap; set None to disable

# 2a) Seeding policy switch:
#    "centroid" -> exactly one seed (the centroid/closest) per polygon
#    "stride"   -> centroid is a seed AND every SEED_STRIDE-th point thereafter
SEED_POLICY = "centroid"   # "centroid" or "stride"
SEED_STRIDE = 10           # used only if SEED_POLICY == "stride" and > 1

# 2b) Vertical layering
VERTICAL_LAYERS = 2            # >= 1 (1 means just the bottom layer)
VERTICAL_LAYER_DZ_CM = 380.0   # vertical offset between layers (in centimeters)
REPLICATE_SEEDS_ACROSS_LAYERS = False  # False: only bottom layer gets seeds

# 3) Map->World parameters
RES_M = 0.5             # meters per pixel
ORIGIN_X_M = 0.0        # world X (m) at image-origin (see ORIGIN_PX)
ORIGIN_Y_M = 0.0        # world Y (m) at image-origin (see ORIGIN_PX)
FLIP_Y_WORLD = False    # set True to invert Y after turning y-down->up
WORLD_SCALE_CM = 100.0  # centimeters per meter for UE (usually 100)

# 3a) Alignment helpers
WORLD_YAW_DEG = -90.0   # +CCW; use -90.0 to correct a 90° clockwise appearance in UE
ORIGIN_PX = None        # None to use image center; or (px_x, px_y) to anchor a specific pixel

# 3b) Post-rotation reflections in UE frame
MIRROR_X_UE = False     # reflect across Y axis -> flips X
MIRROR_Y_UE = True      # reflect across X axis -> flips Y (set True for "mirrored on forward axis")

# 4) UE spawn parameters (used inside emitted snippet)
BP_REF = "/Game/Vefects/Free_Fire/Blueprints/BP_FireCube"
AUTO_ROTATE = True
ASSIGN_TAGS = True
TRACE_TOP_Z = 20000.0
TRACE_BOTTOM_Z = -20000.0
GROUND_Z_OFFSET_CM = 80.0

# 5) Optional timed spread (handled inside the emitted UE snippet)
DO_TIMED_SPREAD   = True     # if False: only seeds burn; others remain off
SPREAD_BATCH_SIZE = 8        # cubes per wave
SPREAD_INTERVAL_S = 0.75     # seconds between waves

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
        sheaf = PolyStore(polys_imgpx=[])
        self.store = sheaf
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

                # >>> Fill the closed polygon for visualization <<<
                try:
                    patch = MplPolygon(
                        self._current,
                        closed=True,
                        facecolor='tab:blue',
                        edgecolor='none',
                        alpha=0.75
                    )
                    self.ax.add_patch(patch)
                except Exception:
                    pass
                # <<< end fill >>>

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
                      world_scale_cm: float, world_yaw_deg: float = 0.0,
                      origin_px: Tuple[float, float] | None = None,
                      mirror_x_ue: bool = False, mirror_y_ue: bool = False) -> np.ndarray:
    """
    Convert image pixel coords to UE world XY (cm) with:
      - optional pixel-origin override (origin_px)
      - image y-down -> up handling
      - optional global yaw rotation (+CCW)
      - optional post-rotation reflections in UE frame (mirror_x_ue / mirror_y_ue)
    """
    # Choose image-origin (in pixels)
    if origin_px is None:
        cx = (img_w - 1) / 2.0
        cy = (img_h - 1) / 2.0
    else:
        cx, cy = origin_px

    # pixel -> meters in the map frame (x right, y up)
    dx_m = (px_xy[:, 0] - cx) * res_m
    dy_img_up_m = (cy - px_xy[:, 1]) * res_m  # image y-down -> up

    # optional flip of the world Y
    dy_m = -dy_img_up_m if flip_y else dy_img_up_m

    # global yaw (+CCW) around the chosen origin, in meters
    if abs(world_yaw_deg) > 1e-9:
        th = math.radians(world_yaw_deg)
        c, s = math.cos(th), math.sin(th)
        rx = c * dx_m - s * dy_m
        ry = s * dx_m + c * dy_m
    else:
        rx, ry = dx_m, dy_m

    # post-rotation reflections in UE frame
    if mirror_x_ue:
        rx = -rx
    if mirror_y_ue:
        ry = -ry

    # translate to UE meters
    wx_m = rx + origin_x_m
    wy_m = ry + origin_y_m

    # meters -> centimeters
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
    dx = spacing_px
    dy = spacing_px * math.sqrt(3) / 2.0

    bbox = path.get_extents().get_points()
    xmin, ymin = bbox[0]
    xmax, ymax = bbox[1]

    pts = []
    row = 0
    y = ymin
    while y <= ymax:
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
    mask_inside = np.array([path.contains_point(tuple(p)) for p in jittered], dtype=bool)
    out = points_px.copy()
    out[mask_inside] = jittered[mask_inside]
    return out

# =========================
# ===== UE5 SNIPPET =======
# =========================

UE_TEMPLATE = r'''# === AUTO-GENERATED: Paste this into the UE5 Python Console ===
import unreal, random, time

# ----- SETTINGS YOU CAN TWEAK -----
BP_REF               = {bp_ref!r}
SEED                 = {seed}
AUTO_ROTATE          = {auto_rotate}
ASSIGN_TAGS          = {assign_tags}
TRACE_TOP_Z          = {trace_top_z}
TRACE_BOTTOM_Z       = {trace_bottom_z}
GROUND_Z_OFFSET_CM   = {ground_z_offset_cm}
DO_TIMED_SPREAD      = {do_timed_spread}
SPREAD_BATCH_SIZE    = {spread_batch}
SPREAD_INTERVAL_S    = {spread_interval}
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
    raise RuntimeError(f"Could not resolve a UClass from '{bp_ref}'.")

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

def ground_project_with_offset(x: float, y: float, z_offset_cm: float) -> unreal.Vector:
    start = unreal.Vector(x, y, TRACE_TOP_Z)
    end   = unreal.Vector(x, y, TRACE_BOTTOM_Z)
    hit, hr = trace_down(start, end)
    if hit:
        p = hr.location
        return unreal.Vector(p.x, p.y, p.z + GROUND_Z_OFFSET_CM + z_offset_cm)
    return unreal.Vector(x, y, 0.0 + GROUND_Z_OFFSET_CM + z_offset_cm)

# ---------- FX control (hard override) ----------
def _get_fx_components(actor):
    fx = []
    try:
        for c in actor.get_components_by_class(unreal.NiagaraComponent):
            fx.append(c)
    except Exception:
        pass
    try:
        for c in actor.get_components_by_class(unreal.ParticleSystemComponent):
        ...
        pass
    return fx

def _deactivate_fx(actor):
    for c in _get_fx_components(actor):
        for m in ("deactivate", "deactivate_immediately", "deactivate_system"):
            if hasattr(c, m):
                try:
                    getattr(c, m)()
                    break
                except Exception:
                    pass
        try: c.set_editor_property("auto_activate", False)
        except Exception: pass
        try: c.set_editor_property("visible", False)
        except Exception: pass

def _activate_fx(actor):
    for c in _get_fx_components(actor):
        try: c.set_editor_property("visible", True)
        except Exception: pass
        try: c.set_editor_property("auto_activate", True)
        except Exception: pass
        for m in ("activate", "activate_system", "reinitialize_system"):
            if hasattr(c, m):
                try:
                    getattr(c, m)()
                    break
                except Exception:
                    pass

def spawn_fire(loc: unreal.Vector, idx: int, is_seed: bool) -> unreal.Actor:
    yaw = random.uniform(0, 360) if AUTO_ROTATE else 0.0
    rot = unreal.Rotator(0.0, yaw, 0.0)
    actor = lvl.spawn_actor_from_class(BP_CLASS, loc, rot)
    actor.set_actor_label(f"BP_FireCube_{idx+1}", True)

    # Replace tags so class defaults can't keep StartFire on non-seeds
    if ASSIGN_TAGS:
        actor.tags = (["Flammable", "StartFire"] if is_seed else ["Flammable"])

    # Hard override FX state:
    if is_seed:
        _activate_fx(actor)     # only seeds burn now
    else:
        _deactivate_fx(actor)   # everyone else OFF regardless of BP defaults

    return actor

# (x_cm, y_cm, z_offset_cm, is_seed) — ordered centroid->boundary per polygon, per layer
FIRE_POINTS = [
{points_block}
]

actors = []
idx = 0
for (x_cm, y_cm, z_off_cm, is_seed) in FIRE_POINTS:
    loc = ground_project_with_offset(x_cm, y_cm, z_off_cm)
    a = spawn_fire(loc, idx, bool(is_seed))
    actors.append((bool(is_seed), a))
    idx += 1

print(f"Spawned {idx} BP_FireCube actors. Seeds burn; non-seeds off.")

# Timed spread: turn ON FX for non-seeds in waves (in given order)
if DO_TIMED_SPREAD:
    print("Timed spread active...")
    pending = [a for (is_seed, a) in actors if not is_seed]
    for i in range(0, len(pending), SPREAD_BATCH_SIZE):
        batch = pending[i:i+SPREAD_BATCH_SIZE]
        for a in batch:
            _activate_fx(a)
        time.sleep(max(0.0, float(SPREAD_INTERVAL_S)))

print("Done. Fire spreads outward from each polygon centroid (with vertical layers).")
# === END AUTO-GENERATED ===
'''

def emit_ue_snippet(points_world_cm_with_seed_and_z: np.ndarray):
    # points_world_cm_with_seed_and_z: Nx4 [x_cm, y_cm, z_offset_cm, seed_flag]
    lines = [f"    ({x:.2f}, {y:.2f}, {z:.2f}, {int(s)})," for x, y, z, s in points_world_cm_with_seed_and_z]
    points_block = "\n".join(lines)

    txt = UE_TEMPLATE.format(
        bp_ref=BP_REF,
        seed=GLOBAL_SEED,
        auto_rotate="True" if AUTO_ROTATE else "False",
        assign_tags="True" if ASSIGN_TAGS else "False",
        trace_top_z=TRACE_TOP_Z,
        trace_bottom_z=TRACE_BOTTOM_Z,
        ground_z_offset_cm=GROUND_Z_OFFSET_CM,
        do_timed_spread="True" if DO_TIMED_SPREAD else "False",
        spread_batch=int(SPREAD_BATCH_SIZE),
        spread_interval=float(SPREAD_INTERVAL_S),
        points_block=points_block,
    )
    print("\n" + "="*72)
    print("PASTE THE FOLLOWING INTO UE5 PYTHON CONSOLE:\n")
    print(txt)
    print("="*72 + "\n")

# =========================
# ===== MAIN & SEEDS ======
# =========================

def compute_seed_flags(n: int, policy: str, stride: int) -> np.ndarray:
    """
    Build a 0/1 array of length n indicating which positions are seeds.
    Assumes points are already sorted centroid->outward; index 0 is centroid.
    """
    seeds = np.zeros((n,), dtype=int)
    if n == 0:
        return seeds
    seeds[0] = 1  # centroid always a seed
    if policy.lower() == "stride":
        if stride and stride > 1:
            seeds[stride::stride] = 1
    elif policy.lower() == "centroid":
        pass
    else:
        raise ValueError("SEED_POLICY must be 'centroid' or 'stride'")
    return seeds

def main():
    # Deterministic behavior
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)

    # Validate vertical layer settings
    if VERTICAL_LAYERS < 1:
        print("[ERROR] VERTICAL_LAYERS must be >= 1.")
        sys.exit(1)

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
    all_points_out = []   # will hold Nx4: [x_cm, y_cm, z_offset_cm, seed_flag]
    total_count = 0

    for poly_px, path in zip(polys_px, paths):
        # Generate hex lattice inside the polygon
        pts_px = hex_fill_points_inside(path, spacing_px)

        # Apply mild jitter
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

        # Convert to world centimeters (XY) with rotation, optional pixel-origin, and mirroring
        pts_cm_xy = imgpx_to_world_cm(
            pts_px_sorted, W, H,
            res_m=RES_M,
            origin_x_m=ORIGIN_X_M,
            origin_y_m=ORIGIN_Y_M,
            flip_y=FLIP_Y_WORLD,
            world_scale_cm=WORLD_SCALE_CM,
            world_yaw_deg=WORLD_YAW_DEG,
            origin_px=ORIGIN_PX,
            mirror_x_ue=MIRROR_X_UE,
            mirror_y_ue=MIRROR_Y_UE
        )

        # Seed flags on the base layer ordering
        n_base = pts_cm_xy.shape[0]
        base_seed_flags = compute_seed_flags(n_base, SEED_POLICY, SEED_STRIDE)

        # Create vertical layers
        for layer_idx in range(VERTICAL_LAYERS):
            z_off = layer_idx * float(VERTICAL_LAYER_DZ_CM)

            # Decide seeds for this layer
            if REPLICATE_SEEDS_ACROSS_LAYERS or layer_idx == 0:
                seed_flags = base_seed_flags.copy()
            else:
                seed_flags = np.zeros_like(base_seed_flags)

            # Respect MAX_CUBES_TOTAL
            n_this_layer = n_base
            if MAX_CUBES_TOTAL is not None and total_count + n_this_layer > MAX_CUBES_TOTAL:
                remaining = MAX_CUBES_TOTAL - total_count
                if remaining <= 0:
                    break
                n_this_layer = remaining
                seed_flags = seed_flags[:n_this_layer]
                pts_layer_xy = pts_cm_xy[:n_this_layer]
            else:
                pts_layer_xy = pts_cm_xy

            z_col = np.full((n_this_layer, 1), z_off, dtype=float)
            s_col = seed_flags[:n_this_layer].reshape(-1, 1).astype(int)
            block = np.concatenate([pts_layer_xy[:n_this_layer], z_col, s_col], axis=1)  # Nx4
            all_points_out.append(block)
            total_count += n_this_layer

            if MAX_CUBES_TOTAL is not None and total_count >= MAX_CUBES_TOTAL:
                break

        if MAX_CUBES_TOTAL is not None and total_count >= MAX_CUBES_TOTAL:
            break

    if len(all_points_out) == 0:
        print("[ERROR] No points generated. Try lowering SPACING_M.")
        sys.exit(1)

    all_points_out = np.concatenate(all_points_out, axis=0)  # (N, 4)

    print(f"[INFO] Generated {all_points_out.shape[0]} spawn points at spacing ~{SPACING_M} m.")
    print(f"[INFO] Seeding policy: {SEED_POLICY} (stride={SEED_STRIDE if SEED_POLICY=='stride' else 'n/a'})")
    print(f"[INFO] Vertical layers: {VERTICAL_LAYERS} @ {VERTICAL_LAYER_DZ_CM} cm "
          f"(replicate_seeds={REPLICATE_SEEDS_ACROSS_LAYERS})")
    print(f"[INFO] Alignment: WORLD_YAW_DEG={WORLD_YAW_DEG}°, ORIGIN_PX={ORIGIN_PX}, "
          f"FLIP_Y_WORLD={FLIP_Y_WORLD}, MIRROR_X_UE={MIRROR_X_UE}, MIRROR_Y_UE={MIRROR_Y_UE}")

    # Emit UE snippet (print only)
    emit_ue_snippet(points_world_cm_with_seed_and_z=all_points_out)

if __name__ == "__main__":
    main()
