import unreal, random, math

# ===== USER SETTINGS =====
BP_REF              = "/Game/Vefects/Free_Fire/Blueprints/BP_FireCube"  # Copy Reference, remove ".BP_FireCube"
TOTAL_COUNT         = 40            # total cubes
NUM_CLUSTERS        = 5             # number of clusters
STEP_MAX            = 250.0         # <= 200–250 units between neighbors
SEED                = 1337          # random seed

# Spawn region (XY) for cluster centers
AREA_MIN_XY         = (-6000.0, -6000.0)
AREA_MAX_XY         = ( 6000.0,  6000.0)

# Ground projection trace (Z range)
TRACE_TOP_Z         = 20000.0
TRACE_BOTTOM_Z      = -20000.0

# >>> Height above ground (in cm). Set this to fix below-ground spawns.
GROUND_Z_OFFSET_CM  = 80.0

# Misc
AUTO_ROTATE         = True          # random yaw
ASSIGN_TAGS         = True          # "Flammable" for all, "StartFire" for cluster seeds
# =========================

# Editor world (non-deprecated)
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
lvl   = unreal.EditorLevelLibrary
random.seed(SEED)

# Use raw ObjectTypeQuery enums (typical defaults: 1=WorldStatic, 2=WorldDynamic)
OBJECT_TYPES = [
    unreal.ObjectTypeQuery.OBJECT_TYPE_QUERY1,
    unreal.ObjectTypeQuery.OBJECT_TYPE_QUERY2
]

# ---------- Blueprint class resolution (robust) ----------
def resolve_bp_class(bp_ref: str) -> unreal.Class:
    """
    Preferred: EditorAssetLibrary.load_blueprint_class('/Game/.../BP_FireCube')
    Fallbacks: load_asset('/Game/.../BP_FireCube.BP_FireCube') → .generated_class
               or '/Game/.../BP_FireCube.BP_FireCube_C' directly.
    """
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

    raise RuntimeError(f"Could not resolve a UClass from '{bp_ref}'. Use Content Browser → Copy Reference and set BP_REF accordingly.")

BP_CLASS = resolve_bp_class(BP_REF)

# ---------- Trace helpers (handles different return signatures) ----------
def trace_down(start: unreal.Vector, end: unreal.Vector):
    """Call line_trace_single_for_objects and normalize return to (hit_bool, HitResult)."""
    res = unreal.SystemLibrary.line_trace_single_for_objects(
        world, start, end, OBJECT_TYPES, False, [], unreal.DrawDebugTrace.NONE, False
    )
    if isinstance(res, tuple):  # some builds return (bool, HitResult)
        return res[0], res[1]
    hr = res                   # some builds return HitResult only
    hit = getattr(hr, "blocking_hit", False) or getattr(hr, "bBlockingHit", False)
    return hit, hr

def ground_project(x: float, y: float) -> unreal.Vector:
    """Project XY to ground and add the adjustable offset."""
    start = unreal.Vector(x, y, TRACE_TOP_Z)
    end   = unreal.Vector(x, y, TRACE_BOTTOM_Z)
    hit, hr = trace_down(start, end)
    if hit:
        p = hr.location
        return unreal.Vector(p.x, p.y, p.z + GROUND_Z_OFFSET_CM)
    # Fallback if nothing hit: place at Z=offset
    return unreal.Vector(x, y, 0.0 + GROUND_Z_OFFSET_CM)

# ---------- Spawn helpers ----------
def rand_unit_vector_2d() -> unreal.Vector:
    ang = random.uniform(0.0, math.tau)
    return unreal.Vector(math.cos(ang), math.sin(ang), 0.0)

def spawn_fire(loc: unreal.Vector, idx: int, is_seed: bool=False) -> unreal.Actor:
    yaw = random.uniform(0, 360) if AUTO_ROTATE else 0.0
    rot = unreal.Rotator(0.0, yaw, 0.0)
    actor = lvl.spawn_actor_from_class(BP_CLASS, loc, rot)  # requires a UClass
    actor.set_actor_label(f"BP_FireCube_{idx+1}", True)     # sequential unique name
    if ASSIGN_TAGS:
        tags = list(actor.tags)
        if "Flammable" not in tags: tags.append("Flammable")
        if is_seed and "StartFire" not in tags: tags.append("StartFire")
        actor.tags = tags
    return actor

# ---------- Cluster sizing ----------
if NUM_CLUSTERS <= 0:
    raise ValueError("NUM_CLUSTERS must be >= 1")

cluster_sizes = [TOTAL_COUNT // NUM_CLUSTERS] * NUM_CLUSTERS
for i in range(TOTAL_COUNT % NUM_CLUSTERS):
    cluster_sizes[i] += 1
cluster_sizes = [c for c in cluster_sizes if c > 0]

# ---------- Cluster centers ----------
centers = []
for _ in range(len(cluster_sizes)):
    cx = random.uniform(AREA_MIN_XY[0], AREA_MAX_XY[0])
    cy = random.uniform(AREA_MIN_XY[1], AREA_MAX_XY[1])
    centers.append(ground_project(cx, cy))

# ---------- Spawn clusters ----------
idx = 0
for ci, count in enumerate(cluster_sizes):
    if count <= 0:
        continue

    seed_loc = centers[ci]
    spawn_fire(seed_loc, idx, is_seed=True); idx += 1

    frontier = [seed_loc]
    for _ in range(count - 1):
        base = random.choice(frontier)
        step = random.uniform(0.0, STEP_MAX)
        off  = rand_unit_vector_2d() * step
        loc  = ground_project(base.x + off.x, base.y + off.y)
        spawn_fire(loc, idx, is_seed=False); idx += 1
        frontier.append(loc)

print(f"Spawned {idx} BP_FireCube actors across {len(cluster_sizes)} cluster(s) with seed {SEED}, offset {GROUND_Z_OFFSET_CM} cm.")
