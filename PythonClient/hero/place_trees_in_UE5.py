import unreal
import random
import math

# ------------------------------------------------------------
# UE 5.2.1 Forest cluster spawner (StaticMeshActor)
# - Circular-ish cluster around ORIGIN
# - Enforces spacing in [MIN_SPACING_M, MAX_SPACING_M]
# - Trees vertical, random yaw only
# - Optional ground snap via SystemLibrary.line_trace_single
# ------------------------------------------------------------

TREE_ASSET_PATH = "/Game/PN_interactiveSpruceForest/Meshes/half/low/spruce_half_01_low.spruce_half_01_low"

NUM_TREES       = 2000
MIN_SPACING_M   = 2.0
MAX_SPACING_M   = 4.0
SUFFIX          = "_TREE"

ORIGIN = unreal.Vector(0.0, 0.0, 0.0)

SNAP_TO_GROUND   = True
TRACE_START_Z_CM = 200000.0
TRACE_END_Z_CM   = -200000.0
Z_OFFSET_CM      = 0.0

TRACE_TYPE = unreal.TraceTypeQuery.TRACE_TYPE_QUERY1

MAX_ATTEMPTS_PER_TREE = 2000
CLUSTER_RADIUS_M = None  # set e.g. 30.0 to force radius

def cm(meters: float) -> float:
    return meters * 100.0

def estimate_cluster_radius_cm(n: int, avg_spacing_cm: float, packing: float = 0.55, margin: float = 1.15) -> float:
    area_cm2 = n * (avg_spacing_cm * avg_spacing_cm) / max(packing, 1e-6)
    radius_cm = math.sqrt(area_cm2 / math.pi)
    return radius_cm * margin

def random_point_in_disk(radius_cm: float) -> unreal.Vector:
    u = random.random()
    r = math.sqrt(u) * radius_cm
    theta = random.uniform(0.0, 2.0 * math.pi)
    x = ORIGIN.x + r * math.cos(theta)
    y = ORIGIN.y + r * math.sin(theta)
    return unreal.Vector(x, y, ORIGIN.z)

def distance_xy_cm(a: unreal.Vector, b: unreal.Vector) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)

def _hitresult_to_bool(hit_result) -> bool:
    if hit_result is None:
        return False
    if hasattr(hit_result, "blocking_hit"):
        try:
            return bool(hit_result.blocking_hit)
        except Exception:
            pass
    if hasattr(hit_result, "is_valid_blocking_hit"):
        try:
            return bool(hit_result.is_valid_blocking_hit())
        except Exception:
            pass
    return False

def snap_z_to_ground(world, x: float, y: float, default_z: float) -> float:
    if not SNAP_TO_GROUND:
        return default_z

    start = unreal.Vector(x, y, TRACE_START_Z_CM)
    end   = unreal.Vector(x, y, TRACE_END_Z_CM)

    res = unreal.SystemLibrary.line_trace_single(
        world_context_object=world,
        start=start,
        end=end,
        trace_channel=TRACE_TYPE,
        trace_complex=False,
        actors_to_ignore=[],
        draw_debug_type=unreal.DrawDebugTrace.NONE,
        ignore_self=True
    )

    hit_bool = False
    hit_result = None

    if isinstance(res, tuple) and len(res) == 2:
        hit_bool, hit_result = bool(res[0]), res[1]
    else:
        hit_result = res
        hit_bool = _hitresult_to_bool(hit_result)

    if hit_bool and hit_result and hasattr(hit_result, "location"):
        return hit_result.location.z + Z_OFFSET_CM

    return default_z

# ----------------------------
# Main
# ----------------------------

unreal_editor_subsys = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
world = unreal_editor_subsys.get_editor_world()
if not world:
    raise RuntimeError("Editor world is not available. Make sure a level is open and you are not in PIE.")

tree_mesh = unreal.load_asset(TREE_ASSET_PATH)
if not tree_mesh:
    unreal.log_error(f"Failed to load asset: {TREE_ASSET_PATH}")
    raise RuntimeError("Tree mesh asset not found. Paste the exact Copy Reference path.")

actor_subsys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

min_spacing_cm = cm(MIN_SPACING_M)
max_spacing_cm = cm(MAX_SPACING_M)
avg_spacing_cm = 0.5 * (min_spacing_cm + max_spacing_cm)

if CLUSTER_RADIUS_M is None:
    cluster_radius_cm = estimate_cluster_radius_cm(NUM_TREES, avg_spacing_cm=avg_spacing_cm)
else:
    cluster_radius_cm = cm(CLUSTER_RADIUS_M)

placed_points = []

unreal.log(f"Spawning {NUM_TREES} trees, radius ~ {cluster_radius_cm/100.0:.2f} m, spacing {MIN_SPACING_M:.2f}-{MAX_SPACING_M:.2f} m")
unreal.log(f"Ground snap: {SNAP_TO_GROUND}, TraceType: {TRACE_TYPE}")

spawned = 0

with unreal.ScopedEditorTransaction("Spawn Forest Cluster"):
    for i in range(NUM_TREES):
        required_spacing_cm = random.uniform(min_spacing_cm, max_spacing_cm)

        candidate = None
        for _attempt in range(MAX_ATTEMPTS_PER_TREE):
            p = random_point_in_disk(cluster_radius_cm)

            ok = True
            for q in placed_points:
                if distance_xy_cm(p, q) < required_spacing_cm:
                    ok = False
                    break

            if ok:
                candidate = p
                break

        if candidate is None:
            unreal.log_warning(
                f"Stopped early at {i}/{NUM_TREES}: cannot satisfy spacing. Increase CLUSTER_RADIUS_M or lower NUM_TREES."
            )
            break

        z = snap_z_to_ground(world, candidate.x, candidate.y, default_z=ORIGIN.z)

        yaw = random.uniform(0.0, 360.0)

        # Critical fix: use keyword args so the random rotation is always YAW (vertical / world Z)
        rot = unreal.Rotator(pitch=0.0, yaw=yaw, roll=0.0)

        loc = unreal.Vector(candidate.x, candidate.y, z)

        actor = actor_subsys.spawn_actor_from_object(tree_mesh, loc, rot)
        if not actor:
            unreal.log_error(f"Spawn failed at index {i+1}.")
            continue

        actor.set_actor_label(f"{tree_mesh.get_name()}_{i+1}{SUFFIX}")
        placed_points.append(unreal.Vector(candidate.x, candidate.y, 0.0))
        spawned += 1

unreal.log(f"Done. Spawned {spawned} tree actors.")
