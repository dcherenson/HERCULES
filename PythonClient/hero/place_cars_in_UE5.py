import unreal
import random

# 1) Asset paths to your Static Meshes
CAR_ASSETS = [
    "/Game/CitySampleVehicles/MergedVehicleMeshes/Bus.Bus",
    "/Game/CitySampleVehicles/MergedVehicleMeshes/Copcar.Copcar",
    "/Game/CitySampleVehicles/MergedVehicleMeshes/Garbagetruck.Garbagetruck",
    "/Game/CitySampleVehicles/MergedVehicleMeshes/Hugetruck.Hugetruck",
    "/Game/CitySampleVehicles/MergedVehicleMeshes/Milkvan.Milkvan",
    "/Game/CitySampleVehicles/MergedVehicleMeshes/Pickuptruck.Pickuptruck",
    "/Game/CitySampleVehicles/MergedVehicleMeshes/Sedan1.Sedan1",
    "/Game/CitySampleVehicles/MergedVehicleMeshes/Sedan2.Sedan2",
    "/Game/CitySampleVehicles/MergedVehicleMeshes/Sportscar.Sportscar",
    "/Game/CitySampleVehicles/MergedVehicleMeshes/SUV1.SUV1",
    "/Game/CitySampleVehicles/MergedVehicleMeshes/SUV2.SUV2",
    "/Game/CitySampleVehicles/MergedVehicleMeshes/Taxi.Taxi",
    "/Game/CitySampleVehicles/MergedVehicleMeshes/Van1.Van1",
]

NUM_TO_SPAWN = 15       # total actors you want
SPACING      = 900.0    # world units between each car on the X-axis
Z_HEIGHT     = 10.0     # above ground

# --- Add your custom suffix here ---
SUFFIX = "_VEHTAG3"       # e.g. "_AI", "_TEST", whatever you need

lvl_lib     = unreal.EditorLevelLibrary
name_counts = {}        # track how many times each mesh has been placed

for idx in range(NUM_TO_SPAWN):
    # 1) Pick & load a random mesh
    asset_path = random.choice(CAR_ASSETS)
    mesh       = unreal.load_asset(asset_path)  # UStaticMesh
    mesh_name  = mesh.get_name()                # e.g. "Bus", "Sedan1", etc.

    # 2) Increment count & build unique label with suffix
    count = name_counts.get(mesh_name, 0) + 1
    name_counts[mesh_name] = count
    label = f"{mesh_name}_{count}{SUFFIX}"

    # 3) Compute line positions (orientation stays zeroed)
    loc = unreal.Vector(idx * SPACING, 0.0, Z_HEIGHT)
    rot = unreal.Rotator(0, 0, 0)

    # 4) Spawn & assign
    actor = lvl_lib.spawn_actor_from_class(
        unreal.StaticMeshActor.static_class(), loc, rot
    )
    actor.static_mesh_component.set_static_mesh(mesh)

    # 5) Rename actor in Outliner
    actor.set_actor_label(label)
