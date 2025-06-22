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

NUM_TO_SPAWN = 13        # or len(CAR_ASSETS), or however many you want
SPACING      = 300.0     # world units between each car on the X-axis
Z_HEIGHT     = 100.0     # above ground

# Shortcuts
lvl_lib = unreal.EditorLevelLibrary

for idx in range(NUM_TO_SPAWN):
    # 1) Pick & load a random mesh
    asset_path = random.choice(CAR_ASSETS)
    mesh       = unreal.load_asset(asset_path)  # UStaticMesh

    # 2) Compute line positions
    x   = idx * SPACING
    y   = 0.0
    loc = unreal.Vector(x, y, Z_HEIGHT)
    rot = unreal.Rotator(0, 0, 0)  # fixed orientation

    # 3) Spawn & assign
    actor = lvl_lib.spawn_actor_from_class(
        unreal.StaticMeshActor.static_class(), loc, rot
    )
    actor.static_mesh_component.set_static_mesh(mesh)

    # 4) Rename the actor to match the mesh
    mesh_name = mesh.get_name()            # e.g. "Bus", "Sedan1", etc.
    actor.set_actor_label(mesh_name)
