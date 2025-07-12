import unreal

# 1) Define the parent asset and load it
#    Convert your file path into a UE content path:
#    "/home/.../Content/CustomHumanPedestrians/BP_VehicleAI.uasset"
#    → "/Game/CustomHumanPedestrians/BP_VehicleAI.BP_VehicleAI"
parent_bp_path = "/Game/CustomHumanPedestrians/BP_VehicleAI.BP_VehicleAI"
parent_bp = unreal.EditorAssetLibrary.load_asset(parent_bp_path)
if not parent_bp:
    unreal.log_error(f"Failed to load parent Blueprint at {parent_bp_path}")
    raise SystemExit

# 2) Derive the UClass from the BlueprintGeneratedClass
parent_bp_generated_path = parent_bp_path + "_C"
parent_bp_generated = unreal.load_object(None, parent_bp_generated_path)
if not parent_bp_generated:
    unreal.log_error(f"Failed to load generated class at {parent_bp_generated_path}")
    raise SystemExit

parent_class = unreal.get_default_object(parent_bp_generated).get_class()

# 3) Set up the Blueprint factory to use that parent class
factory = unreal.BlueprintFactory()
factory.set_editor_property("ParentClass", parent_class)

# 4) Get the AssetTools helper
asset_tools = unreal.AssetToolsHelpers.get_asset_tools()

# 5) Specify where new assets will live
package_path = "/Game/CustomHumanPedestrians"

# 6) Names of the child Blueprints to create
child_names = [
    "BP_VehicleAI_SUV2",
    "BP_VehicleAI_sedan2",
    "BP_VehicleAI_van2",
    "BP_VehicleAI_taxi",
    "BP_VehicleAI_policecar"
]

# 7) Loop and create each child
for name in child_names:
    # Create the asset; this returns a UBlueprint object
    child_bp = asset_tools.create_asset(
        asset_name=name,
        package_path=package_path,
        asset_class=parent_bp.static_class(),
        factory=factory
    )
    if child_bp:
        # Save it to disk
        unreal.EditorAssetLibrary.save_loaded_asset(child_bp)
        unreal.log(f"Created child Blueprint: {child_bp.get_path_name()}")
    else:
        unreal.log_error(f"Failed to create {name}")
