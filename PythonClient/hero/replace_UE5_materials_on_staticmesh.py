import unreal

# ——— UPDATE THESE TO YOUR UE VIRTUAL PATHS ———
# mesh_path = "/Game/CitySampleBuildings/MergedMesh/CHI_1.CHI_1"
# mat_path  = "/Game/CitySampleBuildings/Material/CustomGlassMaterial/CustomGlassRed.CustomGlassRed"

mesh_path = "/Game/CitySampleBuildings/MergedMesh/SFA_2.SFA_2"
mat_path  = "/Game/CitySampleBuildings/Material/CustomGlassMaterial/CustomGlassBlue.CustomGlassBlue"
# ==============================================

# 1. Load the StaticMesh asset
static_mesh = unreal.EditorAssetLibrary.load_asset(mesh_path)
if not static_mesh:
    raise RuntimeError(f"Could not load mesh asset: {mesh_path}")

# 2. Load the Material asset
glass_mat = unreal.EditorAssetLibrary.load_asset(mat_path)
if not glass_mat:
    raise RuntimeError(f"Could not load material asset: {mat_path}")

# 3. Hack: create a transient StaticMeshComponent to read slot names
tmp_comp = unreal.StaticMeshComponent()
tmp_comp.set_static_mesh(static_mesh)
slot_names = unreal.StaticMeshComponent.get_material_slot_names(tmp_comp)
# slot_names is now a Python list of FName, e.g. ["None", "Bldg_glass12", ...]

# 4. Loop through each slot name & replace matching indices
for idx, slot in enumerate(slot_names):
    name_str = str(slot)  # convert FName to Python str
    if name_str.startswith("Bldg_glass"):
        try:
            num = int(name_str.split("Bldg_glass", 1)[1])
        except ValueError:
            continue
        if 12 <= num <= 63:
            static_mesh.set_material(idx, glass_mat)
            unreal.log(f"[{idx}] {slot} -> CustomGlass")

# 5. Save your edits back into the asset
unreal.EditorAssetLibrary.save_asset(mesh_path)
unreal.log("All matching slots updated and mesh saved.")
