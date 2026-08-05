#!/usr/bin/env python3
"""Pin segmentation IDs for the semantic terrain patches (mud/water/grass/debris) + the landscape.

Run ONCE after the sim starts (before/at ROS stack launch):
    cd PythonClient/hero && python3 set_terrain_seg_ids.py

Why: the Dirichlet pipeline classifies seg pixels by EXACT colour match (semantic_ground_extractor
class_colors). Auto-assigned instance-seg IDs shift when the map changes, so we explicitly pin:

  patch actors (spawned in the editor, names Sem<Type>_*):
    SemMud_*    -> ID 400 -> colour (127,110,127)   cost 6.0
    SemWater_*  -> ID 410 -> colour (191,110,159)   cost 9.0
    SemGrass_*  -> ID 420 -> colour (159,110,255)   cost 4.5
    SemDebris_* -> ID 430 -> colour (223,110,95)    cost 7.0
    SemLeaves_* -> ID 440 -> colour (79,110,79)     cost 4.0
  landscape (the ground class the pipeline already uses):
    -> ID 321 -> colour (79,159,95)  cost 3.0

IDs 400-430 are ABOVE this map's auto-assigned range (max ~337 + our 33 patches), so no existing
object collides. The colours MUST match class_colors in mapping.launch.py (K=8).
"""
import setup_path  # noqa: F401
import hercules_cosysairsim as airsim

PATCH_IDS = {
    "SemMud[^ ]*": 400,
    "SemWater[^ ]*": 410,
    "SemGrass[^ ]*": 420,
    "SemDebris[^ ]*": 430,
    "SemLeaves[^ ]*": 440,
}
LANDSCAPE_PATTERNS = ["StaticMeshActor_1$", "LandscapeComponent.*"]
LANDSCAPE_ID = 321

client = airsim.VehicleClient()
client.confirmConnection()

for pattern, sid in PATCH_IDS.items():
    ok = client.simSetSegmentationObjectID(pattern, sid, True)
    print("set %-18s -> ID %d : %s" % (pattern, sid, ok))

# pin the landscape so the existing ground class (79,159,95) survives scene edits
pinned = False
for pattern in LANDSCAPE_PATTERNS:
    if client.simSetSegmentationObjectID(pattern, LANDSCAPE_ID, True):
        print("pinned landscape (%s) -> ID %d" % (pattern, LANDSCAPE_ID))
        pinned = True
if not pinned:
    print("WARNING: no landscape pattern matched -- verify the ground still renders (79,159,95) "
          "in segmentation, else re-record class_colors (segmentation_generate_list.py)")

# sanity: report what a few patch objects now have
objs = client.simListSceneObjects("Sem.*")
print("Sem* objects visible to sim: %d (e.g. %s)" % (len(objs), objs[:4]))
