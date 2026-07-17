# Using HERCULES Blueprint Packages

The [dynamic phenomena](dynamic_phenomena.md) and dynamic-agent Blueprints are distributed as **Blueprint packages**: zip archives of an Unreal `Content/` subfolder containing the HERCULES-authored Blueprints (and any HERCULES-authored support assets such as materials, curves, or data tables), published on the [HERCULES releases page](https://github.com/lunarlab-gatech/HERCULES/releases).

## What is (and is not) in a package

* **Included** — everything authored by the HERCULES team: the Blueprint classes, animation Blueprints, behavior logic, and self-made support assets. These are redistributable and covered by the repository license.
* **Not included** — third-party marketplace/Fab assets (meshes, skeletons, animation packs, VFX, MetaHumans). Marketplace licenses allow *using* these in a project but not *redistributing* them, so each Blueprint page lists the required packs with links; you install them into your project yourself (most are free).

Blueprints reference third-party assets through **instance-editable asset slots** (soft references) wherever possible, so a package loads cleanly even before the third-party content is installed — you then assign the assets in the Details panel or install them at the documented content path.

## Installing a package

1. Download the package zip from the [releases page](https://github.com/lunarlab-gatech/HERCULES/releases) and unzip it.
2. Copy the contained folder into your Unreal project's `Content/` directory (keep the folder name — Blueprint references are by content path).
3. Install the third-party asset packs listed on the Blueprint's page from their store links. If the page documents an expected content path (e.g. `/Game/AnimalVarietyPack/...`), install/move them to that path so hard references resolve.
4. Restart the Unreal Editor so the asset registry picks everything up.
5. Open the Blueprint and check the **asset slot variables** in the Details panel — assign any that are unset (mesh, animation Blueprint, Niagara system, etc.).
6. Drag the Blueprint actor into your level and configure its parameters (see the per-Blueprint page).

!!! tip "Broken references?"
    If the Blueprint opens with blank meshes or compile warnings about missing assets, the third-party pack is either not installed or installed at a different content path than the Blueprint expects. Either move it to the documented path (Fix Up Redirectors afterwards) or assign the assets manually via the exposed slots.

## Making Blueprints visible to the HERCULES sensors

Dynamic actors interact with the HERCULES ground-truth and synthetic sensors through the [instance segmentation](instance_segmentation.md) system:

* **Editor-placed actors** are registered automatically at simulation start (static and skeletal meshes only — see the [limitations](instance_segmentation.md#limitations)).
* **Runtime-spawned actors** (which is what most dynamic phenomena/agent Blueprints do) must be registered explicitly: call `ASimModeBase::AddNewActorToSegmentation(AActor)` from the Blueprint/C++ after spawning, or via RPC call `simAddSegmentationActor` followed by `simSetSegmentationObjectID`. The HERCULES phenomenon Blueprints do this for you where applicable.

Registration also drives the [synthetic thermal and night-vision cameras](synthetic_cameras.md): the [LWIR ThermalIR camera](lwir_camera.md) classifies every registered mesh by **name keywords** (`fire`, `kangaroo`, `animal`, `person`, `car`, ...) to assign temperature and emissivity. Two consequences for Blueprint authors and users:

1. **Name your spawned actors/meshes meaningfully** (`BP_Kangaroo23`, `FireFront_...`) so they classify correctly.
2. For anything that doesn't fit the keyword table, set thermal properties explicitly in settings.json — no renaming or recompiling needed:

```json
"SyntheticCameraSettings": {
  "ThermalIR": {
    "Overrides": [
      { "Match": "floodwater", "TempK": 288.0, "Emissivity": 0.96 },
      { "Match": "elephant", "TempK": 309.0, "Emissivity": 0.98, "IsAnimal": true }
    ]
  }
}
```

See the [synthetic cameras page](synthetic_cameras.md) for the full keyword table, settings reference, and troubleshooting ("object not showing up").

## Authoring and sharing your own Blueprint package

If you build your own phenomenon on top of HERCULES and want to share it the same way:

1. **Audit dependencies** — right-click the Blueprint → *Reference Viewer* / *Size Map* and classify every referenced asset: yours (ship it), engine/starter content (users already have it), third-party (link it, don't ship it).
2. **Break hard third-party references** — replace direct mesh/VFX references with instance-editable variables (`Static Mesh`, `Skeletal Mesh`, `Niagara System` object types, ideally soft references), assigned per-instance or on BeginPlay. This is what keeps the package loadable without the store content.
3. **Migrate** — right-click the Blueprint → *Asset Actions → Migrate* into a **clean project's** `Content/`, and verify there that it opens without the third-party packs installed.
4. **Package** — zip the migrated folder, attach it to a GitHub release, and document: required UE version, required plugins (e.g. Niagara, Cesium), each third-party pack with store link and expected content path, and the exposed parameters.
5. Optionally paste the graph to [blueprintUE](https://blueprintue.com/) and link it from your docs so readers can inspect the logic without downloading anything.
