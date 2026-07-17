# Pedestrian Behavior Blueprint

The **Pedestrian Behavior** Blueprint is a custom, parameterized Unreal Engine Blueprint (MetaHuman-based) that populates a world with realistic walking pedestrians that navigate and react at runtime. This page covers how to download it and use it in your own Unreal project.

## Download

!!! note "Package being finalized"
    The Pedestrian Behavior Blueprint will be published as a Blueprint package on the [HERCULES releases page](https://github.com/lunarlab-gatech/HERCULES/releases). Until the zip lands, this page already documents the requirements, install procedure, and sensor integration.
<!-- TODO: replace with the actual release asset link once published -->

The package contains the HERCULES-authored behavior Blueprints only. MetaHuman characters are generated per-user through Epic's own pipeline and are **not redistributed** — see [Using Blueprint Packages](blueprint_packages.md).

## Requirements

| Requirement | Notes |
|---|---|
| Unreal Engine 5.2.1 | Version HERCULES is built against |
| HERCULES AirSim plugin | For sensing integration ([install](install_precompiled.md)) |
| [MetaHuman](https://www.unrealengine.com/en-US/metahuman) characters | Create in MetaHuman Creator (free) and add to your project via Quixel Bridge; assign to the Blueprint's character slots |
| NavMesh in your level | Pedestrians navigate with Unreal's navigation system — add a `NavMeshBoundsVolume` over the walkable area |

<!-- TODO: confirm final requirements (MetaHuman plugin version, animation set) when the package is cut -->

## Install into your Unreal project

Follow the general steps in [Using Blueprint Packages](blueprint_packages.md#installing-a-package):

1. Download and unzip the Blueprint package, and copy the contained folder into your project's `Content/`.
2. Import one or more MetaHuman characters into the project via Quixel Bridge.
3. Restart the Unreal Editor and assign your MetaHuman(s) to the Blueprint's exposed character slots.
4. Make sure your level has a NavMesh covering sidewalks / walkable areas.

## Usage

1. Place the pedestrian spawner / behavior Blueprint into your level.
2. Configure crowd density, walk paths, and behavior.
3. Play / run the simulation — the pedestrians navigate the world and are observable by all robots' sensors.

## Parameters

<!-- TODO: fill in the real exposed parameters from the packaged Blueprint -->
| Parameter | Description | Default |
|---|---|---|
| Crowd density | How many pedestrians | — |
| Walk paths / area | Where pedestrians move | — |
| Behavior | Walking / idle / crossing behavior | — |

## Sensor integration

* Spawned pedestrians must be registered with [instance segmentation](instance_segmentation.md) (`AddNewActorToSegmentation` on spawn — see [Using Blueprint Packages](blueprint_packages.md#making-blueprints-visible-to-the-hercules-sensors)).
* Mesh names containing `human` or `person` classify as warm bodies (315 K) in the [LWIR thermal camera](synthetic_cameras.md) — name spawned instances accordingly, or add an `Overrides` entry matching your MetaHuman's name.
* Pedestrians at night are a natural target for the [night-vision camera](night_vision.md) in search-and-rescue style scenarios.

See the [Dynamic Environmental Phenomena](dynamic_phenomena.md) overview and the [Environments](index.md#environments) for these agents in context.
