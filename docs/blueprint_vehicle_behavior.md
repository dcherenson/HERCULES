# Vehicle Behavior Blueprint

The **Vehicle Behavior** Blueprint is a custom, parameterized Unreal Engine Blueprint (VehicleAI) that populates a world with autonomous vehicle traffic that drives and reacts at runtime. This page covers how to download it and use it in your own Unreal project.

![Vehicle traffic through the urban world](media/citycarspeds1.gif)

## Download

!!! note "Package being finalized"
    The Vehicle Behavior Blueprint will be published as a Blueprint package on the [HERCULES releases page](https://github.com/lunarlab-gatech/HERCULES/releases). Until the zip lands, this page already documents the requirements, install procedure, and sensor integration.
<!-- TODO: replace with the actual release asset link once published -->

The package contains the HERCULES-authored behavior Blueprints (traffic spawner, VehicleAI logic, route tooling) only. Vehicle model packs are third-party marketplace content — **linked, not shipped** — see [Using Blueprint Packages](blueprint_packages.md).

## Requirements

| Requirement | Notes |
|---|---|
| Unreal Engine 5.2.1 | Version HERCULES is built against |
| HERCULES AirSim plugin | For sensing integration ([install](install_precompiled.md)) |
| Vehicle model pack(s) | Linked from the release notes; assign meshes via the Blueprint's exposed vehicle slots |
| Road network / spline routes | The traffic follows routes defined in your level (spline-based); the urban HERCULES world ships with its own |

<!-- TODO: confirm final third-party asset list + expected content paths when the package is cut -->

## Install into your Unreal project

Follow the general steps in [Using Blueprint Packages](blueprint_packages.md#installing-a-package):

1. Download and unzip the Blueprint package, and copy the contained folder into your project's `Content/`.
2. Install the vehicle model pack(s) from their store links.
3. Restart the Unreal Editor and assign vehicle meshes to the Blueprint's exposed slots.
4. Lay out (or import) the route splines along your road network.

## Usage

1. Place the traffic / vehicle behavior Blueprint into your level along the road network.
2. Configure traffic density, speeds, and routing behavior.
3. Play / run the simulation — the vehicles drive through the world and are observable by all robots' sensors.

## Parameters

<!-- TODO: fill in the real exposed parameters from the packaged Blueprint -->
| Parameter | Description | Default |
|---|---|---|
| Traffic density | How many vehicles | — |
| Speed range | Vehicle speeds | — |
| Routes | Road network the vehicles follow | — |

## Sensor integration

* Spawned vehicles must be registered with [instance segmentation](instance_segmentation.md) (`AddNewActorToSegmentation` on spawn — see [Using Blueprint Packages](blueprint_packages.md#making-blueprints-visible-to-the-hercules-sensors)).
* Mesh names containing `car`, `truck`, or `vehicle` classify as engine-warmed metal (305 K) in the [LWIR thermal camera](synthetic_cameras.md); tune per-model signatures with `Overrides` entries if needed.

See the [Dynamic Environmental Phenomena](dynamic_phenomena.md) overview and the [Environments](index.md#environments) for these agents in context.
