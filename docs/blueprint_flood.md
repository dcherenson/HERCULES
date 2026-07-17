# Flood Blueprint

The **Flood** Blueprint is a custom, parameterized Unreal Engine Blueprint that simulates rising water and flood inundation across terrain at runtime — including over georeferenced [Cesium for Unreal](https://cesium.com/platform/cesium-for-unreal/) city models. This page covers how to download it and use it in your own Unreal project.

![Flood inundation over a Cesium model of Atlanta](media/Atlanta_hercules_flooding_demo1.gif)

## Download

!!! note "Package being finalized"
    The Flood Blueprint will be published as a Blueprint package on the [HERCULES releases page](https://github.com/lunarlab-gatech/HERCULES/releases). Until the zip lands, this page already documents the requirements, install procedure, and sensor integration.
<!-- TODO: replace with the actual release asset link once published -->

The package contains the HERCULES-authored Blueprints and support assets only; third-party content is linked, not shipped — see [Using Blueprint Packages](blueprint_packages.md).

## Requirements

| Requirement | Notes |
|---|---|
| Unreal Engine 5.2.1 | Version HERCULES is built against |
| HERCULES AirSim plugin | For sensing integration ([install](install_precompiled.md)) |
| [Cesium for Unreal](https://cesium.com/platform/cesium-for-unreal/) plugin | Only for flooding georeferenced city models; free |

<!-- TODO: confirm final third-party asset list (water material/plugin) when the package is cut -->

## Install into your Unreal project

Follow the general steps in [Using Blueprint Packages](blueprint_packages.md#installing-a-package):

1. Download and unzip the Blueprint package, and copy the contained folder into your project's `Content/`.
2. For city-scale floods, install and configure Cesium for Unreal with your georeferenced tileset.
3. Restart the Unreal Editor and assign any unset asset slots (water material) in the Details panel.

## Usage

1. Drag the flood Blueprint actor into your level and position the water volume.
2. Configure the target water level, rise rate, and extent.
3. Play / run the simulation — the water level updates the shared world state at runtime and is observable by all robots.

![Flood inundation in a dense jungle](media/jungleenv_flooding_1.gif)

## Parameters

<!-- TODO: fill in the real exposed parameters from the packaged Blueprint -->
| Parameter | Description | Default |
|---|---|---|
| Target water level | Final flood height | — |
| Rise rate | How fast the water rises | — |
| Extent | Region covered by the flood | — |

## Sensor integration

* The rising water is directly visible to the RGB, depth, and [night-vision](night_vision.md) cameras; depth captures make the inundation extent measurable.
* Water has no keyword in the [thermal camera's](synthetic_cameras.md) classifier and defaults to neutral (295 K). To render it distinctly cool in [LWIR](lwir_camera.md), register the water actor with the [instance segmentation](instance_segmentation.md) system and add an override:

```json
"SyntheticCameraSettings": {
  "ThermalIR": { "Overrides": [ { "Match": "flood", "TempK": 288.0, "Emissivity": 0.96 } ] }
}
```

See the [Dynamic Environmental Phenomena](dynamic_phenomena.md) overview for all phenomena in context.
