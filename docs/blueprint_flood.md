# Flood Blueprint

The **Flood** Blueprint is a custom, parameterized Unreal Engine Blueprint that simulates rising water and flood inundation across terrain at runtime — including over georeferenced [Cesium for Unreal](https://cesium.com/platform/cesium-for-unreal/) city models. This page covers how to download it and use it in your own Unreal project.

!!! note "Work in progress"
    The downloadable Blueprint package and full parameter reference are being finalized. This page is a placeholder — the download link and detailed usage guide will land here soon.

![Flood inundation over a Cesium model of Atlanta](media/Atlanta_hercules_flooding_demo1.gif)

## Download

<!-- TODO: replace with the actual release asset link once published -->
*Download link coming soon.* The Flood Blueprint will be published as a downloadable Unreal asset package on the [HERCULES releases page](https://github.com/lunarlab-gatech/HERCULES/releases).

## Install into your Unreal project

1. Download and unzip the Blueprint package.
2. Copy the provided `Content/` assets into your Unreal project's `Content/` directory.
3. Restart the Unreal Editor so the assets are registered.

## Usage

1. Drag the flood Blueprint actor into your level and position the water volume.
2. Configure the target water level, rise rate, and extent.
3. Play / run the simulation — the water level updates the shared world state at runtime and is observable by all robots.

## Parameters

<!-- TODO: fill in the real exposed parameters -->
| Parameter | Description | Default |
|---|---|---|
| Target water level | Final flood height | — |
| Rise rate | How fast the water rises | — |
| Extent | Region covered by the flood | — |

See the [Dynamic Environmental Phenomena](dynamic_phenomena.md) overview for all phenomena in context.
