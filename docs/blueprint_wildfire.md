# Wildfire Blueprint

The **Wildfire** Blueprint is a custom, parameterized Unreal Engine Blueprint that simulates fire igniting and spreading through an environment at runtime. This page covers how to download it and use it in your own Unreal project.

!!! note "Work in progress"
    The downloadable Blueprint package and full parameter reference are being finalized. This page is a placeholder — the download link and detailed usage guide will land here soon.

![Wildfire spreading through a forest](media/forestfire1.gif)

## Download

<!-- TODO: replace with the actual release asset link once published -->
*Download link coming soon.* The Wildfire Blueprint will be published as a downloadable Unreal asset package on the [HERCULES releases page](https://github.com/lunarlab-gatech/HERCULES/releases).

## Install into your Unreal project

1. Download and unzip the Blueprint package.
2. Copy the provided `Content/` assets into your Unreal project's `Content/` directory.
3. Restart the Unreal Editor so the assets are registered.

## Usage

1. Drag the wildfire Blueprint actor into your level.
2. Set the ignition source(s) and configure the spread behavior.
3. Play / run the simulation — the fire updates the shared world state at runtime and is observable by all robots (including the [LWIR thermal](lwir_camera.md) and [night-vision](night_vision.md) sensors).

## Parameters

<!-- TODO: fill in the real exposed parameters -->
| Parameter | Description | Default |
|---|---|---|
| Ignition point(s) | Where the fire starts | — |
| Spread rate | How fast the fire front advances | — |
| Fuel / extent | Region the fire can consume | — |

See the [Dynamic Environmental Phenomena](dynamic_phenomena.md) overview for all phenomena in context.
