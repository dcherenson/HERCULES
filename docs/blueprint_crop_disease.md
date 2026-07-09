# Crop Disease Blueprint

The **Crop Disease** Blueprint is a custom, parameterized Unreal Engine Blueprint that simulates disease transmission spreading across agricultural terrain at runtime — driving precision-agriculture and monitoring scenarios. This page covers how to download it and use it in your own Unreal project.

!!! note "Work in progress"
    The downloadable Blueprint package and full parameter reference are being finalized. This page is a placeholder — the download link and detailed usage guide will land here soon.

![Crop disease transmission across farmland](media/phenom_cropdisease.png)

## Download

<!-- TODO: replace with the actual release asset link once published -->
*Download link coming soon.* The Crop Disease Blueprint will be published as a downloadable Unreal asset package on the [HERCULES releases page](https://github.com/lunarlab-gatech/HERCULES/releases).

## Install into your Unreal project

1. Download and unzip the Blueprint package.
2. Copy the provided `Content/` assets into your Unreal project's `Content/` directory.
3. Restart the Unreal Editor so the assets are registered.

## Usage

1. Drag the crop-disease Blueprint actor into your level over the agricultural terrain.
2. Set the initial infection site(s) and configure the transmission behavior.
3. Play / run the simulation — the disease state spreads across the field at runtime and is observable by the robots' cameras.

## Parameters

<!-- TODO: fill in the real exposed parameters -->
| Parameter | Description | Default |
|---|---|---|
| Initial infection site(s) | Where the disease starts | — |
| Transmission rate | How fast the disease spreads | — |
| Field extent | Region the disease can affect | — |

See the [Dynamic Environmental Phenomena](dynamic_phenomena.md) overview for all phenomena in context.
