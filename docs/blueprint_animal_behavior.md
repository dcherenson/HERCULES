# Animal Behavior Blueprint

The **Animal Behavior** Blueprint is a custom, parameterized Unreal Engine Blueprint (AnimalAI) that populates a world with wildlife — kangaroos, deer, and other animals — that roam and react at runtime. This page covers how to download it and use it in your own Unreal project.

!!! note "Work in progress"
    The downloadable Blueprint package and full parameter reference are being finalized. This page is a placeholder — the download link and detailed usage guide will land here soon.

![Kangaroo wildlife roaming the outback](media/auskangaroo1.gif)

## Download

<!-- TODO: replace with the actual release asset link once published -->
*Download link coming soon.* The Animal Behavior Blueprint will be published as a downloadable Unreal asset package on the [HERCULES releases page](https://github.com/lunarlab-gatech/HERCULES/releases).

## Install into your Unreal project

1. Download and unzip the Blueprint package.
2. Copy the provided `Content/` assets into your Unreal project's `Content/` directory.
3. Restart the Unreal Editor so the assets are registered.

## Usage

1. Place the animal spawner / behavior Blueprint into your level.
2. Choose the species and configure herd size, roam area, and behavior.
3. Play / run the simulation — the animals move through the world and are observable by all robots' sensors.

## Parameters

<!-- TODO: fill in the real exposed parameters -->
| Parameter | Description | Default |
|---|---|---|
| Species | Which animal(s) to spawn | — |
| Count / density | How many animals | — |
| Roam area | Region the animals wander | — |

See the [Dynamic Environmental Phenomena](dynamic_phenomena.md) overview and the [Environments](index.md#environments) for these agents in context.
