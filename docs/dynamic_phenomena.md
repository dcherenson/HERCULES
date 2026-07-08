# Dynamic Environmental Phenomena

HERCULES exposes parameterized **dynamic-environment modules** and **dynamic-agent Blueprints** through plug-and-play interfaces, so the shared world state evolves at runtime for procedural scenario generation. These go beyond the static [dynamic objects](dynamic_objects.md) inherited from Cosys-AirSim.

## Phenomena

=== "Wildfire spread"

    Fire propagates through the environment over time, coupling naturally with the [LWIR thermal](lwir_camera.md) and [night-vision](night_vision.md) sensors for detection and monitoring scenarios.

    ![Wildfire spreading through a forest](media/forestfire1.gif)
    ![Fire over a Cesium model of Pasadena at night](media/caltech_fire_1drone_night.gif)

=== "Flood inundation"

    Water level rises and spreads across terrain, including over georeferenced [Cesium for Unreal](https://cesium.com/platform/cesium-for-unreal/) city models.

    ![Flood inundation over a Cesium model of Atlanta](media/Atlanta_hercules_flooding_demo1.gif)
    ![Flood inundation in a dense jungle](media/jungleenv_flooding_1.gif)

=== "Crop-disease transmission"

    Disease spreads across agricultural terrain, driving precision-agriculture and monitoring scenarios.

    ![Crop disease transmission across farmland](media/phenom_cropdisease.png)

## Dynamic agents

Dynamic-agent Blueprints populate the world with moving entities that update the shared state at runtime:

- **MetaHuman pedestrians** — realistic walking agents.
- **VehicleAI traffic** — autonomous vehicle traffic in the city world.
- **AnimalAI wildlife** — kangaroos, deer, and other wildlife in the natural worlds.

These integrate with the [environments](index.md#environments) and are designed to stress perception and planning under dynamic obstacles.
