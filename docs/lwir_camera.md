# LWIR Thermal Camera

HERCULES adds a **physics-based long-wave infrared (LWIR) thermal camera** that is not present in Cosys-AirSim. Rather than false-coloring an RGB image, it renders apparent temperature from **Planck-law spectral radiance**, so warm bodies (wildlife, people, vehicles, fire) stand out against a cooler background.

![LWIR thermal rendering — kangaroos read hot near a fire](media/lwir.png)

## What it models

- Per-object thermal emission integrated over the LWIR band using the Planck radiation law.
- Distinct thermal signatures for warm agents (wildlife, MetaHuman pedestrians, vehicles) versus the environment.
- Pairs naturally with the [dynamic phenomena](dynamic_phenomena.md) (e.g. wildfire) and the [night-vision camera](night_vision.md) for low-light and search-and-rescue scenarios.

## Usage

The LWIR camera is exposed as a camera image type, captured through the same [Image APIs](image_apis.md) as the RGB, depth, and segmentation streams, and configured in the [settings file](settings.md). See [Sensors](sensors.md) for the full sensor suite and the [Infrared Camera](InfraredCamera.md) page for the related inherited thermal tooling.
