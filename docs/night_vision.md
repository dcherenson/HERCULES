# Night-Vision Camera

HERCULES adds a configurable **night-vision (NVG) camera** with an empirical photometric transfer that reproduces the characteristic low-light, image-intensified look — enabling nighttime perception and exploration experiments.

![Night-vision rendering of a desert scene](media/nvg.png)

## What it models

- Empirical photometric transfer mapping scene luminance to the intensified night-vision response.
- Configurable low-light behavior for nighttime flights and drives.
- Complements the [LWIR thermal camera](lwir_camera.md) for combined thermal + low-light sensing, and the [dynamic phenomena](dynamic_phenomena.md) for scenarios such as fires observed at night.

## Usage

The night-vision camera is exposed as camera image type **`NightVision = 12`**, synthesized server-side from Scene + Segmentation captures rendered in the same `simGetImages` batch (same frame, lockstep-safe) and captured through the standard [Image APIs](image_apis.md):

```python
responses = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.NightVision, False, False)
])
```

For the fused scene-luminance look, the Scene and Segmentation captures must share the same resolution in the [settings file](settings.md) (otherwise the camera falls back to a pure thermal-map view). Gain, blend, and seed are configurable via the `SyntheticCameraSettings` block — see **[Synthetic IR / NVG Cameras](synthetic_cameras.md)** for the full reference and troubleshooting. See [Sensors](sensors.md) for the complete sensor list and [Camera Views](camera_views.md) for viewport configuration.
