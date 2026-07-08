# HERCULES

### An Open-Source Simulation Framework for Heterogeneous Multi-Robot SLAM, Collaborative Perception, and Exploration

HERCULES is a simulation platform for **heterogeneous multi-robot autonomy** that lets unmanned ground vehicles (UGVs) and unmanned aerial vehicles (UAVs) collaboratively explore and understand large-scale, complex environments. It is an advanced fork of [AirSim](https://github.com/microsoft/AirSim) / [Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim) built on **Unreal Engine 5**.

[:material-rocket-launch: Getting Started](install_linux.md){ .md-button .md-button--primary }
[:material-github: Code](https://github.com/lunarlab-gatech/HERCULES){ .md-button }
[:material-file-document: arXiv](https://arxiv.org/abs/2606.22756){ .md-button }
[:material-database: Dataset](https://huggingface.co/datasets/GeorgiaTech/HERCULES){ .md-button }

![HERCULES system overview](media/system_overview.png)

## What HERCULES adds

HERCULES extends Cosys-AirSim with a unified heterogeneous autonomy stack, new sensor modalities, dynamic environments, and ready-to-run multi-robot benchmarks:

- **[Heterogeneous UAV–UGV autonomy](heterogeneous_autonomy.md)** — concurrent UAV and UGV operation in a single simulation session, a unified waypoint-level command interface, an autonomous UGV controller, and coordinated multi-robot sensor logging.
- **[LWIR thermal camera](lwir_camera.md)** — a physics-based long-wave infrared sensor using Planck-law spectral radiance.
- **[Night-vision camera](night_vision.md)** — a configurable low-light sensor with empirical photometric transfer.
- **[Dynamic environmental phenomena](dynamic_phenomena.md)** — wildfire spread, flood inundation, and crop-disease transmission, plus dynamic agents (pedestrians, traffic, wildlife).
- **[Collaborative SLAM (ROMAN)](collaborative_slam.md)** — a ready-to-run multi-robot SLAM benchmark.
- **[Cooperative perception (DAIR-V2X)](cooperative_perception.md)** — V2X-style multi-view 3D detection baselines.
- **[Multimodal dataset](dataset.md)** — synchronized RGB, depth, semantic, and LiDAR export in KITTI-style layouts and ROS 2 bags.

It inherits the full Cosys-AirSim / AirSim sensor suite and API surface — cameras, depth, semantic segmentation, LiDAR, GPU-LiDAR, echo/radar, IMU, GPS, instance segmentation, annotation, skid-steer vehicles, and more — documented under **Simulator**.

## Environments

Three photorealistic worlds — **desert, forest, and city** — each stress a different class of perception and planning failure, plus georeferenced real-world scenes via [Cesium for Unreal](https://cesium.com/platform/cesium-for-unreal/).

<div class="grid cards" markdown>

- ![City](media/city_uavugv1.gif)
- ![Forest](media/forestdeer_uavugv1.gif)

</div>

## Quick links

- **Install:** [Linux](install_linux.md) · [Windows](install_windows.md) · [Precompiled plugin](install_precompiled.md) · [Docker](docker_ubuntu.md)
- **Configure:** [Settings](settings.md)
- **Develop:** [Python APIs](apis.md) · [ROS 2 wrapper](ros_cplusplus.md)

## Citation

```bibtex
@misc{garimella2026hercules,
  title         = {HERCULES: An Open-Source Simulation Framework for Heterogeneous Multi-Robot SLAM, Collaborative Perception, and Exploration},
  author        = {Garimella, Sandilya Sai and Butterfield, Daniel Chase and Wilson, Sean and Gan, Lu},
  year          = {2026},
  eprint        = {2606.22756},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.22756}
}
```

## Acknowledgements

HERCULES builds on [AirSim](https://github.com/microsoft/AirSim) (Microsoft) and [Cosys-AirSim](https://github.com/Cosys-Lab/Cosys-AirSim) (Cosys-Lab, University of Antwerp), and benchmarks [ROMAN](https://github.com/mit-acl/ROMAN) (MIT-ACL) and [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X). Developed at [LunarLab](https://sites.gatech.edu/lunarlab/), Georgia Institute of Technology. Released under the [MIT License](https://github.com/lunarlab-gatech/HERCULES/blob/main/LICENSE).
