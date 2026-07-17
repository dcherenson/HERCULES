# Cooperative Perception (DAIR-V2X)

HERCULES supports **V2X-style collaborative perception**, providing multi-view 3D object-detection baselines in the style of [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X).

A **vehicle-side** agent (a UGV traversing the sidewalk) and an **infrastructure-side** agent (an overhead UAV) observe a shared city intersection from complementary viewpoints. Each agent exports time-synchronized **RGB** and **LiDAR**, providing the overlapping multi-view coverage that cooperative detection and fusion methods rely on.

## Cooperative RGB

![Vehicle-side (left) and infrastructure-side (right) RGB of a shared intersection](media/cp_rgb_sidebyside.gif)

## Per-agent LiDAR

<div class="grid" markdown>
![Vehicle-side LiDAR](media/cp_lidar_vehicle.gif)
![Infrastructure-side LiDAR](media/cp_lidar_infra.gif)
</div>

## Per-agent sensor streams

<div class="grid" markdown>
![Vehicle-side RGB collage](media/cp_collage_vehicle_rgb.png)
![Infrastructure-side RGB collage](media/cp_collage_infra_rgb.png)
![Vehicle-side LiDAR collage](media/cp_collage_vehicle_lidar.png)
![Infrastructure-side LiDAR collage](media/cp_collage_infra_lidar.png)
</div>

## Running it

Cooperative-perception data is exported by the [dataset pipeline](dataset.md) in standard formats and integrated through the [ROS 2 wrapper](ros_cplusplus.md). See [Image APIs](image_apis.md) and [LIDAR](lidar.md) for the underlying sensor configuration.
