#!/usr/bin/env python3
import setup_path                    # ensure cosysairsim is on PYTHONPATH
import cosysairsim as airsim
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')              # force a GUI-capable backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

VEHICLE = "Husky1"
CAM     = "front_center"

# We draw ALL detections; no meshName filtering.
EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

def intrinsics(width, height, fov_deg):
    """Compute focal length and principal point from horizontal FOV."""
    f = width / (2 * math.tan(math.radians(fov_deg) / 2))
    return f, width/2.0, height/2.0

def main():
    client = airsim.CarClient(port=41452)
    client.confirmConnection()
    print("Connected to AirSim")

    # Grab one RGB frame to get w/h and intrinsics
    resp = client.simGetImages([
        airsim.ImageRequest(CAM, airsim.ImageType.Scene, False, False)
    ], vehicle_name=VEHICLE)[0]
    if resp.width == 0 or resp.height == 0:
        raise RuntimeError("No Scene image available; check settings.json & restart.")

    h, w = resp.height, resp.width
    cam_info = client.simGetCameraInfo(CAM, VEHICLE)
    f, cx, cy = intrinsics(w, h, cam_info.fov)
    print(f"RGB {w}×{h}, FOV={cam_info.fov:.2f}°, focal={f:.1f}")

    # Decode RGB
    arr1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
    img_rgb = arr1d.reshape(h, w, 3)[..., ::-1]  # BGR→RGB

    # Fetch detections (no filters)
    dets = client.simGetDetections(CAM, airsim.ImageType.Scene, VEHICLE) or []
    print(f"Detections returned: {len(dets)}")

    # Build corner_list for *all* dets
    corner_list = []
    for i, det in enumerate(dets):
        pmin, pmax = det.box3D.min, det.box3D.max
        corners = np.array([
            [pmax.x_val, pmax.y_val, pmin.z_val],
            [pmax.x_val, pmin.y_val, pmin.z_val],
            [pmin.x_val, pmin.y_val, pmin.z_val],
            [pmin.x_val, pmax.y_val, pmin.z_val],
            [pmax.x_val, pmax.y_val, pmax.z_val],
            [pmax.x_val, pmin.y_val, pmax.z_val],
            [pmin.x_val, pmin.y_val, pmax.z_val],
            [pmin.x_val, pmax.y_val, pmax.z_val],
        ], dtype=float)
        corner_list.append(corners)
        print(f" [{i:2d}] corners X[{corners[:,0].min():.1f}…{corners[:,0].max():.1f}] "
              f"Y[{corners[:,1].min():.1f}…{corners[:,1].max():.1f}] "
              f"Z[{corners[:,2].min():.1f}…{corners[:,2].max():.1f}]")

    # If there truly are zero detections, fallback to dummy
    if not corner_list:
        print("No detections at all—using dummy cube")
        dummy = np.array([
            [10,10,10],[20,10,10],[20,20,10],[10,20,10],
            [10,10,20],[20,10,20],[20,20,20],[10,20,20]
        ], dtype=float)
        corner_list.append(dummy)

    # Auto‐compute axis limits from all corners
    all_pts = np.vstack(corner_list)
    x_min, x_max = all_pts[:,0].min(), all_pts[:,0].max()
    y_min, y_max = all_pts[:,1].min(), all_pts[:,1].max()
    z_min, z_max = all_pts[:,2].min(), all_pts[:,2].max()

    pad = 0.1  # 10% padding
    def pad_range(a, b):
        span = b - a
        return a - pad*span, b + pad*span

    x_lims = pad_range(x_min, x_max)
    y_lims = pad_range(y_min, y_max)
    z_lims = pad_range(z_min, z_max)

    # Plot
    fig = plt.figure(figsize=(12,7))
    ax  = fig.add_subplot(111, projection='3d')

    # RGB background as flat plane at z=0
    ax.plot_surface(
        np.array([[0, w],[0, w]]),
        np.array([[0, 0],[h, h]]),
        np.zeros((2,2)),
        rstride=1, cstride=1, facecolors=img_rgb/255., shade=False
    )

    # Camera tripod
    L = max(x_max, y_max, -z_min, 1.0) * 0.2
    ax.quiver(0,0,0, L,0,0, length=L, color='r'); ax.text(L,0,0,'X',color='r')
    ax.quiver(0,0,0, 0,L,0, length=L, color='g'); ax.text(0,L,0,'Y',color='g')
    ax.quiver(0,0,0, 0,0,L, length=L, color='b'); ax.text(0,0,L,'Z',color='b')

    # Draw all boxes
    for corners in corner_list:
        for i,j in EDGES:
            ax.plot(
                [corners[i,0], corners[j,0]],
                [corners[i,1], corners[j,1]],
                [corners[i,2], corners[j,2]],
                linewidth=2, color='yellow'
            )

    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Y (right)")
    ax.set_zlabel("Z (down)")
    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)
    ax.set_zlim(*z_lims)
    plt.title("3D Detections in Camera Frame (all meshes)")

    plt.savefig('debug_plot.png')
    print("Saved debug_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
