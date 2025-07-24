#!/usr/bin/env python3
import setup_path                    # ensure cosysairsim is on PYTHONPATH
import cosysairsim as airsim
import cv2
import numpy as np
import open3d as o3d                # Open3D for 3D viz
import math

# ─── Configuration ────────────────────────────────────────────────────────────
CAMERA_NAME = "front_center"
IMAGE_TYPE  = airsim.ImageType.Scene

# match your settings.json
WIDTH, HEIGHT, FOV_DEG = 1920, 1080, 128.0

def main():
    # ── Connect ───────────────────────────────────────────────────────────────
    client = airsim.CarClient(port=41452)  # CarClient for SimMode=Car
    client.confirmConnection()

    # ── Set mesh‑name filter to match your static mesh asset prefix ────────────
    client.simClearDetectionMeshNames(CAMERA_NAME, IMAGE_TYPE)
    client.simAddDetectionFilterMeshName(CAMERA_NAME, IMAGE_TYPE, "Cylinder*")
    client.simSetDetectionFilterRadius(CAMERA_NAME, IMAGE_TYPE, 200* 100)   # cm

    # ── Capture one RGB frame ─────────────────────────────────────────────────
    resp = client.simGetImages([
        airsim.ImageRequest(CAMERA_NAME, IMAGE_TYPE, False, False)
    ])[0]
    if resp.width == 0 or resp.height == 0:
        raise RuntimeError("No image returned; check settings.json & restart AirSim")

    # decode into a writable BGR image for OpenCV
    img = np.frombuffer(resp.image_data_uint8, np.uint8) \
            .reshape(resp.height, resp.width, 3).copy()

    # ── Fetch detections ───────────────────────────────────────────────────────
    dets = client.simGetDetections(CAMERA_NAME, IMAGE_TYPE) or []
    print(f"Got {len(dets)} detections")

    # ── Draw 2D boxes with OpenCV ──────────────────────────────────────────────
    for d in dets:
        x0,y0 = int(d.box2D.min.x_val), int(d.box2D.min.y_val)
        x1,y1 = int(d.box2D.max.x_val), int(d.box2D.max.y_val)
        cv2.rectangle(img, (x0,y0), (x1,y1), (0,255,0), 2)
        cv2.putText(img, d.name, (x0, max(0,y0-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imshow("2D Detections", img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # ── Build Open3D geometries ────────────────────────────────────────────────
    geometries = []

    # camera coordinate triad
    triad = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)  # X=red, Y=green, Z=blue :contentReference[oaicite:1]{index=1}
    geometries.append(triad)

    # for each detection, make a LineSet from 8 corners + 12 edges
    for d in dets:
        pmin, pmax = d.box3D.min, d.box3D.max
        # 8 corners in camera FRD frame
        corners = np.array([
            [ pmax.x_val,  pmax.y_val,  pmin.z_val],
            [ pmax.x_val,  pmin.y_val,  pmin.z_val],
            [ pmin.x_val,  pmin.y_val,  pmin.z_val],
            [ pmin.x_val,  pmax.y_val,  pmin.z_val],
            [ pmax.x_val,  pmax.y_val,  pmax.z_val],
            [ pmax.x_val,  pmin.y_val,  pmax.z_val],
            [ pmin.x_val,  pmin.y_val,  pmax.z_val],
            [ pmin.x_val,  pmax.y_val,  pmax.z_val],
        ], dtype=float)

        # 12 edges connecting those corners
        edges = [
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]
        ]
        # one color per edge (green)
        colors = [[0,1,0] for _ in edges]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corners),
            lines=o3d.utility.Vector2iVector(edges)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)

    # ── Visualize interactively ────────────────────────────────────────────────
    o3d.visualization.draw_geometries(
        geometries,
        window_name="3D Detections in Camera Frame",
        width=800, height=600,
        left=50, top=50,
        mesh_show_back_face=True
    )  # supports pan/zoom/rotate with mouse :contentReference[oaicite:2]{index=2}

if __name__ == "__main__":
    main()
