#!/usr/bin/env python3
import setup_path               # ensure cosysairsim is on PYTHONPATH
import cosysairsim as airsim
import cv2
import numpy as np
import open3d as o3d
import multiprocessing as mp
import time

# ─── Configuration ────────────────────────────────────────────────────────────
CAMERA_NAME = "front_center"
IMAGE_TYPE  = airsim.ImageType.Scene
WIDTH, HEIGHT, FOV_DEG = 1920, 1080, 128.0

def open3d_viewer(geoms):
    """Run Open3D visualizer in a separate process."""
    vis = o3d.visualization.Visualizer()
    vis.create_window("3D Detections", width=800, height=600)
    for g in geoms:
        vis.add_geometry(g)
    vis.run()
    vis.destroy_window()

def main():
    # ── Connect ───────────────────────────────────────────────────────────────
    client = airsim.CarClient(port=41452)
    client.confirmConnection()

    # ── Detection filter ──────────────────────────────────────────────────────
    client.simClearDetectionMeshNames(CAMERA_NAME, IMAGE_TYPE)
    client.simAddDetectionFilterMeshName(CAMERA_NAME, IMAGE_TYPE, "Cylinder*")
    client.simSetDetectionFilterRadius(CAMERA_NAME, IMAGE_TYPE, 200*100)

    # ── Capture one compressed PNG frame ──────────────────────────────────────
    resp = client.simGetImages([
        airsim.ImageRequest(CAMERA_NAME, IMAGE_TYPE, False, True)
    ])[0]
    if not resp.image_data_uint8:
        raise RuntimeError("No image returned; check settings.json")

    # decode exact UE5 colors
    img = cv2.imdecode(np.frombuffer(resp.image_data_uint8, np.uint8),
                       cv2.IMREAD_COLOR)

    # ── Fetch detections ───────────────────────────────────────────────────────
    dets = client.simGetDetections(CAMERA_NAME, IMAGE_TYPE) or []
    print(f"Got {len(dets)} detections")

    # ── Draw 2D boxes ─────────────────────────────────────────────────────────
    for d in dets:
        x0,y0 = int(d.box2D.min.x_val), int(d.box2D.min.y_val)
        x1,y1 = int(d.box2D.max.x_val), int(d.box2D.max.y_val)
        cv2.rectangle(img, (x0,y0), (x1,y1), (0,255,0), 2)
        cv2.putText(img, d.name, (x0, max(0,y0-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.namedWindow("2D Detections", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("2D Detections", img)

    # ── Prepare Open3D geometries ─────────────────────────────────────────────
    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)]
    for d in dets:
        pmin, pmax = d.box3D.min, d.box3D.max
        corners = np.array([
            [ pmax.x_val,  pmax.y_val, -pmin.z_val],
            [ pmax.x_val,  pmin.y_val, -pmin.z_val],
            [ pmin.x_val,  pmin.y_val, -pmin.z_val],
            [ pmin.x_val,  pmax.y_val, -pmin.z_val],
            [ pmax.x_val,  pmax.y_val, -pmax.z_val],
            [ pmax.x_val,  pmin.y_val, -pmax.z_val],
            [ pmin.x_val,  pmin.y_val, -pmax.z_val],
            [ pmin.x_val,  pmax.y_val, -pmax.z_val],
        ], dtype=float)
        edges = [[0,1],[1,2],[2,3],[3,0],
                 [4,5],[5,6],[6,7],[7,4],
                 [0,4],[1,5],[2,6],[3,7]]
        colors = [[0,1,0] for _ in edges]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corners),
            lines=o3d.utility.Vector2iVector(edges)
        )
        ls.colors = o3d.utility.Vector3dVector(colors)
        geoms.append(ls)

    # ── Launch 3D viewer in separate process ───────────────────────────────────
    p = mp.Process(target=open3d_viewer, args=(geoms,), daemon=True)
    p.start()

    # ── Main loop: keep 2D window responsive until user quits ────────────────
    print("Press 'q' in the 2D window to exit.")
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        # re‑refresh the window so resizing works
        cv2.imshow("2D Detections", img)

    # ── Tear down ─────────────────────────────────────────────────────────────
    cv2.destroyAllWindows()
    if p.is_alive():
        p.terminate()
        p.join()

if __name__ == "__main__":
    main()
