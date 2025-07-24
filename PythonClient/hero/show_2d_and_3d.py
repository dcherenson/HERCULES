#!/usr/bin/env python3
import setup_path                    # ensure cosysairsim is on PYTHONPATH
import cosysairsim as airsim
import cv2
import numpy as np
import open3d as o3d
import time

CAMERA_NAME = "front_center"
IMAGE_TYPE  = airsim.ImageType.Scene
WIDTH, HEIGHT, FOV_DEG = 1920, 1080, 128.0

def make_line_set(det):
    """Given a single detection, return an Open3D LineSet of its 3D box."""
    pmin, pmax = det.box3D.min, det.box3D.max
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

    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]
    colors = [[0,1,0] for _ in edges]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(edges)
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def main():
    client = airsim.CarClient(port=41452)
    client.confirmConnection()

    # set up mesh filter
    client.simClearDetectionMeshNames(CAMERA_NAME, IMAGE_TYPE)
    client.simAddDetectionFilterMeshName(CAMERA_NAME, IMAGE_TYPE, "Cylinder*")
    client.simSetDetectionFilterRadius(CAMERA_NAME, IMAGE_TYPE, 200*100)

    # OpenCV window
    cv2.namedWindow("2D Detections", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("3D Detections", width=800, height=600)
    # **Prevent far‐plane culling by expanding the view frustum**
    ctr = vis.get_view_control()
    ctr.set_constant_z_near(0.01)
    ctr.set_constant_z_far(1e6)

    # add camera frame once
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    vis.add_geometry(cam_frame)

    # placeholder for the LineSets
    current_ls = []

    try:
        while True:
            # 1) grab image
            resp = client.simGetImages([
                airsim.ImageRequest(CAMERA_NAME, IMAGE_TYPE, False, True)
            ])[0]
            if not resp.image_data_uint8:
                continue
            img = cv2.imdecode(
                np.frombuffer(resp.image_data_uint8, np.uint8),
                cv2.IMREAD_COLOR
            )

            # 2) fetch detections
            dets = client.simGetDetections(CAMERA_NAME, IMAGE_TYPE) or []

            # 3) draw 2D boxes
            for d in dets:
                x0,y0 = int(d.box2D.min.x_val), int(d.box2D.min.y_val)
                x1,y1 = int(d.box2D.max.x_val), int(d.box2D.max.y_val)
                cv2.rectangle(img, (x0,y0), (x1,y1), (0,255,0), 2)
                cv2.putText(img, d.name, (x0, max(0,y0-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow("2D Detections", img)

            # 4) rebuild 3D boxes
            for ls in current_ls:
                vis.remove_geometry(ls, reset_bounding_box=False)
            current_ls = []
            for d in dets:
                ls = make_line_set(d)
                vis.add_geometry(ls, reset_bounding_box=False)
                current_ls.append(ls)

            # 5) poll and render
            vis.poll_events()
            vis.update_renderer()

            # 6) check exit key
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        vis.destroy_window()

if __name__ == "__main__":
    main()
