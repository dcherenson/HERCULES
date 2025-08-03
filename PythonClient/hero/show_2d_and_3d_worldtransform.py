#!/usr/bin/env python3
import setup_path                    # ensure cosysairsim is on PYTHONPATH
import cosysairsim as airsim
import cv2, numpy as np, open3d as o3d

CAMERA_NAME = "front_center"
IMAGE_TYPE  = airsim.ImageType.Scene

def quaternion_to_rot_matrix(q):
    w,x,y,z = q.w_val, q.x_val, q.y_val, q.z_val
    n = np.linalg.norm([w,x,y,z])
    w,x,y,z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),  2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),1-2*(x*x+y*y)]
    ])

def world_T_cam_from_info(cam_info):
    p = cam_info.pose.position
    R = quaternion_to_rot_matrix(cam_info.pose.orientation)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = [p.x_val, p.y_val, p.z_val]
    return T

def _cam_to_o3d(pt):
    # FRD → RUF
    x_fwd, y_right, z_down = pt
    return [ y_right, -z_down, x_fwd ]

def main():
    client = airsim.CarClient(port=41452)
    client.confirmConnection()

    # keep Cylinder* filter
    client.simClearDetectionMeshNames(CAMERA_NAME, IMAGE_TYPE)
    # client.simAddDetectionFilterMeshName(CAMERA_NAME, IMAGE_TYPE, "Cylinder*")
    # client.simAddDetectionFilterMeshName(CAMERA_NAME, IMAGE_TYPE, "Sportscar*")  #works
    # client.simAddDetectionFilterMeshName(CAMERA_NAME, IMAGE_TYPE, "SK_Survival_Character*")  #doesnt work
    # client.simAddDetectionFilterMeshName(CAMERA_NAME, IMAGE_TYPE, "SM_vehTruck_vehicle04_No_Wheel*")  #doesnt work
    client.simAddDetectionFilterMeshName(CAMERA_NAME, IMAGE_TYPE, "BP_SplineHuman_*") #doesnt work


    client.simSetDetectionFilterRadius(CAMERA_NAME, IMAGE_TYPE, 200*100)

    cv2.namedWindow("2D Detections", cv2.WINDOW_NORMAL)
    vis = o3d.visualization.Visualizer()
    vis.create_window("3D Detections")
    ctr = vis.get_view_control()
    ctr.set_constant_z_near(0.01); ctr.set_constant_z_far(1e6)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    geoms = []
    try:
        while True:
            # ------ 2D ------
            img_r = client.simGetImages([
                airsim.ImageRequest(CAMERA_NAME, IMAGE_TYPE, False, True)
            ])[0]
            if not img_r.image_data_uint8:
                continue
            img = cv2.imdecode(np.frombuffer(img_r.image_data_uint8, np.uint8),
                               cv2.IMREAD_COLOR)

            dets = client.simGetDetections(CAMERA_NAME, IMAGE_TYPE) or []
            for d in dets:
                x0,y0 = int(d.box2D.min.x_val), int(d.box2D.min.y_val)
                x1,y1 = int(d.box2D.max.x_val), int(d.box2D.max.y_val)
                cv2.rectangle(img,(x0,y0),(x1,y1),(0,255,0),2)
                cv2.putText(img, d.name, (x0,y0-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            cv2.imshow("2D Detections", img)

            # ------ get transforms ------
            cam_info    = client.simGetCameraInfo(CAMERA_NAME)
            world_T_cam = world_T_cam_from_info(cam_info)
            cam_T_world = np.linalg.inv(world_T_cam)

            # ------ compute ground offset in world ------
            # pick lowest bottom-corner world-Z
            min_z = None
            for d in dets:
                pm, pM = d.box3D.min, d.box3D.max
                xs, ys, zs = sorted([pm.x_val,pM.x_val]), sorted([pm.y_val,pM.y_val]), sorted([pm.z_val,pM.z_val])
                # bottom corners k=0
                for i in (0,1):
                    for j in (0,1):
                        cam_pt = np.array([xs[i], ys[j], zs[0], 1.0])
                        z_w = (world_T_cam @ cam_pt)[2]
                        min_z = z_w if min_z is None or z_w<min_z else min_z
            ground_z = min_z if min_z is not None else 0.0

            # clear old
            for g in geoms:
                vis.remove_geometry(g, reset_bounding_box=False)
            geoms.clear()

            # ------ rebuild in camera frame ------
            for d in dets:
                # 1) corners in camera FRD
                pm, pM = d.box3D.min, d.box3D.max
                xs,ys,zs = sorted([pm.x_val,pM.x_val]), sorted([pm.y_val,pM.y_val]), sorted([pm.z_val,pM.z_val])
                corners_cam = np.array([[xs[i],ys[j],zs[k]]
                                         for i in(0,1) for j in(0,1) for k in(0,1)])
                # 2) world-ground align: cam→world→subtract→back to camera
                corners_world = (world_T_cam @ np.hstack((corners_cam, np.ones((8,1)))).T).T[:,:3]
                corners_world[:,2] -= ground_z
                corners_cam_aligned = (cam_T_world @ np.hstack((corners_world, np.ones((8,1)))).T).T[:,:3]
                # 3) map to Open3D axes & draw
                corners_o3d = np.array([_cam_to_o3d(pt) for pt in corners_cam_aligned])
                edges = [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]]
                ls = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(corners_o3d),
                    lines=o3d.utility.Vector2iVector(edges)
                )
                ls.colors = o3d.utility.Vector3dVector([[0,1,0]]*len(edges))
                vis.add_geometry(ls, reset_bounding_box=False)
                geoms.append(ls)

                # draw the center triad
                rp = d.relative_pose.position
                cam_c = np.array([rp.x_val, rp.y_val, rp.z_val,1.0])
                wc = (world_T_cam @ cam_c)[:3]
                wc[2] -= ground_z
                cc = (cam_T_world @ np.hstack((wc,1.0)))[:3]
                o3d_c = _cam_to_o3d(cc)
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                frame.translate(o3d_c)
                vis.add_geometry(frame, reset_bounding_box=False)
                geoms.append(frame)

            vis.poll_events(); vis.update_renderer()

            if cv2.waitKey(50)&0xFF==ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        vis.destroy_window()

if __name__ == "__main__":
    main()

