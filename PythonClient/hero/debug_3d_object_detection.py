#!/usr/bin/env python3
"""
Minimal script to:
  1. Find an object/actor by name/regex in the world.
  2. Print that object's 6DOF pose and a camera's 6DOF pose in the world frame (NED).
  3. Compute & print the camera intrinsics matrix, assuming AirSim/Unreal returns horizontal FOV.
     Vertical FOV is derived from the aspect ratio.
  4. Query and (optionally) disable any lens distortion parameters.
  5. Define a virtual 3D bounding box (L*W*H) in the actor's local frame, transform it into world
     coordinates using its pose, print the 8 world-frame corners, and visualize.
  6. Optionally project those 3D corners into the image using the camera extrinsics/intrinsics and draw
     the resulting 3D bounding box on the image (can be disabled).
"""

import math
import numpy as np
import setup_path                    # ensure cosysairsim is on PYTHONPATH
import cosysairsim as airsim
import cv2  # required for decoding compressed image

# optional visualization
try:
    import open3d as o3d
except ImportError:
    o3d = None

# === CONFIGURATION ===
ACTOR_PATTERN = "BP_SplineHuman_Type10.*"   # regex to match actor name(s)
CAMERA_NAME   = "front_center"              # camera to query
CLIENT_CLASS  = airsim.MultirotorClient      # or airsim.CarClient depending on your setup
PORT          = 41451                       # adjust if your AirSim instance uses a different port

# virtual bounding box size in meters (length, width, height)
# Length: local forward (x), Width: local right (y), Height: local up (z)
BOX_LENGTH = 0.5
BOX_WIDTH = 0.5
BOX_HEIGHT = 1.8

# toggle whether to do the 3D->2D projection and draw on the image
PROJECTION_ENABLED = False
# =====================

def quaternion_to_euler(q):
    """Convert AirSim quaternion (w, x, y, z) to roll, pitch, yaw in radians."""
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return 0.0, 0.0, 0.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # roll (x-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def quaternion_to_rotation_matrix(q):
    """Convert normalized quaternion to 3x3 rotation matrix (local -> world)."""
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # rotation matrix (Hamilton convention)
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ], dtype=float)
    return R

def print_pose(label, pose):
    if pose is None:
        print(f"{label}: <no pose>")
        return
    p, o = pose.position, pose.orientation
    roll, pitch, yaw = quaternion_to_euler(o)
    print(f"=== {label} ===")
    print(f"Position (NED): x={p.x_val:.6f}, y={p.y_val:.6f}, z={p.z_val:.6f}")
    print("Orientation quaternion (w,x,y,z): "
          f"({o.w_val:.6f}, {o.x_val:.6f}, {o.y_val:.6f}, {o.z_val:.6f})")
    print("Orientation Euler (deg): "
          f"roll={math.degrees(roll):.2f}, pitch={math.degrees(pitch):.2f}, yaw={math.degrees(yaw):.2f}")
    print()

def compute_intrinsics_from_horizontal_fov(hfov_deg, width, height):
    """
    Given horizontal FOV (deg) and image size, compute intrinsic matrix K
    and derive vertical FOV. Assumes square pixels (fx == fy).
    """
    hfov = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx,   0, cx],
                  [ 0,  fy, cy],
                  [ 0,   0,  1]], dtype=float)
    vfov = 2 * math.degrees(math.atan((height / 2.0) / fy))
    return K, vfov

def compute_bounding_box_corners_world(pose, length, width, height):
    """Compute the 8 world-frame corner positions of a box centered on the actor, oriented by its pose.
    Local box axes: +X forward, +Y right, +Z up."""
    half_l = length / 2.0
    half_w = width / 2.0
    half_h = height / 2.0

    # local corners: (±L/2, ±W/2, ±H/2)
    corners_local = np.array([
        [ half_l,  half_w,  half_h],
        [ half_l,  half_w, -half_h],
        [ half_l, -half_w,  half_h],
        [ half_l, -half_w, -half_h],
        [-half_l,  half_w,  half_h],
        [-half_l,  half_w, -half_h],
        [-half_l, -half_w,  half_h],
        [-half_l, -half_w, -half_h],
    ], dtype=float)  # shape (8,3)

    R = quaternion_to_rotation_matrix(pose.orientation)
    t = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val], dtype=float)

    world_corners = (R @ corners_local.T).T + t  # apply rotation then translation
    return world_corners

def project_world_points_to_image(world_pts, cam_pose, K):
    """
    Projects world-frame 3D points into the image using the camera extrinsics and intrinsics.
    AirSim camera frame is (x_forward, y_right, z_down):
      X_cam = R_cam_world * (X_world - C)
    where R_cam_world = (R_world_cam)^T and C is camera center in world.
    Then projection:
      u = (y_right / x_forward) * fx + cx
      v = (z_down / x_forward) * fy + cy
    """
    if cam_pose is None:
        raise ValueError("Camera pose is required for projection.")

    R_world_cam = quaternion_to_rotation_matrix(cam_pose.orientation)  # camera-to-world
    R_cam_world = R_world_cam.T  # world-to-camera

    C = np.array([cam_pose.position.x_val, cam_pose.position.y_val, cam_pose.position.z_val], dtype=float)

    X_cam = (R_cam_world @ ((world_pts - C).T)).T  # (N,3) in camera frame

    x_fwd = X_cam[:, 0]
    y_right = X_cam[:, 1]
    z_down = X_cam[:, 2]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    eps = 1e-8
    valid = x_fwd > eps

    u = np.zeros_like(x_fwd)
    v = np.zeros_like(x_fwd)
    u[valid] = (y_right[valid] / x_fwd[valid]) * fx + cx
    v[valid] = (z_down[valid] / x_fwd[valid]) * fy + cy

    pts2d = np.stack([u, v], axis=1)
    depth = x_fwd  # forward distance

    return pts2d, depth, valid

def draw_projected_box(img, pts2d, depth, valid_mask):
    """Draws the 3D bounding box (edges + corner indices) on the image in-place."""
    h, w = img.shape[:2]
    edges = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # +X face
        [4, 5], [5, 7], [7, 6], [6, 4],  # -X face
        [0, 4], [1, 5], [2, 6], [3, 7]   # connectors
    ]

    for i, j in edges:
        if depth[i] > 0 and depth[j] > 0 and valid_mask[i] and valid_mask[j]:
            pt1 = (int(round(pts2d[i, 0])), int(round(pts2d[i, 1])))
            pt2 = (int(round(pts2d[j, 0])), int(round(pts2d[j, 1])))
            if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)

    for idx in range(8):
        if depth[idx] > 0 and valid_mask[idx]:
            u = int(round(pts2d[idx, 0]))
            v = int(round(pts2d[idx, 1]))
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(img, (u, v), 4, (0, 0, 255), -1)
                cv2.putText(img, str(idx), (u + 3, v - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def main():
    np.set_printoptions(precision=4, suppress=True)

    client = CLIENT_CLASS(port=PORT)
    client.confirmConnection()

    # --- check / disable lens distortion ---
    dist_params = client.simGetDistortionParams(CAMERA_NAME)
    print("Distortion parameters [K1, K2, K3, P1, P2]:", dist_params)
    if any(abs(d) > 1e-9 for d in dist_params):
        print("Non-zero distortion detected; zeroing parameters.")
        client.simSetDistortionParams(
            CAMERA_NAME,
            {"K1": 0.0, "K2": 0.0, "K3": 0.0, "P1": 0.0, "P2": 0.0}
        )
    else:
        print("No lens distortion active (default).")
    print()

    # --- find actor by pattern ---
    scene_objs = client.simListSceneObjects(ACTOR_PATTERN)
    if not scene_objs:
        print(f"No scene objects matched '{ACTOR_PATTERN}'.")
        return
    if len(scene_objs) > 1:
        print(f"Multiple matches, using first: {scene_objs[0]}")
    actor_name = scene_objs[0]

    # --- get actor pose ---
    try:
        actor_pose = client.simGetObjectPose(actor_name, True)
    except TypeError:
        actor_pose = client.simGetObjectPose(actor_name)

    # --- get camera info & pose ---
    cam_info = client.simGetCameraInfo(CAMERA_NAME)
    cam_pose = cam_info.pose if cam_info else None

    # --- print poses ---
    print_pose(f"Actor ({actor_name})", actor_pose)
    print_pose(f"Camera ({CAMERA_NAME})", cam_pose)

    # --- acquire one image to get resolution ---
    img_resp = client.simGetImages([
        airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, True)
    ])[0]
    if not img_resp.image_data_uint8:
        print("Empty image data; cannot infer resolution.")
        return
    img = cv2.imdecode(np.frombuffer(img_resp.image_data_uint8, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to decode image.")
        return
    h, w = img.shape[:2]

    # --- compute & print intrinsics ---
    if not cam_info or cam_info.fov <= 0:
        print("Invalid camera FOV:", getattr(cam_info, "fov", None))
        return

    K, vfov = compute_intrinsics_from_horizontal_fov(cam_info.fov, w, h)
    print(f"Image resolution: width={w}, height={h}")
    print(f"Horizontal FOV (deg): {cam_info.fov:.6f}")
    print(f"Derived vertical FOV (deg): {vfov:.6f}")
    print("Intrinsic matrix K:")
    print(K)
    print()

    # --- compute bounding box corners in world frame ---
    if actor_pose is None:
        print("Actor pose unavailable; cannot compute bounding box.")
        return

    world_corners = compute_bounding_box_corners_world(
        actor_pose, BOX_LENGTH, BOX_WIDTH, BOX_HEIGHT
    )
    print(f"Bounding box dimensions (L*W*H): {BOX_LENGTH} * {BOX_WIDTH} * {BOX_HEIGHT} [meters]")
    print("Bounding box corners in world frame (NED):")
    for i, corner in enumerate(world_corners):
        x, y, z = corner
        print(f"  [{i}] x={x:.6f}, y={y:.6f}, z={z:.6f}")
    print()

    # --- optionally project 3D corners onto image and draw ---
    img_with_box = img.copy()
    if PROJECTION_ENABLED:
        if cam_pose is None:
            print("Camera pose unavailable; cannot project.")
        else:
            pts2d, depth, valid = project_world_points_to_image(world_corners, cam_pose, K)
            print("Projected 2D box corner pixel coordinates (u,v) and depth (forward):")
            for i, ((u, v), d, ok) in enumerate(zip(pts2d, depth, valid)):
                status = "visible" if ok and d > 0 else "behind/invalid"
                print(f"  [{i}] u={u:.1f}, v={v:.1f}, depth={d:.4f} ({status})")
            print()
            draw_projected_box(img_with_box, pts2d, depth, valid)
    else:
        print("Projection disabled; skipping 3D->2D projection and drawing.")
        print()

    # annotate actor name
    cv2.putText(img_with_box, f"Actor: {actor_name}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # show image
    window_name = "Projected 3D Bounding Box" if PROJECTION_ENABLED else "Camera View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img_with_box)
    print("Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- optional 3D visualization (original) ---
    if o3d:
        try:
            actor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            T_actor = np.eye(4)
            T_actor[:3, :3] = quaternion_to_rotation_matrix(actor_pose.orientation)
            T_actor[:3, 3] = np.array([actor_pose.position.x_val,
                                       actor_pose.position.y_val,
                                       actor_pose.position.z_val])
            actor_frame.transform(T_actor)

            cam_frame = None
            if cam_pose:
                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
                T_cam = np.eye(4)
                T_cam[:3, :3] = quaternion_to_rotation_matrix(cam_pose.orientation)
                T_cam[:3, 3] = np.array([cam_pose.position.x_val,
                                          cam_pose.position.y_val,
                                          cam_pose.position.z_val])
                cam_frame.transform(T_cam)

            edges = [
                [0, 1], [1, 3], [3, 2], [2, 0],
                [4, 5], [5, 7], [7, 6], [6, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            box_lines = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(world_corners),
                lines=o3d.utility.Vector2iVector(edges)
            )
            box_lines.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(edges))  # red

            geometries = [box_lines, actor_frame]
            if cam_frame:
                geometries.append(cam_frame)

            print("Displaying 3D bounding box and pose frames in Open3D. Close the window to exit.")
            o3d.visualization.draw_geometries(geometries)
        except Exception as e:
            print(f"Visualization error: {e}")
    else:
        print("open3d not installed; skipping 3D visualization.")

if __name__ == "__main__":
    main()
