#!/usr/bin/env python3
"""
Draw 2‑D (red) and 3‑D (green, rebuilt from depth) bounding boxes
for Hercules / Cosys‑AirSim. Quit with ‘q’.
"""

import setup_path, cosysairsim as airsim
import numpy as np, cv2, math, time

# ------------------------------------------------ CONFIG --------------------
VEHICLE = "Husky1"
CAM     = "front_center"
MESHES  = ["BP_VehicleAI*","Sportscar*","Sedan*","Van1*","Copcar*",
           "Hugetruck*","Sedan1*","SUV1*","Milkvan*","Sedan2*",
           "Pickuptruck*","Garbagetruck*","BP_SplineHuman*"]
RADIUS_CM = 200*100          # 200 m radius – API uses centimetres
MAX_DEPTH = 80.0             # ignore points farther than this (m)
STEP      = 4                # depth subsampling factor

EDGES = [(0,1),(1,2),(2,3),(3,0),
         (4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]

# ------------------------------------------------ HELPERS -------------------
def make_K(w,h,hfov):
    f = w / (2*math.tan(math.radians(hfov)/2))
    return f, np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], np.float32)

def project_xyz(K, xyz):                         # AirSim cam: X fwd, Y right, Z down
    X,Y,Z = xyz[:,0], xyz[:,1], xyz[:,2]
    good  = X > 0
    u = (K[0,0] * (Y/X) + K[0,2]).astype(int)
    v = (K[1,1] * (Z/X) + K[1,2]).astype(int)
    pts = [None]*8
    for i,g in enumerate(good):
        if g: pts[i] = (u[i], v[i])
    return pts

def draw_box(img, pts, color=(0,255,0)):
    for i,j in EDGES:
        if pts[i] and pts[j]:
            cv2.line(img, pts[i], pts[j], color, 2, cv2.LINE_AA)

# ------------------------------------------------ CONNECT -------------------
client = airsim.CarClient(port=41452); client.confirmConnection()

# filters for the meshes we care about
client.simSetDetectionFilterRadius(CAM, airsim.ImageType.Scene,
                                   RADIUS_CM, VEHICLE)
client.simClearDetectionMeshNames(CAM, airsim.ImageType.Scene, VEHICLE)
for p in MESHES:
    client.simAddDetectionFilterMeshName(CAM, airsim.ImageType.Scene, p, VEHICLE)

# one RGB frame → base resolution + intrinsics
png_rgb  = client.simGetImage(CAM, airsim.ImageType.Scene, VEHICLE)
rgb0     = cv2.imdecode(airsim.string_to_uint8_array(png_rgb), cv2.IMREAD_COLOR)
H_rgb, W_rgb = rgb0.shape[:2]
f, K = make_K(W_rgb, H_rgb, client.simGetCameraInfo(CAM, VEHICLE).fov)

cv2.namedWindow("3D‑bbox", cv2.WINDOW_NORMAL)

# ------------------------------------------------ MAIN LOOP -----------------
while True:
    # RGB & depth in one call (no vehicle_name in ImageRequest constructor)
    requests = [
        airsim.ImageRequest(CAM, airsim.ImageType.Scene, False, True),
        airsim.ImageRequest(CAM, airsim.ImageType.DepthPerspective, True, False)
    ]
    rgb_rsp, depth_rsp = client.simGetImages(requests, vehicle_name=VEHICLE)

    rgb = cv2.imdecode(airsim.string_to_uint8_array(rgb_rsp.image_data_uint8),
                       cv2.IMREAD_COLOR)

    # reshape depth with *its own* resolution  :contentReference[oaicite:3]{index=3}
    W_d, H_d = depth_rsp.width, depth_rsp.height
    depth = np.array(depth_rsp.image_data_float, dtype=np.float32
                     ).reshape(H_d, W_d)

    # scale factors from RGB → depth pixel coords
    sx, sy = W_d / W_rgb, H_d / H_rgb

    # 2‑D detections
    dets = client.simGetDetections(CAM, airsim.ImageType.Scene, VEHICLE) or []

    for d in dets:
        # red rectangle in RGB frame
        x1,y1 = int(d.box2D.min.x_val), int(d.box2D.min.y_val)
        x2,y2 = int(d.box2D.max.x_val), int(d.box2D.max.y_val)
        cv2.rectangle(rgb,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(rgb,d.name,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

        # corresponding rectangle in depth frame
        dx1, dy1 = int(x1*sx), int(y1*sy)
        dx2, dy2 = int(x2*sx), int(y2*sy)

        patch = depth[dy1:dy2:STEP, dx1:dx2:STEP]
        mask  = (patch>0) & (patch<MAX_DEPTH)
        if not mask.any(): continue

        vs, us = np.where(mask)
        Z = patch[vs, us]
        us = us*STEP + dx1
        vs = vs*STEP + dy1

        X = Z
        Y = (us - W_d/2) * Z / f
        Z = (vs - H_d/2) * Z / f

        mins = np.array([X.min(), Y.min(), Z.min()], np.float32)
        maxs = np.array([X.max(), Y.max(), Z.max()], np.float32)
        corners = np.array([
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]]
        ], np.float32)

        pts = project_xyz(K, corners)
        draw_box(rgb, pts)

    cv2.imshow("3D‑bbox", cv2.resize(rgb,(1280,720)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
