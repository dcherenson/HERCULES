#!/usr/bin/env python3
# Green 3‑D wire‑frames via depth‑band masking (no Segmentation RPC).

import setup_path, cosysairsim as airsim
import numpy as np, cv2, math, time

VEHICLE   = "Husky1"
CAM       = "front_center"
MESHES    = ["BP_VehicleAI*","Sportscar*","Sedan*","Van1*","Copcar*",
             "Hugetruck*","Sedan1*","SUV1*","Milkvan*","Sedan2*",
             "Pickuptruck*","Garbagetruck*","BP_SplineHuman*"]
RADIUS_CM = 200*100
MAX_D     = 80.0
STEP      = 3              # subsample factor
LOW_PCT   = 15             # keep depths between 15‑th …
HIGH_PCT  = 85             # … and 85‑th percentile

EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
         (0,4),(1,5),(2,6),(3,7)]

def intrinsics(w,h,fov):
    f = w/(2*math.tan(math.radians(fov)/2))
    return f,f,w/2,h/2

def draw_wire(img, P, col=(0,255,0)):
    for i,j in EDGES:
        if P[i] and P[j]:
            cv2.line(img,P[i],P[j],col,2,cv2.LINE_AA)

cli = airsim.CarClient(port=41452); cli.confirmConnection()
cli.simSetDetectionFilterRadius(CAM,airsim.ImageType.Scene,RADIUS_CM,VEHICLE)
cli.simClearDetectionMeshNames(CAM,airsim.ImageType.Scene,VEHICLE)
for p in MESHES:
    cli.simAddDetectionFilterMeshName(CAM,airsim.ImageType.Scene,p,VEHICLE)

rgb0 = cv2.imdecode(airsim.string_to_uint8_array(
        cli.simGetImage(CAM,airsim.ImageType.Scene,VEHICLE)),cv2.IMREAD_COLOR)
Hrgb,Wrgb = rgb0.shape[:2]
fov = cli.simGetCameraInfo(CAM,VEHICLE).fov
fx_r,fy_r,cx_r,cy_r = intrinsics(Wrgb,Hrgb,fov)

cv2.namedWindow("3D‑bbox",cv2.WINDOW_NORMAL)

while True:
    rsp_rgb,rsp_d = cli.simGetImages([
        airsim.ImageRequest(CAM,airsim.ImageType.Scene,False,True),
        airsim.ImageRequest(CAM,airsim.ImageType.DepthPerspective,True,False)
    ], vehicle_name=VEHICLE)

    rgb = cv2.imdecode(airsim.string_to_uint8_array(rsp_rgb.image_data_uint8),
                       cv2.IMREAD_COLOR)
    Wd,Hd = rsp_d.width,rsp_d.height
    depth = np.asarray(rsp_d.image_data_float,np.float32).reshape(Hd,Wd)

    fx_d,fy_d,cx_d,cy_d = intrinsics(Wd,Hd,fov)
    sx,sy = Wd/Wrgb, Hd/Hrgb

    for det in cli.simGetDetections(CAM,airsim.ImageType.Scene,VEHICLE) or []:
        dx1,dy1 = int(det.box2D.min.x_val*sx), int(det.box2D.min.y_val*sy)
        dx2,dy2 = int(det.box2D.max.x_val*sx), int(det.box2D.max.y_val*sy)
        if dx1>=dx2 or dy1>=dy2: continue

        patch = depth[dy1:dy2:STEP, dx1:dx2:STEP]
        z = patch[(patch>0)&(patch<MAX_D)]
        if z.size < 30: continue

        lo, hi = np.percentile(z,[LOW_PCT,HIGH_PCT])
        m = (patch>=lo)&(patch<=hi)
        if not m.any(): continue

        vs,us = np.where(m)
        Z = patch[vs,us]
        us = us*STEP+dx1; vs=vs*STEP+dy1

        X = Z
        Y = (us-cx_d)*Z/fx_d
        Z = (vs-cy_d)*Z/fy_d           # +Z down

        mins = np.array([X.min(), Y.min(), Z.min()])
        maxs = np.array([X.max(), Y.max(), Z.max()])
        c = np.array([[maxs[0],maxs[1],mins[2]],
                      [mins[0],maxs[1],mins[2]],
                      [mins[0],mins[1],mins[2]],
                      [maxs[0],mins[1],mins[2]],
                      [maxs[0],maxs[1],maxs[2]],
                      [mins[0],maxs[1],maxs[2]],
                      [mins[0],mins[1],maxs[2]],
                      [maxs[0],mins[1],maxs[2]]])

        U = (fx_r*c[:,1]/c[:,0] + cx_r).astype(int)
        V = (fy_r*c[:,2]/c[:,0] + cy_r).astype(int)
        pts=[ (U[i],V[i]) if c[i,0]>0 else None for i in range(8) ]
        draw_wire(rgb,pts)

    cv2.imshow("3D‑bbox",cv2.resize(rgb,(1280,720)))
    if cv2.waitKey(1)&0xFF==ord('q'): break

cv2.destroyAllWindows()
