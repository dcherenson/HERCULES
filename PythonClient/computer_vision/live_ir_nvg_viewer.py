"""Live side-by-side FLIR + NVG viewer using the new server-side synthetic
image types (no ROS2). Run with the sim up; q or ESC quits."""

import sys

import cv2
import numpy as np

import hercules_cosysairsim as airsim

CAMERA = sys.argv[1] if len(sys.argv) > 1 else "front_center"

client = airsim.MultirotorClient()
client.confirmConnection()

requests = [
    airsim.ImageRequest(CAMERA, airsim.ImageType.ThermalIR, False, False),
    airsim.ImageRequest(CAMERA, airsim.ImageType.NightVision, False, False),
]

cv2.namedWindow("FLIR | NVG", cv2.WINDOW_NORMAL)
while True:
    responses = client.simGetImages(requests)
    panels = []
    for r in responses:
        if len(r.image_data_uint8) != r.width * r.height * 3:
            print(f"empty response: {r.message!r}")
            continue
        img = np.frombuffer(r.image_data_uint8, dtype=np.uint8).reshape(r.height, r.width, 3)
        panels.append(img[:, :, ::-1])  # RGB -> BGR for cv2
    if panels:
        cv2.imshow("FLIR | NVG", cv2.hconcat(panels))
    if cv2.waitKey(30) & 0xFF in (27, ord("q")):
        break

cv2.destroyAllWindows()
