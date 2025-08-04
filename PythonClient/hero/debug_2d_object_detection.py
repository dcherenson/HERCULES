#!/usr/bin/env python3
"""
Live 2D bounding box from instance segmentation mask.

  • No manual flip — Python’s OpenCV view already matches AirSim’s segmentation image orientation.  
  • Runs in a loop, fetching and displaying fresh RGB + segmentation each frame.  
  • Draws the green 2D box around your actor’s mask, and overlays the mask.

Requires:
    pip install airsim opencv-python numpy
"""

import cv2
import numpy as np
import setup_path                    # ensure cosysairsim is on PYTHONPATH
import cosysairsim as airsim

# === USER SETTINGS ===
ACTOR_NAME   = "BP_SplineHuman_Type10_C_UAID_6C6E07132D49C88102_1970519919"
CAMERA_NAME  = "front_center"
CLIENT_CLASS = airsim.MultirotorClient  # or airsim.CarClient
PORT         = 41451
SEG_ID       = 200
# =====================

def get_mask_from_segmentation(img_seg, seg_id):
    """Build a 0/1 mask for the given segmentation ID."""
    if img_seg.ndim == 3 and img_seg.shape[2] >= 3:
        return (((img_seg[:,:,0] == seg_id) &
                 (img_seg[:,:,1] == seg_id) &
                 (img_seg[:,:,2] == seg_id))
                .astype(np.uint8))
    else:
        return (img_seg == seg_id).astype(np.uint8)


def main():
    client = CLIENT_CLASS(port=PORT)
    print("Connecting to AirSim…", end="")
    client.confirmConnection()
    print("Connected!")

    # assign exactly this ID to the actor
    if client.simSetSegmentationObjectID(ACTOR_NAME, SEG_ID, False):
        print(f"[INFO] segmentation ID {SEG_ID} → {ACTOR_NAME}")
    else:
        print(f"[WARN] couldn't set seg ID {SEG_ID} on '{ACTOR_NAME}'")

    # windows
    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Segmentation", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Overlay + Box", cv2.WINDOW_NORMAL)

    while True:
        # 1) fetch both images compressed
        reqs = [
            airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene,       False, True),
            airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Segmentation, False, True)
        ]
        imgs = client.simGetImages(reqs)
        if len(imgs) < 2:
            print("[ERROR] missing images")
            break

        # 2) decode
        img_rgb = cv2.imdecode(np.frombuffer(imgs[0].image_data_uint8, np.uint8),
                               cv2.IMREAD_COLOR)
        img_seg = cv2.imdecode(np.frombuffer(imgs[1].image_data_uint8, np.uint8),
                               cv2.IMREAD_UNCHANGED)
        if img_rgb is None or img_seg is None:
            print("[ERROR] failed to decode")
            break

        # 3) build mask & find bbox
        mask = get_mask_from_segmentation(img_seg, SEG_ID) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = img_rgb.copy()
        if contours:
            # pick the largest blob
            c = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(overlay, ACTOR_NAME, (x, max(0,y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 4) colored mask overlay
        color_mask = np.zeros_like(img_rgb)
        color_mask[mask>0] = (0,255,0)
        combined = cv2.addWeighted(img_rgb, 0.7, color_mask, 0.3, 0)
        if contours:
            cv2.rectangle(combined, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(combined, ACTOR_NAME, (x, max(0,y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 5) display
        cv2.imshow("RGB", img_rgb)
        # segmentation image may be single-channel; convert for display
        seg_disp = (img_seg if img_seg.ndim==3
                    else cv2.cvtColor(img_seg, cv2.COLOR_GRAY2BGR))
        cv2.imshow("Segmentation", seg_disp)
        cv2.imshow("Overlay + Box", combined)

        # wait 50ms; break on any key
        if cv2.waitKey(50) & 0xFF != 0xFF:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
