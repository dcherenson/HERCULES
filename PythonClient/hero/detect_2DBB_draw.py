#!/usr/bin/env python3
import torch
import cv2
import numpy as np

def main():
    # 1) Load model (downloads weights if needed)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  
    model.conf = 0.5  # confidence threshold

    # 2) Read images
    img_rgb = cv2.imread("rgb.png")                     # H×W×3 BGR 
    img_seg = cv2.imread("seg.png")                     # instance-segmentation mask

    # 3) Run inference on RGB
    results = model(img_rgb)
    detections = results.xyxy[0].cpu().numpy()          # [x1,y1,x2,y2,conf,cls]

    # 4) Draw boxes for “car” and “person”
    for *box, conf, cls in detections:
        class_name = model.names[int(cls)]
        if class_name not in ("car", "person"):
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_rgb, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img_rgb, f"{class_name} {conf:.2f}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,0), 1)

    # 5) Show and save result
    cv2.imshow("Detections", img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("detections.png", img_rgb)

if __name__ == "__main__":
    main()
