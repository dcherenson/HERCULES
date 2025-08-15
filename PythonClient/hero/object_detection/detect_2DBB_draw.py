#!/usr/bin/env python3

import torch
import cv2
import numpy as np

def main():
    # 1) Load YOLOv5s model from PyTorch Hub
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.3  # allow detections down to 30% confidence

    # 2) Read your RGB image and check it loaded correctly
    rgb_path = "/home/sgarimella34/multi-robot-coordination/collaborative-perception-BEVP/datasets/dair_v2x_synth/cooperative/vehicle-side/image/100.png"
    img_bgr  = cv2.imread(rgb_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read RGB image at {rgb_path}")

    # 3) Convert BGR → RGB for YOLO
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 4) Run inference
    results    = model(img_rgb)
    detections = results.xyxy[0].cpu().numpy()  # each row: [x1, y1, x2, y2, conf, cls]

    # 5) Draw boxes for cars and pedestrians on the original BGR image
    for *box, conf, cls in detections:
        class_name = model.names[int(cls)]
        if class_name not in ("car", "person"):
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(
            img_bgr, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    # 6) Display and save the result
    cv2.imshow("Detections", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("detections.png", img_bgr)
    print("Saved detection result to detections.png")

if __name__ == "__main__":
    main()
