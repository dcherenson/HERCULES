#!/usr/bin/env python3

import cv2
import numpy as np

# ============================================================
# USER PARAMETER: Set the input image path here
# ============================================================
INPUT_IMAGE_PATH = "/media/sgarimella34/hercules-collect/raw_data_hercules/test10_forest_2uav_camtilt_calib_752x480_NOISY_VEGETATION/Drone1/rgb/9.850000.png"

# ============================================================
# Load image
# ============================================================
img = cv2.imread(INPUT_IMAGE_PATH)

if img is None:
    raise FileNotFoundError(f"Could not load image: {INPUT_IMAGE_PATH}")

print("Loaded:", INPUT_IMAGE_PATH)

# ============================================================
# Apply edge-preserving denoising (bilateral filter)
# ============================================================
# OpenCV bilateral filter parameters:
# d   — diameter of pixel neighborhood (use 5–9 for moderate smoothing)
# sigmaColor — filter sigma in color space (how dissimilar colors can mix)
# sigmaSpace — filter sigma in coordinate space (how far neighbors influence)
d = 7
sigmaColor = 75
sigmaSpace = 75

output = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

# ============================================================
# Stack images for visual comparison
# ============================================================
comparison = np.hstack((img, output))

# ============================================================
# Display
# ============================================================
cv2.imshow("Before (left)  |  After Bilateral Filter (right)", comparison)
print("Press any key to close window...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# ============================================================
# Save output image
# ============================================================
out_path = INPUT_IMAGE_PATH.replace(".png", "_bilateral.png").replace(".jpg", "_bilateral.jpg")
cv2.imwrite(out_path, output)
print("Saved processed image to:", out_path)
