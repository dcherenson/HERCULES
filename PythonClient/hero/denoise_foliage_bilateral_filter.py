#!/usr/bin/env python3
import os
import cv2
import numpy as np

# ============================================================
# USER PARAMETERS — adjust these paths
# ============================================================
INPUT_DIR = "/media/sgarimella34/hercules-collect/raw_data_hercules/test10_forest_2uav_camtilt_calib_752x480_NOISY_VEGETATION/Drone1/rgb"
OUTPUT_DIR = "/media/sgarimella34/hercules-collect/raw_data_hercules/test10_forest_2uav_camtilt_calib_752x480_NOISY_VEGETATION/Drone1/rgb_denoised"

# Filter parameters
# Strong bilateral filter
d = 9
sigmaColor = 200
sigmaSpace = 200

# Supported extensions
EXTENSIONS = [".png", ".jpg", ".jpeg"]

# ============================================================
# Ensure output directory exists
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Process each image
# ============================================================
for fname in os.listdir(INPUT_DIR):
    base, ext = os.path.splitext(fname)
    if ext.lower() not in EXTENSIONS:
        continue

    in_path  = os.path.join(INPUT_DIR, fname)
    out_path = os.path.join(OUTPUT_DIR, base + "_bilateral" + ext)

    img = cv2.imread(in_path)
    if img is None:
        print("Warning: could not load", in_path)
        continue

    # Apply bilateral filter
    output = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

    # Save filtered image
    cv2.imwrite(out_path, output)

    # Optional: print progress
    print("Processed:", fname, "→", os.path.basename(out_path))

print("Done. All filtered images saved to:", OUTPUT_DIR)
