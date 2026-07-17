#!/usr/bin/env python3
import os
import cv2
import numpy as np
from glob import glob

# =============================================================
# USER PARAMETERS — MODIFY THESE AS NEEDED
# =============================================================

# Directory containing input frames (e.g. RGB images from your drones)
INPUT_DIR = "/media/sgarimella34/hercules-collect/raw_data_hercules/test10_forest_2uav_camtilt_calib_752x480_NOISY_VEGETATION/Drone1/rgb"

# Output directory — processed images will be placed here (subfolders added per method)
OUTPUT_DIR = "./denoised_output"

# Supported image extensions
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]

# Temporal processing window: number of frames before and after current frame
TEMPORAL_WINDOW = 1   # gives window size = 2*TEMPORAL_WINDOW + 1 (e.g. 3 frames)

# Adaptive-median parameters
AMF_MAX_WINDOW = 7    # must be odd (e.g. 3,5,7)
AMF_INITIAL_WINDOW = 3

# Bilateral filter parameters (for the spatial smoothing stage)
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# =============================================================
# Utility functions
# =============================================================

def list_images(dir_path):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob(os.path.join(dir_path, f"*{ext}")))
    files = sorted(files)
    return files

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def adaptive_median(img, max_window=7, initial_window=3):
    """
    Simple Adaptive Median Filter (per-channel).
    Not optimized for speed, but serves as prototype.
    Works on uint8 images.
    """
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    pad = max_window // 2
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)

    for y in range(h):
        for x in range(w):
            for c in range(3):
                window = initial_window
                val = None
                while window <= max_window:
                    half = window // 2
                    region = padded[y + pad - half : y + pad + half + 1,
                                     x + pad - half : x + pad + half + 1,
                                     c]
                    A1 = int(region.min())
                    A2 = int(region.max())
                    Amed = int(np.median(region))
                    Zxy = int(padded[y + pad, x + pad, c])

                    if A1 < Amed < A2:
                        if A1 < Zxy < A2:
                            val = Zxy
                        else:
                            val = Amed
                        break
                    else:
                        window += 2  # increase window by 2 to keep odd size
                if val is None:
                    val = Amed
                out[y, x, c] = val
    return out

def temporal_median_filter(frames, idx, window):
    """
    Given list of frames (as numpy arrays), and current index idx,
    compute median across frames idx-window ... idx+window (clamped).
    """
    h = len(frames)
    start = max(0, idx - window)
    end = min(h - 1, idx + window)
    stack = np.stack(frames[start:end+1], axis=0)
    m = np.median(stack, axis=0).astype(np.uint8)
    return m

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

# =============================================================
# Main processing
# =============================================================

def main():
    image_paths = list_images(INPUT_DIR)
    if not image_paths:
        print("No images found in", INPUT_DIR)
        return

    print("Found {} images".format(len(image_paths)))

    # Load all frames into memory (if too many, you can adapt to streaming)
    frames = [load_image(p) for p in image_paths]

    # Prepare output directories
    methods = ["adaptive_median", "temporal_median", "temporal_median_bilateral"]
    for m in methods:
        ensure_dir(os.path.join(OUTPUT_DIR, m))

    # Process each frame
    for idx, path in enumerate(image_paths):
        base = os.path.splitext(os.path.basename(path))[0]
        img = frames[idx]

        # 1) Adaptive median filter (single frame)
        amf = adaptive_median(img, max_window=AMF_MAX_WINDOW, initial_window=AMF_INITIAL_WINDOW)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "adaptive_median", f"{base}_amf.png"), amf)

        # 2) Temporal median filter
        tm = temporal_median_filter(frames, idx, TEMPORAL_WINDOW)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "temporal_median", f"{base}_tm.png"), tm)

        # 3) Temporal median + bilateral
        tm_bilat = cv2.bilateralFilter(tm, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "temporal_median_bilateral", f"{base}_tmb.png"), tm_bilat)

        if idx == 0:
            # display comparison for first frame
            comp = np.hstack((img, amf, tm, tm_bilat))
            cv2.imshow("orig | adaptive_median | temporal_median | temporal_median + bilateral", comp)
            print("Showing comparison for first frame. Close window to continue saving others.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if idx % 20 == 0:
            print(f"Processed {idx+1}/{len(frames)}")

    print("Done. Outputs in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
