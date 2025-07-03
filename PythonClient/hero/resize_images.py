#!/usr/bin/env python3
"""
resize_images.py

Resize all images in a folder to a specified resolution and optionally convert them to grayscale,
while preserving the original filenames and filesystem timestamps.

Usage:
    python3 resize_images.py INPUT_DIR --width WIDTH --height HEIGHT [--grayscale] [--output-dir OUTPUT_DIR]

Example:
    python3 resize_images.py   /media/sgarimella34/hercules-collect/raw_data_hercules/test1_1husky/rgb   --width 752 
    --height 480   --output-dir /media/sgarimella34/hercules-collect/raw_data_hercules/test1_1husky/rgb_lowres --grayscale
"""
import os
import argparse
from PIL import Image

def process_images(input_dir, width, height, grayscale=False, output_dir=None):
    # Determine where to save processed images
    save_dir = output_dir or input_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue
        src_path = os.path.join(input_dir, fname)
        dst_path = os.path.join(save_dir, fname)

        # Preserve original filesystem timestamps
        stat = os.stat(src_path)
        original_atime = stat.st_atime
        original_mtime = stat.st_mtime

        # Open, convert, resize, and save
        with Image.open(src_path) as img:
            # Convert to desired mode
            if grayscale:
                img = img.convert('L')  # 8-bit pixels, black and white
            else:
                img = img.convert('RGB')
            # Resize using high-quality downsampling filter
            img_resized = img.resize((width, height), Image.LANCZOS)
            img_resized.save(dst_path)

        # Restore timestamps on saved file
        os.utime(dst_path, (original_atime, original_mtime))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resize images to specified size and optionally convert to grayscale.'
    )
    parser.add_argument('input_dir', help='Folder containing images to process')
    parser.add_argument('--width', type=int, required=True, help='Target width (pixels)')
    parser.add_argument('--height', type=int, required=True, help='Target height (pixels)')
    parser.add_argument('--grayscale', action='store_true', help='Convert images to grayscale')
    parser.add_argument('--output-dir', help='Directory to save processed images (default: overwrite in place)')
    args = parser.parse_args()

    process_images(
        args.input_dir,
        args.width,
        args.height,
        grayscale=args.grayscale,
        output_dir=args.output_dir
    )
