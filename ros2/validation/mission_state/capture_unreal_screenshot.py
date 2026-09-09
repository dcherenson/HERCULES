#!/usr/bin/env python3
"""Capture the visible desktop as supplementary Unreal validation evidence."""

import argparse
import pathlib

from PIL import ImageGrab


parser = argparse.ArgumentParser()
parser.add_argument("output", type=pathlib.Path)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
image = ImageGrab.grab(all_screens=True)
image.save(args.output)
print(args.output)
