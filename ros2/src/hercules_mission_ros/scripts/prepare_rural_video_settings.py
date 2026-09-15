#!/usr/bin/env python3
"""Create an AirSim settings override with the ROS recording cameras."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def apply_recording_settings(settings: dict[str, Any], record_uav: str, record_ugv: str,
                             width: int, height: int, fps: float) -> dict[str, Any]:
    """Add only recording cameras, preserving perception camera entries."""
    vehicles = settings.setdefault("Vehicles", {})
    for name in (record_uav, record_ugv):
        if name not in vehicles:
            raise ValueError("vehicle {!r} is absent from AirSim settings".format(name))
    camera_defaults = settings.setdefault("CameraDefaults", {})
    captures = camera_defaults.setdefault("CaptureSettings", [])
    scene = next((item for item in captures
                  if int(item.get("ImageType", 0)) == 0), None)
    if scene is None:
        scene = {"ImageType": 0}
        captures.append(scene)
    scene["Width"], scene["Height"] = int(width), int(height)
    scene.setdefault("MotionBlurAmount", 0)
    cameras = vehicles[record_uav].setdefault("Cameras", {})
    if not isinstance(cameras, dict):
        raise ValueError("recording UAV Cameras setting must be an object")
    cameras["mission_follow"] = {
        "External": True,
        "ExternalLocal": False,
        "X": 0.0, "Y": 0.0, "Z": 0.0,
        "Roll": 0.0, "Pitch": 0.0, "Yaw": 0.0,
        "CaptureSettings": [{
            "ImageType": 0,
            "Width": int(width),
            "Height": int(height),
            "FOV_Degrees": 90.0,
            "MotionBlurAmount": 0,
            "LumenGIEnable": False,
            "LumenReflectionEnable": False,
        }],
    }
    recording = dict(settings.get("Recording") or {})
    recording.update({
        "Enabled": False,
        "RecordOnMove": False,
        "RecordInterval": 1.0 / float(fps),
        "Folder": "",
        "Cameras": [
            {"CameraName": "mission_follow", "ImageType": 0,
             "Compress": True, "VehicleName": record_uav},
            {"CameraName": "front_center", "ImageType": 0,
             "Compress": True, "VehicleName": record_uav},
            {"CameraName": "front_center", "ImageType": 0,
             "Compress": True, "VehicleName": record_ugv},
        ],
    })
    settings["Recording"] = recording
    return settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-uav", default="Drone1")
    parser.add_argument("--record-ugv", default="Husky1")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("width, height, and fps must be positive")
    settings = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        parser.error("source settings must contain a JSON object")
    output = apply_recording_settings(
        copy.deepcopy(settings), args.record_uav, args.record_ugv,
        args.width, args.height, args.fps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
