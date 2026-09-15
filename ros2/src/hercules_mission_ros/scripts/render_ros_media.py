#!/usr/bin/env python3
"""Render one ROS mission's media into a self-contained run directory.

The top-down renderer and camera encoder are shared with the Python mission.
This command intentionally writes files for one run only; a reference log is
used solely to derive a common display configuration for later comparisons.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


def _source_modules() -> tuple[Any, Any, Any, Any]:
    repo = Path(os.environ.get("HERCULES_REPO_ROOT", "/workspaces/hercules"))
    roots = [repo]
    roots.extend(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "PythonClient" / "distributed_mission").is_dir()
    )
    for root in roots:
        for path in (root / "PythonClient", root / "PythonClient" / "distributed_mission"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
    from modules.mission_plots import (  # pylint: disable=import-outside-toplevel
        _topdown_limits,
        load_mission_records,
        plot_topdown_animation,
    )
    from modules.video_recording import render_recordings  # pylint: disable=import-outside-toplevel
    return load_mission_records, plot_topdown_animation, render_recordings, _topdown_limits


def _route_heading(records: Sequence[Mapping[str, Any]]) -> float:
    for record in records:
        target = record.get("target_truth") or record.get("target")
        if isinstance(target, Mapping):
            pattern = target.get("pattern")
            if isinstance(pattern, Mapping) and pattern.get("route_heading") is not None:
                return float(pattern["route_heading"])
    return 0.0


def _initial_origin(records: Sequence[Mapping[str, Any]]) -> list[float]:
    states = records[0].get("states") or {}
    positions = [state.get("position") for state in states.values()
                 if isinstance(state, Mapping) and state.get("position") is not None]
    if not positions:
        return [0.0, 0.0]
    return np.asarray(positions, dtype=float).reshape((-1, 3)).mean(axis=0)[:2].tolist()


def _display_config(records: Sequence[Mapping[str, Any]], references: Sequence[Sequence[Mapping[str, Any]]],
                    heading: float | None, origin: Sequence[float] | None) -> dict[str, Any]:
    _, _, _, topdown_limits = _source_modules()
    all_records = list(records)
    for reference in references:
        all_records.extend(reference)
    route_heading = float(heading) if heading is not None else _route_heading(records)
    display_origin = [float(value) for value in (origin if origin is not None else _initial_origin(records))]
    names = sorted({str(name) for item in all_records for name in (item.get("states") or {})})
    lower, upper = topdown_limits(
        all_records, names, route_heading, np.asarray(display_origin, dtype=float), include_goal=False,
    )
    return {
        "route_heading": route_heading,
        "origin": display_origin,
        "bounds": [[float(lower[0]), float(lower[1])], [float(upper[0]), float(upper[1])]],
    }


def _read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("display configuration must be a JSON object")
    for key in ("route_heading", "origin", "bounds"):
        if key not in value:
            raise ValueError("display configuration is missing {}".format(key))
    return dict(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="ROS mission JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, help="optional timestamped AirSim PNG directory")
    parser.add_argument("--reference-log", type=Path, action="append", default=[],
                        help="additional log used only to compute shared bounds")
    parser.add_argument("--display-config", type=Path, help="existing shared display JSON")
    parser.add_argument("--display-config-out", type=Path, help="write computed shared display JSON")
    parser.add_argument("--route-heading", type=float)
    parser.add_argument("--origin", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--uav", default="Drone1")
    parser.add_argument("--ugv", default="Husky1")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--gif-fps", type=float, default=10.0)
    parser.add_argument("--gif-height", type=int, default=540)
    parser.add_argument("--playback-speed", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()
    if args.video_fps <= 0 or args.gif_fps <= 0 or args.gif_height <= 0 or args.playback_speed <= 0:
        parser.error("video rates, GIF height, and playback speed must be positive")
    if args.width <= 0 or args.height <= 0:
        parser.error("video dimensions must be positive")

    load_records, plot_topdown_animation, render_recordings, _ = _source_modules()
    records = load_records(str(args.log))
    if not records:
        raise SystemExit("mission JSONL is empty: {}".format(args.log))
    references = [load_records(str(path)) for path in args.reference_log]
    if any(not value for value in references):
        raise SystemExit("a reference mission JSONL is empty")
    if args.display_config:
        config = _read_config(args.display_config)
    else:
        config = _display_config(records, references, args.route_heading, args.origin)
    if args.display_config_out:
        args.display_config_out.parent.mkdir(parents=True, exist_ok=True)
        args.display_config_out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    topdown_mp4 = args.output_dir / "topdown.mp4"
    topdown_gif = args.output_dir / "topdown.gif"
    plot_topdown_animation(
        records, str(topdown_mp4), str(topdown_gif),
        fps=1.0 / max(float(records[0].get("dt", 0.1)), 1e-6),
        playback_speed=args.playback_speed,
        route_heading=float(config["route_heading"]),
        display_origin=config["origin"],
        display_bounds=config["bounds"],
    )
    outputs = [str(topdown_mp4), str(topdown_gif)]
    if args.staging_dir:
        outputs.extend(render_recordings(
            str(args.staging_dir), str(args.output_dir), "mission",
            records, args.uav, args.ugv,
            width=args.width, height=args.height,
            fps=args.video_fps, gif_height=args.gif_height,
            gif_fps=args.gif_fps, keep_frames=args.keep_frames,
            playback_speed=args.playback_speed,
            map_name="rural_australia",
        ))
    manifest = {
        "implementation": "ros2",
        "log": str(args.log),
        "reference_logs": [str(path) for path in args.reference_log],
        "rows": len(records),
        "display": config,
        "video": {
            "width": args.width,
            "height": args.height,
            "capture_fps": args.video_fps,
            "gif_fps": args.gif_fps,
            "playback_speed": args.playback_speed,
        },
        "outputs": outputs,
    }
    (args.output_dir / "media_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
