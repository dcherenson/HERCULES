#!/usr/bin/env python3
"""Compare a truth-target baseline with distributed camera tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from render_validation import CONTROLLED, load, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("truth_jsonl", type=Path)
    parser.add_argument("camera_jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    truth = load(args.truth_jsonl)
    camera = load(args.camera_jsonl)
    if not truth or not camera:
        raise SystemExit("both comparison logs must be non-empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "interpretation": (
            "Diagnostic comparison only: realistic estimator delay/noise means trajectories "
            "are not expected to be identical."
        ),
        "truth_target_nominal": metrics(truth),
        "distributed_camera_tracking": metrics(camera),
    }
    (args.output_dir / "mode_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    for records, style, label in ((truth, "-", "truth target"),
                                  (camera, "--", "distributed camera")):
        time = np.asarray([row["timestamp"] for row in records])
        maximum_slot = [max(float(row["slot_errors"][name]) for name in CONTROLLED)
                        for row in records]
        target = np.asarray([row["target_truth"]["position"][:2] for row in records])
        axes[0].plot(time, maximum_slot, style, label=label)
        axes[1].plot(target[:, 0], target[:, 1], style, label=label)
    axes[0].set_xlabel("Mission time (s)")
    axes[0].set_ylabel("Maximum slot error (m)")
    axes[0].set_title("Formation comparison")
    axes[1].set_xlabel("AirSim world NED X (m)")
    axes[1].set_ylabel("AirSim world NED Y (m)")
    axes[1].set_title("Target path comparison")
    axes[1].set_aspect("equal", adjustable="box")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.savefig(args.output_dir / "mode_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
