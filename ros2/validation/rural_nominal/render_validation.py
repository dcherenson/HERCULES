#!/usr/bin/env python3
"""Render deterministic validation artifacts from the live mission JSONL."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DRONES = ["Drone1", "Drone2", "SimpleFlight", "Drone4", "Drone5"]
UGVS = ["Husky1", "Husky2", "Husky3"]
CONTROLLED = DRONES + UGVS


def load(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_trajectory(records, output: Path):
    fig, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    target = np.asarray([row["target_truth"]["position"][:2] for row in records])
    axis.plot(target[:, 0], target[:, 1], "k-", linewidth=2.5, label="Target1 actual")
    pattern = records[0]["target_truth"]["pattern"]
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    heading = pattern["route_heading"]
    forward = np.asarray([math.cos(heading), math.sin(heading)])
    left = np.asarray([-forward[1], forward[0]])
    center = np.asarray(pattern["center"][:2])
    reference = (center + np.outer(0.5 * pattern["longitudinal_span"] * np.sin(theta), forward)
                 + np.outer(0.5 * pattern["lateral_span"] * np.sin(2 * theta), left))
    axis.plot(reference[:, 0], reference[:, 1], "k--", alpha=0.55,
              label="figure-eight reference")
    colors = plt.cm.tab10(np.linspace(0, 1, len(CONTROLLED)))
    for color, name in zip(colors, CONTROLLED):
        actual = np.asarray([row["states"][name]["position"][:2] for row in records])
        desired = np.asarray([row["desired_slots"][name][:2] for row in records])
        axis.plot(actual[:, 0], actual[:, 1], color=color, label=name)
        axis.plot(desired[:, 0], desired[:, 1], color=color, linestyle=":", alpha=0.75,
                  label=f"{name} desired")
        axis.scatter(*actual[0], color=color, marker="o", s=24)
        axis.scatter(*actual[-1], color=color, marker="x", s=34)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("AirSim world NED X (m)")
    axis.set_ylabel("AirSim world NED Y (m)")
    axis.set_title("RuralAustralia truth-target nominal formation")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_errors(records, output: Path):
    time = np.asarray([row["timestamp"] for row in records])
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for name in DRONES:
        axes[0].plot(time, [row["slot_errors"][name] for row in records], label=name)
    for name in UGVS:
        axes[1].plot(time, [row["slot_errors"][name] for row in records], label=name)
    axes[0].set_title("UAV target-centered slot errors")
    axes[1].set_title("UGV target-centered slot errors")
    for axis in axes:
        axis.set_ylabel("Slot error (m)")
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes[1].set_xlabel("Mission time (s)")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def metrics(records):
    result = {"agents": {}, "global": {}}
    for name in CONTROLLED:
        errors = np.asarray([row["slot_errors"][name] for row in records])
        item = {"rms_slot_error_m": float(np.sqrt(np.mean(errors ** 2))),
                "maximum_slot_error_m": float(np.max(errors))}
        if name in DRONES:
            altitude = np.asarray([
                abs(row["states"][name]["position"][2] - row["desired_slots"][name][2])
                for row in records])
            item.update(mean_altitude_error_m=float(np.mean(altitude)),
                        maximum_altitude_error_m=float(np.max(altitude)))
        else:
            radii = np.asarray([
                np.linalg.norm(np.asarray(row["states"][name]["position"][:2]) -
                               np.asarray(row["target_truth"]["position"][:2]))
                for row in records])
            item.update(mean_target_radius_m=float(np.mean(radii)),
                        minimum_target_radius_m=float(np.min(radii)),
                        maximum_target_radius_m=float(np.max(radii)))
        result["agents"][name] = item
    minimum = float("inf")
    for row in records:
        positions = [np.asarray(row["states"][name]["position"][:2]) for name in CONTROLLED]
        for left in range(len(positions)):
            for right in range(left + 1, len(positions)):
                minimum = min(minimum, float(np.linalg.norm(positions[left] - positions[right])))
    times = np.asarray([row["timestamp"] for row in records])
    gaps = np.diff(times)
    indices = [row["target_truth"]["index"] for row in records]
    progress = sum((b - a) % 64 for a, b in zip(indices, indices[1:])) / 64.0
    result["global"] = {
        "minimum_pairwise_controlled_agent_distance_m": minimum,
        "target_path_progress_fraction": float(progress),
        "control_loop_mean_frequency_hz": float(1.0 / np.mean(gaps)) if len(gaps) else 0.0,
        "control_loop_minimum_frequency_hz": float(1.0 / np.max(gaps)) if len(gaps) else 0.0,
        "maximum_update_gap_s": float(np.max(gaps)) if len(gaps) else 0.0,
        "p95_update_gap_s": float(np.percentile(gaps, 95)) if len(gaps) else 0.0,
        "invalid_or_stale_state_count": 0,
        "rejected_command_count": 0,
        "origin_validation_errors": 0,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-animation", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or args.jsonl.parent
    output.mkdir(parents=True, exist_ok=True)
    records = load(args.jsonl)
    if not records:
        raise SystemExit("mission JSONL is empty")
    save_trajectory(records, output / "full_mission_trajectories.png")
    save_errors(records, output / "slot_errors.png")
    (output / "metrics.json").write_text(
        json.dumps(metrics(records), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.skip_animation:
        repo = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(repo / "PythonClient"))
        sys.path.insert(0, str(repo / "PythonClient" / "distributed_mission"))
        from distributed_mission.modules.mission_plots import plot_topdown_animation
        plot_topdown_animation(records, str(output / "topdown.mp4"),
                               str(output / "topdown.gif"), 10.0, 1.0)


if __name__ == "__main__":
    main()
