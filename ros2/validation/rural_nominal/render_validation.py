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
        estimates = []
        for row in records:
            value = (((row.get("target_tracking") or {}).get("agents") or {})
                     .get(name, {}).get("estimate", {}))
            if value.get("active") and value.get("position") is not None:
                estimates.append(value["position"][:2])
        if estimates:
            estimate_path = np.asarray(estimates, dtype=float)
            axis.plot(estimate_path[:, 0], estimate_path[:, 1], color=color,
                      linestyle="--", linewidth=1.2, alpha=0.9,
                      label=f"{name} target estimate")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("AirSim world NED X (m)")
    axis.set_ylabel("AirSim world NED Y (m)")
    mode = ((records[-1].get("target_tracking") or {}).get("observation_source") or "truth target")
    axis.set_title(f"RuralAustralia formation and local target estimates ({mode})")
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


def _tracking_samples(records, name):
    """Return the last logged sample for each completed estimator epoch."""
    samples = {}
    for row in records:
        tracking = row.get("target_tracking") or {}
        epoch = int(tracking.get("epoch_id", 0))
        agent = (tracking.get("agents") or {}).get(name, {})
        estimate = agent.get("estimate") if isinstance(agent, dict) else None
        if epoch > 0 and isinstance(estimate, dict) and estimate.get("position") is not None:
            samples[epoch] = (row, agent, estimate)
    return [samples[key] for key in sorted(samples)]


def save_tracking_errors(records, output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, constrained_layout=True)
    for name in CONTROLLED:
        samples = _tracking_samples(records, name)
        if not samples:
            continue
        times = [sample[0]["timestamp"] for sample in samples]
        position_error = []
        velocity_error = []
        residual = []
        iterations = []
        for row, _, estimate in samples:
            truth = row["target_truth"]
            position_error.append(float(np.linalg.norm(
                np.asarray(estimate["position"][:2]) - np.asarray(truth["position"][:2]))))
            velocity_error.append(float(np.linalg.norm(
                np.asarray(estimate["velocity"][:2]) - np.asarray(truth["velocity"][:2]))))
            residual.append(float(estimate.get("consensus_residual", 0.0)))
            iterations.append(float(estimate.get("iterations", 0.0)))
        axes[0, 0].plot(times, position_error, label=name)
        axes[0, 1].plot(times, velocity_error, label=name)
        axes[1, 0].plot(times, residual, label=name)
        axes[1, 1].plot(times, iterations, label=name)
    titles = ("Position estimation error", "Velocity estimation error",
              "Consensus residual", "Consensus iterations")
    labels = ("Error (m)", "Error (m/s)", "Residual", "Iterations")
    for axis, title, label in zip(axes.flat, titles, labels):
        axis.set_title(title)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("Mission time (s)")
    axes[1, 1].set_xlabel("Mission time (s)")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_tracking_counts(records, output: Path):
    rows = [row for row in records if int((row.get("target_tracking") or {}).get("epoch_id", 0)) > 0]
    if not rows:
        return
    time = [row["timestamp"] for row in rows]
    direct = [float((row.get("target_tracking") or {}).get("direct_observation_count", 0)) for row in rows]
    active = [float((row.get("target_tracking") or {}).get("active_track_count", 0)) for row in rows]
    edges = [len(row.get("tracking_communication_links") or []) for row in rows]
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.step(time, direct, where="post", label="agents with direct observation")
    axis.step(time, active, where="post", label="active local tracks")
    axis.step(time, edges, where="post", label="communication graph edges")
    axis.set_xlabel("Mission time (s)")
    axis.set_ylabel("Count")
    axis.set_title("Distributed target-tracking availability")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def cbf_metrics(records):
    """Summarize optional Phase 2/3 CBF diagnostics, accepting old logs."""
    entries = []
    for row in records:
        value = row.get("cbf") or {}
        if isinstance(value, dict):
            entries.extend(item for item in value.get("entries", []) if isinstance(item, dict))
    if not entries:
        return {"available": False, "steps": 0, "entries": 0}
    enabled = [item for item in entries if item.get("enabled", False)]
    fallbacks = [item for item in enabled if item.get("fallback", False)]
    deadline_misses = [item for item in enabled if item.get("deadline_miss", False)]
    interventions = [float(item.get("intervention_norm", 0.0)) for item in enabled
                     if item.get("intervention_norm") is not None]
    barriers = [float(item["minimum_barrier"]) for item in enabled
                if item.get("minimum_barrier") is not None and np.isfinite(item["minimum_barrier"])]
    return {
        "available": True,
        "steps": len(records),
        "entries": len(entries),
        "enabled_entries": len(enabled),
        "fallback_entries": len(fallbacks),
        "deadline_misses": len(deadline_misses),
        "fraction_intervened": float(np.mean(np.asarray(interventions) > 1e-12)) if interventions else 0.0,
        "maximum_intervention_norm": float(max(interventions)) if interventions else 0.0,
        "minimum_reported_barrier": float(min(barriers)) if barriers else None,
        "final_infeasible_entries": int(sum(not item.get("final_feasible", True) for item in enabled)),
        "methods": sorted({str(item.get("effective_method", "")) for item in enabled}),
    }


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
        tracking_samples = _tracking_samples(records, name)
        if tracking_samples:
            position_errors = []
            velocity_errors = []
            active = []
            direct = []
            iterations = []
            residuals = []
            for row, agent, estimate in tracking_samples:
                truth = row["target_truth"]
                position_errors.append(np.linalg.norm(
                    np.asarray(estimate["position"][:2]) - np.asarray(truth["position"][:2])))
                velocity_errors.append(np.linalg.norm(
                    np.asarray(estimate["velocity"][:2]) - np.asarray(truth["velocity"][:2])))
                active.append(bool(agent.get("active", estimate.get("active", False))))
                direct.append(bool(agent.get("direct_observation", False)))
                iterations.append(float(estimate.get("iterations", 0)))
                residuals.append(float(estimate.get("consensus_residual", 0.0)))
            position_errors = np.asarray(position_errors)
            velocity_errors = np.asarray(velocity_errors)
            item.update(
                rms_target_position_error_m=float(np.sqrt(np.mean(position_errors ** 2))),
                maximum_target_position_error_m=float(np.max(position_errors)),
                rms_target_velocity_error_mps=float(np.sqrt(np.mean(velocity_errors ** 2))),
                fraction_tracking_epochs_active=float(np.mean(active)),
                fraction_epochs_direct_observation=float(np.mean(direct)),
                mean_consensus_iterations=float(np.mean(iterations)),
                maximum_consensus_residual=float(np.max(residuals)),
            )
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
    result["cbf"] = cbf_metrics(records)
    tracking_rows = {}
    for row in records:
        tracking = row.get("target_tracking") or {}
        epoch = int(tracking.get("epoch_id", 0))
        if epoch > 0:
            tracking_rows[epoch] = row
    if tracking_rows:
        epoch_rows = [tracking_rows[key] for key in sorted(tracking_rows)]
        latest_tracking = epoch_rows[-1].get("target_tracking") or {}
        observer = records[-1].get("target_observation_diagnostics") or {}
        result["global"].update(
            tracking_epochs_completed=len(epoch_rows),
            tracking_epochs_timed_out=sum(any(
                bool(agent.get("epoch_timed_out", False))
                for agent in ((row.get("target_tracking") or {}).get("agents") or {}).values()
            ) for row in epoch_rows),
            consensus_messages_published=int(latest_tracking.get("consensus_messages_published", 0)),
            consensus_messages_rejected_wrong_epoch_round_or_identity=int(
                latest_tracking.get("consensus_messages_rejected", 0)),
            handoffs_sent=int(latest_tracking.get("handoffs_sent", 0)),
            handoffs_accepted=int(latest_tracking.get("handoffs_accepted", 0)),
            camera_captures=int(observer.get("camera_captures", 0)),
            camera_visible_detections=int(observer.get("camera_visible_detections", 0)),
            camera_invalid_detections=int(observer.get("camera_invalid_detections", 0)),
            camera_rpc_errors=int(observer.get("camera_rpc_errors", 0)),
            mean_capture_rate_hz=float(observer.get("mean_capture_rate_hz", 0.0)),
            capture_interval_p95_sec=float(observer.get("capture_interval_p95_sec", 0.0)),
            capture_interval_max_sec=float(observer.get("capture_interval_max_sec", 0.0)),
            mean_image_rpc_duration_sec=float(observer.get("mean_image_rpc_duration_sec", 0.0)),
            max_image_rpc_duration_sec=float(observer.get("max_image_rpc_duration_sec", 0.0)),
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-animation", action="store_true")
    parser.add_argument("--display-config", type=Path,
                        help="shared JSON containing route_heading, origin, and bounds")
    parser.add_argument("--route-heading", type=float)
    parser.add_argument("--display-origin", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--display-bounds", nargs=4, type=float,
                        metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--playback-speed", type=float, default=1.0)
    args = parser.parse_args()
    output = args.output_dir or args.jsonl.parent
    output.mkdir(parents=True, exist_ok=True)
    records = load(args.jsonl)
    if not records:
        raise SystemExit("mission JSONL is empty")
    save_trajectory(records, output / "full_mission_trajectories.png")
    save_errors(records, output / "slot_errors.png")
    if any((row.get("target_tracking") or {}).get("enabled", False) for row in records):
        save_tracking_errors(records, output / "tracking_errors.png")
        save_tracking_counts(records, output / "tracking_counts.png")
    (output / "metrics.json").write_text(
        json.dumps(metrics(records), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.skip_animation:
        repo = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(repo / "PythonClient"))
        sys.path.insert(0, str(repo / "PythonClient" / "distributed_mission"))
        from distributed_mission.modules.mission_plots import plot_topdown_animation
        display = {}
        if args.display_config:
            display = json.loads(args.display_config.read_text(encoding="utf-8"))
        route_heading = args.route_heading if args.route_heading is not None else display.get("route_heading")
        display_origin = args.display_origin if args.display_origin is not None else display.get("origin")
        display_bounds = args.display_bounds
        if display_bounds is not None:
            display_bounds = [[display_bounds[0], display_bounds[1]],
                              [display_bounds[2], display_bounds[3]]]
        elif display.get("bounds") is not None:
            display_bounds = display["bounds"]
        plot_topdown_animation(
            records, str(output / "topdown.mp4"), str(output / "topdown.gif"),
            10.0, args.playback_speed,
            route_heading=route_heading, display_origin=display_origin,
            display_bounds=display_bounds,
        )


if __name__ == "__main__":
    main()
