#!/usr/bin/env python3
"""Summarize matched 500-step Python and ROS RuralAustralia runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PYTHON_LOG = ROOT / "artifacts/python_500_truth_full/mestres_rural_australia_1789261218.jsonl"
ROS_LOG = ROOT / "artifacts/ros_500_mestres_truth_live.jsonl"


def load(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def trajectory(rows):
    result = {}
    for agent in rows[0]["states"]:
        points = np.asarray([r["states"][agent]["position"] for r in rows], dtype=float)
        result[agent] = {
            "path_length_m": float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum()),
            "displacement_m": float(np.linalg.norm(points[-1, :2] - points[0, :2])),
            "start": points[0].tolist(), "end": points[-1].tolist(),
        }
    return result


def python_summary(rows):
    phases = {name: pct([r["timing"]["phase_ms"][name] for r in rows])
              for name in ("state", "estimate", "perception", "cbf", "actuation")}
    cycle = pct([r["timing"]["cycle_ms"] for r in rows])
    cycle["total_s"] = float(sum(r["timing"]["cycle_ms"] for r in rows) / 1000.0)
    cycle["deadline_misses"] = int(sum(r["timing"]["deadline_miss"] for r in rows))
    cbf = {}
    for agent in rows[0]["cbf"]:
        cbf[agent] = pct([r["cbf"][agent]["solve_time_ms"] for r in rows])
    active = int(sum(r["target_tracking"]["agents"]["Drone1"]["active"] for r in rows))
    return {"rows": len(rows), "mission_time_s": rows[-1]["timestamp"],
            "wall_span_s": rows[-1]["wall_timestamp"] - rows[0]["wall_timestamp"],
            "phases_ms": phases, "cycle_ms": cycle, "cbf_solver_ms": cbf,
            "tracking_active_steps": active,
            "tracking_iterations": sorted({r["target_tracking"]["iterations"] for r in rows}),
            "trajectory": trajectory(rows),
            "relevant_collision_records": int(sum(
                c.get("relevant", False) for r in rows for c in r.get("collisions", {}).values()))}


def ros_summary(rows):
    filters = {}
    for agent in {e["agent_id"] for r in rows for e in r["cbf"]["entries"]}:
        filters[agent] = pct([e["filter_time_ms"] for r in rows for e in r["cbf"]["entries"]
                              if e["agent_id"] == agent])
    gaps = np.diff([r["wall_timestamp"] for r in rows]) * 1000.0
    obs = rows[-1]["target_observation_diagnostics"]
    tracking = rows[-1]["target_tracking"]
    return {"rows": len(rows), "mission_time_s": rows[-1]["timestamp"],
            "wall_span_s": rows[-1]["wall_timestamp"] - rows[0]["wall_timestamp"],
            "control_period_ms": pct(gaps), "cbf_solver_ms": filters,
            "target_observation": obs, "tracking": {
                "epochs": tracking["epoch_id"],
                "active_track_count": tracking["active_track_count"],
                "direct_observation_count": tracking["direct_observation_count"],
                "consensus_messages_published": tracking["consensus_messages_published"],
                "epochs_timed_out": tracking["epochs_timed_out"]},
            "trajectory": trajectory(rows),
            "relevant_collision_records": int(sum(
                c.get("relevant", False) for r in rows for c in r.get("collisions", {}).values()))}


def plot_trajectories(py_rows, ros_rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, rows, title in ((axes[0], py_rows, "Python orchestrator"),
                            (axes[1], ros_rows, "ROS integration")):
        for agent in rows[0]["states"]:
            xy = np.asarray([r["states"][agent]["position"][:2] for r in rows], dtype=float)
            xy -= xy[0]
            ax.plot(xy[:, 0], xy[:, 1], label=agent, linewidth=1.0)
        ax.set_title(title + " (XY, translated to each run's start)")
        ax.set_xlabel("north/east frame x (m)")
        ax.set_ylabel("north/east frame y (m)")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
    axes[1].legend(fontsize=7, loc="best")
    fig.savefig(ROOT / "artifacts/comparison_500_trajectories.png", dpi=160)
    plt.close(fig)


def main():
    py_rows, ros_rows = load(PYTHON_LOG), load(ROS_LOG)
    output = {"python": python_summary(py_rows), "ros": ros_summary(ros_rows)}
    common = sorted(set(output["python"]["trajectory"]) & set(output["ros"]["trajectory"]))
    n = min(len(py_rows), len(ros_rows))
    output["aligned_xy_rmse_m"] = {}
    for agent in common:
        py = np.asarray([r["states"][agent]["position"][:2] for r in py_rows[:n]], dtype=float)
        ros = np.asarray([r["states"][agent]["position"][:2] for r in ros_rows[:n]], dtype=float)
        py -= py[0]; ros -= ros[0]
        output["aligned_xy_rmse_m"][agent] = float(np.sqrt(np.mean(np.sum((py - ros) ** 2, axis=1))))
    (ROOT / "artifacts/comparison_500.json").write_text(json.dumps(output, indent=2) + "\n")
    plot_trajectories(py_rows, ros_rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
