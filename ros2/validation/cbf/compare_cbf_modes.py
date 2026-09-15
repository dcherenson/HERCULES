#!/usr/bin/env python3
"""Compare no-CBF, Mestres and Wang logs without assuming identical trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path as _Path
import sys
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "rural_nominal"))
from render_validation import cbf_metrics, load  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path, help="mode=path JSONL arguments")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    modes = {}
    for item in args.logs:
        if "=" not in str(item):
            raise SystemExit("each log must be mode=path")
        mode, path = str(item).split("=", 1)
        records = load(Path(path))
        if not records:
            raise SystemExit(f"empty log: {path}")
        modes[mode] = records
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {mode: cbf_metrics(records) for mode, records in modes.items()}
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for mode, records in modes.items():
        times = np.asarray([float(row.get("timestamp", index)) for index, row in enumerate(records)])
        interventions = []
        barriers = []
        for row in records:
            entries = ((row.get("cbf") or {}).get("entries") or [])
            enabled = [entry for entry in entries if entry.get("enabled", False)]
            interventions.append(max((float(entry.get("intervention_norm", 0.0)) for entry in enabled), default=0.0))
            values = [float(entry["minimum_barrier"]) for entry in enabled
                      if entry.get("minimum_barrier") is not None]
            barriers.append(min(values) if values else np.nan)
        axes[0].plot(times, interventions, label=mode)
        axes[1].plot(times, barriers, label=mode)
    axes[0].set_ylabel("max intervention norm")
    axes[1].set_ylabel("minimum reported barrier")
    axes[1].set_xlabel("mission time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    figure.savefig(args.output_dir / "cbf_modes.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
