#!/usr/bin/env python3
"""Run and summarize one ten-second paper mission (inside the ROS container)."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from hercules_maicp.reset_scene import reset_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['collision', 'tracking', 'localization'], required=True)
    parser.add_argument('--host', default='192.168.65.254')
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--margin', type=float, default=0.0)
    parser.add_argument('--dynamics-file', default='')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--observation-source', default='camera', choices=['camera', 'truth'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError('Choose a new output name; mission evidence is never overwritten')
    before = time.monotonic()
    manifest = reset_scene(args.seed, args.case, host=args.host)
    args.output.with_suffix('.reset.json').write_text(json.dumps(manifest, indent=2))
    reset_seconds = time.monotonic() - before
    command = ['ros2', 'launch', 'hercules_maicp', 'case_study.launch.py',
               f'case:={args.case}', f'host:={args.host}', f'seed:={args.seed}',
               f'margin_uav:={args.margin}', f'margin_ugv:={args.margin}',
               f'training:={str(args.training).lower()}',
               f'observation_source:={args.observation_source}', f'log_path:={args.output}']
    if args.dynamics_file:
        command.append(f'dynamics_file:={args.dynamics_file}')
    started = time.monotonic()
    with Path(str(args.output) + '.stdout').open('w') as stdout:
        process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while not Path(str(args.output) + '.done').exists():
                failed = Path(str(args.output) + '.failed')
                if failed.exists():
                    raise RuntimeError(failed.read_text())
                if process.poll() is not None:
                    raise RuntimeError(f'Launch exited {process.returncode}; inspect {args.output}.stdout')
                if time.monotonic() - started > 90:
                    raise TimeoutError(f'No completion within 90s; inspect {args.output}.stdout')
                time.sleep(.1)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    summary = dict(reset_seconds=reset_seconds, launch_seconds=time.monotonic()-started,
                   samples=len(rows), horizon_seconds=rows[-1]['timestamp'], agents={})
    for name in manifest['controlled_agents']:
        positions = [row['states'][name]['position'][:2] for row in rows]
        distances = [math.dist(a,b) for a,b in zip(positions,positions[1:])]
        summary['agents'][name] = dict(distance_m=sum(distances),
            return_error_m=math.dist(positions[0], positions[-1]),
            return_error_xyz_m=math.dist(rows[0]['states'][name]['position'],
                                         rows[-1]['states'][name]['position']),
            mean_speed_mps=sum(math.hypot(*row['states'][name]['velocity'][:2]) for row in rows)/len(rows),
            returned_home=rows[-1]['maicp']['returned_home'][name])
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
