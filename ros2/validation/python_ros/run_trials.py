#!/usr/bin/env python3
"""Run paired Python/ROS trials with explicit simulator ownership.

The runner is intentionally command-driven: the Python baseline and the ROS
container invocation are supplied by the caller, while this tool owns only
the Unreal process it starts.  It writes a manifest before each trial, hashes
the shared AirSim settings, restarts Unreal between implementations, and
invokes the existing perception-parity reset helper when requested.  Use
``--dry-run`` to inspect the complete trial matrix without starting anything.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import platform
from pathlib import Path
import shlex
import signal
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


AGENTS = ["Drone1", "Drone2", "SimpleFlight",
          "Husky1", "Husky2", "Husky3", "Target1"]
ROWS = (
    ("mestres_truth", "mestres", "truth", "none"),
    ("wang_truth", "wang", "truth", "none"),
    ("mestres_camera", "mestres", "camera", "perception"),
    ("wang_camera", "wang", "camera", "perception"),
)
COMPARATOR = Path(__file__).with_name("compare_python_ros.py")
ROS_MEDIA_RENDERER = (
    Path(__file__).resolve().parents[2] / "src" / "hercules_mission_ros"
    / "scripts" / "render_ros_media.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_docker_metadata(requested_digest: str) -> Dict[str, str]:
    """Record the local image identity when the caller did not supply one."""
    image = os.environ.get("HERCULES_DOCKER_IMAGE", "hercules-ros2:humble")
    try:
        value = subprocess.check_output(
            ["docker", "image", "inspect", image,
             "--format", "{{.Id}} {{.Architecture}}"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().split()
    except (OSError, subprocess.CalledProcessError):
        value = []
    if len(value) >= 2:
        return {
            "digest": requested_digest if requested_digest and requested_digest != "unknown" else value[0],
            "architecture": value[1],
        }
    return {"digest": requested_digest or "unknown", "architecture": "unknown"}


def python_package_versions() -> Dict[str, str]:
    """Capture the host oracle versions without importing optional packages."""
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - Python 3.7 compatibility
        return {}
    versions: Dict[str, str] = {}
    for package in ("numpy", "scipy", "scikit-learn", "osqp"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


def git_worktree_metadata(repo: Path) -> Dict[str, Any]:
    """Capture dirty-state evidence so an uncommitted trial is reproducible."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True,
            stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--binary"], cwd=repo,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"dirty": None, "diff_sha256": None}
    return {
        "dirty": bool(status.strip()),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def command_text(value: str, replacements: Dict[str, str]) -> List[str]:
    rendered = value.format(**replacements)
    return shlex.split(rendered)


def command_argv_env(value: str, replacements: Dict[str, str]) -> Tuple[List[str], Optional[Dict[str, str]]]:
    """Split a command and honor leading ``NAME=value`` assignments.

    Trial commands are commonly supplied as shell-style strings so callers
    can select the Docker network or RPC host inline.  ``subprocess.Popen``
    does not interpret those assignments when ``shell=False`` (our deliberate
    default), so peel them off and pass them through the child environment.
    An explicit ``env ...`` command remains supported unchanged.
    """
    argv = command_text(value, replacements)
    updates: Dict[str, str] = {}
    while argv and "=" in argv[0]:
        key, val = argv[0].split("=", 1)
        if not key or not (key[0].isalpha() or key[0] == "_") or not all(
            char.isalnum() or char == "_" for char in key
        ):
            break
        updates[key] = val
        argv.pop(0)
    return argv, (updates or None)


def container_workspace_path(path: Path) -> str:
    """Translate a host workspace path to the compose bind-mount path."""
    host_root = Path.cwd().resolve()
    try:
        relative = path.resolve().relative_to(host_root)
    except ValueError:
        return str(path)
    return str(Path("/workspaces/hercules") / relative)


def popen_command(value: str, replacements: Dict[str, str]) -> Tuple[subprocess.Popen[Any], List[str]]:
    argv, updates = command_argv_env(value, replacements)
    environment = None
    if updates:
        environment = os.environ.copy()
        environment.update(updates)
    return subprocess.Popen(argv, env=environment, start_new_session=True), argv


def wait_for_airsim(host: str, ports: List[int], timeout_sec: float = 120.0) -> None:
    """Wait until every owned AirSim RPC endpoint accepts a TCP connection.

    Unreal is started asynchronously by ``OwnedProcess``.  A process being
    present is not enough: Hero mode opens the multirotor and car services at
    different points during map load.  Waiting here prevents the parity reset
    or either implementation from racing the simulator startup.  On macOS,
    ``host.docker.internal`` is a container-only alias, so the native host's
    loopback address is also tried for readiness.
    """
    candidates = [str(host or "127.0.0.1")]
    if candidates[0] in {"host.docker.internal", "docker.for.mac.host.internal"}:
        candidates.append("127.0.0.1")
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    last_error: Optional[BaseException] = None
    while time.monotonic() < deadline:
        all_ready = True
        for port in ports:
            connected = False
            for candidate in candidates:
                try:
                    with socket.create_connection((candidate, int(port)), timeout=1.0):
                        connected = True
                        break
                except OSError as error:
                    last_error = error
            if not connected:
                all_ready = False
                break
        if all_ready:
            return
        time.sleep(0.5)
    raise TimeoutError(
        "AirSim endpoints {}:{} did not become ready within {:.1f}s: {}".format(
            ",".join(candidates), ",".join(str(port) for port in ports),
            float(timeout_sec), last_error,
        )
    )


class OwnedProcess:
    def __init__(self, command: Optional[str], dry_run: bool = False) -> None:
        self.command = command
        self.dry_run = dry_run
        self.process: Optional[subprocess.Popen[Any]] = None

    def start(self, replacements: Optional[Dict[str, str]] = None) -> None:
        if not self.command:
            return
        rendered = command_text(self.command, replacements or {})
        if self.dry_run:
            print("$ " + " ".join(shlex.quote(item) for item in rendered), flush=True)
            return
        self.process, _ = popen_command(self.command, replacements or {})

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                process.wait(timeout=5)


def run_command(command: str, replacements: Dict[str, str], dry_run: bool,
                timeout_sec: Optional[float] = None) -> bool:
    rendered = command_text(command, replacements)
    print("$ " + " ".join(shlex.quote(item) for item in rendered), flush=True)
    if dry_run:
        return False
    process, _ = popen_command(command, replacements)
    timed_out = False
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        timed_out = True
        print("command reached its bounded trial timeout; sending SIGINT", flush=True)
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                process.wait(timeout=5)
    if process.returncode not in (0, 130, -signal.SIGINT):
        raise subprocess.CalledProcessError(process.returncode, rendered)
    return timed_out


def ensure_mission_log(output_dir: Path) -> Path:
    """Expose the primary logger output under the stable comparison name.

    The Python orchestrator intentionally timestamps its artifact stem, while
    ROS launch commands commonly accept an explicit ``log_path``.  Keep both
    implementations' raw files intact and copy the sole JSONL mission log to
    ``mission.jsonl`` so the comparator has a deterministic input contract.
    """
    target = output_dir / "mission.jsonl"
    if target.is_file() and target.stat().st_size > 0:
        return target
    candidates = [
        path for path in output_dir.glob("*.jsonl")
        if path.is_file() and path.stat().st_size > 0 and path.name != target.name
    ]
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise RuntimeError(
            f"expected exactly one primary JSONL mission log in {output_dir}; found {names}"
        )
    shutil.copy2(candidates[0], target)
    return target


def compare_trial(trial: Path, dry_run: bool) -> None:
    """Normalize and compare the two logs produced by one paired trial."""
    python_log = trial / "python" / "mission.jsonl"
    ros_log = trial / "ros" / "mission.jsonl"
    output_dir = trial / "comparison"
    command = [sys.executable, str(COMPARATOR), str(python_log), str(ros_log),
               "--output-dir", str(output_dir), "--dt", "0.1"]
    print("$ " + " ".join(shlex.quote(item) for item in command), flush=True)
    if dry_run:
        return
    if not python_log.is_file() or not ros_log.is_file():
        raise RuntimeError(
            "paired commands must write python/mission.jsonl and ros/mission.jsonl "
            f"under {trial} before comparison"
        )
    subprocess.run(command, check=True)


def render_ros_media(trial: Path, dry_run: bool) -> None:
    """Create ROS top-down media using bounds shared with the Python run."""
    python_log = trial / "python" / "mission.jsonl"
    ros_log = trial / "ros" / "mission.jsonl"
    output_dir = trial / "ros" / "media"
    command = [sys.executable, str(ROS_MEDIA_RENDERER), "--log", str(ros_log),
               "--output-dir", str(output_dir), "--reference-log", str(python_log)]
    staging = trial / "ros" / "recording_frames"
    if staging.is_dir():
        command.extend(["--staging-dir", str(staging)])
    print("$ " + " ".join(shlex.quote(item) for item in command), flush=True)
    if dry_run:
        return
    if not ROS_MEDIA_RENDERER.is_file():
        raise RuntimeError(f"ROS media renderer is missing: {ROS_MEDIA_RENDERER}")
    subprocess.run(command, check=True)


def analyze_pending_trials(output_root: Path, dry_run: bool) -> None:
    """Render and compare completed raw pairs after collection.

    Running analysis after the simulator campaign allows independent pairs to
    be normalized concurrently.  Existing complete bundles are left intact,
    which also makes ``--resume --defer-analysis`` safe after an interruption.
    """
    trials = sorted(output_root.glob("*-trial[0-9][0-9]"))
    pending = []
    for trial in trials:
        if (trial / "comparison" / "comparison.json").is_file() and all(
            (trial / "ros" / "media" / name).is_file()
            for name in ("topdown.mp4", "topdown.gif")
        ):
            continue
        pending.append(trial)
    if not pending:
        return
    if dry_run:
        for trial in pending:
            render_ros_media(trial, True)
            compare_trial(trial, True)
        return

    def analyze(trial: Path) -> None:
        render_ros_media(trial, False)
        compare_trial(trial, False)

    workers = min(4, len(pending))
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="trial-analysis") as executor:
        futures = [executor.submit(analyze, trial) for trial in pending]
        for future in futures:
            future.result()


def consolidate_trials(output_root: Path, expected_count: int) -> None:
    """Write one index over every completed pair in ``output_root``.

    Individual comparator directories remain the source of truth (including
    their normalized rows and plots).  This report is deliberately compact:
    it records the run identity, coverage, warnings, and global difference
    statistics while linking back to each complete comparison directory.
    """
    entries: List[Dict[str, Any]] = []
    for comparison_path in sorted(output_root.glob("*-trial*/comparison/comparison.json")):
        trial = comparison_path.parent.parent
        metadata_path = trial / "metadata.json"
        try:
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            entries.append({
                "trial": trial.name,
                "status": "invalid",
                "error": str(exc),
                "comparison": str(comparison_path.relative_to(output_root)),
            })
            continue
        entries.append({
            "trial": trial.name,
            "row": metadata.get("row"),
            "repetition": metadata.get("repetition"),
            "status": "complete",
            "duration_s": comparison.get("duration_s"),
            "grid_count": comparison.get("grid_count"),
            "common_agents": comparison.get("common_agents", []),
            "warnings": comparison.get("warnings", []),
            "global_differences": comparison.get("global", {}),
            "python_summary": comparison.get("python_summary", {}),
            "ros_summary": comparison.get("ros_summary", {}),
            "comparison": str(comparison_path.relative_to(output_root)),
        })

    complete = sum(entry.get("status") == "complete" for entry in entries)
    summary: Dict[str, Any] = {
        "expected_trial_count": expected_count,
        "discovered_trial_count": len(entries),
        "completed_trial_count": complete,
        "trials": entries,
    }
    (output_root / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Python-vs-ROS consolidated comparison",
        "",
        f"Completed **{complete}** of **{expected_count}** expected paired trials.",
        "",
        "Each row links to the complete per-trial comparator output. Numeric "
        "differences are descriptive; nondeterministic simulator runs are not "
        "treated as exact-equivalence tests.",
        "",
        "| Trial | Row | Rep | Duration (s) | Grid | Common agents | Warnings |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for entry in entries:
        trial = str(entry.get("trial", ""))
        row = str(entry.get("row", ""))
        repetition = entry.get("repetition", "")
        duration = entry.get("duration_s", "")
        grid = entry.get("grid_count", "")
        agents = len(entry.get("common_agents", [])) if isinstance(entry.get("common_agents"), list) else ""
        warnings = len(entry.get("warnings", [])) if isinstance(entry.get("warnings"), list) else ""
        link = entry.get("comparison", "")
        lines.append(
            f"| [{trial}]({link}) | {row} | {repetition} | {duration} | {grid} | {agents} | {warnings} |"
        )
    if not entries:
        lines.extend(["", "No completed comparison directories were found."])
    (output_root / "comparison_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True,
                        help="canonical AirSim settings JSON shared by both implementations")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-command", required=True,
                        help="command template; output_dir is trial/python; also supports "
                             "{python_output_dir}, {trial_dir}, {duration}, {row}, {method}, {observation}, "
                             "{obstacles}, {python_obstacle_args}, {python}")
    parser.add_argument("--ros-command", required=True,
                        help="command template; output_dir is trial/ros; also supports "
                             "{ros_output_dir}, {trial_dir}, {duration}, {row}, {method}, {observation}, {obstacles}")
    parser.add_argument("--unreal-command",
                        help="optional native Unreal command template owned by this runner")
    parser.add_argument("--prepare-reset-command",
                        help="optional command template run before ROS; supports {python_log}, "
                             "{output_dir}, {ros_output_dir}, {trial_dir}, {airsim_host}, "
                             "{drone_port}, and {ugv_port}")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--sleep-between", type=float, default=2.0)
    parser.add_argument("--python-commit", default="ce742c81")
    parser.add_argument("--ros-commit", default="HEAD")
    parser.add_argument("--docker-image-digest", default="unknown")
    parser.add_argument("--resume", action="store_true",
                        help="skip completed pairs in an existing output root and rerun incomplete pairs")
    parser.add_argument("--defer-analysis", action="store_true",
                        help="collect all raw pairs first, then render/compare pending pairs concurrently")
    parser.add_argument("--airsim-host", default=os.environ.get("AIRSIM_HOST", "127.0.0.1"),
                        help="AirSim RPC host passed to command templates")
    parser.add_argument("--drone-port", type=int,
                        default=int(os.environ.get("AIRSIM_MULTIROTOR_PORT", "41451")))
    parser.add_argument("--ugv-port", type=int,
                        default=int(os.environ.get("AIRSIM_CAR_PORT", "41452")))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repetitions < 1 or args.duration <= 0 or args.sleep_between < 0:
        raise SystemExit("repetitions, duration, and sleep-between must be nonnegative/positive")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    settings = args.settings.expanduser().resolve()
    if not settings.is_file():
        raise SystemExit(f"settings file does not exist: {settings}")
    # Freeze one settings file for every implementation and repetition.  The
    # source path may be edited later, but an existing output directory remains
    # self-contained and its recorded hash stays tied to the actual trials.
    canonical_settings = output_root / "canonical_settings.json"
    if not canonical_settings.exists():
        shutil.copy2(settings, canonical_settings)
    settings = canonical_settings
    settings_hash = sha256(settings)
    def resolve_git_commit(value: str) -> str:
        if value != "HEAD":
            return value
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True,
                cwd=Path(__file__).resolve().parents[3],
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return value

    docker_metadata = resolve_docker_metadata(args.docker_image_digest)
    repository = Path(__file__).resolve().parents[3]
    worktree_metadata = git_worktree_metadata(repository)
    runner_manifest = {
        "settings": str(settings),
        "settings_sha256": settings_hash,
        "python_commit": args.python_commit,
        "ros_commit": resolve_git_commit(args.ros_commit),
        "ros_worktree_dirty": worktree_metadata["dirty"],
        "ros_diff_sha256": worktree_metadata["diff_sha256"],
        "docker_image_digest": docker_metadata["digest"],
        "docker_architecture": docker_metadata["architecture"],
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_packages": python_package_versions(),
        "airsim_host": args.airsim_host,
        "drone_port": args.drone_port,
        "ugv_port": args.ugv_port,
        "duration_sec": args.duration,
        "repetitions": args.repetitions,
        "rows": [name for name, *_ in ROWS],
        "agents": AGENTS,
        "parameters": {
            "dt": 0.1, "target_speed": 0.10, "target_pattern_length": 10.0,
            "target_pattern_width": 8.0, "target_sample_count": 64,
            "target_start_sample_index": 5, "tracking_rate": 4.0,
            "tracking_window": 5.0, "tracking_process_noise": 0.20,
            "tracking_measurement_std": 0.25, "tracking_admm_rho": 1.0,
            "tracking_admm_max_iterations": 20, "tracking_admm_tolerance": 0.001,
        },
    }
    (output_root / "runner_manifest.json").write_text(
        json.dumps(runner_manifest, indent=2) + "\n", encoding="utf-8")

    unreal = OwnedProcess(args.unreal_command, args.dry_run)
    try:
        for repetition in range(1, args.repetitions + 1):
            for row_name, method, observation, obstacles in ROWS:
                trial = output_root / f"{row_name}-trial{repetition:02d}"
                if args.resume:
                    comparison_result = trial / "comparison" / "comparison.json"
                    if comparison_result.is_file() and comparison_result.stat().st_size > 0:
                        print(f"Skipping completed trial {trial.name} (--resume)", flush=True)
                        continue
                    if trial.exists():
                        # Preserve failed/incomplete evidence outside the
                        # consolidated trial glob before recreating the pair.
                        backup_root = output_root / "incomplete"
                        backup_root.mkdir(exist_ok=True)
                        backup = backup_root / f"{trial.name}-{int(time.time())}"
                        shutil.move(str(trial), str(backup))
                        print(f"Moved incomplete trial to {backup}", flush=True)
                trial.mkdir(parents=True, exist_ok=True)
                (trial / "python").mkdir(exist_ok=True)
                (trial / "ros").mkdir(exist_ok=True)
                metadata = dict(runner_manifest, row=row_name, repetition=repetition,
                                method=method, observation=observation,
                                obstacles=obstacles, trial_dir=str(trial))
                (trial / "metadata.json").write_text(
                    json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                base_replacements = {
                    "trial_dir": str(trial),
                    "python_output_dir": str(trial / "python"),
                    "ros_output_dir": str(trial / "ros"),
                    "duration": str(args.duration),
                    "row": row_name,
                    "method": method, "observation": observation,
                    "obstacles": obstacles, "settings": str(settings),
                    "python": sys.executable,
                    "airsim_host": args.airsim_host,
                    "drone_port": str(args.drone_port),
                    "ugv_port": str(args.ugv_port),
                    # ROS executes inside the compose container, where the
                    # checkout is mounted at /workspaces/hercules.  Keep the
                    # host path for Python/reset/comparison, and expose this
                    # explicit form for ROS log/media arguments.
                    "ros_output_dir_container": container_workspace_path(trial / "ros"),
                    "python_obstacle_args": "--no-spawn-obstacles" if obstacles == "none" else "",
                }
                # Python owns the first simulator session for this row.
                python_replacements = dict(base_replacements,
                                           output_dir=str(trial / "python"))
                unreal.start(base_replacements)
                if not args.dry_run:
                    wait_for_airsim(args.airsim_host, [args.drone_port, args.ugv_port])
                run_command(args.python_command, python_replacements, args.dry_run,
                            # The historical Python oracle advances a 10 Hz
                            # mission slower than real time on this Mac while
                            # writing its compressed perception sidecar.  Give
                            # it enough bounded wall time to finish all rows;
                            # the mission itself remains exactly duration_sec.
                            timeout_sec=max(120.0, args.duration + 240.0))
                if not args.dry_run:
                    ensure_mission_log(trial / "python")
                python_log = trial / "python" / "mission.jsonl"
                unreal.stop()
                if args.sleep_between and not args.dry_run:
                    time.sleep(args.sleep_between)

                # ROS is always run against a fresh Unreal process, optionally
                # staged from Python's initial record by the reset helper.
                ros_replacements = dict(base_replacements,
                                        output_dir=str(trial / "ros"),
                                        python_log=str(python_log))
                unreal.start(base_replacements)
                if not args.dry_run:
                    wait_for_airsim(args.airsim_host, [args.drone_port, args.ugv_port])
                if args.prepare_reset_command:
                    run_command(args.prepare_reset_command, ros_replacements, args.dry_run,
                                timeout_sec=120.0)
                # A healthy ROS mission finishes shortly after its requested
                # duration.  Keep a bounded grace period for launch/shutdown,
                # but do not hold the next paired trial for two minutes when
                # a startup gate has already failed.
                run_command(args.ros_command, ros_replacements, args.dry_run,
                            timeout_sec=max(90.0, args.duration + 30.0))
                if not args.dry_run:
                    ensure_mission_log(trial / "ros")
                    if not args.defer_analysis:
                        render_ros_media(trial, args.dry_run)
                if not args.defer_analysis:
                    compare_trial(trial, args.dry_run)
                unreal.stop()
                if args.sleep_between and not args.dry_run:
                    time.sleep(args.sleep_between)
    finally:
        unreal.stop()
    if not args.dry_run:
        if args.defer_analysis:
            analyze_pending_trials(output_root, args.dry_run)
        consolidate_trials(output_root, args.repetitions * len(ROWS))
    print(f"Trial manifest: {output_root / 'runner_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
