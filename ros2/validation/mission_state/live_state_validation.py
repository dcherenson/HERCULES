#!/usr/bin/env python3
"""Validation-only synchronized ROS/direct-AirSim recorder and plotter."""

import argparse
import csv
import json
import math
import pathlib
import subprocess
import sys
import time

import rclpy
from hercules_interfaces.msg import GroundTruthState

import hercules_cosysairsim as airsim


FIELDNAMES = [
    "receipt_wall_time", "sample_stamp_ns", "phase", "vehicle",
    "origin_x", "origin_y", "origin_z",
    "spawn_origin_x", "spawn_origin_y", "spawn_origin_z",
    "ros_x", "ros_y", "ros_z", "direct_x", "direct_y", "direct_z",
    "local_x", "local_y", "local_z",
    "ros_vx", "ros_vy", "ros_vz",
    "direct_vx", "direct_vy", "direct_vz", "ros_yaw", "direct_yaw",
]

TOLERANCES = {
    "max_stationary_position_error_m": 0.05,
    "post_motion_position_error_m": 0.05,
    "displacement_error_m": 0.05,
    "stationary_velocity_error_mps": 0.05,
    "yaw_error_rad": 0.02,
    "spawn_origin_error_m": 0.05,
    "minimum_direction_cosine": 0.95,
}


def vector(value):
    return [float(value.x_val), float(value.y_val), float(value.z_val)]


def norm(values):
    return math.sqrt(sum(value * value for value in values))


def difference(left, right):
    return [a - b for a, b in zip(left, right)]


def mean_vector(rows, prefix):
    return [
        sum(float(row[f"{prefix}_{axis}"]) for row in rows) / len(rows)
        for axis in "xyz"
    ]


def wrapped_error(left, right):
    return abs((left - right + math.pi) % (2.0 * math.pi) - math.pi)


class LiveRecorder:
    def __init__(self, args):
        self.args = args
        self.phase = "pre_stationary"
        self.rows = []
        self.rpc_errors = 0
        client_type = airsim.MultirotorClient if args.vehicle_type == "drone" else airsim.CarClient
        self.client = client_type(ip=args.host, port=args.port)
        self.client.confirmConnection()
        self.node = rclpy.create_node(f"validate_{args.vehicle.lower()}_mission_state")
        self.subscription = self.node.create_subscription(
            GroundTruthState, args.topic, self.receive, 20)

    def receive(self, message):
        if not message.valid or message.agent_id != self.args.vehicle:
            return
        try:
            pose = self.client.simGetObjectPose(self.args.vehicle, True)
            kinematics = self.client.simGetGroundTruthKinematics(
                vehicle_name=self.args.vehicle)
            direct_position = vector(pose.position)
            local_position = vector(kinematics.position)
            direct_velocity = vector(kinematics.linear_velocity)
            _, _, direct_yaw = airsim.quaternion_to_euler_angles(
                kinematics.orientation)
        except Exception as error:
            self.rpc_errors += 1
            print(f"direct AirSim sample failed: {error}", file=sys.stderr)
            return
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec)
        values = {
            "receipt_wall_time": time.time(),
            "sample_stamp_ns": stamp_ns,
            "phase": self.phase,
            "vehicle": self.args.vehicle,
            "origin_x": self.args.origin[0],
            "origin_y": self.args.origin[1],
            "origin_z": self.args.origin[2],
            "spawn_origin_x": self.args.spawn_origin[0],
            "spawn_origin_y": self.args.spawn_origin[1],
            "spawn_origin_z": self.args.spawn_origin[2],
            "ros_yaw": float(message.yaw),
            "direct_yaw": float(direct_yaw),
        }
        for axis, value in zip("xyz", message.position):
            values[f"ros_{axis}"] = float(value)
        for axis, value in zip("xyz", direct_position):
            values[f"direct_{axis}"] = value
        for axis, value in zip("xyz", local_position):
            values[f"local_{axis}"] = value
        for axis, value in zip("xyz", message.velocity):
            values[f"ros_v{axis}"] = float(value)
        for axis, value in zip("xyz", direct_velocity):
            values[f"direct_v{axis}"] = value
        self.rows.append(values)

    def spin_for(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def run_smoke(self):
        command = [
            "ros2", "run", "hercules_control", "smoke_node", "--ros-args",
            "-p", f"vehicle_type:={self.args.vehicle_type}",
            "-p", "mode:=motion",
        ]
        print("Starting existing smoke node: " + " ".join(command))
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + self.args.smoke_timeout
        while process.poll() is None and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
            raise RuntimeError("existing smoke node exceeded validation timeout")
        output = process.stdout.read()
        print(output, end="")
        if process.returncode != 0 or "SMOKE_RESULT success=true" not in output:
            raise RuntimeError("existing smoke motion test failed")

    def close(self):
        self.node.destroy_node()


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def calculate_metrics(rows, rpc_errors):
    phases = {
        phase: [row for row in rows if row["phase"] == phase]
        for phase in ("pre_stationary", "motion", "post_stationary")
    }
    for phase, samples in phases.items():
        if len(samples) < 3:
            raise RuntimeError(f"only {len(samples)} samples captured for {phase}")
    stationary = phases["pre_stationary"] + phases["post_stationary"]

    def position_error(row):
        return norm(difference(
            [float(row[f"ros_{axis}"]) for axis in "xyz"],
            [float(row[f"direct_{axis}"]) for axis in "xyz"],
        ))

    def velocity_error(row):
        return norm(difference(
            [float(row[f"ros_v{axis}"]) for axis in "xyz"],
            [float(row[f"direct_v{axis}"]) for axis in "xyz"],
        ))

    def spawn_origin_error(row):
        observed = [
            float(row[f"direct_{axis}"]) - float(row[f"local_{axis}"])
            for axis in "xyz"
        ]
        configured = [float(row[f"spawn_origin_{axis}"]) for axis in "xyz"]
        return norm(difference(observed, configured))

    ros_start = mean_vector(phases["pre_stationary"], "ros")
    ros_end = mean_vector(phases["post_stationary"], "ros")
    direct_start = mean_vector(phases["pre_stationary"], "direct")
    direct_end = mean_vector(phases["post_stationary"], "direct")
    ros_displacement = difference(ros_end, ros_start)
    direct_displacement = difference(direct_end, direct_start)
    ros_xy = ros_displacement[:2]
    direct_xy = direct_displacement[:2]
    denominator = norm(ros_xy) * norm(direct_xy)
    direction_cosine = (
        sum(a * b for a, b in zip(ros_xy, direct_xy)) / denominator
        if denominator > 1e-8 else 1.0
    )
    moving_velocity_rows = []
    for row in phases["motion"]:
        ros_velocity = [float(row[f"ros_v{axis}"]) for axis in "xyz"]
        direct_velocity = [float(row[f"direct_v{axis}"]) for axis in "xyz"]
        if max(norm(ros_velocity), norm(direct_velocity)) > 0.05:
            moving_velocity_rows.append(
                sum(a * b for a, b in zip(ros_velocity, direct_velocity)) >= 0.0)

    metrics = {
        "sample_count": len(rows),
        "pre_stationary_samples": len(phases["pre_stationary"]),
        "motion_samples": len(phases["motion"]),
        "post_stationary_samples": len(phases["post_stationary"]),
        "direct_rpc_errors": rpc_errors,
        "max_stationary_position_error_m": max(map(position_error, stationary)),
        "post_motion_position_error_m": max(
            map(position_error, phases["post_stationary"])),
        "max_motion_position_error_m": max(
            map(position_error, phases["motion"])),
        "displacement_error_m": norm(difference(
            ros_displacement, direct_displacement)),
        "stationary_velocity_error_mps": max(map(velocity_error, stationary)),
        "yaw_error_rad": max(
            wrapped_error(float(row["ros_yaw"]), float(row["direct_yaw"]))
            for row in rows),
        "spawn_origin_error_m": max(map(spawn_origin_error, rows)),
        "xy_displacement_direction_cosine": direction_cosine,
        "moving_velocity_direction_agreement_fraction": (
            sum(moving_velocity_rows) / len(moving_velocity_rows)
            if moving_velocity_rows else 1.0),
        "canonical_displacement_m": norm(ros_displacement),
        "direct_displacement_m": norm(direct_displacement),
    }
    failures = []
    for name, tolerance in TOLERANCES.items():
        if name == "minimum_direction_cosine":
            if metrics["xy_displacement_direction_cosine"] < tolerance:
                failures.append(
                    f"direction cosine {metrics['xy_displacement_direction_cosine']:.6g} < {tolerance}")
        elif metrics[name] > tolerance:
            failures.append(f"{name} {metrics[name]:.6g} > {tolerance}")
    if metrics["moving_velocity_direction_agreement_fraction"] < 0.95:
        failures.append("moving velocity direction agreement below 95 percent")
    if rpc_errors:
        failures.append(f"{rpc_errors} direct AirSim RPC samples failed")
    return metrics, failures


def plot_vehicle(rows, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vehicle = rows[0]["vehicle"]
    ros_x = [float(row["ros_x"]) for row in rows]
    ros_y = [float(row["ros_y"]) for row in rows]
    direct_x = [float(row["direct_x"]) for row in rows]
    direct_y = [float(row["direct_y"]) for row in rows]
    origin = [float(rows[0]["origin_x"]), float(rows[0]["origin_y"])]
    spawn_origin = [
        float(rows[0]["spawn_origin_x"]),
        float(rows[0]["spawn_origin_y"]),
    ]
    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    axis.plot(ros_x, ros_y, label="canonical ROS airsim_world_ned", linewidth=2.0)
    axis.plot(direct_x, direct_y, "--", label="direct AirSim world NED", linewidth=1.6)
    axis.scatter(*spawn_origin, marker="*", s=130,
                 label="configured vehicle world origin")
    axis.scatter(*origin, marker="D", s=45,
                 label="configured odometry mapping origin")
    axis.scatter(ros_x[0], ros_y[0], marker="o", s=55, label="start")
    axis.scatter(ros_x[-1], ros_y[-1], marker="X", s=70, label="end")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("AirSim world NED X (m)")
    axis.set_ylabel("AirSim world NED Y (m)")
    axis.set_title(f"{vehicle} canonical ROS vs direct AirSim trajectory")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.subplots_adjust(left=0.17, right=0.98, bottom=0.12, top=0.93)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_combined(paths, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.0, 6.5))
    colors = {"Drone1": "tab:blue", "Husky1": "tab:orange"}
    for path in paths:
        rows = read_csv(path)
        vehicle = rows[0]["vehicle"]
        color = colors.get(vehicle)
        ros_x = [float(row["ros_x"]) for row in rows]
        ros_y = [float(row["ros_y"]) for row in rows]
        direct_x = [float(row["direct_x"]) for row in rows]
        direct_y = [float(row["direct_y"]) for row in rows]
        axis.plot(ros_x, ros_y, color=color, linewidth=2.0,
                  label=f"{vehicle} canonical ROS")
        axis.plot(direct_x, direct_y, "--", color=color, linewidth=1.4,
                  label=f"{vehicle} direct AirSim")
        axis.scatter(float(rows[0]["spawn_origin_x"]),
                     float(rows[0]["spawn_origin_y"]),
                     color=color, marker="*", s=120)
        axis.scatter(float(rows[0]["origin_x"]), float(rows[0]["origin_y"]),
                     color=color, marker="D", s=40)
        axis.scatter(ros_x[0], ros_y[0], color=color, marker="o", s=45)
        axis.scatter(ros_x[-1], ros_y[-1], color=color, marker="X", s=60)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("AirSim world NED X (m)")
    axis.set_ylabel("AirSim world NED Y (m)")
    axis.set_title("Drone1 and Husky1 canonical ROS vs direct AirSim")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.11, top=0.93)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle", choices=("Drone1", "Husky1"), required=True)
    parser.add_argument("--vehicle-type", choices=("drone", "ugv"), required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--origin", nargs=3, type=float, required=True,
                        metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--spawn-origin", nargs=3, type=float, metavar=("X", "Y", "Z"),
        help="requested settings spawn used only to check actor-minus-ground-truth kinematics")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pre-seconds", type=float, default=2.0)
    parser.add_argument("--post-seconds", type=float, default=2.0)
    parser.add_argument("--smoke-timeout", type=float, default=90.0)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path(__file__).parent)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.spawn_origin is None:
        args.spawn_origin = list(args.origin)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.vehicle.lower()
    rclpy.init()
    recorder = LiveRecorder(args)
    try:
        recorder.spin_for(args.pre_seconds)
        recorder.phase = "motion"
        recorder.run_smoke()
        recorder.phase = "post_stationary"
        recorder.spin_for(args.post_seconds)
    finally:
        recorder.close()
        rclpy.shutdown()
    csv_path = args.output_dir / f"{stem}_state_samples.csv"
    plot_path = args.output_dir / f"{stem}_state_comparison.png"
    metrics_path = args.output_dir / f"{stem}_state_metrics.json"
    write_csv(recorder.rows, csv_path)
    metrics, failures = calculate_metrics(recorder.rows, recorder.rpc_errors)
    metrics_path.write_text(json.dumps({
        "vehicle": args.vehicle,
        "tolerances": TOLERANCES,
        "metrics": metrics,
        "passed": not failures,
        "failures": failures,
    }, indent=2) + "\n", encoding="utf-8")
    plot_vehicle(recorder.rows, plot_path)
    other = args.output_dir / (
        "husky1_state_samples.csv" if args.vehicle == "Drone1"
        else "drone1_state_samples.csv")
    if other.exists():
        plot_combined([csv_path, other],
                      args.output_dir / "combined_state_comparison.png")
    print("MISSION_STATE_LIVE_METRICS " + json.dumps(metrics, sort_keys=True))
    print(f"CSV: {csv_path}\nPlot: {plot_path}\nMetrics: {metrics_path}")
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
