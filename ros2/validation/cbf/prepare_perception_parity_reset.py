#!/usr/bin/env python3
"""Reset a fresh AirSim RuralAustralia session to Python-oracle startup state.

This is a small launch-boundary helper for the Python/ROS comparison.  It
uses the same formation pose, target sample, takeoff, and body-frame origin
conversion as ``distributed_mission/orchestrator.py``; it does not run a
controller or synthesize sensor data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "PythonClient"))
sys.path.insert(0, str(ROOT / "PythonClient" / "distributed_mission"))

from modules.formation_control import FormationConfig  # noqa: E402
from modules.target_motion import FigureEightTargetController  # noqa: E402
from orchestrator import (  # noqa: E402
    AirSimFacade,
    AirSimLaunchConfig,
    _pose,
    heading_to_goal,
    map_vehicle_ground_z,
    rotate_xy_heading,
)


AGENTS = ["Drone1", "Drone2", "SimpleFlight",
          "Husky1", "Husky2", "Husky3"]
UAVS = ["Drone1", "Drone2", "SimpleFlight"]
UGVS = ["Husky1", "Husky2", "Husky3"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-log", type=Path, required=True)
    parser.add_argument("--airsim-host", default="127.0.0.1",
                        help="AirSim RPC host (Docker Desktop bridge: host.docker.internal)")
    parser.add_argument("--rpc-port", type=int, default=41451)
    parser.add_argument("--car-port", type=int, default=41452)
    args = parser.parse_args()

    record = json.loads(args.python_log.read_text(encoding="utf-8").splitlines()[0])
    goal = np.asarray(record["goal"], dtype=float)
    pattern = record["target"]["pattern"]
    route_heading = float(pattern["route_heading"])
    initial_yaw = heading_to_goal(np.zeros(3), goal)
    target_center = np.asarray(pattern["center"], dtype=float)
    target_index = 5
    target_controller = FigureEightTargetController(
        center=target_center.copy(),
        route_heading=route_heading,
        longitudinal_span=float(pattern["longitudinal_span"]),
        lateral_span=float(pattern["lateral_span"]),
        speed=float(pattern["speed"]),
        sample_count=int(pattern["sample_count"]),
        waypoint_radius=float(pattern.get("waypoint_radius", 1.0)),
        heading_gain=float(pattern.get("heading_gain", 2.0)),
        max_yaw_rate=float(pattern.get("max_yaw_rate", 1.5)),
        minimum_alignment=float(pattern.get("minimum_alignment", 0.75)),
        direction=int(pattern.get("direction", 1)),
    )
    target_controller.index = target_index
    target_start_position = target_controller.points[target_index]
    target_start_yaw = heading_to_goal(
        target_start_position,
        target_controller.reference(1),
        fallback=initial_yaw + np.pi,
    )

    formation = FormationConfig(uav_altitude=-5.0)
    uav_base = np.array([0.0, 0.0, -1.0])
    ugv_base = np.array([0.0, 0.0, map_vehicle_ground_z("rural_australia")])
    types = {name: "drone" for name in UAVS} | {name: "ugv" for name in UGVS}
    initial_positions = {
        name: (uav_base if types[name] == "drone" else ugv_base) +
        rotate_xy_heading(formation.slots[name], initial_yaw)
        for name in AGENTS
    }

    config = AirSimLaunchConfig(
        launch_mode="existing",
        host=args.airsim_host,
        multirotor_port=args.rpc_port,
        car_port=args.car_port,
    )
    facade = AirSimFacade(config)
    facade.connect()
    facade.pause(True)
    origins = {}
    for name in AGENTS:
        origin = facade.vehicle_frame_origin(name)
        origins[name] = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
        facade.stop_ugv(name) if types[name] == "ugv" else None
    for name in AGENTS:
        request = initial_positions[name] - origins[name]
        facade.set_vehicle_pose(name, _pose(facade.airsim, request, initial_yaw))
    facade.set_vehicle_pose("Target1", _pose(facade.airsim, target_start_position, target_start_yaw))
    for name in AGENTS:
        facade.enable(name, types[name], True)
    facade.enable("Target1", "ugv", True)
    for name in UGVS + ["Target1"]:
        facade.stop_ugv(name)
    facade.pause(False)

    for name in UAVS:
        facade.multirotor.takeoffAsync(vehicle_name=name).join()
    for name in UAVS:
        facade.multirotor.moveToZAsync(-5.0, 3.0, vehicle_name=name).join()
    for name in UAVS:
        facade.multirotor.moveByVelocityZAsync(
            0.0, 0.0, -5.0, 1.0, vehicle_name=name).join()
        facade.multirotor.hoverAsync(vehicle_name=name).join()

    facade.pause(True)
    for name in UGVS:
        facade.set_vehicle_pose(
            name,
            _pose(facade.airsim, initial_positions[name] - origins[name], initial_yaw),
        )
        facade.stop_ugv(name)
    facade.pause(False)
    facade.set_vehicle_pose("Target1", _pose(facade.airsim, target_start_position, target_start_yaw))
    facade.stop_ugv("Target1")

    print(json.dumps({
        "python_log": str(args.python_log),
        "goal": goal.tolist(),
        "route_heading_rad": route_heading,
        "target_center": target_center.tolist(),
        "target_start_position_request": target_start_position.tolist(),
        "target_start_yaw": float(target_start_yaw),
        "initial_positions_request": {name: initial_positions[name].tolist() for name in AGENTS},
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
