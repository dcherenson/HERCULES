#!/usr/bin/env python3
"""Reset one seeded six-robot MA-ICP AirSim fixture.

The reset is deliberately a launch-boundary operation.  It does not fit a
model, run a controller, or draw any calibration samples.  ``scene_plan`` is
pure and is also the source of the positions used by the live reset, so a
calibration and a paired sensitivity run can use the same seed without
changing the random draws when the case changes.

The controlled roster is three UAVs and three UGVs.  ``Target1`` is a
separate AirSim CPHusky actor used by tracking; it is never API-controlled by
this module and is not part of the MA-ICP update graph.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence


# The ROS package is installed in some runs and imported from the source tree
# in others.  Keep the AirSim dependency local to this repository instead of
# relying on a globally installed ``airsim`` package.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_PYTHON_CLIENT = _REPO_ROOT / "PythonClient"
if str(_PYTHON_CLIENT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_CLIENT))
if str(_PYTHON_CLIENT / "distributed_mission") not in sys.path:
    sys.path.insert(0, str(_PYTHON_CLIENT / "distributed_mission"))


CONTROLLED_AGENTS = (
    "Drone1",
    "Drone2",
    "SimpleFlight",
    "Husky1",
    "Husky2",
    "Husky3",
)
UAVS = CONTROLLED_AGENTS[:3]
UGVS = CONTROLLED_AGENTS[3:]
TARGET_NAME = "Target1"
CASES = ("collision", "tracking", "localization")
# The route metadata remains zero for the MA-ICP case API.  The closed-loop
# fixture starts tangent to +Y, so every body is reset with a +pi/2 yaw.
ROUTE_HEADING_RAD = 0.0
START_YAW_RAD = math.pi / 2.0
GROUND_Z = -1.0
UAV_Z = -5.0
START_X = 0.0
START_YS = (-9.0, 0.0, 9.0)
JITTER_LIMIT = 0.1
OWNED_PREFIX = "maicp_"
# FlyingCPP's one-metre obstacle mesh is the same asset used by the Python
# research missions. EditorCube has different geometry and is not a substitute.
BLOCK_ASSET = "1M_Cube_Chamfer"
COLUMN_PLANAR_RADIUS = math.sqrt(0.5**2 + 0.5**2)
CORRIDOR_INNER_WALL_Y = 14.75


def _validate_case(case: str) -> str:
    value = str(case).strip().lower()
    if value not in CASES:
        raise ValueError("case must be one of: {}".format(", ".join(CASES)))
    return value


def _as_float_list(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def _collision_geometry_plan() -> list[dict[str, Any]]:
    """Return the requested collision fixture in world NED coordinates."""

    columns = [
        ("maicp_column_0", (2.5, -9.0, -4.0)),
        ("maicp_column_1", (2.5, 0.0, -4.0)),
        ("maicp_column_2", (2.5, 9.0, -4.0)),
    ]
    geometry: list[dict[str, Any]] = []
    for name, center in columns:
        geometry.append({
            "name": name,
            "kind": "column",
            "asset": BLOCK_ASSET,
            "center": _as_float_list(center),
            "dimensions": [1.0, 1.0, 8.0],
            "physics_enabled": False,
            "collision_enabled": True,
            "planar_radius": float(COLUMN_PLANAR_RADIUS),
        })

    # A 0.5 m wall thickness at y=+/-15 leaves the controller's documented
    # inner corridor faces at y=+/-14.75.
    walls = [
        ("maicp_wall_y_neg", (7.0, -15.0, -4.0), -CORRIDOR_INNER_WALL_Y),
        ("maicp_wall_y_pos", (7.0, 15.0, -4.0), CORRIDOR_INNER_WALL_Y),
    ]
    for name, center, inner_face_y in walls:
        geometry.append({
            "name": name,
            "kind": "wall",
            "asset": BLOCK_ASSET,
            "center": _as_float_list(center),
            "dimensions": [20.0, 0.5, 8.0],
            "physics_enabled": False,
            "collision_enabled": True,
            "inner_face_y": float(inner_face_y),
        })
    return geometry


def scene_plan(seed: int, case: str) -> dict[str, Any]:
    """Build a deterministic, margin-independent fixture description.

    The RNG is initialized once and consumed only for the six controlled
    agents.  Target placement and collision geometry are case constants, so
    ``scene_plan(seed, case_a)["agents"]`` equals the corresponding value for
    every other case with the same seed.
    """

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    case_name = _validate_case(case)
    rng = random.Random(int(seed))
    jitter = [
        (rng.uniform(-JITTER_LIMIT, JITTER_LIMIT), rng.uniform(-JITTER_LIMIT, JITTER_LIMIT))
        for _ in CONTROLLED_AGENTS
    ]

    agents: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(CONTROLLED_AGENTS):
        slot = index % len(START_YS)
        class_name = "uav" if name in UAVS else "ugv"
        z = UAV_Z if class_name == "uav" else GROUND_Z
        jitter_xy = jitter[index]
        position = [START_X + jitter_xy[0], START_YS[slot] + jitter_xy[1], z]
        agents[name] = {
            "name": name,
            "class": class_name,
            "slot_index": int(slot),
            "base_position": [float(START_X), float(START_YS[slot]), float(z)],
            "jitter_xy": _as_float_list(jitter_xy),
            "position": _as_float_list(position),
        }

    target_position = [2.0, 6.0, GROUND_Z] if case_name == "tracking" else [-8.0, 0.0, GROUND_Z]
    return {
        "seed": int(seed),
        "case": case_name,
        "route_heading_rad": float(ROUTE_HEADING_RAD),
        "route_heading": float(ROUTE_HEADING_RAD),
        "start_yaw_rad": float(START_YAW_RAD),
        "owned_prefix": OWNED_PREFIX,
        "controlled_agents": list(CONTROLLED_AGENTS),
        "uavs": list(UAVS),
        "ugvs": list(UGVS),
        "agents": agents,
        "target": {
            "name": TARGET_NAME,
            "controlled": False,
            "actor_type": "target_ugv",
            "position": _as_float_list(target_position),
        },
        "collision_geometry": _collision_geometry_plan() if case_name == "collision" else [],
        "column_planar_radius": float(COLUMN_PLANAR_RADIUS),
        "corridor_inner_wall_y": float(CORRIDOR_INNER_WALL_Y),
        "notes": (
            "Seeded AirSim fixture adaptation for MA-ICP; coordinates are this "
            "fixture's world-NED layout and are not claimed to reproduce paper geography."
        ),
    }


def _pose(airsim: Any, position: Sequence[float], yaw: float = START_YAW_RAD) -> Any:
    return airsim.Pose(
        airsim.Vector3r(*_as_float_list(position)),
        airsim.to_quaternion(0.0, 0.0, float(yaw)),
    )


def _vector3(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        result = [float(value.x_val), float(value.y_val), float(value.z_val)]
    except (AttributeError, TypeError, ValueError):
        return None
    return result if all(math.isfinite(value) for value in result) else None


def _quaternion(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        result = [
            float(value.x_val),
            float(value.y_val),
            float(value.z_val),
            float(value.w_val),
        ]
    except (AttributeError, TypeError, ValueError):
        return None
    return result if all(math.isfinite(component) for component in result) else None


def _refresh_vehicle_names(facade: Any) -> set[str]:
    names: set[str] = set()
    for client in (facade.multirotor, facade.car):
        if client is None:
            continue
        try:
            names.update(str(name) for name in (client.listVehicles() or []))
        except Exception:
            continue
    # AirSimFacade intentionally keeps this cache private; refreshing it is
    # needed after sim.reset() because a runtime vehicle may be removed by the
    # simulator while the facade still has the previous session's cache.
    facade._vehicle_names = names  # type: ignore[attr-defined]
    return names


def _list_owned_objects(client: Any) -> list[str]:
    try:
        objects = client.simListSceneObjects(r"^maicp_.*$") or []
    except Exception as error:
        raise RuntimeError("unable to enumerate owned MA-ICP scene objects") from error
    return sorted(str(name) for name in objects if str(name).startswith(OWNED_PREFIX))


def _clean_owned_objects(client: Any) -> None:
    """Remove only actors owned by this fixture and verify the cleanup."""

    owned = _list_owned_objects(client)
    for name in owned:
        result = client.simDestroyObject(name)
        if result is False:
            raise RuntimeError("AirSim refused removal of owned object {!r}".format(name))

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        remaining = _list_owned_objects(client)
        if not remaining:
            return
        time.sleep(0.02)
    raise RuntimeError("owned scene cleanup incomplete: {}".format(remaining))


def _spawn_geometry(client: Any, airsim: Any, requested: Mapping[str, Any]) -> dict[str, Any]:
    name = str(requested["name"])
    center = [float(value) for value in requested["center"]]
    dimensions = [float(value) for value in requested["dimensions"]]
    scale = airsim.Vector3r(*dimensions)
    # A zero-yaw cube keeps the 20 m wall dimension along world X.  Vehicle
    # startup yaw is pi/2, but scene geometry has its own world orientation.
    pose = _pose(airsim, center, yaw=0.0)

    try:
        listed_assets = client.simListAssets()
    except Exception:
        listed_assets = None
    if listed_assets and BLOCK_ASSET not in {str(asset) for asset in listed_assets}:
        raise RuntimeError("required AirSim asset {!r} is unavailable".format(BLOCK_ASSET))

    try:
        result = client.simSpawnObject(
            name,
            BLOCK_ASSET,
            pose,
            scale,
            bool(requested.get("physics_enabled", False)),
            False,
        )
    except Exception as error:
        raise RuntimeError("failed to spawn required geometry {!r}".format(name)) from error
    if result is False:
        raise RuntimeError("AirSim returned failure while spawning {!r}".format(name))
    actual_name = str(result) if isinstance(result, str) and result else name
    if not actual_name.startswith(OWNED_PREFIX):
        raise RuntimeError(
            "AirSim returned an unowned name {!r} for requested object {!r}".format(
                actual_name, name
            )
        )

    try:
        visible = set(_list_owned_objects(client))
    except RuntimeError:
        visible = set()
    if actual_name not in visible and name not in visible:
        raise RuntimeError("spawned geometry {!r} is not present in AirSim".format(actual_name))

    try:
        object_pose = client.simGetObjectPose(actual_name, True)
        measured_center = _vector3(getattr(object_pose, "position", None))
    except Exception as error:
        raise RuntimeError("unable to read spawned geometry {!r} pose".format(actual_name)) from error
    if measured_center is None:
        raise RuntimeError("spawned geometry {!r} has an invalid pose".format(actual_name))
    try:
        measured_scale = _vector3(client.simGetObjectScale(actual_name))
    except Exception as error:
        raise RuntimeError("unable to read spawned geometry {!r} scale".format(actual_name)) from error
    if measured_scale is None:
        raise RuntimeError("spawned geometry {!r} has an invalid scale".format(actual_name))

    record = dict(requested)
    record.update({
        "name": actual_name,
        "requested_name": name,
        "actual_center": _as_float_list(measured_center),
        "actual_dimensions": _as_float_list(measured_scale),
        "physics_enabled": bool(requested.get("physics_enabled", False)),
        "collision_enabled": bool(requested.get("collision_enabled", True)),
    })
    return record


def _zero_vehicle_kinematics(facade: Any, name: str, vehicle_type: str) -> None:
    """Clear linear/angular velocity while preserving the teleported pose."""

    client = facade.multirotor if vehicle_type == "drone" else facade.car
    if client is None:
        raise RuntimeError("AirSim client is unavailable for {!r}".format(name))
    # This AirSim fork serializes NaN position fields as literal NaNs rather
    # than interpreting them as "leave unchanged".  Read the current local
    # kinematics (which includes the just-applied local frame correction) and
    # send finite pose fields back with zero twists.
    current_kinematics = client.simGetGroundTruthKinematics(vehicle_name=name)
    if (
        _vector3(getattr(current_kinematics, "position", None)) is not None
        and _quaternion(getattr(current_kinematics, "orientation", None)) is not None
    ):
        current_position = current_kinematics.position
        current_orientation = current_kinematics.orientation
    else:
        current_pose = client.simGetVehiclePose(vehicle_name=name)
        current_position = current_pose.position
        current_orientation = current_pose.orientation
    state = facade.airsim.KinematicsState()
    state.position = current_position
    state.orientation = current_orientation
    state.linear_velocity = facade.airsim.Vector3r(0.0, 0.0, 0.0)
    state.angular_velocity = facade.airsim.Vector3r(0.0, 0.0, 0.0)
    state.linear_acceleration = facade.airsim.Vector3r(0.0, 0.0, 0.0)
    state.angular_acceleration = facade.airsim.Vector3r(0.0, 0.0, 0.0)
    client.simSetKinematics(state, True, vehicle_name=name)


def _position_vehicle(
    facade: Any,
    name: str,
    vehicle_type: str,
    world_position: Sequence[float],
    origin: Sequence[float] | None = None,
) -> list[float]:
    """Set a world pose using the vehicle's local frame origin."""

    if origin is None:
        origin = facade.vehicle_frame_origin(name)
        if origin is None:
            raise RuntimeError("missing world frame origin for {!r}".format(name))
    origin_values = [float(value) for value in origin]
    if len(origin_values) != 3 or not all(math.isfinite(value) for value in origin_values):
        raise RuntimeError("invalid frame origin for {!r}".format(name))
    local_position = [
        float(value) - origin_values[index] for index, value in enumerate(world_position)
    ]
    facade.set_vehicle_pose(name, _pose(facade.airsim, local_position))
    return origin_values


def _ensure_vehicle(
    facade: Any,
    name: str,
    vehicle_type: str,
    world_position: Sequence[float],
) -> bool:
    if facade.has_vehicle(name):
        return False
    created = facade.spawn_vehicle(name, vehicle_type, _pose(facade.airsim, world_position))
    if not created and not facade.has_vehicle(name):
        raise RuntimeError("required AirSim vehicle {!r} is unavailable".format(name))
    return bool(created)


def _state_record(facade: Any, name: str) -> dict[str, Any]:
    state = facade.state(name)
    record: dict[str, Any] = {}
    for key in ("position", "velocity", "kinematics_position", "actor_position"):
        value = state.get(key)
        if value is None:
            record[key] = None
        else:
            values = _as_float_list(value)
            record[key] = values if all(math.isfinite(item) for item in values) else None
    yaw = float(state.get("yaw", 0.0))
    record["yaw"] = yaw if math.isfinite(yaw) else None
    return record


def _validate_reset_states(plan: Mapping[str, Any], states: Mapping[str, Mapping[str, Any]]) -> None:
    """Fail closed if the live pose/velocity snapshot is not usable."""

    expected_positions = {
        name: details["position"] for name, details in plan["agents"].items()
    }
    expected_positions[TARGET_NAME] = plan["target"]["position"]
    for name, expected in expected_positions.items():
        state = states.get(name)
        if state is None or state.get("position") is None or state.get("velocity") is None:
            raise RuntimeError("post-reset state for {!r} is incomplete".format(name))
        position = [float(value) for value in state["position"]]
        velocity = [float(value) for value in state["velocity"]]
        if not all(math.isfinite(value) for value in (*position, *velocity)):
            raise RuntimeError("post-reset state for {!r} is non-finite".format(name))
        if math.sqrt(sum((position[i] - float(expected[i])) ** 2 for i in range(3))) > 0.5:
            raise RuntimeError(
                "post-reset pose for {!r} moved too far from its seeded position: {}".format(
                    name, position
                )
            )
        if math.sqrt(sum(value * value for value in velocity)) > 0.25:
            raise RuntimeError(
                "post-reset velocity for {!r} is not zero: {}".format(name, velocity)
            )


def reset_scene(seed: int, case: str, *, host: str = "127.0.0.1", rpc_port: int = 41451,
                car_port: int = 41452) -> dict[str, Any]:
    """Reset AirSim and return the live manifest for one seeded fixture."""

    plan = scene_plan(seed, case)
    # Import the AirSim runtime only for a live reset.  Keeping it lazy makes
    # scene_plan usable in a plain Python test environment without RPC or the
    # optional simulator dependency installed.
    from simulation.airsim_runtime import AirSimFacade, AirSimLaunchConfig

    config = AirSimLaunchConfig(
        launch_mode="existing",
        host=str(host),
        multirotor_port=int(rpc_port),
        car_port=int(car_port),
    )
    facade = AirSimFacade(config)
    facade.connect()
    facade.multirotor.reset()
    facade.pause(True)
    try:
        _refresh_vehicle_names(facade)
        _clean_owned_objects(facade.multirotor)

        created: list[str] = []
        origins: dict[str, list[float]] = {}
        for name in CONTROLLED_AGENTS:
            vehicle_type = "drone" if name in UAVS else "ugv"
            position = plan["agents"][name]["position"]
            if _ensure_vehicle(facade, name, vehicle_type, position):
                created.append(name)
        if _ensure_vehicle(facade, TARGET_NAME, "ugv", plan["target"]["position"]):
            created.append(TARGET_NAME)

        for name in CONTROLLED_AGENTS:
            vehicle_type = "drone" if name in UAVS else "ugv"
            origins[name] = _as_float_list(
                _position_vehicle(facade, name, vehicle_type, plan["agents"][name]["position"])
            )
            if vehicle_type == "ugv":
                facade.stop_ugv(name)
            facade.enable(name, vehicle_type, True)
            _zero_vehicle_kinematics(facade, name, vehicle_type)

        # Target1 is intentionally an actor-only target.  It has a vehicle
        # body for AirSim sensors and mission motion, but no update-graph
        # membership in this reset.
        origins[TARGET_NAME] = _as_float_list(
            _position_vehicle(facade, TARGET_NAME, "ugv", plan["target"]["position"])
        )
        facade.enable(TARGET_NAME, "ugv", True)
        facade.stop_ugv(TARGET_NAME)
        _zero_vehicle_kinematics(facade, TARGET_NAME, "ugv")

        spawned_geometry: list[dict[str, Any]] = []
        for requested in plan["collision_geometry"]:
            spawned_geometry.append(_spawn_geometry(facade.multirotor, facade.airsim, requested))

        # Takeoff/hover is the only unpaused preparation stage.  MoveToZ keeps
        # the seeded XY target while establishing the paper's UAV altitude.
        facade.pause(False)
        takeoff_futures = [
            facade.multirotor.takeoffAsync(vehicle_name=name) for name in UAVS
        ]
        for future in takeoff_futures:
            future.join()
        altitude_futures = [
            facade.multirotor.moveToZAsync(UAV_Z, 3.0, vehicle_name=name)
            for name in UAVS
        ]
        for future in altitude_futures:
            future.join()
        hold_futures = [
            facade.multirotor.moveByVelocityZAsync(
                0.0, 0.0, UAV_Z, 1.0, vehicle_name=name
            )
            for name in UAVS
        ]
        for future in hold_futures:
            future.join()
        hover_futures = [
            facade.multirotor.hoverAsync(vehicle_name=name) for name in UAVS
        ]
        for future in hover_futures:
            future.join()

        # Reapply every seeded world pose after takeoff.  The pause makes this
        # correction deterministic and avoids accumulating launch drift in a
        # paired calibration run; a final hover resumes the UAV hold controller.
        facade.pause(True)
        for name in CONTROLLED_AGENTS:
            vehicle_type = "drone" if name in UAVS else "ugv"
            _position_vehicle(
                facade,
                name,
                vehicle_type,
                plan["agents"][name]["position"],
                origin=origins[name],
            )
            if vehicle_type == "ugv":
                facade.stop_ugv(name)
            _zero_vehicle_kinematics(facade, name, vehicle_type)
        _position_vehicle(
            facade,
            TARGET_NAME,
            "ugv",
            plan["target"]["position"],
            origin=origins[TARGET_NAME],
        )
        facade.stop_ugv(TARGET_NAME)
        _zero_vehicle_kinematics(facade, TARGET_NAME, "ugv")
        # Set the final pose while paused, then issue hover after releasing the
        # pause so the asynchronous command can complete against live physics.
        # A short settle window keeps the returned state at the seeded z=-5.
        facade.pause(False)
        hover_futures = [
            facade.multirotor.hoverAsync(vehicle_name=name) for name in UAVS
        ]
        for future in hover_futures:
            future.join()
        time.sleep(0.25)
        facade.pause(True)
        # Freeze the settled local kinematics one final time.  In particular,
        # a CPHusky can acquire a small vertical contact velocity while its
        # wheels settle onto the map even though its handbrake is engaged.
        for name in CONTROLLED_AGENTS:
            vehicle_type = "drone" if name in UAVS else "ugv"
            if vehicle_type == "ugv":
                facade.stop_ugv(name)
            _zero_vehicle_kinematics(facade, name, vehicle_type)
        facade.stop_ugv(TARGET_NAME)
        _zero_vehicle_kinematics(facade, TARGET_NAME, "ugv")
        pre_calibration_states = {
            name: _state_record(facade, name)
            for name in (*CONTROLLED_AGENTS, TARGET_NAME)
        }
        final_states = {
            name: _state_record(facade, name)
            for name in (*CONTROLLED_AGENTS, TARGET_NAME)
        }
        _validate_reset_states(plan, final_states)
        facade.pause(False)

        manifest = dict(plan)
        manifest.update({
            "reset": {
                "sim_reset": True,
                "host": str(host),
                "rpc_port": int(rpc_port),
                "car_port": int(car_port),
                "route_heading_rad": float(ROUTE_HEADING_RAD),
                "route_heading": float(ROUTE_HEADING_RAD),
                "start_yaw_rad": float(START_YAW_RAD),
                "created_vehicles": created,
                "vehicle_frame_origins": origins,
                "velocities_zeroed_before_calibration": True,
                "uav_takeoff_hover_completed": True,
                "pre_calibration_states": pre_calibration_states,
                "final_states": final_states,
            },
            "collision_geometry": spawned_geometry,
        })
        return manifest
    finally:
        # A failed preparation must not leave an existing AirSim session
        # frozen.  On success this is idempotent and leaves the fixture ready
        # for the calibration runner.
        try:
            facade.pause(False)
        except Exception:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="deterministic mission seed")
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--host", default="127.0.0.1", help="AirSim RPC host")
    parser.add_argument("--output", type=Path, default=None, help="optional manifest JSON path")
    parser.add_argument("--rpc-port", type=int, default=41451, help="AirSim multirotor RPC port")
    parser.add_argument("--car-port", type=int, default=41452, help="AirSim car RPC port")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = reset_scene(
        args.seed,
        args.case,
        host=args.host,
        rpc_port=args.rpc_port,
        car_port=args.car_port,
    )
    encoded = json.dumps(manifest, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"Reset {args.case} seed {args.seed}: {args.output}", flush=True)
    else:
        print(encoded, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
