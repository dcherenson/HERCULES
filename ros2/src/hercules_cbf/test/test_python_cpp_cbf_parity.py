"""Row and control parity harness for the native Phase 1 CBF core."""
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

source = pathlib.Path(os.environ.get("HERCULES_SOURCE_DIR", "")).resolve()
if str(source) not in sys.path:
    sys.path.insert(0, str(source))
from modules.cbf import AgentState, CBFConfig, CBFRequest, DistributedCBFModule, ObstacleProxy


def s(vehicle, ident, p, v=(0, 0, 0), yaw=0.0, acceleration=None):
    return AgentState(ident, np.asarray(p, float), np.asarray(v, float), yaw,
                      None if acceleration is None else np.asarray(acceleration, float), vehicle)


def o(ident, c, radius, velocity=None):
    return ObstacleProxy(ident, np.asarray(c, float), radius,
                         velocity=None if velocity is None else np.asarray(velocity, float))


def cases():
    n = s("drone", "Drone2", (3, 1, -5), (.2, -.1, 0), acceleration=(.3, -.2, .1))
    wang = CBFConfig(method="wang")
    mestres = CBFConfig(method="mestres")
    def d(p=(0, 0, -5), v=(0, 0, 0), u=(0, 0, 0)):
        return CBFRequest(s("drone", "Drone1", p, v), np.asarray(u, float))
    r = d((0, 0, -5), (1, -.5, .2), (.4, -.2, .1)); r.neighbors = [n]
    yield "uav_neighbor", r, wang
    r = d((0, 0, -5), (2, 0, 0), (4, 0, 0)); r.obstacles = [o("wall", (2, 0, -5), 1)]; yield "uav_obstacle_static", r, wang
    r.obstacles = [o("wall", (2, 0, -5), 1, (-1, 0, 0))]; yield "uav_obstacle_moving", r, wang
    r = d((0, 0, 0)); r.uncertainty_radius = .25; yield "altitude_uncertainty", r, wang
    r = CBFRequest(s("ugv", "Husky1", (0, 0, 0), yaw=.4), np.array([1., 0.])); yield "ugv_no_obstacle", r, mestres
    r.neighbors = [s("ugv", "Husky2", (2, 1, 0), (.1, .2, 0))]; yield "ugv_neighbor", r, mestres
    r.neighbors = []; r.obstacles = [o("target", (3, 0, 0), .5, (-2, 0, 0))]; yield "ugv_moving_obstacle", r, mestres
    r = d((0, 0, -5), (1, 0, 0)); r.neighbors = [n]; yield "wang_split", r, wang
    r = d((0, 0, -5), u=(2, -3, 4)); r.control_bounds = (np.array([-1, -2, -3.]), np.array([1, 2, 3.])); yield "custom_bounds", r, wang
    r = d(); r.obstacles = [o("occupied", (0, 0, -5), 2)]; yield "infeasible", r, wang
    r.sensor_valid = False; yield "invalid_sensor", r, wang
    r = d(); r.neighbors = [s("ugv", "Husky1", (0, 0, -5))]; yield "cross_type_exclusion", r, wang
    r = d(); r.obstacles = [o("target_Target1", (3, -2, -1), 2.25, (.5, 1, 0))]; yield "moving_target_proxy", r, wang


def python_case(request, config):
    result = DistributedCBFModule(request.ego.agent_id, request.ego.vehicle_type, config).filter(request)
    constraints = DistributedCBFModule(request.ego.agent_id, request.ego.vehicle_type, config)
    # Build via the public filter's internal result and use a second invocation
    # only for the stable, public diagnostics. A row-level oracle is generated
    # by mirroring the reference's private builders through diagnostics below.
    module = constraints
    rows, rhs, barriers, robust = [], [], [], []
    if module._is_unicycle():
        module._build_unicycle_constraints(request, rows, rhs, barriers, robust)
    else:
        module._build_double_integrator_constraints(request, rows, rhs, barriers, robust)
    lower, upper = module._bounds(request, 2 if module._is_unicycle() else 3)
    return {"rows": np.asarray(rows).tolist(), "rhs": rhs, "barriers": barriers,
            "robust_terms": robust, "labels": ["neighbor:" + x.agent_id for x in request.neighbors if x.vehicle_type == request.ego.vehicle_type] +
                      ["obstacle:" + x.obstacle_id for x in request.obstacles] + ([] if module._is_unicycle() or request.ego.vehicle_type != "drone" else ["altitude"]),
            "lower": lower.tolist(), "upper": upper.tolist(), "safe_control": result.safe_control.tolist(),
            "success": result.success, "status": result.status, "fallback": result.fallback,
            "active_constraints": result.active_constraints, "constraint_count": result.constraint_count,
            "distributed_rounds": result.distributed_rounds}


def assert_close(actual, expected, path):
    if isinstance(expected, list) and (not expected or isinstance(expected[0], (int, float, list))):
        np.testing.assert_allclose(np.asarray(actual, float), np.asarray(expected, float), atol=1e-9, rtol=1e-9, err_msg=path)
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def main():
    exe = os.environ["CBF_PARITY_EXE"]
    native = json.loads(subprocess.check_output([exe], text=True))
    expected = {name: python_case(req, cfg) for name, req, cfg in cases()}
    for item in native:
        name = item["name"]
        assert name in expected
        for key in ("rows", "rhs", "barriers", "robust_terms", "lower", "upper", "safe_control"):
            assert_close(item[key], expected[name][key], f"{name}.{key}")
        for key in ("labels", "success", "fallback", "active_constraints", "constraint_count", "distributed_rounds"):
            assert_close(item[key], expected[name][key], f"{name}.{key}")
        if name not in ("infeasible", "invalid_sensor"):
            assert item["status"] == expected[name]["status"], f"{name}.status"
    assert len(native) == len(expected)


if __name__ == "__main__":
    main()
