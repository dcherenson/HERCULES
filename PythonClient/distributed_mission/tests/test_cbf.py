import numpy as np

from modules.cbf import AgentState, CBFConfig, CBFRequest, DistributedCBFModule, ObstacleProxy
from orchestrator import cbf_sensor_valid_for_target, target_obstacle_proxy_for_agent


def test_double_integrator_passes_nominal_when_far_from_neighbor():
    config = CBFConfig(method="wang", uncertainty_radius=0.0)
    controller = DistributedCBFModule("Drone1", "drone", config)
    request = CBFRequest(
        AgentState("Drone1", np.array([0.0, 0.0, -5.0]), vehicle_type="drone"),
        np.array([0.1, 0.0, 0.0]),
        [AgentState("Drone2", np.array([20.0, 0.0, -5.0]), vehicle_type="drone")],
    )
    result = controller.filter(request)
    assert result.success
    assert np.allclose(result.safe_control, request.nominal_control, atol=1e-3)


def test_double_integrator_turns_away_from_obstacle():
    config = CBFConfig(method="wang", uncertainty_radius=0.0, uav_radius=1.0)
    controller = DistributedCBFModule("Drone1", "drone", config)
    request = CBFRequest(
        AgentState("Drone1", np.array([0.0, 0.0, -5.0]), np.array([2.0, 0.0, 0.0]), vehicle_type="drone"),
        np.array([4.0, 0.0, 0.0]),
        obstacles=[ObstacleProxy("wall", np.array([2.0, 0.0, -5.0]), 1.0)],
    )
    result = controller.filter(request)
    assert result.success
    assert result.safe_control[0] < request.nominal_control[0]


def test_mestres_ugv_returns_speed_and_yaw_rate():
    controller = DistributedCBFModule("Husky1", "ugv", CBFConfig(method="mestres", uncertainty_radius=0.0))
    result = controller.filter(CBFRequest(
        AgentState("Husky1", np.zeros(3), yaw=0.0, vehicle_type="ugv"),
        np.array([1.0, 0.0]),
    ))
    assert result.success
    assert result.safe_control.shape == (2,)
    assert np.allclose(result.safe_control, [1.0, 0.0], atol=1e-3)


def test_infeasible_constraint_uses_fail_safe():
    config = CBFConfig(method="wang", uncertainty_radius=0.0, solver_max_iter=100)
    controller = DistributedCBFModule("Drone1", "drone", config)
    # At the same location as a same-type obstacle, a stationary bounded
    # acceleration cannot satisfy the barrier constraint.
    result = controller.filter(CBFRequest(
        AgentState("Drone1", np.zeros(3), np.zeros(3), vehicle_type="drone"),
        np.array([1.0, 0.0, 0.0]),
        obstacles=[ObstacleProxy("occupied", np.zeros(3), 2.0)],
    ))
    assert not result.success
    assert result.fallback
    assert np.allclose(result.safe_control, np.zeros(3))


def test_invalid_sensor_uses_fail_safe_even_without_obstacles():
    controller = DistributedCBFModule("Drone1", "drone", CBFConfig(method="wang"))
    result = controller.filter(CBFRequest(
        AgentState("Drone1", np.array([0.0, 0.0, -5.0]), np.zeros(3), vehicle_type="drone"),
        np.array([1.0, 0.0, 0.0]),
        sensor_valid=False,
    ))
    assert not result.success
    assert result.status == "invalid_or_stale_sensor"


def test_cross_type_neighbors_are_not_added():
    controller = DistributedCBFModule("Drone1", "drone", CBFConfig(method="wang", uncertainty_radius=0.0))
    result = controller.filter(CBFRequest(
        AgentState("Drone1", np.array([0.0, 0.0, -5.0]), vehicle_type="drone"),
        np.zeros(3),
        neighbors=[AgentState("Husky1", np.zeros(3), vehicle_type="ugv")],
    ))
    assert result.success
    assert result.constraint_count == 1  # altitude constraint only


def test_uav_altitude_floor_is_configurable():
    controller = DistributedCBFModule(
        "Drone1", "drone", CBFConfig(method="wang", uncertainty_radius=0.0, uav_altitude_floor=1.0)
    )
    result = controller.filter(CBFRequest(
        AgentState("Drone1", np.array([0.0, 0.0, 0.0]), vehicle_type="drone"),
        np.zeros(3),
    ))
    assert result.success
    assert result.minimum_barrier >= 0.0


def test_moving_obstacle_velocity_enters_existing_relative_motion_barrier():
    config = CBFConfig(method="mestres", uncertainty_radius=0.0, ugv_radius=1.25)
    controller = DistributedCBFModule("Husky1", "ugv", config)
    ego = AgentState("Husky1", np.zeros(3), yaw=0.0, vehicle_type="ugv")
    static = controller.filter(CBFRequest(
        ego, np.array([1.0, 0.0]),
        obstacles=[ObstacleProxy("target", np.array([3.0, 0.0, 0.0]), 0.5)],
    ))
    closing = controller.filter(CBFRequest(
        ego, np.array([1.0, 0.0]),
        obstacles=[ObstacleProxy("target", np.array([3.0, 0.0, 0.0]), 0.5, velocity=np.array([-2.0, 0.0, 0.0]))],
    ))
    assert static.success
    assert closing.success
    assert np.linalg.norm(closing.safe_control) < np.linalg.norm(static.safe_control)


def test_target_proxy_is_only_added_to_ugv_cbf_requests():
    estimate = {
        "active": True,
        "position": [3.0, -2.0],
        "velocity": [0.5, 1.0],
        "covariance": [[0.25, 0.0], [0.0, 0.25]],
        "timestamp": 7.5,
    }
    assert target_obstacle_proxy_for_agent(
        "drone", "Target1", estimate, -2.0, 1.25, 8.0
    ) is None

    proxy = target_obstacle_proxy_for_agent(
        "ugv", "Target1", estimate, -2.0, 1.25, 8.0
    )
    assert proxy is not None
    assert proxy.obstacle_id == "target_Target1"
    assert np.allclose(proxy.center, [3.0, -2.0, -2.0])
    assert np.allclose(proxy.velocity, [0.5, 1.0, 0.0])
    assert proxy.source == "target_tracking"


def test_active_ugv_target_proxy_keeps_cbf_enabled_when_static_sensor_is_stale():
    proxy = ObstacleProxy("target_Target1", np.array([3.0, 0.0, 0.0]), 1.5)
    assert cbf_sensor_valid_for_target("ugv", False, proxy)
    assert not cbf_sensor_valid_for_target("drone", False, proxy)
    assert not cbf_sensor_valid_for_target("ugv", False, None)
    assert cbf_sensor_valid_for_target("ugv", True, None)
