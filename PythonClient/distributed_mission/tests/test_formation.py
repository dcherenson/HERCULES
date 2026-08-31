import numpy as np

from modules.cbf import AgentState
from modules.formation_control import FormationController, FormationConfig, wrap_angle


def test_slot_rotates_with_leader_heading():
    controller = FormationController()
    states = {"Husky1": AgentState("Husky1", np.zeros(3), yaw=np.pi / 2, vehicle_type="ugv")}
    position, _, _ = controller.reference_for("Husky2", states)
    assert np.allclose(position, np.array([2.0, -2.0 * np.sqrt(3.0), 0.0]))


def test_uav_slots_form_box_with_center():
    slots = FormationConfig().slots
    corners = np.array([slots[name][:2] for name in ("Drone1", "Drone2", "Drone4", "Drone5")])
    assert np.allclose(np.min(corners, axis=0), [-2.0, -2.0])
    assert np.allclose(np.max(corners, axis=0), [2.0, 2.0])
    assert np.allclose(slots["SimpleFlight"][:2], [0.0, 0.0])


def test_uav_reference_uses_fixed_global_hover_altitude():
    controller = FormationController(FormationConfig(uav_altitude=-5.0))
    states = {"Husky1": AgentState("Husky1", np.array([0.0, 0.0, 1.7]), vehicle_type="ugv")}
    position, velocity, _ = controller.reference_for("Drone1", states)
    assert np.isclose(position[2], -5.0)
    assert np.isclose(velocity[2], 0.0)


def test_ugv_slots_form_equilateral_triangle_with_leader_at_front():
    slots = FormationConfig().slots
    positions = np.array([slots[name][:2] for name in ("Husky1", "Husky2", "Husky3")])
    distances = [
        np.linalg.norm(positions[0] - positions[1]),
        np.linalg.norm(positions[0] - positions[2]),
        np.linalg.norm(positions[1] - positions[2]),
    ]
    assert np.allclose(distances, [4.0, 4.0, 4.0])
    assert positions[0, 0] > positions[1:, 0].max()


def test_leader_and_followers_get_distinct_nominal_references():
    controller = FormationController()
    states = {
        "Husky1": AgentState("Husky1", np.zeros(3), vehicle_type="ugv"),
        "Husky2": AgentState("Husky2", FormationConfig().slots["Husky2"], vehicle_type="ugv"),
    }
    leader = controller.nominal_unicycle_control(states["Husky1"], states, np.array([10.0, 0.0, 0.0]))
    follower = controller.nominal_unicycle_control(states["Husky2"], states, np.array([10.0, 0.0, 0.0]))
    assert leader[0] > 0.0
    assert follower[0] <= controller.config.max_speed


def test_heading_wraps_to_shortest_direction():
    assert np.isclose(wrap_angle(3.0 * np.pi), -np.pi)


def test_single_preplanned_waypoint_is_used_before_goal():
    controller = FormationController(FormationConfig(intermediate_waypoint=np.array([16.0, -10.0, -1.0])))
    states = {"Husky1": AgentState("Husky1", np.zeros(3), vehicle_type="ugv")}
    nominal = controller.nominal_unicycle_control(states["Husky1"], states, np.array([16.0, 0.0, -1.0]))
    assert nominal[0] > 0.0
    assert nominal[1] < 0.0


def test_uavs_use_fixed_waypoint_center_before_leader_arrives():
    waypoint = np.array([10.0, -10.0, -1.0])
    controller = FormationController(FormationConfig(intermediate_waypoint=waypoint))
    states = {"Husky1": AgentState("Husky1", np.zeros(3), vehicle_type="ugv")}
    position, velocity, _ = controller.reference_for("SimpleFlight", states)
    assert np.allclose(position, waypoint + controller.config.slots["SimpleFlight"])
    assert np.allclose(velocity, np.zeros(3))
