import numpy as np

from modules.target_motion import FigureEightTargetController, route_basis, target_centered_slot


def test_figure_eight_is_route_aligned_and_starts_off_crossing():
    controller = FigureEightTargetController(np.zeros(3), np.pi / 2.0, sample_count=64)
    forward, left = route_basis(np.pi / 2.0)
    points = controller.points
    assert points.shape == (64, 3)
    assert np.isclose(np.max(np.dot(points[:, :2], forward)), 5.0)
    assert np.isclose(np.max(np.dot(points[:, :2], left)), 4.0)
    assert not np.allclose(points[controller.index, :2], [0.0, 0.0])


def test_target_path_wraps_and_produces_bounded_ugv_command():
    controller = FigureEightTargetController(np.zeros(3), 0.0, sample_count=8, speed=1.5)
    controller.index = 7
    command = controller.update(controller.points[7], 0.0, 0.1)
    assert controller.index == 0
    assert 0.0 <= command[0] <= 1.5
    assert abs(command[1]) <= controller.max_yaw_rate


def test_target_path_progresses_after_a_physical_overshoot():
    controller = FigureEightTargetController(np.zeros(3), 0.0, sample_count=64)
    controller.index = 8
    controller.update(controller.points[11], 0.0, 0.1)
    assert controller.index >= 11


def test_target_reverse_direction_faces_and_progresses_back_along_path():
    controller = FigureEightTargetController(
        np.zeros(3), 0.0, sample_count=64, direction=-1, waypoint_radius=0.0,
    )
    controller.index = 16
    points = controller.points
    assert np.allclose(controller.reference(1), points[15])

    delta = points[15, :2] - points[16, :2]
    yaw = float(np.arctan2(delta[1], delta[0]))
    command = controller.update(points[16], yaw, 0.1)

    assert controller.index == 15
    assert command[0] > 0.0


def test_target_start_phase_moves_toward_chase_camera_right():
    controller = FigureEightTargetController(
        np.zeros(3), 0.0, sample_count=64, direction=1,
    )
    controller.index = 5
    controller.place_start_at(np.zeros(3))
    start = controller.points[controller.index].copy()

    # The fixed startup update may consume already-reached samples, but the
    # selected segment must still have positive camera-right (route-basis
    # lateral) displacement.
    controller.update(start, 0.0, 0.1)
    displacement = controller.reference()[:2] - start[:2]
    _, camera_right = route_basis(0.0)
    assert float(np.dot(displacement, camera_right)) > 0.0


def test_target_start_can_be_exactly_placed_without_removing_lateral_lobe():
    controller = FigureEightTargetController(np.zeros(3), 0.0, sample_count=64)
    desired = np.array([12.0, -4.0, 0.7])
    controller.place_start_at(desired)
    assert np.allclose(controller.points[controller.index], desired)
    assert not np.allclose(controller.points[0, :2], desired[:2])


def test_target_keeps_a_nonzero_command_while_turning_from_zero_yaw():
    controller = FigureEightTargetController(
        np.array([11.2, 0.7, 0.7]), 0.0, sample_count=64, speed=0.1,
    )
    controller.index = 12
    command = controller.update(controller.points[12], 0.0, 0.1)
    assert command[0] >= 0.1 * controller.minimum_alignment - 1e-9


def test_uav_slots_follow_target_and_rotate_with_motion():
    position = np.array([10.0, 20.0, -1.0])
    velocity = np.array([0.0, 3.0, 0.0])
    drone, feedforward, heading = target_centered_slot(
        "Drone1", "drone", position, velocity, 0.0, -5.0
    )
    # Drone1's (-2,-2) body-frame offset rotates to (+2,-2) for +Y motion.
    assert np.allclose(drone, [12.0, 18.0, -5.0])
    assert np.allclose(feedforward, velocity)
    assert np.isclose(heading, np.pi / 2.0)


def test_ugv_slots_are_equilateral_with_husky1_ahead():
    target = np.array([4.0, 8.0, 0.0])
    velocity = np.array([3.0, 0.0, 0.0])
    slots = [
        target_centered_slot(name, "ugv", target, velocity, np.pi / 2.0, -5.0, target_z=0.0, ugv_circumradius=6.0)[0]
        for name in ("Husky1", "Husky2", "Husky3")
    ]
    distances = [np.linalg.norm(slots[index][:2] - slots[(index + 1) % 3][:2]) for index in range(3)]
    assert np.allclose(distances, [6.0 * np.sqrt(3.0)] * 3)
    assert np.allclose(np.mean(np.asarray(slots), axis=0), target)
    assert slots[0][0] > target[0]


def test_target_slot_falls_back_to_route_heading_at_low_speed():
    _, _, heading = target_centered_slot(
        "Husky1", "ugv", np.zeros(3), np.zeros(3), np.pi / 2.0, -5.0
    )
    assert np.isclose(heading, np.pi / 2.0)


def test_target_slot_does_not_rotate_on_weak_directional_signal():
    _, _, heading = target_centered_slot(
        "Husky1", "ugv", np.zeros(3), np.array([1.0, 0.0, 0.0]), np.pi / 2.0, -5.0
    )
    assert np.isclose(heading, np.pi / 2.0)
