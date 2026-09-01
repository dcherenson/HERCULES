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


def test_uav_slots_follow_target_and_rotate_with_motion():
    position = np.array([10.0, 20.0, -1.0])
    velocity = np.array([0.0, 2.0, 0.0])
    drone, feedforward, heading = target_centered_slot(
        "Drone1", "drone", position, velocity, 0.0, -5.0
    )
    # Drone1's (-2,-2) body-frame offset rotates to (+2,-2) for +Y motion.
    assert np.allclose(drone, [12.0, 18.0, -5.0])
    assert np.allclose(feedforward, velocity)
    assert np.isclose(heading, np.pi / 2.0)


def test_ugv_slots_are_equilateral_with_husky1_ahead():
    target = np.array([4.0, 8.0, 0.0])
    velocity = np.array([1.0, 0.0, 0.0])
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
