import numpy as np

from core.robot_sim import TrueRobot


def test_straight_line_motion_matches_ideal_kinematics():
    """With zero actuation noise and zero angular velocity, position should
    advance exactly as v * dt along the current heading."""
    robot = TrueRobot(x0=0.0, y0=0.0, theta0=0.0, actuation_noise_std=(0.0, 0.0))
    v, omega, dt = 2.0, 0.0, 0.1

    for _ in range(10):
        robot.step(v, omega, dt)

    x, y, theta = robot.true_state()
    assert np.isclose(theta, 0.0, atol=1e-9)
    assert np.isclose(x, v * dt * 10, atol=1e-6)
    assert np.isclose(y, 0.0, atol=1e-6)


def test_pure_rotation_matches_ideal_kinematics():
    """With zero actuation noise and zero linear velocity, position should
    not change and heading should advance exactly as omega * dt."""
    robot = TrueRobot(x0=0.0, y0=0.0, theta0=0.0, actuation_noise_std=(0.0, 0.0))
    omega, dt = 1.0, 0.1

    for _ in range(5):
        robot.step(0.0, omega, dt)

    x, y, theta = robot.true_state()
    assert np.isclose(theta, omega * dt * 5, atol=1e-9)
    assert np.isclose(x, 0.0, atol=1e-6)
    assert np.isclose(y, 0.0, atol=1e-6)


def test_last_actual_velocity_matches_commanded_when_noiseless():
    robot = TrueRobot(x0=0.0, y0=0.0, theta0=0.0, actuation_noise_std=(0.0, 0.0))
    robot.step(1.5, -0.3, 0.1)
    actual_v, actual_omega = robot.last_actual_velocity()
    assert np.isclose(actual_v, 1.5)
    assert np.isclose(actual_omega, -0.3)
