import numpy as np

from core.controller import PIDController, heading_error


def test_heading_error_normalized_to_pi_range():
    # Target directly behind the robot: magnitude should be exactly pi.
    err = heading_error(current_theta=0.0, target_x=-1.0, target_y=0.0,
                         current_x=0.0, current_y=0.0)
    assert -np.pi <= err <= np.pi
    assert np.isclose(abs(err), np.pi, atol=1e-6)

    # A heading that requires wrapping around +-pi should still normalize.
    err2 = heading_error(current_theta=-3.0, target_x=1.0, target_y=0.1,
                          current_x=0.0, current_y=0.0)
    assert -np.pi <= err2 <= np.pi


def test_heading_error_zero_when_facing_target():
    err = heading_error(current_theta=np.pi / 4, target_x=1.0, target_y=1.0,
                         current_x=0.0, current_y=0.0)
    assert np.isclose(err, 0.0, atol=1e-6)


def test_controller_drives_heading_error_toward_zero():
    """Over repeated calls with a stationary target, PID output should steer
    heading error toward zero rather than away from it or oscillating forever."""
    pid = PIDController(kp=1.5, ki=0.0, kd=0.3, dt=0.1)
    theta = -1.0
    target = np.array([5.0, 0.0])
    pos = np.array([0.0, 0.0])

    errors = []
    for _ in range(50):
        err = heading_error(theta, target[0], target[1], pos[0], pos[1])
        errors.append(abs(err))
        omega = pid.compute(err)
        theta += omega * 0.1

    assert errors[-1] < errors[0]
    assert errors[-1] < 0.05
