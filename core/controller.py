"""PID waypoint controller. Steers using the EKF's estimate — never the true state."""
import numpy as np


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float):
        raise NotImplementedError

    def compute(self, error: float) -> float:
        """Standard PID output for a single error signal (here, heading error)."""
        raise NotImplementedError


def heading_error(current_theta: float, target_x: float, target_y: float,
                   current_x: float, current_y: float) -> float:
    """Angle (radians, normalized to [-pi, pi]) between current heading and
    the direction toward the target waypoint."""
    raise NotImplementedError


def waypoint_controller(estimate: np.ndarray, waypoint: np.ndarray,
                         pid: PIDController, cruise_speed: float = 1.0,
                         arrival_radius: float = 0.3) -> tuple[float, float, bool]:
    """Return (v_cmd, omega_cmd, reached_waypoint). Uses the ESTIMATE, never
    the true state."""
    raise NotImplementedError
