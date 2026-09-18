"""Extended Kalman Filter state estimator — fuses noisy odometry with
intermittent noisy position fixes into a best estimate of [x, y, theta]."""
import numpy as np


class EKF:
    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray,
                 process_noise: np.ndarray, measurement_noise: np.ndarray):
        raise NotImplementedError

    def predict(self, v_measured: float, omega_measured: float, dt: float) -> None:
        """Propagate state and covariance using the noisy odometry reading
        as the control input. Needs the motion model's Jacobian with respect
        to [x, y, theta]."""
        raise NotImplementedError

    def update(self, position_measurement: np.ndarray) -> None:
        """Correct the state estimate using a noisy absolute position reading."""
        raise NotImplementedError

    def estimate(self) -> np.ndarray:
        """Return the current best estimate [x, y, theta]."""
        raise NotImplementedError
