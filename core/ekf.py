"""Extended Kalman Filter state estimator — fuses noisy odometry with
intermittent noisy position fixes into a best estimate of [x, y, theta]."""
import numpy as np


class EKF:
    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray,
                 process_noise: np.ndarray, measurement_noise: np.ndarray):
        self._state = np.array(initial_state, dtype=float)
        self._P = np.array(initial_covariance, dtype=float)
        self._Q = np.array(process_noise, dtype=float)
        self._R = np.array(measurement_noise, dtype=float)

    def predict(self, v_measured: float, omega_measured: float, dt: float) -> None:
        """Propagate state and covariance using the noisy odometry reading
        as the control input. Needs the motion model's Jacobian with respect
        to [x, y, theta]."""
        x, y, theta = self._state

        # Jacobian of the motion model w.r.t. [x, y, theta], linearized at
        # the pre-propagation state.
        F = np.array([
            [1.0, 0.0, -v_measured * np.sin(theta) * dt],
            [0.0, 1.0, v_measured * np.cos(theta) * dt],
            [0.0, 0.0, 1.0],
        ])

        theta_new = theta + omega_measured * dt
        self._state = np.array([
            x + v_measured * np.cos(theta) * dt,
            y + v_measured * np.sin(theta) * dt,
            np.arctan2(np.sin(theta_new), np.cos(theta_new)),
        ])

        self._P = F @ self._P @ F.T + self._Q

    def update(self, position_measurement: np.ndarray) -> None:
        """Correct the state estimate using a noisy absolute position reading."""
        H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        innovation = np.asarray(position_measurement, dtype=float) - H @ self._state
        S = H @ self._P @ H.T + self._R
        K = self._P @ H.T @ np.linalg.inv(S)

        self._state = self._state + K @ innovation
        self._state[2] = np.arctan2(np.sin(self._state[2]), np.cos(self._state[2]))
        self._P = (np.eye(3) - K @ H) @ self._P

    def estimate(self) -> np.ndarray:
        """Return the current best estimate [x, y, theta]."""
        return self._state.copy()
