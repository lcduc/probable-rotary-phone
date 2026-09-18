"""True robot dynamics (ground truth). Never queried by the controller or EKF directly."""
import numpy as np


class TrueRobot:
    def __init__(self, x0: float, y0: float, theta0: float,
                 actuation_noise_std: tuple[float, float] = (0.02, 0.01)):
        """actuation_noise_std: (v_noise, omega_noise) — how much the robot's
        actual motion deviates from the commanded (v, omega) each step."""
        self._x = x0
        self._y = y0
        self._theta = theta0
        self._v_noise_std, self._omega_noise_std = actuation_noise_std
        self._last_v = 0.0
        self._last_omega = 0.0

    def step(self, v_cmd: float, omega_cmd: float, dt: float) -> None:
        """Advance the TRUE state by one time step, given commanded (v, omega).
        Add actuation noise before integrating — the robot doesn't perfectly
        execute what it's told. Store the post-noise (v, omega) so
        last_actual_velocity() can expose it."""
        v = v_cmd + np.random.normal(0.0, self._v_noise_std)
        omega = omega_cmd + np.random.normal(0.0, self._omega_noise_std)

        self._x += v * np.cos(self._theta) * dt
        self._y += v * np.sin(self._theta) * dt
        self._theta = np.arctan2(np.sin(self._theta + omega * dt),
                                  np.cos(self._theta + omega * dt))

        self._last_v = v
        self._last_omega = omega

    def true_state(self) -> np.ndarray:
        """Return [x, y, theta] — for logging/plotting only. The controller
        and filter must never call this directly."""
        return np.array([self._x, self._y, self._theta])

    def last_actual_velocity(self) -> tuple[float, float]:
        """Return the (v, omega) actually executed on the most recent step
        (post-actuation-noise, pre-sensor-noise). simulate.py feeds this,
        not the commanded values, into noisy_odometry() — the robot's
        encoders measure what it did, not what it was told to do."""
        return self._last_v, self._last_omega
