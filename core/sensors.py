"""Noisy odometry + noisy position-fix sensor models. Fully synthetic."""
import numpy as np


def noisy_odometry(true_v: float, true_omega: float,
                    v_noise_std: float = 0.05, omega_noise_std: float = 0.02) -> tuple[float, float]:
    """Simulate a wheel-encoder-style reading of the robot's actual velocity —
    this is what the EKF's predict step will use."""
    v_meas = true_v + np.random.normal(0.0, v_noise_std)
    omega_meas = true_omega + np.random.normal(0.0, omega_noise_std)
    return v_meas, omega_meas


def noisy_position_fix(true_x: float, true_y: float,
                        noise_std: float = 0.5) -> np.ndarray:
    """Simulate an intermittent absolute position sensor (a stand-in for GPS
    or a landmark detector). Only called every N simulation steps, not every step."""
    return np.array([true_x, true_y]) + np.random.normal(0.0, noise_std, size=2)
