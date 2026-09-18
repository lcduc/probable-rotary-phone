import numpy as np

from core.ekf import EKF


def _make_ekf():
    return EKF(initial_state=np.array([0.0, 0.0, 0.0]),
               initial_covariance=np.eye(3) * 0.1,
               process_noise=np.diag([0.01, 0.01, 0.005]),
               measurement_noise=np.eye(2) * 0.25)


def test_odometry_only_drifts_from_ground_truth():
    """With no position updates, dead-reckoning should accumulate error over
    time when the odometry reading is biased relative to the true motion."""
    ekf = _make_ekf()
    true_x, true_y, true_theta = 0.0, 0.0, 0.0
    v, omega, dt = 1.0, 0.1, 0.1
    biased_v = v + 0.05  # odometry consistently overreports speed

    for _ in range(200):
        ekf.predict(biased_v, omega, dt)
        true_x += v * dt * np.cos(true_theta)
        true_y += v * dt * np.sin(true_theta)
        true_theta += omega * dt

    est_x, est_y, _ = ekf.estimate()
    error = np.hypot(est_x - true_x, est_y - true_y)
    assert error > 0.5


def test_periodic_updates_reduce_final_error_vs_odometry_only():
    """Adding periodic position updates should measurably reduce final
    position error compared to odometry-only dead-reckoning."""
    np.random.seed(0)
    v, omega, dt = 1.0, 0.1, 0.1
    biased_v = v + 0.05

    # Odometry-only baseline.
    ekf_odom_only = _make_ekf()
    true_x, true_y, true_theta = 0.0, 0.0, 0.0
    for _ in range(200):
        ekf_odom_only.predict(biased_v, omega, dt)
        true_x += v * dt * np.cos(true_theta)
        true_y += v * dt * np.sin(true_theta)
        true_theta += omega * dt
    odom_only_error = np.hypot(
        ekf_odom_only.estimate()[0] - true_x,
        ekf_odom_only.estimate()[1] - true_y,
    )

    # Same run, with periodic noisy position updates.
    ekf_corrected = _make_ekf()
    true_x, true_y, true_theta = 0.0, 0.0, 0.0
    for step in range(200):
        ekf_corrected.predict(biased_v, omega, dt)
        true_x += v * dt * np.cos(true_theta)
        true_y += v * dt * np.sin(true_theta)
        true_theta += omega * dt
        if step % 10 == 0:
            noisy_fix = np.array([true_x, true_y]) + np.random.normal(0, 0.1, size=2)
            ekf_corrected.update(noisy_fix)
    corrected_error = np.hypot(
        ekf_corrected.estimate()[0] - true_x,
        ekf_corrected.estimate()[1] - true_y,
    )

    assert corrected_error < odom_only_error
