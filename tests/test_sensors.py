import numpy as np

from core.sensors import noisy_odometry, noisy_position_fix


def test_noisy_odometry_zero_noise_returns_exact_reading():
    v_meas, omega_meas = noisy_odometry(1.0, 0.5, v_noise_std=0.0, omega_noise_std=0.0)
    assert np.isclose(v_meas, 1.0)
    assert np.isclose(omega_meas, 0.5)


def test_noisy_odometry_matches_specified_noise_statistics():
    np.random.seed(0)
    true_v, true_omega = 2.0, 0.3
    v_std, omega_std = 0.1, 0.05
    samples = np.array([noisy_odometry(true_v, true_omega, v_std, omega_std)
                         for _ in range(5000)])

    assert np.isclose(samples[:, 0].mean(), true_v, atol=0.02)
    assert np.isclose(samples[:, 0].std(), v_std, atol=0.02)
    assert np.isclose(samples[:, 1].mean(), true_omega, atol=0.02)
    assert np.isclose(samples[:, 1].std(), omega_std, atol=0.02)


def test_noisy_position_fix_zero_noise_returns_exact_reading():
    fix = noisy_position_fix(3.0, -2.0, noise_std=0.0)
    assert np.allclose(fix, [3.0, -2.0])


def test_noisy_position_fix_matches_specified_noise_statistics():
    np.random.seed(1)
    true_x, true_y = 1.0, -1.0
    std = 0.5
    samples = np.array([noisy_position_fix(true_x, true_y, noise_std=std)
                         for _ in range(5000)])

    assert np.isclose(samples[:, 0].mean(), true_x, atol=0.05)
    assert np.isclose(samples[:, 0].std(), std, atol=0.05)
    assert np.isclose(samples[:, 1].mean(), true_y, atol=0.05)
    assert np.isclose(samples[:, 1].std(), std, atol=0.05)
