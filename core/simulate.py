"""Main simulation loop tying the true robot, sensors, EKF, and controller
together. Mechanical glue — the interesting work lives in the modules it
imports, not here."""
import numpy as np

from .robot_sim import TrueRobot
from .sensors import noisy_odometry, noisy_position_fix
from .ekf import EKF
from .controller import PIDController, waypoint_controller


def run_simulation(waypoints: list[np.ndarray], n_steps: int, dt: float,
                    gps_update_interval: int = 10,
                    noise_params: dict | None = None,
                    use_estimate: bool = True) -> dict:
    """Ties everything together. Logs true state, estimated state, and
    waypoint progress at every step for later plotting.

    noise_params: optional overrides for {"v_noise", "omega_noise",
        "position_noise"} — used by experiments.py to sweep noise levels.
    use_estimate: if False, the controller steers using ground truth instead
        of the EKF estimate — the "oracle" baseline experiments.py compares
        against.
    """
    noise_params = noise_params or {}
    v_noise = noise_params.get("v_noise", 0.05)
    omega_noise = noise_params.get("omega_noise", 0.02)
    position_noise = noise_params.get("position_noise", 0.5)

    robot = TrueRobot(x0=0, y0=0, theta0=0,
                       actuation_noise_std=(v_noise, omega_noise))
    ekf = EKF(initial_state=np.zeros(3), initial_covariance=np.eye(3) * 0.1,
              process_noise=np.diag([0.01, 0.01, 0.005]),
              measurement_noise=np.eye(2) * (position_noise ** 2))
    pid = PIDController(kp=1.5, ki=0.0, kd=0.3, dt=dt)

    history = {"true": [], "estimate": [], "waypoint_idx": []}
    waypoint_idx = 0

    for step in range(n_steps):
        if waypoint_idx >= len(waypoints):
            break

        control_state = ekf.estimate() if use_estimate else robot.true_state()
        v_cmd, omega_cmd, reached = waypoint_controller(
            control_state, waypoints[waypoint_idx], pid)
        if reached:
            waypoint_idx += 1
            continue

        robot.step(v_cmd, omega_cmd, dt)
        true_x, true_y, true_theta = robot.true_state()

        # Odometry reflects the robot's ACTUAL v/omega this step (post-actuation
        # -noise), not the commanded values — see TrueRobot.last_actual_velocity().
        actual_v, actual_omega = robot.last_actual_velocity()
        v_meas, omega_meas = noisy_odometry(actual_v, actual_omega,
                                             v_noise_std=v_noise,
                                             omega_noise_std=omega_noise)
        ekf.predict(v_meas, omega_meas, dt)

        if step % gps_update_interval == 0:
            position_meas = noisy_position_fix(true_x, true_y, noise_std=position_noise)
            ekf.update(position_meas)

        history["true"].append([true_x, true_y, true_theta])
        history["estimate"].append(ekf.estimate())
        history["waypoint_idx"].append(waypoint_idx)

    return history
