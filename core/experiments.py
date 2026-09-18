"""Noise-level sweep: oracle vs. estimate, across low/medium/high noise.
Writes results.csv — the only source for numbers cited in the CV or letter."""
import numpy as np

from .simulate import run_simulation
from .metrics import path_length, deviation_from_ideal, completion_time, record_run

NOISE_LEVELS = {
    "low":    {"v_noise": 0.02, "omega_noise": 0.01, "position_noise": 0.2},
    "medium": {"v_noise": 0.05, "omega_noise": 0.02, "position_noise": 0.5},
    "high":   {"v_noise": 0.10, "omega_noise": 0.05, "position_noise": 1.5},
}


def run_all_experiments(waypoints, n_steps, dt, csv_path: str = "data/results.csv"):
    for level_name, params in NOISE_LEVELS.items():
        for mode in ["oracle", "estimate"]:
            history = run_simulation(waypoints, n_steps, dt,
                                      noise_params=params,
                                      use_estimate=(mode == "estimate"))
            true_path = np.array(history["true"])
            length = path_length(true_path)
            deviation = deviation_from_ideal(true_path, np.array(waypoints))
            time_taken = completion_time(history, dt)
            record_run(label=mode, noise_level=level_name,
                       path_len=length, deviation=deviation, time_taken=time_taken,
                       csv_path=csv_path)


if __name__ == "__main__":
    waypoints = [np.array([5, 0]), np.array([5, 5]), np.array([0, 5]), np.array([0, 0])]
    run_all_experiments(waypoints, n_steps=2000, dt=0.1)
