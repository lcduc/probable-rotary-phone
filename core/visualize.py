"""Plotting. Spend your time on the algorithms, not matplotlib."""
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(history: dict, waypoints: list[np.ndarray], title: str = "Trajectory"):
    true_path = np.array(history["true"])
    est_path = np.array(history["estimate"])
    wp = np.array(waypoints)

    plt.figure(figsize=(8, 8))
    plt.plot(true_path[:, 0], true_path[:, 1], label="True path", linewidth=2)
    plt.plot(est_path[:, 0], est_path[:, 1], label="Estimated path", linestyle="--")
    plt.scatter(wp[:, 0], wp[:, 1], color="red", marker="x", s=100, label="Waypoints")
    plt.legend()
    plt.axis("equal")
    plt.title(title)

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{title.replace(' ', '_').lower()}.png", dpi=150)
    plt.show()
