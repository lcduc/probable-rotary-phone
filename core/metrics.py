"""Path length, deviation, timing — real recorded results, not eyeballed plots."""
import csv
import os

import numpy as np


def path_length(path: np.ndarray) -> float:
    """Total distance traveled along a [N,2] or [N,3] path."""
    diffs = np.diff(path[:, :2], axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def deviation_from_ideal(path: np.ndarray, waypoints: np.ndarray) -> float:
    """Average perpendicular distance from the path to the straight-line
    route connecting consecutive waypoints. Measures 'wobble'."""
    raise NotImplementedError  # for each path point, find distance to the
    # nearest segment of the waypoint-to-waypoint straight line, then average


def completion_time(history: dict, dt: float) -> float:
    """Number of simulation steps taken to reach the final waypoint, times dt."""
    return len(history["true"]) * dt


def record_run(label: str, noise_level: str, path_len: float, deviation: float,
                time_taken: float, csv_path: str = "data/results.csv") -> None:
    """Append one experiment's results to a CSV so nothing gets lost between
    runs. Write the header once if the file doesn't exist yet."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["label", "noise_level", "path_len", "deviation", "time_taken"])
        writer.writerow([label, noise_level, path_len, deviation, time_taken])
