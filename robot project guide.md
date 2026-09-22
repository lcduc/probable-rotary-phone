# Project: Closed-Loop Simulated Robot — Noisy Localization + Waypoint Following
**Repo:** `probable-rotary-phone` · **Budget:** 8 weeks, 8–10 hrs/week (core: ~40–55 hrs, optional stretch: +15–20 hrs) · **Stack:** Python, NumPy, Matplotlib, pytest

## Research statement (for your CV and the letter's self-study paragraph)

> A simulated differential-drive robot navigates toward a sequence of waypoints using only noisy sensors — it never has access to its true position. An Extended Kalman Filter fuses noisy odometry (dead-reckoning) with an intermittent noisy absolute position fix into a best estimate, and a PID controller steers the robot using that estimate, not ground truth. The project closes the full sense → estimate → control → act loop, and quantifies how much localization uncertainty degrades navigation performance compared to an oracle baseline with perfect position knowledge.

This replaces the placeholder in your letter's self-study paragraph with something specific and true.

---

## Why this design

Everything here is fully synthetic and under your control — no real-world data formats to fight, no silent coordinate-frame bugs from a dataset you didn't create. But unlike five disconnected exercises, this is **one integrated loop** where each math-phase skill does real work and depends on the others: transforms represent pose, the Kalman filter estimates it, the PID controller acts on that estimate, and bad localization visibly produces bad driving. That last part is what makes it feel like a robot rather than an offline script.

---

## Architecture

```
┌─────────────┐      commands (v, ω)       ┌──────────────┐
│  Controller │ ────────────────────────►  │  True Robot  │
│   (PID)     │                            │  Simulation  │
└──────▲──────┘                            └──────┬───────┘
       │                                          │ true state
       │ state estimate                           ▼
┌──────┴──────┐    noisy odometry    ┌──────────────────────┐
│     EKF     │ ◄──────────────────  │   Sensor Models      │
│  (predict + │                      │ (odometry + fix)     │
│   update)   │ ◄──────── noisy position fix (every N steps)│
└─────────────┘                      └──────────────────────┘
```

The controller only ever sees the EKF's estimate — never the true state. That's the whole point.

---

## Repo structure

```
probable-rotary-phone/
├── README.md
├── core/
│   ├── robot_sim.py        # true robot dynamics (ground truth)
│   ├── sensors.py          # noisy odometry + noisy position-fix models
│   ├── ekf.py              # state estimator
│   ├── controller.py       # PID waypoint controller
│   ├── simulate.py         # main loop tying everything together
│   ├── metrics.py          # path length, deviation, timing — real recorded results
│   ├── visualize.py        # plotting (mostly provided for you below)
│   └── experiments.py      # noise-level sweep: oracle vs. estimate, writes data/results.csv
├── data/
│   └── results.csv         # actual numbers from running experiments.py — cite these, nothing else
├── plots/                  # generated trajectory plot images
└── tests/
    ├── test_robot_sim.py
    ├── test_ekf.py
    └── test_controller.py
```

---

## Component 1 — True robot simulation (ground truth)

Standard differential-drive kinematics. This is the "real world" the robot lives in — it will add small random disturbances so commanded and actual motion aren't identical, which is realistic and gives the filter something meaningful to correct for.

```python
# core/robot_sim.py
import numpy as np

class TrueRobot:
    def __init__(self, x0: float, y0: float, theta0: float,
                 actuation_noise_std: tuple[float, float] = (0.02, 0.01)):
        """actuation_noise_std: (v_noise, omega_noise) — how much the robot's
        actual motion deviates from the commanded (v, omega) each step."""
        raise NotImplementedError

    def step(self, v_cmd: float, omega_cmd: float, dt: float) -> None:
        """Advance the TRUE state by one time step, given commanded (v, omega).
        Add actuation noise before integrating — the robot doesn't perfectly
        execute what it's told."""
        raise NotImplementedError

    def true_state(self) -> np.ndarray:
        """Return [x, y, theta] — for logging/plotting only. The controller
        and filter must never call this directly."""
        raise NotImplementedError
```

---

## Component 2 — Sensor models

Two sensors, both synthetic and fully under your control:

```python
# core/sensors.py
import numpy as np

def noisy_odometry(true_v: float, true_omega: float,
                    v_noise_std: float = 0.05, omega_noise_std: float = 0.02) -> tuple[float, float]:
    """Simulate a wheel-encoder-style reading of the robot's actual velocity —
    this is what the EKF's predict step will use."""
    raise NotImplementedError

def noisy_position_fix(true_x: float, true_y: float,
                        noise_std: float = 0.5) -> np.ndarray:
    """Simulate an intermittent absolute position sensor (a stand-in for GPS
    or a landmark detector). Only called every N simulation steps, not every step."""
    raise NotImplementedError
```

---

## Component 3 — EKF state estimator

This is the same structure as a full EKF localization filter, just on synthetic sensors instead of real data — reuse the motion-model-plus-Jacobian pattern from your earlier Jacobian work.

```python
# core/ekf.py
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
```

**Test to write:** with no position updates at all (odometry only), the estimate should drift away from ground truth over time — verifying dead-reckoning behaves as expected before you trust the update step to correct it.

---

## Component 4 — PID waypoint controller

The controller's only input is the EKF's estimate. Steer heading toward the next waypoint; drive forward at a roughly constant speed, slowing as it approaches.

```python
# core/controller.py
import numpy as np

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float):
        raise NotImplementedError

    def compute(self, error: float) -> float:
        """Standard PID output for a single error signal (here, heading error)."""
        raise NotImplementedError

def heading_error(current_theta: float, target_x: float, target_y: float,
                   current_x: float, current_y: float) -> float:
    """Angle (radians, normalized to [-pi, pi]) between current heading and
    the direction toward the target waypoint."""
    raise NotImplementedError

def waypoint_controller(estimate: np.ndarray, waypoint: np.ndarray,
                         pid: PIDController, cruise_speed: float = 1.0,
                         arrival_radius: float = 0.3) -> tuple[float, float, bool]:
    """Return (v_cmd, omega_cmd, reached_waypoint). Uses the ESTIMATE, never
    the true state."""
    raise NotImplementedError
```

---

## Component 5 — Main simulation loop

This wiring is mostly mechanical — provided here so your time goes into Components 1–4, not glue code.

```python
# core/simulate.py
import numpy as np

def run_simulation(waypoints: list[np.ndarray], n_steps: int, dt: float,
                    gps_update_interval: int = 10):
    """Ties everything together. Logs true state, estimated state, and
    commands at every step for later plotting."""
    robot = TrueRobot(x0=0, y0=0, theta0=0)
    ekf = EKF(initial_state=np.zeros(3), initial_covariance=np.eye(3) * 0.1,
              process_noise=np.diag([0.01, 0.01, 0.005]),
              measurement_noise=np.eye(2) * 0.25)
    pid = PIDController(kp=1.5, ki=0.0, kd=0.3, dt=dt)

    history = {"true": [], "estimate": [], "waypoint_idx": []}
    waypoint_idx = 0

    for step in range(n_steps):
        if waypoint_idx >= len(waypoints):
            break
        v_cmd, omega_cmd, reached = waypoint_controller(
            ekf.estimate(), waypoints[waypoint_idx], pid)
        if reached:
            waypoint_idx += 1
            continue

        robot.step(v_cmd, omega_cmd, dt)
        true_x, true_y, true_theta = robot.true_state()

        # NOTE: odometry should reflect the robot's ACTUAL v/omega this step,
        # which you'll need to expose from TrueRobot — a small but important
        # design detail to get right.
        v_meas, omega_meas = noisy_odometry(v_cmd, omega_cmd)
        ekf.predict(v_meas, omega_meas, dt)

        if step % gps_update_interval == 0:
            position_meas = noisy_position_fix(true_x, true_y)
            ekf.update(position_meas)

        history["true"].append([true_x, true_y, true_theta])
        history["estimate"].append(ekf.estimate())
        history["waypoint_idx"].append(waypoint_idx)

    return history
```

---

## Visualization (provided — spend your time on the algorithms, not matplotlib)

```python
# core/visualize.py
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
    plt.savefig(f"plots/{title.replace(' ', '_').lower()}.png", dpi=150)
    plt.show()
```

---

## Core experiment: how much does noisy localization actually cost?

**Don't just eyeball the plots — compute and save real numbers.** Add a metrics module:

```python
# core/metrics.py
import numpy as np
import csv

def path_length(path: np.ndarray) -> float:
    """Total distance traveled along a [N,2] or [N,3] path."""
    diffs = np.diff(path[:, :2], axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))

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
    raise NotImplementedError
```

**Run the comparison across at least three noise levels — low, medium, high — not just once.** This is the actual experiment, not a single run:

```python
# core/experiments.py
NOISE_LEVELS = {
    "low":    {"v_noise": 0.02, "omega_noise": 0.01, "position_noise": 0.2},
    "medium": {"v_noise": 0.05, "omega_noise": 0.02, "position_noise": 0.5},
    "high":   {"v_noise": 0.10, "omega_noise": 0.05, "position_noise": 1.5},
}

def run_all_experiments(waypoints, n_steps, dt):
    for level_name, params in NOISE_LEVELS.items():
        for mode in ["oracle", "estimate"]:
            history = run_simulation(waypoints, n_steps, dt,
                                      noise_params=params, use_estimate=(mode == "estimate"))
            true_path = np.array(history["true"])
            length = path_length(true_path)
            deviation = deviation_from_ideal(true_path, np.array(waypoints))
            time_taken = completion_time(history, dt)
            record_run(label=mode, noise_level=level_name,
                       path_len=length, deviation=deviation, time_taken=time_taken)
```

This produces one `data/results.csv` with six rows (3 noise levels × 2 modes: oracle vs. estimate) — real, saved numbers you can drop straight into a table in your report, and defend honestly if anyone asks how you measured them. **Do not put any number in your CV or letter until it comes out of this file.**

---

## Optional stretch goal — lightweight confidence weighting

If the core loop is solid with time to spare, extend `EKF.update()` to accept a `reliability` parameter, and simulate a period where the position fix becomes unreliable (larger noise, or occasional big outliers). Compare a fixed-noise EKF against one that inflates its measurement noise when the innovation is unexpectedly large (same idea as the full R&D proposal, at a much smaller scale). Mention in your letter that this is exactly what you plan to develop further during the R&D project itself — a natural, honest bridge from what you've built to what you're proposing.

---

## Suggested 8-week schedule

| Weeks | Focus |
|---|---|
| 1 | `TrueRobot` + `sensors.py`, tested in isolation (no filter yet — just verify the robot moves correctly and sensors add noise as expected) |
| 2–3 | `EKF` — predict step first (verify dead-reckoning drift), then the update step (verify it corrects the drift) |
| 4 | `PIDController` + `waypoint_controller`, tested by driving toward a single fixed waypoint using ground truth (skip the EKF for now, to isolate controller bugs from filter bugs) |
| 5 | Wire everything together in `simulate.py`; get the full loop running end-to-end on a short multi-waypoint route |
| 6 | Run the core experiment (realistic vs. oracle), generate comparison plots and numbers |
| 7 | Optional stretch goal, if time allows |
| 8 | Report, README, final polish and commits |

---

## Tests worth writing

- `test_robot_sim.py`: with zero actuation noise, the robot's motion exactly matches the ideal kinematic equations.
- `test_ekf.py`: odometry-only (no updates) causes estimate drift over time; adding periodic position updates measurably reduces final position error compared to odometry-only.
- `test_controller.py`: heading error is correctly normalized to `[-π, π]`; the controller drives heading error toward zero over repeated calls with a stationary target.