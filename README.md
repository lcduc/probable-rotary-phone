# Closed-Loop Simulated Robot — Noisy Localization + Waypoint Following

A simulated differential-drive robot navigates toward a sequence of
waypoints using only noisy sensors — it never has access to its true
position. An Extended Kalman Filter fuses noisy odometry with an
intermittent noisy absolute position fix into a best estimate, and a PID
controller steers the robot using that estimate, not ground truth.

See [`robot project guide.md`](robot%20project%20guide.md) for the full
design rationale, architecture, and week-by-week plan.

## Status

`core/robot_sim.py`, `core/sensors.py`, and `core/ekf.py` are implemented
and tested. `core/controller.py` and `core/metrics.deviation_from_ideal`
still raise `NotImplementedError` and are the remaining project work (see
the guide's 8-week schedule).

## Layout

- `core/robot_sim.py` — true robot dynamics (ground truth), never queried by
  the controller or filter
- `core/sensors.py` — noisy odometry + noisy position-fix models
- `core/ekf.py` — EKF state estimator (predict + update)
- `core/controller.py` — PID waypoint controller, steers off the estimate only
- `core/simulate.py` — main loop wiring the above together (implemented)
- `core/metrics.py` — path length, deviation, timing, CSV logging
- `core/visualize.py` — trajectory plotting (implemented)
- `core/experiments.py` — noise-level sweep (low/medium/high × oracle/estimate),
  writes `data/results.csv`
- `plots/` — generated trajectory plot images
- `data/results.csv` — generated experiment numbers (produced by `experiments.py`,
  cite these and nothing else)
- `tests/` — one test file per core component

## Setup

```bash
pip install -r requirements.txt
```

## Running tests

From the repo root:

```bash
pytest
```

Tests will fail until the corresponding `NotImplementedError` stubs are
filled in — that's expected at this stage.

## Running the experiment sweep

Once `core/robot_sim.py`, `core/sensors.py`, `core/ekf.py`,
`core/controller.py`, and `core/metrics.deviation_from_ideal` are
implemented, from the repo root:

```bash
python -m core.experiments
```

(Run as a module, not a script directly, so `core`'s internal relative
imports resolve.) This writes `data/results.csv` (6 rows: 3 noise levels ×
oracle/estimate) — the only source for any numbers cited in a CV or letter,
per the guide.
