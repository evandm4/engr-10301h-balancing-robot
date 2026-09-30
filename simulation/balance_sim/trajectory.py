"""Saving a run's trajectory to CSV and reading it back, for replaying or analysis later.

File layout: one comment line of JSON metadata (label, gains, whether it fell,
the scenario, ...), then an ordinary CSV table with a header row. The numbers
are written with 17 significant digits, so a loaded run is exactly the run
that was saved.
"""

import json
from pathlib import Path
from typing import Tuple

import numpy as np

from .simulator import SimResult

COLUMNS = ("t", "x", "x_dot", "theta", "theta_dot", "torque", "pitch_reference", "velocity_goal")


def save_trajectory(result: SimResult, path, **metadata) -> Path:
    """Write a SimResult to `path` as CSV. Keyword arguments go in the metadata line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = result.pitch_reference if result.pitch_reference is not None else np.zeros(len(result.t))
    data = np.column_stack([result.t, result.states[:, :4], result.torques, reference, result.velocity_goal])
    meta = {"fell": bool(result.fell), "fall_time": result.fall_time, "duration": result.duration, **metadata}
    with open(path, "w", newline="") as f:
        f.write("# " + json.dumps(meta, default=float) + "\n")
        f.write(",".join(COLUMNS) + "\n")
        np.savetxt(f, data, delimiter=",", fmt="%.17g")
    return path


def load_trajectory(path) -> Tuple[SimResult, dict]:
    """Read a file written by save_trajectory: returns (SimResult, metadata)."""
    path = Path(path)
    with open(path) as f:
        first = f.readline()
    meta = json.loads(first[1:]) if first.startswith("#") else {}
    data = np.loadtxt(path, delimiter=",", skiprows=2 if meta else 1, ndmin=2)
    t = data[:, 0]
    result = SimResult(
        t=t,
        states=data[:, 1:5],
        torques=data[:, 5],
        velocity_goal=data[:, 7],
        fell=bool(meta.get("fell", False)),
        fall_time=meta.get("fall_time"),
        duration=float(meta.get("duration", t[-1])),
        pitch_reference=data[:, 6],
    )
    return result, meta
