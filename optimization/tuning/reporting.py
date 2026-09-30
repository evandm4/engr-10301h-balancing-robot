"""Saving and presenting tuning results, shared by run_tuning.py and run_batch_tuning.py."""

import csv
import json
from pathlib import Path
from typing import Dict, Sequence

from .optimizers import TuningResult


def save_result(result: TuningResult, folder, extra: dict = None) -> Path:
    """Write <method>.json (summary) and <method>_history.csv into `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    summary = {
        "method": result.method,
        "gains": result.gains_dict(),
        "best_training_cost": result.best_cost,
        "n_evals": result.n_evals,
        "elapsed_s": round(result.elapsed_s, 2),
    }
    if extra:
        summary.update(extra)
    json_path = folder / f"{result.method}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    with open(folder / f"{result.method}_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["evals", "best_cost"])
        writer.writerows(result.history)
    return json_path


def save_example_runs(folder, runs: Dict[str, object], gains: Dict[str, Sequence[float]], **scenario) -> Path:
    """Save each example run as <folder>/trajectories/<label>.csv, for simulation/animate.py to replay.

    runs: label -> SimResult; gains: label -> the gains that produced it;
    scenario: e.g. start_pitch and seed, recorded in each file's metadata.
    """
    from balance_sim import save_trajectory

    out = Path(folder) / "trajectories"
    for label, run in runs.items():
        save_trajectory(run, out / f"{label.replace(' ', '_')}.csv", label=label,
                        gains=[float(g) for g in gains[label]], **scenario)
    return out


# ---- results table ----------------------------------------------------------

GAIN_COLUMN_WIDTH = 8


def gains_header(names: Sequence[str]) -> str:
    """Column headings for a row of gains, matching gains_cells."""
    return "".join(f"{n:>{GAIN_COLUMN_WIDTH}}" for n in names)


def gains_cells(gains: Sequence[float]) -> str:
    """One row of gains, 3 decimals each, for however many gains there are."""
    return "".join(f"{float(g):>{GAIN_COLUMN_WIDTH}.3f}" for g in gains)


# ---- summary figure -----------------------------------------------------------

def tuning_summary_figure(histories: Dict[str, Sequence], baseline_cost: float, example_runs: Dict[str, object],
                          objective_label: str = "best training cost so far", log_evaluations: bool = False,
                          example_title: str = "Tracking on an unseen noise seed"):
    """The 2 x 2 tuning summary: convergence, then velocity, pitch and torque of example runs.

    histories: method name -> [(evaluations so far, best cost so far), ...]
    baseline_cost: drawn as a dashed line (the hand-tuned gains' score)
    example_runs: label -> SimResult, typically hand-tuned vs the best gains on one scenario
    log_evaluations: log x-axis on the convergence plot, for budgets in the tens of thousands and up
    Returns the matplotlib figure; the caller saves or shows it.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax = axes[0, 0]
    for name, history in histories.items():
        ax.plot(*zip(*history), label=name)
    ax.axhline(baseline_cost, color="gray", ls="--", label="hand-tuned")
    ax.set_xlabel("evaluations")
    ax.set_ylabel(objective_label)
    ax.set_yscale("log")
    if log_evaluations:
        ax.set_xscale("log")
    ax.set_title("Convergence")
    ax.legend()

    for name, r in example_runs.items():
        axes[0, 1].plot(r.t, r.states[:, 1], label=name)
        axes[1, 0].plot(r.t, np.degrees(r.states[:, 2]), label=name)
        axes[1, 1].plot(r.t, r.torques, label=name)
    first = next(iter(example_runs.values()))
    axes[0, 1].plot(first.t, first.velocity_goal, "k--", label="goal")
    axes[0, 1].set_ylabel("velocity (m/s)")
    axes[0, 1].set_title(example_title)
    axes[0, 1].legend()
    axes[1, 0].set_ylabel("pitch (deg)")
    axes[1, 1].set_ylabel("torque (N m)")
    for a in (axes[0, 1], axes[1, 0], axes[1, 1]):
        a.set_xlabel("time (s)")
    fig.tight_layout()
    return fig
