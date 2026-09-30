"""Saving tuning results so runs can be compared and cited later."""

import csv
import json
from pathlib import Path

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
