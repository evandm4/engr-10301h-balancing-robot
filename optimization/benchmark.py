"""Time the simulation and the tuner's objective, reliably enough to compare versions.

The elapsed times saved with tuning runs are one-shot wall-clock numbers, and
they swing a lot (the same 840-evaluation random search has taken anywhere
from 68 s to 185 s) because of background load, CPU boost clocks, and OneDrive
syncing the results folder. This script repeats each measurement, reports the
median and spread, and saves the numbers with the machine and library
versions, so it can serve as the baseline for the GPU version.

For the steadiest numbers: plug in the laptop, close other programs, and pause
OneDrive syncing while it runs.

From the optimization/ folder:
    python benchmark.py                 default: 7 repeats
    python benchmark.py --repeats 15
    python benchmark.py --no-save
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "simulation"))
sys.path.insert(0, str(HERE.parent))

from config import CONFIG
from tuning import Evaluator


def measure(fn, repeats: int):
    fn()   # warm-up (imports, caches)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "repeats": repeats,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    evaluator = Evaluator(CONFIG.build_eval_config(), CONFIG.build_search_space())
    gains = CONFIG.controller.hand_tuned_gains
    start_pitch, seed = evaluator.scenarios()[0]
    steps_per_run = int(round(CONFIG.scenario.duration_s / CONFIG.timing.dt_s))
    runs_per_eval = len(evaluator.scenarios())

    one_run = measure(lambda: evaluator.run(gains, start_pitch, seed), args.repeats)
    one_eval = measure(lambda: evaluator.evaluate_gains(gains), args.repeats)

    sims_per_s = 1.0 / one_run["median_s"]
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "workload": {
            "physics_steps_per_run": steps_per_run,
            "runs_per_evaluation": runs_per_eval,
            "dt_s": CONFIG.timing.dt_s,
            "duration_s": CONFIG.scenario.duration_s,
        },
        "single_run": one_run,
        "single_evaluation": one_eval,
        "derived": {
            "runs_per_second_single_core": sims_per_s,
            "physics_steps_per_second_single_core": sims_per_s * steps_per_run,
            "estimated_840_eval_search_single_core_s": 840 * one_eval["median_s"],
        },
    }

    print(f"One {CONFIG.scenario.duration_s:g} s run ({steps_per_run} physics steps): "
          f"median {one_run['median_s'] * 1e3:.1f} ms  (min {one_run['min_s'] * 1e3:.1f}, "
          f"max {one_run['max_s'] * 1e3:.1f})")
    print(f"One evaluation ({runs_per_eval} runs): median {one_eval['median_s'] * 1e3:.0f} ms  "
          f"(min {one_eval['min_s'] * 1e3:.0f}, max {one_eval['max_s'] * 1e3:.0f})")
    print(f"Throughput: {sims_per_s:.1f} runs/s, {sims_per_s * steps_per_run:,.0f} physics steps/s (one core)")
    print(f"An 840-evaluation search on one core would take about "
          f"{report['derived']['estimated_840_eval_search_single_core_s'] / 60:.1f} min")
    if "onedrive" in str(HERE).lower():
        print("Note: this folder is inside OneDrive; pause syncing for the steadiest timings.")

    if not args.no_save:
        folder = HERE.parent / "results" / "benchmarks"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"cpu_{time.strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps(report, indent=2))
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()
