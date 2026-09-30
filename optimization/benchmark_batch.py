"""Throughput of the batched simulator at several batch sizes, against the reference.

Runs the real tuning scenario from config.py and reports time per simulation
at each batch size, so you can see where batching pays off. Compares against
the latest one-at-a-time baseline from benchmark.py (results/benchmarks/cpu_*.json)
and saves its own numbers, with machine and library versions, next to it.

From the project folder:
    python optimization/benchmark_batch.py                    NumPy on the CPU
    python optimization/benchmark_batch.py --backend torch    PyTorch on the GPU
    python optimization/benchmark_batch.py --sizes 120 1200 12000 --repeats 5
    python optimization/benchmark_batch.py --float32          also time float32
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # config.py; importing it puts simulation/ and optimization/ on the path

import numpy as np

from config import CONFIG
from batch_tuning import BatchEvaluator, decode_units


def machine_info(backend: str) -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "numpy": np.__version__}
    if backend == "torch":
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="simulations per batch (rounded up to a multiple of the 9 scenarios)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--float32", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    if args.sizes is None:
        args.sizes = ([9, 120, 1200, 12000, 48000] if args.backend == "numpy"
                      else [9, 1200, 12000, 120000, 400000])

    space = CONFIG.build_search_space()
    config = CONFIG.build_eval_config()
    n_scen = len(config.start_pitches) * len(config.train_seeds)
    # Near-hand-tuned gains, so the robots stay up and every run goes the full length.
    rng = np.random.default_rng(0)
    base = space.encode(CONFIG.controller.hand_tuned_gains)

    cpu_baseline = None
    folder = ROOT / "results" / "benchmarks"
    if folder.exists():
        files = sorted(folder.glob("cpu_*.json"))
        if files:
            cpu_baseline = json.loads(files[-1].read_text())
    cpu_ms = cpu_baseline["single_run"]["median_s"] * 1e3 if cpu_baseline else None

    dtypes = [np.float64] + ([np.float32] if args.float32 else [])
    rows = []
    print(f"{'dtype':<9}{'sims':>8}{'median s':>11}{'ms / sim':>11}{'sims / s':>11}"
          + (f"{'vs CPU':>9}" if cpu_ms else ""))
    for dtype in dtypes:
        for size in args.sizes:
            n_candidates = max(1, -(-size // n_scen))
            units = np.clip(base + rng.normal(0, 0.02, (n_candidates, space.dim)), 0, 1)
            gains = decode_units(space, units)
            evaluator = BatchEvaluator(config, space, dtype=dtype, max_batch=n_candidates * n_scen,
                                       backend=args.backend)
            evaluator.evaluate_population(gains)   # warm-up (and, for torch, graph capture)
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                evaluator.evaluate_population(gains)
                times.append(time.perf_counter() - start)
            sims = n_candidates * n_scen
            med = statistics.median(times)
            row = {"dtype": np.dtype(dtype).name, "sims": sims, "median_s": med, "min_s": min(times),
                   "max_s": max(times), "ms_per_sim": med / sims * 1e3, "sims_per_s": sims / med}
            rows.append(row)
            print(f"{row['dtype']:<9}{sims:>8}{med:>11.3f}{row['ms_per_sim']:>11.4f}{row['sims_per_s']:>11,.0f}"
                  + (f"{cpu_ms / row['ms_per_sim']:>8.3g}x" if cpu_ms else ""))

    if cpu_ms:
        print(f"\nCPU baseline ({cpu_baseline['timestamp']}): {cpu_ms:.1f} ms per simulation, one at a time.")

    if not args.no_save:
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": args.backend,
            "machine": machine_info(args.backend),
            "workload": {"physics_steps_per_run": int(round(config.duration / config.dt)),
                         "dt_s": config.dt, "duration_s": config.duration},
            "cpu_baseline_ms_per_sim": cpu_ms,
            "results": rows,
        }
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{args.backend}_{time.strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps(report, indent=2))
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()
