"""Compare saved controllers on one evaluation tier, with the statistics of docs/ml-baselines-plan.md 3.4.

Controllers come from three kinds of file:
    the existing comparison (results/lqr_comparison/<run>/comparison.json): its 7 PID and LQR
        controllers, one run each (--existing; default the reference run in config.py);
    learned policies (.npz from run_neuroevolution.py), grouped into methods:
        --policies "A=results/neuroevolution/mlp/*/final.npz"   (one method, one run per file);
    re-runs of a gain search (.json with "gains", and "kind": "lqr" for LQR gains), the same way:
        --gain-runs "PID re-runs=results/pid_reruns/*/pid.json".

Every controller is scored on the same scenarios (the tier's seeds x 3 start tilts, randomized
robots and latencies, the chaos-penalty twins), in float64 with NumPy by default, on the 60 s
validation scenario. Reported, against a reference controller (default the ML-tuned PID):
    per run: mean cost, objective, falls with a Clopper-Pearson interval, the median of the
        per-scenario cost ratio, and an exact McNemar test of its falls against the reference's
        on the same scenarios;
    per method: the headline, the median over runs of each run's median ratio; its two-level
        bootstrap interval with scenarios crossed with runs (one scenario resample per replicate,
        shared by every run); the distribution-free order-statistic interval for the median over
        runs; fall-rate and McNemar summaries (median and range over runs); every run's numbers;
    per scenario (per_scenario.csv): cost, fell and fall time for every controller, so each fall
        can be traced to the start tilt, a push, braking or standstill drift.
With --latency-sweep, each controller also runs at every latency of a 2.5 ms grid from 15 to
45 ms (no chaos twins, every scenario at that one latency), and reports the first latency with
4 or more falls (pre-registered; plan 3.4) and the full fall-count curve.

The validation and fall-rate tiers are frozen: used once, on final controllers. The script
refuses them without --final, and nobody inspects per-scenario results there until every
controller in the comparison is frozen. Use --tier dev (or smoke) while developing.

From the project folder:
    python optimization/compare_controllers.py --tier dev --policies "L=results/neuroevolution/linear/*/final.npz"
    python optimization/compare_controllers.py --tier fall_rate --final --latency-sweep --policies ...

Results: results/comparison/<timestamp>-<tier>/ with comparison.json, per_scenario.csv and,
with the sweep, latency_sweep.csv.
"""

import argparse
import csv
import glob
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))   # config.py; importing it puts simulation/ and optimization/ on the path

import numpy as np
from scipy import stats

from config import CONFIG
from balance_sim import load_policy
from batch_tuning import BatchEvaluator
from tuning import provenance

TIERS = ("smoke", "dev", "validation", "fall_rate")
FROZEN = ("validation", "fall_rate")
SWEEP_LATENCIES_MS = tuple(15.0 + 2.5 * i for i in range(13))    # 15, 17.5, ..., 45 ms
SWEEP_FALL_RATE_SEEDS = 333          # 6000-6332 x 3 tilts = 999 scenarios
SWEEP_FALL_THRESHOLD = 4             # Clopper-Pearson upper bound passes 1% at about 4 / 1,000


# ---- statistics -----------------------------------------------------------------------------

def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact two-sided interval for a binomial proportion k / n."""
    low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return low, high


def mcnemar_exact(fell_a, fell_b) -> dict:
    """Exact McNemar test on paired fall flags: b = only A fell, c = only B fell; two-sided p."""
    fell_a, fell_b = np.asarray(fell_a, bool), np.asarray(fell_b, bool)
    b, c = int(np.sum(fell_a & ~fell_b)), int(np.sum(~fell_a & fell_b))
    p = 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)
    return {"only_this_fell": b, "only_reference_fell": c, "p_value": p}


def order_statistic_interval(values, k: Optional[int] = None) -> dict:
    """Distribution-free interval for the median: the k-th smallest to the k-th largest.

    Coverage is 1 - 2 P(Binomial(n, 1/2) <= k - 1). k defaults to the largest k with coverage at
    least 95% (for n = 10, k = 2: the 2nd to 9th values, 97.9%). None for n < 6, where even
    (min, max) covers less than 95%.
    """
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    coverage = lambda k: 1.0 - 2.0 * float(stats.binom.cdf(k - 1, n, 0.5))
    if k is None:
        ks = [k for k in range(1, n // 2 + 1) if coverage(k) >= 0.95]
        if not ks:
            return {"low": None, "high": None, "coverage": None, "k": None}
        k = max(ks)
    return {"low": float(v[k - 1]), "high": float(v[n - k]), "coverage": coverage(k), "k": k}


def crossed_bootstrap(ratios, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Two-level bootstrap of median over runs (of the median over scenarios) of ratios (R, N).

    Scenarios are crossed with runs: every run was scored on the same scenarios, so each replicate
    draws run indices and ONE scenario index vector shared by all runs. Resampling scenarios
    separately per run would average away the shared scenario-difficulty component and give
    intervals that are too narrow (plan 3.4).
    """
    ratios = np.asarray(ratios, dtype=float)
    R, N = ratios.shape
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        runs = rng.integers(0, R, R)
        scen = rng.integers(0, N, N)
        out[b] = np.median(np.median(ratios[runs][:, scen], axis=1))
    low, high = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return {"low": float(low), "high": float(high), "n_boot": n_boot}


def first_latency_with_falls(latencies_ms, falls, threshold: int = SWEEP_FALL_THRESHOLD) -> Optional[float]:
    """The pre-registered sweep metric: the first latency with `threshold` or more falls; None if none."""
    for ms, n in zip(latencies_ms, falls):
        if n >= threshold:
            return float(ms)
    return None


# ---- controllers ----------------------------------------------------------------------------

@dataclass
class Entry:
    method: str          # what the paper's table groups by
    run: str             # one trained (or designed) controller
    kind: str            # "cascaded", "lqr" or "policy"
    gains: np.ndarray    # PID / LQR gains, or policy parameters
    spec: object = None  # PolicySpec, for policies
    source: str = ""

    @property
    def label(self) -> str:
        return self.method if self.run == self.method else f"{self.method} / {self.run}"


def existing_controllers(path) -> List[Entry]:
    results = json.loads(Path(path).read_text())["results"]
    return [Entry(name, name, "lqr" if r["kind"] == "lqr" else "cascaded", np.asarray(r["gains"], float),
                  source=str(path)) for name, r in results.items()]


def _expand(pattern: str) -> List[Path]:
    if "=" not in pattern:
        raise SystemExit(f"expected NAME=GLOB, got {pattern!r}")
    return [Path(p) for p in sorted(glob.glob(pattern.split("=", 1)[1], recursive=True))]


def policy_runs(pattern: str) -> List[Entry]:
    method = pattern.split("=", 1)[0]
    entries = []
    for path in _expand(pattern):
        spec, params, _ = load_policy(path)
        entries.append(Entry(method, path.parent.name, "policy", params, spec, str(path)))
    if not entries:
        raise SystemExit(f"no files match {pattern!r}")
    return entries


def gain_runs(pattern: str) -> List[Entry]:
    method = pattern.split("=", 1)[0]
    entries = []
    for path in _expand(pattern):
        data = json.loads(path.read_text())
        entries.append(Entry(method, path.parent.name, data.get("kind", "cascaded"),
                             np.asarray(data["gains"], float), source=str(path)))
    if not entries:
        raise SystemExit(f"no files match {pattern!r}")
    return entries


# ---- scoring --------------------------------------------------------------------------------

def score(entries: Sequence[Entry], config, robust, seeds, backend="numpy", device="cuda",
          dtype=np.float64) -> Dict[str, dict]:
    """Every entry on the same scenarios. Returns label -> per-scenario arrays and summaries."""
    space = CONFIG.build_search_space()
    out = {}

    def keep(group, result):
        for i, e in enumerate(group):
            out[e.label] = {"costs": result.costs[i], "fell": result.fell[i], "fall_time": result.fall_time[i],
                            "mean_cost": float(result.mean_cost[i]), "objective": float(result.objective[i]),
                            "sensitivity": float(result.sensitivity[i]), "n_fell": int(result.n_fell[i])}

    common = dict(robustness=robust, backend=backend, device=device, dtype=dtype)
    cascaded = [e for e in entries if e.kind == "cascaded"]
    if cascaded:
        ev = BatchEvaluator(config, space, controller="cascaded", **common)
        keep(cascaded, ev.evaluate_population(np.array([e.gains for e in cascaded]), seeds))
    lqr = [e for e in entries if e.kind == "lqr"]
    if lqr:
        width = max(len(e.gains) for e in lqr)   # extra zero gains on commands in flight change nothing
        gains = np.array([np.concatenate([e.gains, np.zeros(width - len(e.gains))]) for e in lqr])
        keep(lqr, BatchEvaluator(config, space, controller="lqr", **common).evaluate_population(gains, seeds))
    by_spec: Dict[object, List[Entry]] = {}
    for e in entries:
        if e.kind == "policy":
            by_spec.setdefault(e.spec, []).append(e)
    for spec, group in by_spec.items():
        ev = BatchEvaluator(config, space, controller="policy", policy=spec, **common)
        keep(group, ev.evaluate_policies(np.array([e.gains for e in group]), seeds))
    return out


def summarize(entries: Sequence[Entry], scores: Dict[str, dict], reference: str, n_boot: int) -> dict:
    """Per-run and per-method statistics against the reference controller (plan 3.4)."""
    ref = scores[reference]
    runs = {}
    for e in entries:
        s = scores[e.label]
        n = len(s["costs"])
        low, high = clopper_pearson(s["n_fell"], n)
        ratio = s["costs"] / ref["costs"]
        runs[e.label] = {"method": e.method, "run": e.run, "kind": e.kind, "source": e.source,
                         "mean_cost": s["mean_cost"], "objective": s["objective"], "sensitivity": s["sensitivity"],
                         "worst_cost": float(np.max(s["costs"])), "falls": s["n_fell"], "scenarios": n,
                         "fall_rate": s["n_fell"] / n, "fall_rate_95ci": [low, high],
                         "median_ratio": float(np.median(ratio)),
                         "scenarios_better": int(np.sum(s["costs"] < ref["costs"])),
                         "mcnemar_vs_reference": mcnemar_exact(s["fell"], ref["fell"])}
    methods = {}
    for method in dict.fromkeys(e.method for e in entries):
        group = [e for e in entries if e.method == method]
        medians = [runs[e.label]["median_ratio"] for e in group]
        ratios = np.array([scores[e.label]["costs"] / ref["costs"] for e in group])
        falls = [runs[e.label]["falls"] for e in group]
        p_values = [runs[e.label]["mcnemar_vs_reference"]["p_value"] for e in group]
        methods[method] = {
            "runs": [e.run for e in group],
            "headline_median_ratio": float(np.median(medians)),
            "bootstrap_95ci": crossed_bootstrap(ratios, n_boot) if len(group) > 1 else None,
            "order_statistic_interval": order_statistic_interval(medians),
            "run_median_ratios": medians,
            "falls": {"median": float(np.median(falls)), "min": int(min(falls)), "max": int(max(falls))},
            "mcnemar_p": {"median": float(np.median(p_values)), "min": float(min(p_values)),
                          "max": float(max(p_values))},
        }
    return {"reference": reference, "runs": runs, "methods": methods}


def latency_sweep(entries: Sequence[Entry], config, robust, seeds, latencies_ms=SWEEP_LATENCIES_MS,
                  backend="numpy", device="cuda") -> Dict[str, dict]:
    """Falls and mean cost at each latency: every scenario at that one latency, no chaos twins."""
    sweep_robust = replace(robust, latency_range=None, sensitivity_weight=0.0)
    curves = {e.label: {"latency_ms": [], "falls": [], "mean_cost": [], "fall_rate_95ci": []} for e in entries}
    for ms in latencies_ms:
        scores = score(entries, replace(config, latency=ms / 1000), sweep_robust, seeds, backend, device)
        for e in entries:
            s, c = scores[e.label], curves[e.label]
            c["latency_ms"].append(ms)
            c["falls"].append(s["n_fell"])
            c["mean_cost"].append(s["mean_cost"])
            c["fall_rate_95ci"].append(list(clopper_pearson(s["n_fell"], len(s["costs"]))))
    for c in curves.values():
        c["first_latency_with_falls"] = first_latency_with_falls(c["latency_ms"], c["falls"])
        c["scenarios"] = len(seeds) * len(config.start_pitches)
    return curves


def sweep_by_method(entries, curves) -> dict:
    """The sweep metric per method: each run's, and their median (None = no latency reached it)."""
    out = {}
    for method in dict.fromkeys(e.method for e in entries):
        values = [curves[e.label]["first_latency_with_falls"] for e in entries if e.method == method]
        finite = [v if v is not None else math.inf for v in values]
        med = float(np.median(finite))
        out[method] = {"per_run": values, "median": None if math.isinf(med) else med}
    return out


# ---- main -----------------------------------------------------------------------------------

def tier_seeds(robust, tier: str) -> Tuple[int, ...]:
    return tuple({"smoke": robust.smoke_seeds, "dev": robust.dev_seeds, "validation": robust.validation_seeds,
                  "fall_rate": robust.fall_rate_seeds}[tier])


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=TIERS, default="dev")
    parser.add_argument("--final", action="store_true", help="allow the frozen validation and fall-rate tiers")
    parser.add_argument("--existing", default=str(ROOT / CONFIG.learning.reference_pid_file),
                        help="comparison.json with the existing controllers ('' for none)")
    parser.add_argument("--policies", action="append", default=[], help="NAME=GLOB of policy .npz files")
    parser.add_argument("--gain-runs", action="append", default=[], help="NAME=GLOB of gain .json files")
    parser.add_argument("--reference", default=CONFIG.learning.reference_pid_entry)
    parser.add_argument("--latency-sweep", action="store_true")
    parser.add_argument("--seeds", type=int, default=None, help="only the tier's first N seeds (development)")
    parser.add_argument("--duration", type=float, default=None, help="run length (default: 60 s validation scenario)")
    parser.add_argument("--latencies", type=str, default=None, help="sweep latencies in ms, e.g. 15,30,45")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy",
                        help="numpy float64 is the designated validation path; torch float64 for development")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.tier in FROZEN and not args.final:
        parser.error(f"the {args.tier} tier is frozen: pass --final once every controller is final")

    entries = existing_controllers(args.existing) if args.existing else []
    for pattern in args.policies:
        entries += policy_runs(pattern)
    for pattern in args.gain_runs:
        entries += gain_runs(pattern)
    labels = [e.label for e in entries]
    if args.reference not in labels:
        parser.error(f"reference {args.reference!r} is not among the controllers")
    if len(set(labels)) != len(labels):
        parser.error("two controllers share a label; give the methods different names")

    robust = CONFIG.build_robustness()
    seeds = tier_seeds(robust, args.tier)
    if args.seeds:
        seeds = seeds[:args.seeds]
    config = CONFIG.build_eval_config(validation=True)
    if args.duration:
        config = replace(config, duration=args.duration)

    folder = Path(args.out) if args.out else ROOT / "results" / "comparison" / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.tier}"
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    scores = score(entries, config, robust, seeds, args.backend, args.device)
    report = summarize(entries, scores, args.reference, args.bootstrap)
    report.update(tier=args.tier, seeds=[seeds[0], seeds[-1]], n_seeds=len(seeds), duration_s=config.duration,
                  backend=args.backend, provenance=provenance(), config=CONFIG.snapshot())

    scenarios = [(seed, pitch) for seed in seeds for pitch in config.start_pitches]
    with open(folder / "per_scenario.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller", "method", "run", "seed", "start_pitch", "cost", "fell", "fall_time_s"])
        for e in entries:
            s = scores[e.label]
            for (seed, pitch), c, fell, ft in zip(scenarios, s["costs"], s["fell"], s["fall_time"]):
                w.writerow([e.label, e.method, e.run, seed, pitch, c, int(fell), "" if np.isnan(ft) else ft])

    if args.latency_sweep:
        sweep_seeds = seeds[:SWEEP_FALL_RATE_SEEDS] if args.tier == "fall_rate" else seeds
        latencies = (tuple(float(x) for x in args.latencies.split(",")) if args.latencies
                     else SWEEP_LATENCIES_MS)
        curves = latency_sweep(entries, config, robust, sweep_seeds, latencies, args.backend, args.device)
        report["latency_sweep"] = {"seeds": [sweep_seeds[0], sweep_seeds[-1]], "threshold_falls": SWEEP_FALL_THRESHOLD,
                                   "curves": curves, "by_method": sweep_by_method(entries, curves)}
        with open(folder / "latency_sweep.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["controller", "latency_ms", "falls", "mean_cost"])
            for label, c in curves.items():
                for ms, n, mc in zip(c["latency_ms"], c["falls"], c["mean_cost"]):
                    w.writerow([label, ms, n, mc])
    report["elapsed_s"] = time.time() - t0
    (folder / "comparison.json").write_text(json.dumps(report, indent=2, default=float))

    print(f"{args.tier}: {len(seeds)} seeds x {len(config.start_pitches)} tilts, {config.duration:g} s, "
          f"against {args.reference}")
    print(f"{'method':<28}{'runs':>5}{'median ratio':>14}{'interval':>22}{'falls (med)':>13}")
    for method, m in report["methods"].items():
        ci = m["bootstrap_95ci"] or m["order_statistic_interval"]
        interval = "" if ci is None or ci.get("low") is None else f"[{ci['low']:.3f}, {ci['high']:.3f}]"
        print(f"{method:<28}{len(m['runs']):>5}{m['headline_median_ratio']:>14.4f}{interval:>22}"
              f"{m['falls']['median']:>13.1f}")
    print(f"saved {folder}")
    return folder


if __name__ == "__main__":
    main()
