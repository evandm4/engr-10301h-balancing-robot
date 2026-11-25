"""Compare saved controllers on one evaluation tier, with the statistics of docs/ml-baselines-plan.md 3.4.

Controllers come from three kinds of file:
    an existing comparison.json, from run_lqr_comparison.py (the Pololu robot's 7 PID and LQR
        controllers) or run_batch_tuning.py ("PID, cma-es", "PID, hand-tuned", ...), one run each
        (--existing; default the robot's reference file in config.py). A robot with no tuned
        reference yet (the LEGO robot before E1/E2) gets its own hand-tuned and model-based PIDs
        instead, and the reference (its hand-tuned PID) is reported as provisional;
    learned policies (.npz from run_neuroevolution.py), grouped into methods:
        --policies "A=C:/balancebot_training/pololu/neuroevolution/mlp/*/final.npz"   (one method, one run per file);
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
        runs, which covers run-seed variation ON THESE SCENARIOS only (conditional on the scenario
        set: far too narrow as a statement about new scenarios, so never quote it in place of the
        bootstrap interval); fall-rate and McNemar summaries (median and range over runs); every
        run's numbers;
    method against method: for every pair where at least one method has several runs, the ratio
        of their headlines with a paired bootstrap interval (one scenario resample shared by both
        methods, run indices resampled within each). Overlapping per-method intervals say little
        about a difference, because both are measured on the same scenarios;
    per scenario (per_scenario.csv): cost, fell and fall time for every controller, so each fall
        can be traced to the start tilt, a push, braking or standstill drift;
    side metrics that no cost weight touches, per run (median over scenarios, and the worst):
        torque roughness (RMS change per control tick, total variation per second), RMS torque,
        time at the torque limit, peak pitch, RMS velocity error;
    weight sensitivity: every controller re-scored, from the same simulations, with each Bryson
        limit of CostConfig halved and doubled, the effort term off, and the torque limit at the
        motors' continuous (thermal) rating (the robot's ComparisonConfig: Pololu 0.253 N m, LEGO
        0.50 N m). Costs are rebuilt from each scenario's unweighted
        integrals, so nothing is re-simulated. The controllers were trained on the configured
        cost, so this tests whether a ranking depends on the weights, not how each method would
        do if trained on another cost.
With --latency-sweep, each controller also runs at every latency of the robot's sweep grid
(config.ComparisonConfig; no chaos twins, every scenario at that one latency). Pololu: the design
latency (10 ms) and a 2.5 ms grid from 15 to 45 ms; LEGO: 7.5 to 45 ms in 2.5 ms steps, so its
7.5-12.5 ms training range is in it. The metrics are measured from the baseline latency: the robot's
nominal latency (TimingConfig.latency_s, 10 ms on both) or --baseline-latency, never the grid's first
entry. The baseline is always swept; latencies below it are reported but no metric uses them. Two
pre-registered metrics (plan 3.4), a relative fall metric, and the full curves:
    falls: the first latency with 4 or more falls. If the baseline already has that many,
        latency isn't what makes it fall, and the metric is reported as at or below the baseline
        (left-censored: every LEGO controller, whose baselines fall about 4% of the time);
    excess falls: the first latency above the baseline with 5 or more falls more than the
        controller's own at the baseline, on the same scenarios, where an exact one-sided McNemar
        test of the paired fall flags gives p < 0.05 (5 is the fewest that can: 5 new falls and
        none recovered give p = 0.03125, 4 give 0.0625). Readable whatever the baseline's fall rate,
        unless the baseline falls in so many scenarios that even every other one falling couldn't
        pass (flagged excess_falls_ceiling_censored, and then None means unknown, not never). The
        0.05 is POINTWISE and unadjusted: it is tested at every latency above the baseline (13 on
        Pololu's grid, 14 on LEGO's), so with no latency effect at all the chance of a false flag
        somewhere on the grid is well above 5% (about 16-22% by the 2026-11-25 review; the
        latencies' tests are correlated). Read a flag near the baseline with that in mind;
    cost: the first latency at which the median per-scenario cost reaches 1.5x the controller's
        own at the baseline. A controller past its delay margin can oscillate for a whole run without
        falling, which only this metric sees;
plus, at every latency, the median per-scenario cost ratio against the reference at that same
latency, and the first latency where a controller that beats the reference at the baseline stops
beating it.

The validation and fall-rate tiers are frozen: used once, on final controllers. The script
refuses them without --final, and nobody inspects per-scenario results there until every
controller in the comparison is frozen. Use --tier dev (or smoke) while developing.
--tier confirm (6,000,000-6,000,099, RobustnessSettings.confirm_seeds; registered for the LEGO E2
plan's D2) needs no --final, but it is a confirmation tier, not a selection tier: it checks picks
already made on dev (E2_PLAN.md D3's rule) and never chooses among candidates itself.

From the project folder (ROBOT=legobot, or --robot legobot, for the LEGO robot):
    python optimization/compare_controllers.py --tier dev --policies "L=C:/balancebot_training/pololu/neuroevolution/linear/*/final.npz"
    python optimization/compare_controllers.py --tier fall_rate --final --latency-sweep --policies ...

Results: results/<robot>/comparison/<timestamp>-<tier>/ with comparison.json, per_scenario.csv and,
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

from config import (CONFIG, POLOLU_COMPARISON, add_robot_argument, load_comparison_config, read_pid_comparison,
                    results_dir)
from balance_sim import load_policy
from batch_tuning import BatchEvaluator
from tuning import provenance

TIERS = ("smoke", "dev", "confirm", "validation", "fall_rate")
FROZEN = ("validation", "fall_rate")
# The latency sweep's grid, baseline and thresholds, and the thermal torque limit, are the robot's
# (config.ComparisonConfig, through comparison_settings()); the Pololu values were constants here.


def comparison_settings():
    """The selected robot's ComparisonConfig (looked up through CONFIG, so tests can swap the robot)."""
    return load_comparison_config(CONFIG.name)


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

    The coverage is over run seeds only, conditional on the scenarios every run was scored on.
    Where scenario difficulty dominates, it covers the median for new scenarios far less often
    (29% in the PR #10 review's simulation, against 93% for crossed_bootstrap).
    """
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    coverage = lambda k: 1.0 - 2.0 * float(stats.binom.cdf(k - 1, n, 0.5))
    if k is None:
        ks = [k for k in range(1, n // 2 + 1) if coverage(k) >= 0.95]
        if not ks:
            return {"low": None, "high": None, "coverage": None, "k": None}
        k = max(ks)
    return {"low": float(v[k - 1]), "high": float(v[n - k]), "coverage": coverage(k), "k": k,
            "conditional_on_scenarios": True}


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


def paired_method_bootstrap(ratios_a, ratios_b, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Interval for headline(A) / headline(B), each headline the median over runs of the median
    over scenarios of ratios (R_a, N) and (R_b, N) on the same N scenarios.

    Each replicate draws ONE scenario index vector, shared by both methods (they were scored on the
    same scenarios, so their difference is paired), and run indices within each method separately
    (their runs are independent). Below 1: A's typical cost is lower than B's.
    """
    a, b = np.asarray(ratios_a, dtype=float), np.asarray(ratios_b, dtype=float)
    if a.shape[1] != b.shape[1]:
        raise ValueError("both methods must be scored on the same scenarios")
    head = lambda r: np.median(np.median(r, axis=1))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        scen = rng.integers(0, a.shape[1], a.shape[1])
        out[i] = head(a[rng.integers(0, len(a), len(a))][:, scen]) / head(b[rng.integers(0, len(b), len(b))][:, scen])
    low, high = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return {"ratio": float(head(a) / head(b)), "low": float(low), "high": float(high), "n_boot": n_boot}


# ---- weight sensitivity and side metrics ------------------------------------------------------

def cost_variants() -> Dict[str, Tuple[float, float, float]]:
    """name -> (pitch_limit_rad, velocity_limit, torque_limit): the configured limits, each halved
    and doubled, effort off, and the torque limit at the motors' continuous rating (the robot's
    ComparisonConfig.thermal_torque: Pololu 25% of its 6.0 A stall current, LEGO its 0.25 N m rating)."""
    c = CONFIG.cost
    base = (c.pitch_limit_rad, c.velocity_limit, c.torque_limit)
    thermal = comparison_settings().thermal_torque(CONFIG.actuator)
    variants = {"configured": base}
    for i, name in enumerate(("pitch", "velocity", "torque")):
        for factor in (0.5, 2.0):
            limits = list(base)
            limits[i] *= factor
            variants[f"{name} limit x{factor:g}"] = tuple(limits)
    variants["effort off"] = (base[0], base[1], math.inf)
    variants[f"torque limit {thermal:.3f} (thermal)"] = (base[0], base[1], thermal)
    return variants


def rescore(s: dict, limits: Tuple[float, float, float], duration: float) -> np.ndarray:
    """Per-scenario costs under other Bryson limits, from the unweighted integrals (balance_sim.cost)."""
    pitch_limit, velocity_limit, torque_limit = limits
    velocity_weight = (pitch_limit / velocity_limit) ** 2
    effort_weight = 0.0 if math.isinf(torque_limit) else (pitch_limit / torque_limit) ** 2
    total = s["pitch_error"] + velocity_weight * s["velocity_error"] + effort_weight * s["effort"]
    fall_time = np.where(s["fell"], s["fall_time"], duration)
    penalty = CONFIG.cost.fall_penalty * (1.0 + (duration - fall_time) / duration)
    return np.where(s["fell"], total + penalty, total)


def weight_sensitivity(entries, scores, reference: str, duration: float) -> dict:
    """Each method's headline (median over runs of the median per-scenario ratio) under every variant.

    Also checks, per variant, that the fall penalty still ranks every fall below every run that
    stayed up: the costliest non-fall scenario scored here must cost less than the cheapest
    possible fall (the penalty alone, for a fall at the last instant). The analytic bound only
    holds for 20 s runs under the configured weights (CostConfig), so this is checked on the data.
    """
    out = {}
    for name, limits in cost_variants().items():
        costs = {e.label: rescore(scores[e.label], limits, duration) for e in entries}
        ref = costs[reference]
        heads = {}
        for method in dict.fromkeys(e.method for e in entries):
            runs = [float(np.median(costs[e.label] / ref)) for e in entries if e.method == method]
            heads[method] = float(np.median(runs))
        stayed_up = [costs[e.label][~scores[e.label]["fell"]] for e in entries]
        worst_up = max((float(c.max()) for c in stayed_up if c.size), default=0.0)
        out[name] = {"limits": {"pitch_rad": limits[0], "velocity_m_s": limits[1], "torque_n_m": limits[2]},
                     "headline_median_ratio": heads, "max_non_fall_cost": worst_up,
                     "falls_rank_below_every_non_fall": worst_up < CONFIG.cost.fall_penalty}
    return out


def side_metrics(s: dict, control_dt: float) -> dict:
    """Weight-free behavior per scenario, summarized as the median and the worst over scenarios."""
    ticks = np.maximum(s["alive_ticks"], 1.0)
    seconds = ticks * control_dt
    per_scenario = {
        "torque_rms_change_per_tick_n_m": np.sqrt(s["torque_rate_sq"] / ticks),
        "torque_variation_per_s_n_m": s["torque_variation"] / seconds,
        "torque_rms_n_m": np.sqrt(s["effort"] / seconds),
        "time_at_torque_limit_fraction": s["saturated_ticks"] / ticks,
        "peak_pitch_deg": np.degrees(s["peak_pitch"]),
        "velocity_rms_error_m_s": np.sqrt(s["velocity_error"] / seconds),
    }
    return {k: {"median": float(np.median(v)), "worst": float(np.max(v))} for k, v in per_scenario.items()}


def first_latency_with_falls(latencies_ms, falls, threshold: int = POLOLU_COMPARISON.sweep_fall_threshold
                             ) -> Optional[float]:
    """The pre-registered fall metric: the first latency with `threshold` or more falls; None if none.
    The grid starts at the baseline. When the baseline itself reaches it, latency isn't what makes it
    fall: report it as at or below that latency (latency_sweep does)."""
    for ms, n in zip(latencies_ms, falls):
        if n >= threshold:
            return float(ms)
    return None


def first_latency_with_cost(latencies_ms, median_costs, factor: float = POLOLU_COMPARISON.sweep_cost_factor
                            ) -> Optional[float]:
    """The pre-registered cost metric: the first latency whose median per-scenario cost is at least
    `factor` times the controller's own at the first grid point (the baseline); None if none."""
    base = median_costs[0]
    for ms, c in zip(latencies_ms[1:], median_costs[1:]):
        if c >= factor * base:
            return float(ms)
    return None


def mcnemar_excess(fell, fell_baseline) -> dict:
    """Exact one-sided McNemar test that a controller falls MORE at some latency than at its own
    baseline latency, on the same scenarios: b = fell only here, c = fell only at the baseline,
    p = P(Binomial(b + c, 1/2) >= b). excess = b - c, the difference in fall counts."""
    fell, fell_baseline = np.asarray(fell, bool), np.asarray(fell_baseline, bool)
    b, c = int(np.sum(fell & ~fell_baseline)), int(np.sum(~fell & fell_baseline))
    p = 1.0 if b == 0 else float(stats.binomtest(b, b + c, 0.5, alternative="greater").pvalue)
    return {"only_here": b, "only_at_baseline": c, "excess": b - c, "p_value": p}


def first_latency_with_excess_falls(latencies_ms, excess, p_values,
                                    min_excess: int = POLOLU_COMPARISON.sweep_excess_falls,
                                    alpha: float = POLOLU_COMPARISON.sweep_excess_alpha) -> Optional[float]:
    """The relative fall metric: over the grid above the baseline, the first latency with at least
    `min_excess` more falls than the controller's own at the baseline (same scenarios) and a one-sided
    McNemar p below `alpha` (mcnemar_excess); None if none. `alpha` applies at each latency on its
    own (pointwise, no multiplicity adjustment across the grid).

    Unlike first_latency_with_falls it isn't left-censored by a baseline that already falls some
    (the LEGO robot's ~4%), but it has a ceiling: if so few scenarios survive the baseline that even
    all of them falling can't pass (excess_ceiling_censored), None means unknown, not never."""
    for ms, e, p in zip(latencies_ms, excess, p_values):
        if e >= min_excess and p < alpha:
            return float(ms)
    return None


def excess_ceiling_censored(survivors_at_baseline: int,
                            min_excess: int = POLOLU_COMPARISON.sweep_excess_falls,
                            alpha: float = POLOLU_COMPARISON.sweep_excess_alpha) -> bool:
    """True when the relative fall metric can't fire at any latency: the best case, every scenario
    that survived the baseline falling and none of its falls recovering (b = survivors, c = 0, so
    p = 0.5 ** survivors), still misses `min_excess` or `alpha`."""
    best = int(survivors_at_baseline)
    return not (best >= min_excess and 0.5 ** best < alpha)


def sweep_grid(latencies_ms, baseline_ms: float) -> Tuple[float, ...]:
    """The sweep's latencies, ascending, with the baseline always among them (added if missing).
    The baseline is named, never taken from a position: '--latencies 7.5,10,15' keeps a 10 ms baseline."""
    grid = sorted({round(float(ms), 9) for ms in latencies_ms} | {round(float(baseline_ms), 9)})
    if grid[0] < 0:
        raise ValueError(f"negative latency in the sweep: {grid[0]} ms")
    return tuple(grid)


def first_latency_losing(latencies_ms, median_ratios) -> Optional[float]:
    """For a controller that beats the reference at the baseline (median ratio < 1), the first latency
    where it no longer does; None if it never stops, or if it didn't beat the reference to begin with."""
    if not median_ratios[0] < 1.0:
        return None
    for ms, r in zip(latencies_ms[1:], median_ratios[1:]):
        if r >= 1.0:
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
    warnings: List[str] = field(default_factory=list)   # e.g. trained under a different scoring config

    @property
    def label(self) -> str:
        return self.method if self.run == self.method else f"{self.method} / {self.run}"


# Snapshot sections that only matter to learned policies: a PID or LQR run saved under other LEARNING
# scales is still comparable (the scales are re-measured as the reference PID changes).
NOT_FOR_GAINS = ("learning",)


def _run_warnings(path: Path, saved_max_torque: Optional[float] = None,
                  embedded_config: Optional[dict] = None, ignore: tuple = ()) -> List[str]:
    """Why a saved controller may not be comparable with today's config: its run's snapshot.json
    (next to it), or failing that the config embedded in the file itself (the tuning scripts' per-method
    json and comparison.json), differs in something that changes training or scoring, or its saved
    torque limit differs from the one it will be scored with. `ignore`: sections that don't apply to it."""
    out = []
    snapshot = path.parent / "snapshot.json"
    if snapshot.exists():
        differences = CONFIG.scoring_differences(json.loads(snapshot.read_text())["config"], ignore)
        if differences:
            out.append("trained under a different config: " + ", ".join(differences))
    elif isinstance(embedded_config, dict):
        differences = CONFIG.scoring_differences(embedded_config, ignore)
        if differences:
            out.append("trained under a different config: " + ", ".join(differences))
    else:
        out.append("no snapshot.json next to it: its training config can't be checked")
    if saved_max_torque is not None and saved_max_torque != CONFIG.robot.max_torque:
        out.append(f"saved with max_torque {saved_max_torque}, scored with {CONFIG.robot.max_torque}")
    return out


def existing_controllers(path=None) -> List[Entry]:
    """One entry per controller of a comparison.json (either tuning script's layout), or, with no
    path, the selected robot's reference controllers (config.ProjectConfig.reference_controllers:
    its reference file, or its own baselines while it has no tuned reference)."""
    if path is None:
        controllers = CONFIG.reference_controllers()
        source = CONFIG.learning.reference_pid_file or f"{CONFIG.name} baselines (config.py)"
    else:
        controllers, source = read_pid_comparison(path), str(path)
    warnings = []
    if path is not None:   # comparison.json embeds the config it was run under: old (pre-B5, tick-locked) files differ
        embedded = json.loads(Path(path).read_text()).get("config")
        if isinstance(embedded, dict):
            differences = CONFIG.scoring_differences(embedded, NOT_FOR_GAINS)   # PID / LQR entries only
            if differences:
                warnings.append("trained under a different config: " + ", ".join(differences))
    return [Entry(label, label, kind, np.asarray(gains, float), source=str(source), warnings=list(warnings))
            for label, (kind, gains) in controllers.items()]


def _expand(pattern: str) -> List[Path]:
    if "=" not in pattern:
        raise SystemExit(f"expected NAME=GLOB, got {pattern!r}")
    return [Path(p) for p in sorted(glob.glob(pattern.split("=", 1)[1], recursive=True))]


def policy_runs(pattern: str) -> List[Entry]:
    method = pattern.split("=", 1)[0]
    entries = []
    for path in _expand(pattern):
        spec, params, max_torque = load_policy(path)
        entries.append(Entry(method, path.parent.name, "policy", params, spec, str(path),
                             _run_warnings(path, max_torque)))
    if not entries:
        raise SystemExit(f"no files match {pattern!r}")
    return entries


def gain_runs(pattern: str) -> List[Entry]:
    method = pattern.split("=", 1)[0]
    entries = []
    for path in _expand(pattern):
        data = json.loads(path.read_text())
        gains = data["gains"]
        if isinstance(gains, dict):   # save_result writes {"kp": ..., "ki": ..., ...}
            gains = [gains[k] for k in ("kp", "ki", "kd", "kv_p", "kv_i") if k in gains]
        entries.append(Entry(method, path.parent.name, data.get("kind", "cascaded"),
                             np.asarray(gains, float), source=str(path),
                             warnings=_run_warnings(path, embedded_config=data.get("config"),
                                                    ignore=NOT_FOR_GAINS)))
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
            out[e.label] = {"costs": result.costs[i], "fell": result.fell[i],
                            "mean_cost": float(result.mean_cost[i]), "objective": float(result.objective[i]),
                            "sensitivity": float(result.sensitivity[i]), "n_fell": int(result.n_fell[i]),
                            **{f: getattr(result, f)[i] for f in BatchEvaluator.REPORT_FIELDS}}

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
                         "warnings": list(e.warnings),
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
    # Method against method: only where at least one side has several runs (two single
    # controllers are compared exactly by their per-run numbers, or by --reference).
    ratios_of = {m: np.array([scores[e.label]["costs"] / ref["costs"] for e in entries if e.method == m])
                 for m in methods}
    pairs = {}
    names = list(methods)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if len(ratios_of[a]) > 1 or len(ratios_of[b]) > 1:
                pairs[f"{a} vs {b}"] = paired_method_bootstrap(ratios_of[a], ratios_of[b], n_boot)
    return {"reference": reference, "runs": runs, "methods": methods, "method_pairs": pairs}


def latency_sweep(entries: Sequence[Entry], config, robust, seeds, reference: str,
                  latencies_ms=None, backend="numpy", device="cuda", baseline_ms: Optional[float] = None,
                  settings=None) -> Dict[str, dict]:
    """Falls and cost at each latency: every scenario at that one latency, no chaos twins.

    settings: a config.ComparisonConfig (default: the robot's). latencies_ms defaults to its grid and
    baseline_ms, the latency every metric is measured from, to its baseline (the robot's nominal
    latency). The grid is swept in ascending order with the baseline added if missing (sweep_grid);
    the metrics use the baseline and the latencies above it only.
    """
    settings = comparison_settings() if settings is None else settings
    baseline_ms = settings.baseline_ms(CONFIG) if baseline_ms is None else float(baseline_ms)
    grid = sweep_grid(settings.sweep_latencies_ms if latencies_ms is None else latencies_ms, baseline_ms)
    i0 = grid.index(round(baseline_ms, 9))
    sweep_robust = replace(robust, latency_range=None, sensitivity_weight=0.0)
    fields = ("latency_ms", "falls", "mean_cost", "median_cost", "median_ratio_vs_reference", "fall_rate_95ci")
    curves = {e.label: {f: [] for f in fields} for e in entries}
    fell = {e.label: [] for e in entries}
    for ms in grid:
        scores = score(entries, replace(config, latency=ms / 1000), sweep_robust, seeds, backend, device)
        ref = scores[reference]["costs"]
        for e in entries:
            s, c = scores[e.label], curves[e.label]
            c["latency_ms"].append(ms)
            c["falls"].append(s["n_fell"])
            c["mean_cost"].append(s["mean_cost"])
            c["median_cost"].append(float(np.median(s["costs"])))
            c["median_ratio_vs_reference"].append(float(np.median(s["costs"] / ref)))
            c["fall_rate_95ci"].append(list(clopper_pearson(s["n_fell"], len(s["costs"]))))
            fell[e.label].append(np.asarray(s["fell"], bool))
    for label, c in curves.items():
        up = {k: c[k][i0:] for k in ("latency_ms", "falls", "median_cost", "median_ratio_vs_reference")}
        first = first_latency_with_falls(up["latency_ms"], up["falls"], settings.sweep_fall_threshold)
        c["first_latency_with_falls"] = first
        # Already falling that often at the baseline: latency isn't the cause, so the true
        # threshold is unknown below the baseline (left-censored).
        c["falls_at_or_below_baseline"] = first is not None and first == c["latency_ms"][i0]
        c["falls_at_baseline"] = c["falls"][i0]
        c["first_latency_with_cost"] = first_latency_with_cost(up["latency_ms"], up["median_cost"],
                                                               settings.sweep_cost_factor)
        c["first_latency_losing_to_reference"] = first_latency_losing(up["latency_ms"], up["median_ratio_vs_reference"])
        # The relative fall metric: paired against this controller's own falls at the baseline.
        tests = [mcnemar_excess(f, fell[label][i0]) for f in fell[label]]
        c["excess_falls_vs_baseline"] = [t["excess"] for t in tests]
        c["falls_only_here_vs_baseline"] = [t["only_here"] for t in tests]
        c["falls_only_at_baseline"] = [t["only_at_baseline"] for t in tests]
        c["mcnemar_p_vs_baseline"] = [t["p_value"] for t in tests]
        c["first_latency_with_excess_falls"] = first_latency_with_excess_falls(
            c["latency_ms"][i0 + 1:], c["excess_falls_vs_baseline"][i0 + 1:], c["mcnemar_p_vs_baseline"][i0 + 1:],
            settings.sweep_excess_falls, settings.sweep_excess_alpha)
        c["excess_falls_ceiling_censored"] = excess_ceiling_censored(
            int(np.sum(~fell[label][i0])), settings.sweep_excess_falls, settings.sweep_excess_alpha)
        c["scenarios"] = len(seeds) * len(config.start_pitches)
    return curves


SWEEP_METRICS = ("first_latency_with_falls", "first_latency_with_cost", "first_latency_losing_to_reference",
                 "first_latency_with_excess_falls")


# A metric's None means "no latency on the grid reached it" (ranked above every latency in the median)
# unless the run's flag here says the metric couldn't have fired at all: then it is unknown.
SWEEP_CENSORED_FLAGS = {"first_latency_with_excess_falls": "excess_falls_ceiling_censored"}


def sweep_by_method(entries, curves) -> dict:
    """The sweep metrics per method: each run's value, and their median (None = no latency reached it).
    A ceiling-censored None (SWEEP_CENSORED_FLAGS) is unknown, not "never": a method with any such run
    gets median None and censored_runs, its count."""
    out = {}
    for method in dict.fromkeys(e.method for e in entries):
        group = [curves[e.label] for e in entries if e.method == method]
        out[method] = {}
        for metric in SWEEP_METRICS:
            values = [c[metric] for c in group]
            flag = SWEEP_CENSORED_FLAGS.get(metric)
            censored = sum(1 for c in group if flag and c.get(flag) and c[metric] is None)
            if censored:
                out[method][metric] = {"per_run": values, "median": None, "censored_runs": censored}
                continue
            med = float(np.median([v if v is not None else math.inf for v in values]))
            out[method][metric] = {"per_run": values, "median": None if math.isinf(med) else med}
        out[method]["falls_at_or_below_baseline"] = [c["falls_at_or_below_baseline"] for c in group]
        out[method]["excess_falls_ceiling_censored"] = [c["excess_falls_ceiling_censored"] for c in group]
    return out


# ---- main -----------------------------------------------------------------------------------

def tier_seeds(robust, tier: str) -> Tuple[int, ...]:
    return tuple({"smoke": robust.smoke_seeds, "dev": robust.dev_seeds, "confirm": robust.confirm_seeds,
                  "validation": robust.validation_seeds, "fall_rate": robust.fall_rate_seeds}[tier])


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=TIERS, default="dev",
                        help="dev (or smoke) while developing; confirm checks dev picks (never selects); "
                             "validation and fall_rate are frozen")
    parser.add_argument("--final", action="store_true", help="allow the frozen validation and fall-rate tiers")
    parser.add_argument("--existing", default=None,
                        help="comparison.json with the existing controllers (default: the robot's reference "
                             "file, or its own baselines if it has none; '' for none)")
    parser.add_argument("--policies", action="append", default=[], help="NAME=GLOB of policy .npz files")
    parser.add_argument("--gain-runs", action="append", default=[], help="NAME=GLOB of gain .json files")
    parser.add_argument("--reference", default=CONFIG.learning.reference_pid_entry)
    parser.add_argument("--latency-sweep", action="store_true")
    parser.add_argument("--seeds", type=int, default=None, help="only the tier's first N seeds (development)")
    parser.add_argument("--duration", type=float, default=None, help="run length (default: 60 s validation scenario)")
    parser.add_argument("--latencies", type=str, default=None,
                        help="sweep latencies in ms, e.g. 7.5,10,15,30,45 (default: the robot's grid, "
                             "config.ComparisonConfig); the baseline is added if missing")
    parser.add_argument("--baseline-latency", type=float, default=None,
                        help="the sweep's baseline latency in ms, the one every sweep metric is measured from "
                             "(default: the robot's, its nominal TimingConfig.latency_s); never the first of "
                             "--latencies")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy",
                        help="numpy float64 is the designated validation path; torch float64 for development")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    add_robot_argument(parser)
    args = parser.parse_args(argv)
    if args.tier in FROZEN and not args.final:
        parser.error(f"the {args.tier} tier is frozen: pass --final once every controller is final")

    entries = existing_controllers(args.existing) if args.existing else \
        ([] if args.existing == "" else existing_controllers())
    # The robot's reference while it has no tuned one (config.py reference_is_provisional).
    provisional = (args.existing is None and CONFIG.reference_is_provisional
                   and args.reference == CONFIG.learning.reference_pid_entry)
    if provisional:
        print(f"NOTE: {CONFIG.name} has no tuned reference PID yet; {args.reference!r} is a provisional "
              "reference, and every ratio against it is provisional")
    for pattern in args.policies:
        entries += policy_runs(pattern)
    for pattern in args.gain_runs:
        entries += gain_runs(pattern)
    labels = [e.label for e in entries]
    if args.reference not in labels:
        parser.error(f"reference {args.reference!r} is not among the controllers")
    if len(set(labels)) != len(labels):
        parser.error("two controllers share a label; give the methods different names")
    for e in entries:
        for warning in e.warnings:
            print(f"WARNING {e.label}: {warning}")

    robust = CONFIG.build_robustness()
    seeds = tier_seeds(robust, args.tier)
    if args.seeds:
        seeds = seeds[:args.seeds]
    config = CONFIG.build_eval_config(validation=True)
    if args.duration:
        config = replace(config, duration=args.duration)

    folder = Path(args.out) if args.out else results_dir("comparison", f"{time.strftime('%Y%m%d-%H%M%S')}-{args.tier}")
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    scores = score(entries, config, robust, seeds, args.backend, args.device)
    report = summarize(entries, scores, args.reference, args.bootstrap)
    for e in entries:   # rebuilding the costs from their parts must give the scored costs back
        np.testing.assert_allclose(rescore(scores[e.label], cost_variants()["configured"], config.duration),
                                   scores[e.label]["costs"], rtol=1e-9)
        report["runs"][e.label]["side_metrics"] = side_metrics(scores[e.label], config.control_dt)
    report["weight_sensitivity"] = weight_sensitivity(entries, scores, args.reference, config.duration)
    report.update(robot=CONFIG.name, reference_provisional=provisional, tier=args.tier, seeds=[seeds[0], seeds[-1]], n_seeds=len(seeds), duration_s=config.duration,
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
        settings = comparison_settings()
        sweep_seeds = seeds[:settings.sweep_fall_rate_seeds] if args.tier == "fall_rate" else seeds
        baseline = settings.baseline_ms(CONFIG) if args.baseline_latency is None else args.baseline_latency
        latencies = sweep_grid(tuple(float(x) for x in args.latencies.split(",")) if args.latencies
                               else settings.sweep_latencies_ms, baseline)
        curves = latency_sweep(entries, config, robust, sweep_seeds, args.reference, latencies,
                               args.backend, args.device, baseline_ms=baseline, settings=settings)
        report["latency_sweep"] = {"seeds": [sweep_seeds[0], sweep_seeds[-1]], "baseline_ms": baseline,
                                   "threshold_falls": settings.sweep_fall_threshold,
                                   "cost_factor": settings.sweep_cost_factor,
                                   "curves": curves, "by_method": sweep_by_method(entries, curves),
                                   "latencies_ms": list(latencies),
                                   "excess_falls": {"min_excess": settings.sweep_excess_falls,
                                                    "alpha": settings.sweep_excess_alpha,
                                                    "alpha_scope": "pointwise_unadjusted",
                                                    "n_tested_latencies": sum(1 for ms in latencies if ms > baseline),
                                                    "test": "exact one-sided McNemar of the fall flags at each "
                                                            "latency against the controller's own at baseline_ms, "
                                                            "same scenarios; alpha at each latency on its own, "
                                                            "not adjusted for the n_tested_latencies tests"}}
        with open(folder / "latency_sweep.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["controller", "latency_ms", "falls", "mean_cost", "median_cost", "median_ratio_vs_reference"])
            for label, c in curves.items():
                for row in zip(c["latency_ms"], c["falls"], c["mean_cost"], c["median_cost"],
                               c["median_ratio_vs_reference"]):
                    w.writerow([label, *row])
    report["elapsed_s"] = time.time() - t0
    (folder / "comparison.json").write_text(json.dumps(report, indent=2, default=float))

    print(f"{CONFIG.name}, {args.tier}: {len(seeds)} seeds x {len(config.start_pitches)} tilts, "
          f"{config.duration:g} s, against {args.reference}" + (" (PROVISIONAL reference)" if provisional else ""))
    print(f"{'method':<28}{'runs':>5}{'median ratio':>14}{'bootstrap 95%':>20}{'falls (med)':>13}")
    for method, m in report["methods"].items():
        ci = m["bootstrap_95ci"]   # the order-statistic interval is conditional on these scenarios: not shown here
        interval = "" if ci is None else f"[{ci['low']:.3f}, {ci['high']:.3f}]"
        print(f"{method:<28}{len(m['runs']):>5}{m['headline_median_ratio']:>14.4f}{interval:>20}"
              f"{m['falls']['median']:>13.1f}")
    if report["method_pairs"]:
        print("\nMethod against method: ratio of headlines, paired bootstrap 95% (below 1: the first is better)")
        for pair, p in report["method_pairs"].items():
            print(f"  {pair:<50}{p['ratio']:>8.4f}  [{p['low']:.3f}, {p['high']:.3f}]")
    print("\nWeight sensitivity: headline median ratio vs the reference, re-scored under other Bryson limits")
    methods = list(report["methods"])
    print(f"{'variant':<32}" + "".join(f"{m[:14]:>16}" for m in methods))
    for name, v in report["weight_sensitivity"].items():
        flag = "" if v["falls_rank_below_every_non_fall"] else \
            f"   WARNING: a non-fall run cost {v['max_non_fall_cost']:.0f}, above the fall penalty"
        print(f"{name:<32}" + "".join(f"{v['headline_median_ratio'][m]:>16.3f}" for m in methods) + flag)
    if args.latency_sweep:
        sweep = report["latency_sweep"]
        print(f"\nLatency sweep (baseline {baseline:g} ms; {', '.join(f'{ms:g}' for ms in latencies)} ms): first latency "
              f"with >= {sweep['threshold_falls']} falls; with >= {settings.sweep_excess_falls} falls more than at "
              f"the baseline (McNemar p < {settings.sweep_excess_alpha:g} pointwise, unadjusted over "
              f"{sweep['excess_falls']['n_tested_latencies']} latencies); with median cost >= "
              f"{sweep['cost_factor']}x baseline; losing to the reference")
        show = lambda v: "none" if v is None else f"{v:g} ms"
        for label, c in sweep["curves"].items():
            falls = show(c["first_latency_with_falls"]) + (" (at/below baseline)" if c["falls_at_or_below_baseline"] else "")
            excess = "censored" if c["excess_falls_ceiling_censored"] and c["first_latency_with_excess_falls"] is None \
                else show(c["first_latency_with_excess_falls"])
            print(f"  {label[:40]:<42}{falls:>26}{excess:>10}"
                  f"{show(c['first_latency_with_cost']):>10}{show(c['first_latency_losing_to_reference']):>10}"
                  f"   ({c['falls_at_baseline']} falls at baseline)")
    print("\nSide metrics (median over scenarios; not in the cost)")
    keys = [("torque_rms_change_per_tick_n_m", "dU/tick"), ("torque_variation_per_s_n_m", "TV/s"),
            ("torque_rms_n_m", "U rms"), ("time_at_torque_limit_fraction", "sat."),
            ("peak_pitch_deg", "pk pitch"), ("velocity_rms_error_m_s", "v err")]
    print(f"{'controller':<34}" + "".join(f"{label:>10}" for _, label in keys))
    for label, run in report["runs"].items():
        m = run["side_metrics"]
        print(f"{label[:33]:<34}" + "".join(f"{m[k]['median']:>10.4f}" for k, _ in keys))
    print(f"saved {folder}")
    return folder


if __name__ == "__main__":
    main()
