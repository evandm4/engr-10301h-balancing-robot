"""Phase E2.2: pick the reference-PID finalists on dev and confirm them on the D2 slice with D3's rule
(hardware/legobot/E2_PLAN.md D1-D3, E2.1-E2.3).

1. Collect the 20 E2.1 runs under results/legobot/batch_tuning/: identified by provenance, not by timestamp.
   A run counts when its comparison.json records the E2.1 commit (--commit, default ecb666e), robot legobot,
   150 training seeds (--train-seeds; len(robustness.train_seeds), cross-checked against --seeds), float64
   searches (no --float32; the per-method json's dtype), the torch backend, robust non-final scoring on the dev
   seeds, CMA-ES 30 x 120, and one of the two E2.1 recipes:
       mid:  --cma-x0 midpoint     (DE 30 gens x pop 24, then CMA-ES)          the reference procedure
       warm: --cma-x0 model-based  --de-iters 2 --de-pop 6                      as in E1
   Every other folder (the four E1 runs, benchmarks, ...) is listed with the reason it was left out. The kept
   runs must be exactly one per (recipe, run seed 0-9), none with a dirty tree and none trained under a config
   that differs from today's in anything scored (config.scoring_differences); otherwise the script stops,
   unless --allow-partial.

2. Candidates, two per run (E2_PLAN E2.2 "each run's cma-es (mean) and best"):
       "cma-es (mean)"  CMA-ES's final distribution mean (xfavorite);
       "cma-es"         the run's best: CMA-ES's best sample, i.e. the lowest training objective the search
                        saw. This is the reading of "best" used here, following E1_NOTES.md section 5 item 4
                        ("both cma-es (mean) and best-sample finalists"). DE and random search are not
                        candidates by default: --best overall swaps the CMA-ES best sample for whichever search
                        entry (DE, CMA-ES best, random search) has the lowest float64 re-scored training
                        objective in the run's comparison.json; --include-entry adds others outright.
   So 40 tuned candidates for the full E2.1 set, plus "PID, model-based" and "PID, hand-tuned".

3. Dev pick (E2.2): every candidate and both baselines on the dev seeds 5000-5049 (x 3 start tilts = 150
   scenarios), 60 s, float64, with compare_controllers.score / summarize (the NumPy engine compare_controllers
   uses for every final score, in this one process on this one machine). Ranked by the robust objective (mean
   cost with the fall penalty; the LEGO chaos weight is 0, so objective = mean cost), also reporting median
   cost, falls, the paired per-scenario median cost ratio to model-based with its bootstrap 95% CI
   (compare_controllers.crossed_bootstrap on the one-run ratio matrix: one shared scenario resample per
   replicate, i.e. the paired bootstrap) and the exact McNemar test on falls. The finalists are the top
   --n-finalists (default 3, as E1 carried its top three Step 1 picks to the fresh slice) TUNED candidates by
   dev objective; ties go to the lower median cost. Model-based is not a finalist: it is the comparator
   and the default.

4. Confirm (D2/D3): the finalists and model-based on the confirm tier (6,000,000-6,000,099 x 3 = 300
   scenarios, 60 s, same engine). D3, as written before the runs: a finalist QUALIFIES iff
       (a) the upper end of its paired median-cost ratio's 95% bootstrap CI against model-based is below 1, and
       (b) McNemar does not show more falls, where "shows more falls" means: the finalist fell in more of the
           discordant scenarios than model-based (only_this_fell > only_reference_fell, equivalently more falls
           in total on the same 300 scenarios) AND the exact two-sided McNemar p-value is below 0.05.
   If none qualifies, the reference is model-based. If several qualify, the one ranked best on DEV (lowest
   dev objective) is adopted: the confirm tier never chooses among candidates; it only accepts or rejects.
   No multiplicity correction across finalists (D3 verbatim); the count is reported.

Outputs, in results/legobot/e2_selection/<timestamp>/ (with -dry appended when --seeds/--duration shorten it):
    selection.json         everything: runs and their provenance (commit, dirty, hostname, gpu), candidates and
                           gains, dev and confirm statistics, finalists, the D3 verdict per finalist, the adopted
                           reference, settings and this process's provenance;
    dev_per_scenario.csv, confirm_per_scenario.csv
    e2_notes_tables.md     tables ready to paste into E2_NOTES.md.

From the project folder (the venv):
    python hardware/legobot/e2_select.py                                   the real E2.2 (after E2.1)
    python hardware/legobot/e2_select.py --allow-partial --seeds 2 --duration 5     cheap dry run
    python hardware/legobot/e2_select.py --commit 59065a3 --train-seeds 50 --allow-partial --seeds 2 --duration 5
                                                                           dry run on the four E1 runs

Seeds: dev (5000-5049) and confirm (6,000,000-6,000,099) only, checked against every held-out tier at start;
never the frozen validation (3000-3099) or fall-rate (6000-6999) sets. The confirm slice is touched only by the
real run: a dry run (--seeds N and/or --duration S) scores its dev stage on the first N dev seeds and its
confirm stage on the LAST N dev seeds as a stand-in (N <= 25, default 25), so D2's slice stays fresh.
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "simulation", REPO / "optimization"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

DEFAULT_COMMITS = ("ecb666e",)
DEFAULT_TRAIN_SEEDS = 150
RUN_SEEDS = tuple(range(10))
RECIPES = {   # E2_PLAN E2.1: name -> the run_arguments that define it
    "mid": {"cma_x0": "midpoint", "de_iters": 30, "de_pop": 24},
    "warm": {"cma_x0": "model-based", "de_iters": 2, "de_pop": 6},
}
COMMON_ARGS = {"robot": "legobot", "backend": "torch", "float32": False, "plain": False, "final": False,
               "no_randomize": False, "sensitivity_weight": None, "cma_iters": 30, "cma_pop": 120}
MEAN, BEST = "cma-es (mean)", "cma-es"
SEARCH_ENTRIES = ("differential evolution", "cma-es", "random search")
ENTRY_TAG = {"cma-es (mean)": "CMA-{r} (mean)", "cma-es": "CMA-{r}", "differential evolution": "DE-{r}",
             "random search": "RS-{r}"}
GAIN_KEYS = ("kp", "ki", "kd", "kv_p", "kv_i")   # config.read_pid_comparison's order
MB, HT = "PID, model-based", "PID, hand-tuned"
DEFAULT_N_FINALISTS = 3
ALPHA = 0.05
DRY_DEFAULT_SEEDS = 25   # a dry run without --seeds: dev 5000-5024, confirm stand-in 5025-5049

D3_RULE = ("D3 (E2_PLAN.md, written before the runs): adopt a tuned finalist only if, on the confirm slice "
           "(6,000,000-6,000,099 x 3 tilts, 60 s), (a) its paired per-scenario median-cost ratio to model-based has a "
           "95% bootstrap CI entirely below 1, and (b) McNemar does not show more falls. 'Shows more falls' := the "
           "finalist fell in more discordant scenarios than model-based (only_this_fell > only_reference_fell, i.e. "
           "more falls in total on the same scenarios) AND the exact two-sided McNemar p < 0.05. Otherwise "
           "model-based. If several finalists qualify, the one with the lowest DEV objective is adopted: the confirm "
           "tier accepts or rejects, it never chooses among candidates. No multiplicity correction across finalists.")
BEST_READING = ("'best' = the run's CMA-ES best sample (comparison.json entry 'cma-es': the lowest training objective "
                "the CMA-ES search saw), alongside 'cma-es (mean)'; E1_NOTES.md 5.4 ('cma-es (mean) and best-sample "
                "finalists'). DE and random search are excluded unless --best overall / --include-entry.")


# ---- collection ---------------------------------------------------------------------------------

@dataclass
class Run:
    folder: Path
    recipe: Optional[str] = None
    seed: Optional[int] = None
    commit: Optional[str] = None
    dirty: Optional[bool] = None
    hostname: Optional[str] = None
    gpu: Optional[str] = None
    n_train_seeds: Optional[int] = None
    dtype: Optional[str] = None
    timestamp: Optional[str] = None
    results: dict = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)   # why it isn't an E2.1 run (excluded)
    flags: List[str] = field(default_factory=list)     # an E2.1 run, but suspect (refused without --allow-partial)

    @property
    def name(self) -> str:
        return self.folder.name

    def summary(self) -> dict:
        return {"folder": self.name, "path": str(self.folder), "recipe": self.recipe, "run_seed": self.seed,
                "git_commit": self.commit, "git_dirty": self.dirty, "hostname": self.hostname, "gpu": self.gpu,
                "n_train_seeds": self.n_train_seeds, "search_dtype": self.dtype, "timestamp": self.timestamp,
                "flags": list(self.flags)}


def read_run(folder: Path, commits: Sequence[str], n_train_seeds: int, cfg=None) -> Run:
    """One run folder's provenance, checked against the E2.1 definition (module docstring, step 1)."""
    run = Run(Path(folder))
    try:
        data = json.loads((run.folder / "comparison.json").read_text())
    except (OSError, ValueError) as exc:
        run.reasons.append(f"no readable comparison.json ({type(exc).__name__})")
        return run
    prov, args = data.get("provenance") or {}, data.get("run_arguments") or {}
    run.commit, run.dirty = prov.get("git_commit"), prov.get("git_dirty")
    run.hostname, run.gpu, run.timestamp = prov.get("hostname"), prov.get("gpu"), data.get("timestamp")
    run.results = data.get("results") or {}
    run.seed = args.get("seed")
    train_seeds = (data.get("robustness") or {}).get("train_seeds")
    run.n_train_seeds = None if train_seeds is None else len(train_seeds)
    method_json = run.folder / "cma_es.json"
    try:
        run.dtype = json.loads(method_json.read_text()).get("dtype")
    except (OSError, ValueError):
        run.dtype = None

    if not run.commit or not any(run.commit.startswith(c) for c in commits):
        run.reasons.append(f"commit {str(run.commit)[:9]} not in {list(commits)}")
    robot = args.get("robot", (data.get("config") or {}).get("name"))
    if robot != "legobot" or (data.get("config") or {}).get("name", "legobot") != "legobot":
        run.reasons.append(f"robot {robot}")
    if run.n_train_seeds != n_train_seeds:
        run.reasons.append(f"{run.n_train_seeds} training seeds, not {n_train_seeds}")
    if args.get("seeds") is not None and args.get("seeds") != run.n_train_seeds:
        run.reasons.append(f"--seeds {args.get('seeds')} but {run.n_train_seeds} training seeds recorded")
    for key, want in COMMON_ARGS.items():
        if args.get(key, want if key == "robot" else None) != want:
            run.reasons.append(f"--{key.replace('_', '-')} {args.get(key)!r}, E2.1 has {want!r}")
    if run.dtype not in (None, "float64"):
        run.reasons.append(f"searched in {run.dtype}")
    if data.get("robust_checked_on") != "dev":
        run.reasons.append(f"robust checks on {data.get('robust_checked_on')!r}, not dev")
    run.recipe = next((name for name, want in RECIPES.items()
                       if all(args.get(k) == v for k, v in want.items())), None)
    if run.recipe is None:
        run.reasons.append("not an E2.1 recipe (cma_x0 {}, de {} x {})".format(
            args.get("cma_x0"), args.get("de_iters"), args.get("de_pop")))
    missing = [e for e in (MEAN, BEST) if e not in run.results]
    if missing:
        run.reasons.append(f"no {', '.join(missing)} entry")

    if run.dirty is None:
        run.flags.append("git_dirty unknown")
    elif run.dirty:
        run.flags.append("trained from a dirty tree")
    if cfg is not None and isinstance(data.get("config"), dict):
        drift = cfg.scoring_differences(data["config"], ("learning",))
        if drift:
            run.flags.append("trained under a different config: " + ", ".join(drift))
    if run.hostname is None:
        run.flags.append("no hostname in provenance (pre-ecb666e run)")
    return run


def collect_runs(root: Path, commits: Sequence[str] = DEFAULT_COMMITS, n_train_seeds: int = DEFAULT_TRAIN_SEEDS,
                 cfg=None) -> Tuple[List[Run], List[Run]]:
    """(E2.1 runs, everything else) among root's run folders, each sorted by folder name."""
    kept, excluded = [], []
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        run = read_run(folder, commits, n_train_seeds, cfg)
        (excluded if run.reasons else kept).append(run)
    return kept, excluded


def design_problems(runs: Sequence[Run], run_seeds: Sequence[int] = RUN_SEEDS) -> List[str]:
    """Why the kept runs aren't E2.1's full design (one run per recipe x run seed, all clean)."""
    problems = []
    want = len(RECIPES) * len(run_seeds)
    if len(runs) != want:
        problems.append(f"{len(runs)} E2.1 runs found, expected {want}")
    cells: Dict[Tuple[str, int], List[str]] = {}
    for r in runs:
        cells.setdefault((r.recipe, r.seed), []).append(r.name)
    for recipe in RECIPES:
        for s in run_seeds:
            names = cells.get((recipe, s), [])
            if not names:
                problems.append(f"missing {recipe} seed {s}")
            elif len(names) > 1:
                problems.append(f"{len(names)} runs for {recipe} seed {s}: {', '.join(names)}")
    for (recipe, s), names in cells.items():
        if s not in run_seeds:
            problems.append(f"unexpected run seed {s} ({recipe}): {', '.join(names)}")
    for r in runs:
        for flag in r.flags:
            if not flag.startswith("no hostname"):
                problems.append(f"{r.name}: {flag}")
    return problems


@dataclass
class Candidate:
    label: str
    run: str            # folder name
    entry: str          # comparison.json entry
    gains: Tuple[float, ...]
    train_objective: Optional[float] = None
    recorded_dev_objective: Optional[float] = None   # run_batch_tuning's robust_validation_objective (dev, 60 s)
    recorded_dev_falls: Optional[int] = None


def gains_of(entry: dict) -> Tuple[float, ...]:
    g = entry["gains"]
    return tuple(float(g[k]) for k in GAIN_KEYS if k in g) if isinstance(g, dict) else tuple(map(float, g))


def run_entries(run: Run, best: str = "cma-sample", include: Sequence[str] = ()) -> List[str]:
    """The comparison.json entries a run contributes (module docstring, step 2)."""
    if best == "cma-sample":
        best_entry = BEST
    elif best == "overall":
        searched = [e for e in SEARCH_ENTRIES if e in run.results]
        best_entry = min(searched, key=lambda e: run.results[e]["train_objective"])
    else:
        raise ValueError(f"unknown --best {best!r}")
    return list(dict.fromkeys([MEAN, best_entry, *[e for e in include if e in run.results]]))


def candidates_of(runs: Sequence[Run], best: str = "cma-sample", include: Sequence[str] = ()) -> List[Candidate]:
    out = []
    taken = {(r.recipe, r.seed) for r in runs}
    dup = {k for k in taken if sum((r.recipe, r.seed) == k for r in runs) > 1}
    for r in runs:
        for entry in run_entries(r, best, include):
            e = r.results[entry]
            label = f"PID, {ENTRY_TAG.get(entry, entry + ' {r}').format(r=r.recipe)} / s{r.seed}"
            if (r.recipe, r.seed) in dup:
                label += f" [{r.name}]"
            out.append(Candidate(label, r.name, entry, gains_of(e), e.get("train_objective"),
                                 e.get("robust_validation_objective"), e.get("robust_validation_falls")))
    return out


# ---- statistics (compare_controllers' machinery) -------------------------------------------------

def stage_stats(scores: Dict[str, dict], labels: Sequence[str], report: dict, n_boot: int) -> Dict[str, dict]:
    """Per controller: compare_controllers.summarize's run numbers against model-based, plus median cost and
    the paired bootstrap CI of the median per-scenario ratio (crossed_bootstrap with one run)."""
    from compare_controllers import crossed_bootstrap

    ref = scores[MB]["costs"]
    out = {}
    for label in labels:
        s, r = scores[label], report["runs"][label]
        ratio = s["costs"] / ref
        ci = crossed_bootstrap(ratio[None, :], n_boot) if label != MB else {"low": 1.0, "high": 1.0, "n_boot": 0}
        out[label] = {"objective": r["objective"], "mean_cost": r["mean_cost"],
                      "median_cost": float(np.median(s["costs"])), "worst_cost": r["worst_cost"],
                      "falls": r["falls"], "scenarios": r["scenarios"], "fall_rate_95ci": r["fall_rate_95ci"],
                      "median_ratio_vs_model_based": r["median_ratio"], "ratio_95ci": [ci["low"], ci["high"]],
                      "scenarios_better": r["scenarios_better"], "mcnemar_vs_model_based": r["mcnemar_vs_reference"]}
    return out


def rank_dev(stats: Dict[str, dict]) -> List[str]:
    """Labels by dev objective (lower first), ties to the lower median cost, then the label."""
    return sorted(stats, key=lambda k: (stats[k]["objective"], stats[k]["median_cost"], k))


def pick_finalists(ranked: Sequence[str], tuned: Sequence[str], n: int) -> List[str]:
    tuned = set(tuned)
    return [k for k in ranked if k in tuned][:n]


def shows_more_falls(mcnemar: dict, alpha: float = ALPHA) -> bool:
    """D3 (b), operationalized: more discordant falls than model-based AND exact two-sided McNemar p < alpha."""
    return mcnemar["only_this_fell"] > mcnemar["only_reference_fell"] and mcnemar["p_value"] < alpha


def d3_decide(confirm: Dict[str, dict], finalists_dev_order: Sequence[str], alpha: float = ALPHA) -> dict:
    """D3 on the confirm statistics. finalists_dev_order: the finalists, best dev objective first; the
    adopted finalist is the first qualifying one in THAT order (confirm numbers never break the tie)."""
    verdicts = {}
    for label in finalists_dev_order:
        s = confirm[label]
        ci_below = s["ratio_95ci"][1] < 1.0
        more = shows_more_falls(s["mcnemar_vs_model_based"], alpha)
        reasons = []
        if not ci_below:
            reasons.append(f"ratio CI upper {s['ratio_95ci'][1]:.4f} >= 1")
        if more:
            m = s["mcnemar_vs_model_based"]
            reasons.append(f"more falls ({m['only_this_fell']} vs {m['only_reference_fell']} discordant, "
                           f"p = {m['p_value']:.3g} < {alpha:g})")
        verdicts[label] = {"ratio_ci_below_1": ci_below, "shows_more_falls": more,
                           "qualifies": ci_below and not more, "why_not": reasons}
    qualifying = [k for k in finalists_dev_order if verdicts[k]["qualifies"]]
    adopted = qualifying[0] if qualifying else MB
    if qualifying:
        why = (f"{adopted} qualifies under D3" + (f"; {len(qualifying)} qualify, so the best on dev is adopted "
                                                  "(confirm never selects among them)" if len(qualifying) > 1 else ""))
    else:
        why = "no finalist qualifies under D3, so the reference is model-based"
    return {"rule": D3_RULE, "alpha": alpha, "verdicts": verdicts, "qualifying_in_dev_order": qualifying,
            "adopted": adopted, "adopted_is_model_based": adopted == MB, "why": why,
            "n_finalists_tested": len(finalists_dev_order)}


# ---- output ---------------------------------------------------------------------------------------

def _gains_cell(g) -> str:
    return "`[" + ", ".join(f"{x:.4f}" for x in g) + "]`"


def _ratio_cell(s: dict) -> str:
    lo, hi = s["ratio_95ci"]
    return f"`{s['median_ratio_vs_model_based']:.3f} [{lo:.3f}, {hi:.3f}]`"


def markdown(sel: dict) -> str:
    st, dev, conf = sel["settings"], sel["dev"], sel["confirm"]
    cands = {c["label"]: c for c in sel["candidates"]}
    gains = {**{k: c["gains"] for k, c in cands.items()}, **sel["baseline_gains"]}
    lines = [f"### E2.2 run collection ({len(sel['runs'])} runs; commit filter {st['commits']}, "
             f"{st['train_seeds']} training seeds)", "",
             "| Folder | Recipe | Run seed | Commit | Dirty | Host | GPU | Train obj (mean / best) |",
             "|---|---|---:|---|---|---|---|---:|"]
    for r in sel["runs"]:
        mine = [c for c in sel["candidates"] if c["run"] == r["folder"]]
        tr = " / ".join("—" if c["train_objective"] is None else f"{c['train_objective']:.2f}" for c in mine)
        lines.append(f"| `{r['folder']}` | {r['recipe']} | {r['run_seed']} | `{str(r['git_commit'])[:7]}` | "
                     f"{r['git_dirty']} | {r['hostname'] or '—'} | {r['gpu'] or '—'} | {tr} |")
    lines += ["", f"### E2.2 dev pick: {dev['n_scenarios']} scenarios (seeds {dev['seeds'][0]}–{dev['seeds'][1]}), "
              f"{st['duration_s']:g} s, {st['backend']} float64, against model-based",
              "", f"Ranked by robust objective (mean cost with the fall penalty). Finalists: the top {st['n_finalists']} "
              "tuned candidates. Ratio = median per-scenario cost ratio to model-based, paired bootstrap 95% CI "
              "(< 1 favours the candidate); McNemar = exact two-sided p on falls.", "",
              "| Rank | Controller | Gains `[kp, ki, kd, kv_p]` | Objective | Median | Falls | Ratio vs MB `[95% CI]` | McNemar p | Finalist |",
              "|---:|---|---|---:|---:|---:|---|---:|---|"]
    for i, label in enumerate(dev["ranking"], 1):
        s = dev["stats"][label]
        lines.append(f"| {i} | `{label}` | {_gains_cell(gains[label])} | {s['objective']:.2f} | {s['median_cost']:.2f} | "
                     f"{s['falls']}/{s['scenarios']} | {_ratio_cell(s)} | {s['mcnemar_vs_model_based']['p_value']:.3f} | "
                     f"{'**yes**' if label in sel['finalists'] else ''} |")
    d3 = sel["decision"]
    stand_in = "" if conf.get("is_d2_slice", True) else " (dry-run stand-in on dev seeds, NOT the D2 slice)"
    lines += ["", f"### E2.2 confirm (D2/D3): {conf['n_scenarios']} scenarios (seeds {conf['seeds'][0]:,}–"
              f"{conf['seeds'][1]:,}){stand_in}, {st['duration_s']:g} s, same engine", "",
              "| Controller | Dev rank | Mean | Median | Falls | Ratio vs MB `[95% CI]` | McNemar (only this / only MB, p) | CI < 1 | More falls | Qualifies |",
              "|---|---:|---:|---:|---:|---|---|---|---|---|"]
    for label in [*sel["finalists"], MB]:
        s = conf["stats"][label]
        m = s["mcnemar_vs_model_based"]
        v = d3["verdicts"].get(label)
        tail = ("— | — | (default)" if v is None else
                f"{'yes' if v['ratio_ci_below_1'] else 'no'} | {'yes' if v['shows_more_falls'] else 'no'} | "
                f"{'**yes**' if v['qualifies'] else 'no'}")
        lines.append(f"| `{label}` | {dev['ranking'].index(label) + 1} | {s['mean_cost']:.2f} | {s['median_cost']:.2f} | "
                     f"{s['falls']}/{s['scenarios']} | {_ratio_cell(s)} | {m['only_this_fell']} / {m['only_reference_fell']}, "
                     f"{m['p_value']:.3f} | {tail} |")
    lines += ["", f"**Reference PID (D3): `{d3['adopted']}`** {_gains_cell(gains[d3['adopted']])}: {d3['why']}.", "",
              f"*{D3_RULE}*", "", f"*Candidates: {BEST_READING}*"]
    if st["dry_run"]:
        lines.insert(0, f"> DRY RUN ({sel['dev']['n_seeds']} seeds per stage, "
                        f"{st['duration_s']:g} s): not E2.2's result.\n")
    return "\n".join(lines) + "\n"


def write_per_scenario(path: Path, labels, scores, seeds, pitches) -> None:
    scenarios = [(seed, p) for seed in seeds for p in pitches]   # the evaluator's order (compare_controllers)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller", "seed", "start_pitch", "cost", "fell", "fall_time_s"])
        for label in labels:
            s = scores[label]
            for (seed, p), c, fell, ft in zip(scenarios, s["costs"], s["fell"], s["fall_time"]):
                w.writerow([label, seed, p, float(c), int(fell), "" if np.isnan(ft) else float(ft)])


# ---- main -----------------------------------------------------------------------------------------

def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=REPO / "results" / "legobot" / "batch_tuning")
    parser.add_argument("--commit", action="append", default=None,
                        help=f"git commit prefix of the E2.1 runs (repeatable; default {DEFAULT_COMMITS[0]})")
    parser.add_argument("--train-seeds", type=int, default=DEFAULT_TRAIN_SEEDS, help="E2.1's --seeds (D1)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="go on with fewer than 20 runs, duplicates, dirty or config-drifted runs (flagged)")
    parser.add_argument("--best", choices=("cma-sample", "overall"), default="cma-sample",
                        help="each run's 'best': CMA-ES's best sample (default) or the lowest-training-objective search")
    parser.add_argument("--include-entry", action="append", default=[],
                        help="also score this comparison.json entry of every run (e.g. 'differential evolution')")
    parser.add_argument("--n-finalists", type=int, default=DEFAULT_N_FINALISTS)
    parser.add_argument("--seeds", type=int, default=None, help="dry run: only each tier's first N seeds")
    parser.add_argument("--duration", type=float, default=None, help="dry run: run length (default 60 s)")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy",
                        help="one engine for both stages; numpy float64 is compare_controllers' scoring path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    commits = tuple(args.commit) if args.commit else DEFAULT_COMMITS

    os.environ["ROBOT"] = "legobot"
    from batch_tuning import first_train_seeds
    from config import CONFIG, load_config
    from compare_controllers import Entry, score, summarize, tier_seeds
    from tuning import provenance
    if CONFIG.name != "legobot":
        raise SystemExit(f"config was already loaded for {CONFIG.name}; run this script on its own")
    cfg = load_config("legobot")

    robust = CONFIG.build_robustness()
    dev_seeds, confirm_seeds = tier_seeds(robust, "dev"), tier_seeds(robust, "confirm")
    assert dev_seeds == tuple(range(5000, 5050)) and confirm_seeds == tuple(range(6_000_000, 6_000_100))
    dry = bool(args.seeds or args.duration)
    if dry:   # keep the D2 slice fresh: a dry run's confirm stage runs on a stand-in (the dev tier's LAST N seeds)
        n = args.seeds or DRY_DEFAULT_SEEDS
        if not 0 < n <= len(dev_seeds) // 2:
            raise SystemExit(f"--seeds must be 1-{len(dev_seeds) // 2} for a dry run")
        dev_seeds, confirm_seeds = dev_seeds[:n], dev_seeds[-n:]
    held = set(robust.validation_seeds) | set(robust.fall_rate_seeds) | set(robust.retired_seeds) | \
        set(robust.dev_extended_seeds) | set(first_train_seeds(args.train_seeds, robust.held_out_seeds))
    for seeds in (dev_seeds, confirm_seeds):
        assert not set(seeds) & held, "a scored seed lies in a frozen, retired or training tier"
    assert not set(dev_seeds) & set(confirm_seeds)

    eval_config = CONFIG.build_eval_config(validation=True)
    if args.duration:
        from dataclasses import replace
        eval_config = replace(eval_config, duration=args.duration)

    # 1. collection
    runs, excluded = collect_runs(args.runs_root, commits, args.train_seeds, cfg)
    print(f"Run folders under {args.runs_root}: {len(runs)} E2.1 runs, {len(excluded)} left out")
    print(f"{'folder':<18}{'recipe':<7}{'seed':>5}  {'commit':<9}{'dirty':<7}{'host':<18}{'gpu':<40}")
    for r in runs:
        print(f"{r.name:<18}{str(r.recipe):<7}{str(r.seed):>5}  {str(r.commit)[:7]:<9}{str(r.dirty):<7}"
              f"{str(r.hostname)[:17]:<18}{str(r.gpu)[:39]:<40}" + ("  FLAGS: " + "; ".join(r.flags) if r.flags else ""))
    for r in excluded:
        print(f"  left out {r.name}: " + "; ".join(r.reasons))
    problems = design_problems(runs)
    if problems:
        print("Not E2.1's full, clean design:\n  " + "\n  ".join(problems))
        if not args.allow_partial:
            raise SystemExit("refusing to select on these runs (pass --allow-partial to go on, flagged)")
        print("--allow-partial: going on; the problems are recorded in selection.json")
    if not runs:
        raise SystemExit("no E2.1 runs to select from")

    # 2. candidates
    cands = candidates_of(runs, args.best, args.include_entry)
    baselines = {MB: tuple(map(float, CONFIG.model_based_gains())), HT: tuple(map(float, CONFIG.controller.hand_tuned_gains))}
    by_label = {c.label: c for c in cands}
    assert len(by_label) == len(cands) and not set(by_label) & set(baselines)
    entries = [Entry(k, k, "cascaded", np.asarray(g, float)) for k, g in baselines.items()] + \
              [Entry(c.label, c.label, "cascaded", np.asarray(c.gains, float), source=c.run) for c in cands]
    pitches = [float(p) for p in eval_config.start_pitches]

    # 3. dev pick
    print(f"\nDev: {len(entries)} controllers x {len(dev_seeds) * len(pitches)} scenarios x {eval_config.duration:g} s "
          f"({args.backend} float64) ...", flush=True)
    t0 = time.perf_counter()
    dev_scores = score(entries, eval_config, robust, dev_seeds, args.backend, args.device)
    dev_elapsed = time.perf_counter() - t0
    labels = [e.label for e in entries]
    dev_stats = stage_stats(dev_scores, labels, summarize(entries, dev_scores, MB, 0), args.bootstrap)
    ranking = rank_dev(dev_stats)
    finalists = pick_finalists(ranking, [c.label for c in cands], args.n_finalists)
    full_dev = not dry
    recorded = {c.label: (c.recorded_dev_objective, dev_stats[c.label]["objective"]) for c in cands
                if c.recorded_dev_objective is not None}
    max_rec_diff = max((abs(a - b) for a, b in recorded.values()), default=None)
    print(f"  {dev_elapsed:.1f} s")
    print(f"{'rank':>4}  {'controller':<30}{'objective':>10}{'median':>9}{'falls':>8}{'ratio vs MB [95% CI]':>28}{'McNemar p':>11}")
    for i, k in enumerate(ranking, 1):
        s = dev_stats[k]
        lo, hi = s["ratio_95ci"]
        print(f"{i:>4}  {k:<30}{s['objective']:>10.2f}{s['median_cost']:>9.2f}{s['falls']:>4}/{s['scenarios']:<3}"
              f"{s['median_ratio_vs_model_based']:>10.3f} [{lo:.3f}, {hi:.3f}]{s['mcnemar_vs_model_based']['p_value']:>11.3f}"
              + ("  <- finalist" if k in finalists else ""))
    if full_dev and max_rec_diff is not None:
        print(f"  cross-check: largest |dev objective - the run's own robust_validation_objective| = {max_rec_diff:.3g}"
              + ("" if max_rec_diff < 1e-6 else "  (WARNING: the runs' end-of-run dev checks differ from this machine's)"))

    # 4. confirm
    conf_entries = [e for e in entries if e.label in finalists or e.label == MB]
    print(f"\nConfirm{' (dry-run STAND-IN: dev seeds, not the D2 slice)' if dry else ''}: {len(conf_entries)} controllers "
          f"x {len(confirm_seeds) * len(pitches)} scenarios x {eval_config.duration:g} s ...", flush=True)
    t0 = time.perf_counter()
    conf_scores = score(conf_entries, eval_config, robust, confirm_seeds, args.backend, args.device)
    conf_elapsed = time.perf_counter() - t0
    conf_labels = [e.label for e in conf_entries]
    conf_stats = stage_stats(conf_scores, conf_labels, summarize(conf_entries, conf_scores, MB, 0), args.bootstrap)
    decision = d3_decide(conf_stats, finalists)
    print(f"  {conf_elapsed:.1f} s")
    for k in [*finalists, MB]:
        s, v = conf_stats[k], decision["verdicts"].get(k)
        lo, hi = s["ratio_95ci"]
        m = s["mcnemar_vs_model_based"]
        print(f"  {k:<30}mean {s['mean_cost']:8.2f}  median {s['median_cost']:7.2f}  falls {s['falls']:3d}/{s['scenarios']}  "
              f"ratio {s['median_ratio_vs_model_based']:.3f} [{lo:.3f}, {hi:.3f}]  McNemar {m['only_this_fell']}/"
              f"{m['only_reference_fell']} p {m['p_value']:.3f}  "
              + ("(model-based)" if v is None else ("QUALIFIES" if v["qualifies"] else "fails: " + "; ".join(v["why_not"]))))
    adopted_gains = baselines.get(decision["adopted"]) or by_label[decision["adopted"]].gains
    print(f"\nReference PID: {decision['adopted']} {list(adopted_gains)}\n  {decision['why']}")

    # outputs
    stamp = time.strftime("%Y%m%d-%H%M%S") + ("-dry" if dry else "")
    folder = args.out or (REPO / "results" / "legobot" / "e2_selection" / stamp)
    folder.mkdir(parents=True, exist_ok=True)
    selection = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": provenance(),
        "settings": {"commits": list(commits), "train_seeds": args.train_seeds, "allow_partial": args.allow_partial,
                     "best": args.best, "include_entries": args.include_entry, "n_finalists": args.n_finalists,
                     "duration_s": eval_config.duration, "backend": args.backend, "dtype": "float64",
                     "device": args.device if args.backend == "torch" else None, "bootstrap": args.bootstrap,
                     "dry_run": dry, "seeds_override": args.seeds, "start_pitches_rad": pitches,
                     "latency_range_s": list(robust.latency_range) if robust.latency_range else None,
                     "argv": sys.argv[1:] if argv is None else list(argv)},
        "best_reading": BEST_READING,
        "runs": [r.summary() for r in runs],
        "excluded_runs": [{"folder": r.name, "git_commit": r.commit, "reasons": r.reasons} for r in excluded],
        "design_problems": problems,
        "machines": sorted({f"{r.hostname} / {r.gpu}" for r in runs}),
        "candidates": [{"label": c.label, "run": c.run, "entry": c.entry, "gains": list(c.gains),
                        "train_objective": c.train_objective, "recorded_dev_objective": c.recorded_dev_objective,
                        "recorded_dev_falls": c.recorded_dev_falls} for c in cands],
        "baseline_gains": {k: list(g) for k, g in baselines.items()},
        "dev": {"seeds": [dev_seeds[0], dev_seeds[-1]], "n_seeds": len(dev_seeds),
                "n_scenarios": len(dev_seeds) * len(pitches), "elapsed_s": round(dev_elapsed, 2),
                "ranking": ranking, "stats": dev_stats,
                "max_abs_diff_vs_recorded_dev_objective": max_rec_diff if full_dev else None},
        "finalists": finalists,
        "confirm": {"seeds": [confirm_seeds[0], confirm_seeds[-1]], "n_seeds": len(confirm_seeds),
                    "is_d2_slice": not dry,
                    "n_scenarios": len(confirm_seeds) * len(pitches), "elapsed_s": round(conf_elapsed, 2),
                    "stats": conf_stats},
        "decision": decision,
        "reference": {"label": decision["adopted"], "kind": "cascaded", "gains": dict(zip(GAIN_KEYS, adopted_gains)),
                      "source_run": None if decision["adopted"] in baselines else by_label[decision["adopted"]].run,
                      "source_entry": None if decision["adopted"] in baselines else by_label[decision["adopted"]].entry},
        "config": CONFIG.snapshot(),
    }
    (folder / "selection.json").write_text(json.dumps(selection, indent=2, default=float))
    write_per_scenario(folder / "dev_per_scenario.csv", labels, dev_scores, dev_seeds, pitches)
    write_per_scenario(folder / "confirm_per_scenario.csv", conf_labels, conf_scores, confirm_seeds, pitches)
    (folder / "e2_notes_tables.md").write_text(markdown(selection), encoding="utf-8")
    print(f"saved {folder}")
    return folder


if __name__ == "__main__":
    main()
