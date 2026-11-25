# PR #15 audit, prompt 1 of 2: for the window that ran E1

Paste everything below the line into the Antigravity window that ran Phase E1. Run this one first; prompt 2
(`pr15-handoff-2-fresh-window.md`) goes to a fresh window after this one's commit lands.

---

An independent audit of PR #15 (branch `claude/legobot-tick-timing-574623`, head `fae53bb`) found that part of
Phase E1 can't be reproduced from the repository. You ran E1, so you're the only one who knows exactly what
produced those results. This task is **recovery only**: put what you actually ran, and what it produced, into the
repo. A fresh window will handle the interpretation and doc fixes afterwards.

## Do not

- Edit `hardware/legobot/E1_NOTES.md`, `E1_HANDOFF.md`, `A4_BASELINES.md`, `SOFTWARE_PLAN.md`, `DATASHEET.md`,
  `PHYSICS.md`, `docs/`, `legobot_config.py` or `config.py`, or change any conclusion. Another window does that.
- Change any gains, or re-tune anything.
- Score the frozen seed sets (validation 3000–3099, fall-rate 6000–6999) or pass `--final`.
- Push, merge, or open a PR.

## What's missing

1. No committed script produces any of these:
   - `results/legobot/e1_ablations/ablation_summary.json` (Step 2: the 12 variants, their re-tuning, the mean vs
     best-sample pick, the `dev150_60s` / `dev450_20s` cross-evaluation, the bootstrap CIs and McNemar tests);
   - `results/legobot/e1_learning_scales_check.json`;
   - `results/legobot/comparison/e1_staged_gains/` (all the per-method JSON files staged for
     `compare_controllers.py --gain-runs`);
   - the "Fresh Out-of-Sample Confirmation" on seeds 5050–5149 (`E1_NOTES.md` section 3), the 5050–5149 slice of
     `dev450_20s`, and the pre/post-H9 baselines on 5050–5149 in `A4_BASELINES.md:520`. These have no saved output
     at all.
2. `ablation_summary.json` doesn't record each variant's seeds, config overrides or robustness settings.
3. `e1_staged_gains/cma_lat10_15/seed0/cma.json` embeds `robustness.latency_range_s = [0.0075, 0.0125]`, but the
   gains were tuned at (0.010, 0.015) (Step 2's `wider_latency_10_15ms` variant). So `compare_controllers.py`'s
   config-drift warning stays silent for it.
4. `e1_staged_gains/*_mean/seed*/cma_es_mean.json` carry the best sample's `best_training_cost` (e.g. 24.06),
   not a value that belongs to the mean gains.

## Do

1. **Commit the scripts exactly as you ran them**, from your scratch or session folders, under
   `hardware/legobot/` with clear names (for example `e1_ablations.py`, `e1_fresh_check.py`,
   `e1_scales_check.py`, `e1_stage_gains.py`). Fix only what's needed for them to run from the repo root with
   `C:\venvs\balancebot\Scripts\python.exe` (paths, imports). Put a docstring at the top of each: what it
   produces, the command line you used, and its runtime.
2. **Record the inputs for each Step 2 variant**: write `results/legobot/e1_ablations/variants.json` (or add to
   `ablation_summary.json` without changing its existing fields) with, per variant, the exact config overrides,
   the robustness settings (`latency_range_s`, `randomize_range`, seeds for tuning, picking and each
   cross-evaluation), the optimizer seed, and the `CONFIG.snapshot()` the variant was tuned under.
3. **Save the 5050–5149 results**: if the per-scenario costs and falls still exist anywhere, save them as
   `results/legobot/comparison/e1-fresh-5050-5149/per_scenario.csv` plus a `summary.json` with seeds, durations,
   latencies, gains and `provenance()`. If they don't exist, re-run only that scoring (NumPy float64, robust,
   seeds 5050–5149) with the committed script, and say so.
4. **Fix the staged gain files' provenance** without changing any gains: regenerate `e1_staged_gains/` with the
   committed staging script so that each file embeds the config and robustness it was really tuned under
   (`cma_lat10_15` gets `latency_range_s = [0.010, 0.015]`), and the mean files either drop
   `best_training_cost` or carry a clearly named field for the mean.
5. **Check**: if anything you re-ran differs from the numbers in `E1_NOTES.md` or `A4_BASELINES.md` by more than
   chaotic noise, list it. Do not edit the notes.
6. Run `python -m pytest`, `ROBOT=legobot python -m pytest` and
   `python -m pytest hardware/legobot/robot/test_control.py hardware/legobot/robot/measure/test_fit.py`
   (all passed at `fae53bb`: 663 / 534 / 43).
7. Commit on `claude/legobot-tick-timing-574623` in small commits (scripts; variant records; 5050–5149 results;
   staged-file provenance). Don't push.

## Report back (to the user, short)

- Each script committed, and the artifact it produces.
- What you saved from existing output and what you had to re-run.
- Anything you couldn't recover, and any numbers that didn't match.
- The commit hashes.
