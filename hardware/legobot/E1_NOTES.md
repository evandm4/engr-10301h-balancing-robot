# E1 Notes: Development Tuning, Model Ablations, and Latency Sweep on the LEGO Robot

- **Date:** 2026-11-24
- **Branch & revision:** `claude/legobot-tick-timing-574623` (`59065a34` + Round-2/3/4 pre-E1 fixes, `docs/legobot-review/review-v2.md`)
- **Environment:** `C:\venvs\balancebot` (Python 3.12.10, NumPy 2.5.3, SciPy 1.18.1, `cma` 4.5.0, PyTorch 2.11.0+cu128)
- **GPU & power budget:** NVIDIA GeForce RTX 5070 Ti Laptop GPU at an enforced **`33.0 W`** power limit (`keep_awake_ok: true`). All LEGO searches ran strictly in **`float64`** (no `--float32`), achieving **`6.8 evals/s`** at `popsize = 64` and **`12.3 evals/s`** at `popsize = 256` (`150` scenarios × `20 s` per evaluation, `dt = 2.5 ms`).
- **Config under test:** Post-Round-2 `legobot_config.py` (`gearbox_backlash` `±1.3°` nominal / `±1.1–1.6°` randomized, `coupling_backlash` `±1.5°` nominal / `±0–3.0°` randomized, `latency_range_s = (0.0075, 0.0125)` with free `pwm_tick_phase`, pushes `0.73 N × 50 ms` at `5.5 s` and `7.5 s`, `ROBUSTNESS.sensitivity_weight = 0.0`, `SEARCH.ki_max = 10.0`, post-`H9`/`H11`).
- **Seed hygiene:** Strictly development tiers only — `train` (`1–50`, 150 scenarios × 20 s), `dev` (`5000–5049`, 150 scenarios × 20 s or 60 s), and extended `dev` (`5050–5149`, giving a 450-scenario `5000–5149` 20 s slice for Step 2 cross-evaluation). The frozen **`validation` (`3000–3099`)** and **`fall-rate` (`6000–6999`)** sets were **never touched** (`--final` was never passed).

---

## 1. Step 1 — Default Robust Float64 Tuning (DE, CMA-ES Midpoint vs Warm-Start)

Manifest: `results/legobot/e1_step1_manifest.json` (`03:03:22`–`04:17:20`, `4,428.8 s` / `73.8 min` total across 4 runs).

### Commands, Budgets, and Wall Clock

| Run label | Folder (`results/legobot/batch_tuning/`) | Command | Evals (`DE / CMA / RS`) | Wall clock |
|---|---|---|---|---:|
| `full_seed0_midpoint` | `20261124-032414` | `python optimization/run_batch_tuning.py --robot legobot --backend torch --no-show --seed 0 --cma-x0 midpoint` | `2,976 / 3,600 / 3,600` | `1,255.5 s` (`20.9 min`) |
| `full_seed1_midpoint` | `20261124-034505` | `python optimization/run_batch_tuning.py --robot legobot --backend torch --no-show --seed 1 --cma-x0 midpoint` | `2,976 / 3,600 / 3,600` | `1,248.1 s` (`20.8 min`) |
| `warm_seed0_model_based` | `20261124-040107` | `python optimization/run_batch_tuning.py --robot legobot --backend torch --no-show --seed 0 --cma-x0 model-based --de-iters 2 --de-pop 6` | `72 / 3,600 / 3,600` | `960.4 s` (`16.0 min`) |
| `warm_seed1_model_based` | `20261124-041714` | `python optimization/run_batch_tuning.py --robot legobot --backend torch --no-show --seed 1 --cma-x0 model-based --de-iters 2 --de-pop 6` | `72 / 3,600 / 3,600` | `964.9 s` (`16.1 min`) |

*(DE defaults: `de_iters = 30, de_pop = 24`; CMA-ES defaults: `cma_iters = 30, cma_pop = 120`. The two warm-start runs used a `72`-eval DE stub so compute went to CMA-ES from `--cma-x0 model-based`; their `random search` rows repeat seeds `0` and `1` identically.)*

### Full Step 1 Results (NumPy Float64 Re-Scoring)

| Run | Candidate | Gains `[kp, ki, kd, kv_p]` | Train 20 s obj (`1–50`) | Plain 60 s (`run_tuning.py`, seeds `100–109`, retired; `30` runs) | Dev 60 s obj (`5000–5049`) | Dev 60 s falls (`/150`) | Dev 60 s worst |
|---|---|---|---:|---:|---:|---:|---:|
| *Baselines* | `hand-tuned` | `[2.5000, 1.5000, 0.2000, 0.5000]` | `115.73` | `54.19 (0)` | `217.91` | `13` | `1948.8` |
| *Baselines* | `model-based` | `[1.9797, 5.2360, 0.1509, 0.5153]` | `93.41` | `28.81 (0)` | `105.31` | `6` | `1998.7` |
| `20261124-032414` | `DE` (seed 0) | `[2.3284, 5.5560, 0.1365, 0.4201]` | `58.09` | `31.42 (0)` | `117.74` | `7` | `1998.2` |
| `20261124-032414` | `CMA-ES` best (mid, s0) | `[2.0618, 7.4220, 0.1457, 0.4307]` | `69.08` | `30.52 (0)` | `92.83` | `5` | `1998.7` |
| `20261124-032414` | `CMA-ES (mean)` (mid, s0) | `[2.0953, 7.3573, 0.1511, 0.4342]` | `69.50` | `31.23 (0)` | `93.93` | `5` | `1998.7` |
| `20261124-032414` | `Random` (seed 0) | `[2.9929, 8.2189, 0.1787, 0.3602]` | `76.75` | `43.39 (0)` | `79.74` | `3` | `1935.0` |
| `20261124-034505` | `DE` (seed 1) | `[2.1739, 8.8011, 0.1503, 0.3825]` | `47.96` | `33.56 (0)` | `107.54` | `6` | `1998.4` |
| `20261124-034505` | `CMA-ES` best (mid, s1) | `[2.0764, 6.0194, 0.1418, 0.4598]` | `23.90` | `29.14 (0)` | `104.49` | `6` | `1998.6` |
| `20261124-034505` | `CMA-ES (mean)` (mid, s1) | `[2.1204, 6.9571, 0.1475, 0.4391]` | `46.79` | `30.67 (0)` | `93.45` | `5` | `1998.7` |
| `20261124-034505` | `Random` (seed 1) | `[2.3278, 3.1778, 0.1438, 0.4100]` | `71.36` | `31.58 (0)` | `107.17` | `6` | `1997.7` |
| `20261124-040107` | `DE` stub (`72 evals`, s0) | `[1.8278, 5.9389, 0.1538, 0.5374]` | `59.00` | `29.42 (0)` | `119.57` | `7` | `1999.2` |
| `20261124-040107` | `CMA-ES` best (warm, s0) | `[2.2381, 6.6895, 0.1375, 0.4227]` | `24.08` | `31.00 (0)` | `117.35` | `7` | `1998.5` |
| `20261124-040107` | `CMA-ES (mean)` (warm, s0) | `[2.1866, 7.1457, 0.1476, 0.4262]` | `58.95` | `31.36 (0)` | `93.75` | `5` | `1998.6` |
| `20261124-041714` | `DE` stub (`72 evals`, s1) | `[4.5983, 8.5718, 0.2351, 0.4582]` | `156.15` | `123.74 (0)` | `234.68` | `10` | `1958.3` |
| `20261124-041714` | `CMA-ES` best (warm, s1) | `[2.0067, 3.3848, 0.1510, 0.5026]` | `58.09` | `28.14 (0)` | `93.26` | `5` | `1998.0` |
| `20261124-041714` | `CMA-ES (mean)` (warm, s1) | `[2.0173, 3.9543, 0.1641, 0.5388]` | `59.41` | `31.49 (0)` | `83.91` | `4` | `1997.9` |

*(No candidate hit any search bound: `at_search_bound == []` for every row. The "Plain 60 s" column is `run_tuning.py`'s nominal-robot scoring on `ScenarioConfig.validation_seeds = 100–109` — retired from robust tiers, printed automatically by existing `run_batch_tuning.py` `ref val` behaviour and never used for selection.)*

### Key Step 1 Findings
1. **No Step 1 candidate beats the untuned `model-based` PID out of sample:** On `300` fresh `60 s` runs (`5050–5149`, NumPy float64, Section 3), `model-based` scores `mean = 60.11`, `5/300` falls, `median = 27.80`, whereas every Step 1 full-budget candidate has at least as many falls (`5–13/300`, `mean = 58.87–110.81`, all McNemar $p \ge 0.057$) and typical tracking cost equal within noise (e.g., `5000–5049`-selected `CMA-mid / seed0` scores `mean = 72.01`, `7/300` falls, `med = 27.33`, paired ratio `0.985 [0.976, 0.993]` vs `model-based` on exact JSON gains, or `72.2` / `7/300` / `27.59` on 4-d.p. rounded table gains; 2-run `model-based` vs `CMA-mid (mean)` ratio `0.991 [0.972, 1.010]`; vs `CMA-warm (mean)` `0.931 [0.875, 0.994]`, favoring `model-based`). `CMA-mid`'s `1.046 [1.006, 1.084]` edge in Section 3 holds on the selection seeds `5000–5049` only.
2. **`H1` (`SEARCH.ki_max = 10.0`) removed an active clip (`K4`):** **`8 of 10`** full-budget non-random optimizer candidates (`2` DE + `4` CMA-mid best/mean + `2` CMA-warm seed 0 best/mean) converged to **`ki = 5.56–8.80`** — well above the old `ki_max = 5.0` bound and cleanly inside `10.0`. However, `warm_seed1` landed in a distinct lower-`ki` basin (`3.38` best / `3.95` mean, with `83.91` / `4` falls on `dev150_60s`), inside the old box; thus `ki_max = 10.0` prevented boundary clipping rather than proving `ki > 5` strictly beats `ki < 5`.
3. **`cma-es (mean)` (`xfavorite`) showed lower across-seed dev variance (`K5`):** Across the four CMA-ES runs, `cma-es (mean)` spanned `83.91–93.93` (`4–5/150` falls) on `dev150_60s`, compared with `92.83–117.35` (`5–7/150` falls) for single-sample `cma-es` best (whose `full_seed1` and `warm_seed0` train objectives of `23.90` and `24.08` regressed to `104.49` and `117.35` on dev). Note that a `1–2` fall difference sits inside the `±4`-fall chaotic band, and on paired median tracking cost `CMA-mid` best sample slightly edged `CMA-mid (mean)` (`0.963 [0.921, 0.999]`, Section 3).
4. **`J2` chaotic-variance lesson (`Random Search` seed 0):** Although `Random` seed 0 posted the lowest single-tier `dev150_60s` mean (`79.74`, `3/150` falls), its typical tracking cost was **`+48.6%` worse** than `full_model` (`median = 41.85` vs `27.82`, 95% paired CI `[1.473, 1.520]`, plain 60 s `43.39` vs `30.52–31.49`), and on the larger 450-scenario dev slice (`5000–5149`, 20 s) it fell **`12/450`** (`3` on selection seeds `5000–5049`, `9` on fresh seeds `5050–5149`; `mean = 63.02`, median ratio `1.518`), trailing `full_model` (`10/450`, `mean = 52.41`, McNemar `p = 0.824`). Never pick a finalist from a single 150-scenario fall count without inspecting median cost and a wider out-of-sample dev slice.

---

## 2. Step 2 — Model Ablations (Which Models Matter for E2?)

Artifact: `results/legobot/e1_ablations/ablation_summary.json` (`2026-11-24 05:09:49`).

Each of the 12 variants ran:
1. Fixed-controller rescore on the variant (`dev150_20s`, `5000–5049`) for `hand-tuned`, `model-based`, and `step1_best` (`Random / seed0`),
2. Re-tuning on the variant with float64 GPU CMA-ES (`max_iter = 25, popsize = 96, n_evals = 2,400`, `x0 = model-based`, picking whichever of `mean` or `best_sample` scored lower on the variant's own `dev150_20s`), and
3. Cross-evaluation of the variant's picked gains on the **true full realistic LEGO model** across both `dev150_60s` (`5000–5049`, 60 s) and `dev450_20s` (`5000–5149`, 20 s), with paired bootstrap 95% CIs on median cost ratio and exact McNemar $p$-values against the `full_model` control.

### Complete 12-Variant Ablation Table

| Variant | Wall clock | Fixed rescore on variant (`model-based`, dev20) | Re-tuned gains `[kp, ki, kd, kv_p]` (pick) | Own dev20 cost (falls) | True full `dev150_60s` ratio `[95% CI]` (falls) | True full `dev450_20s` mean (falls `/450`) | `dev450` McNemar $p$ vs `full_model` |
|---|---:|---:|---|---:|---|---:|---:|
| **`full_model`** *(control)* | `280.0 s` | `83.35 (6)` | `[2.157, 6.994, 0.144, 0.432]` (`mean`) | `74.77 (5)` | `1.000` (`5`) | `52.41` (`10`, `2.2%`) | `1.0000` |
| `ideal_pitch_sensor` | `227.6 s` | `56.91 (4)` | `[1.015, 2.298, 0.084, 0.325]` (`best`) | `6.15 (0)` | `1.065 [0.996, 1.309]` (`63`) | `861.57` (**`199`**, `44.2%`) | **`2.5e-57`** |
| `continuous_velocity` | `255.0 s` | `33.85 (2)` | `[1.355, 2.152, 0.164, 0.792]` (`best`) | `19.64 (1)` | `1.235 [1.204, 1.279]` (`24`) | `300.67` (**`73`**, `16.2%`) | **`3.1e-17`** |
| `no_backlash` | `169.4 s` | `6.60 (0)` | `[1.406, 4.032, 0.115, 0.435]` (`mean`) | `5.22 (0)` | `0.762 [0.747, 0.776]` (`20`) | `286.33` (**`64`**, `14.2%`) | **`2.6e-13`** |
| `averaged_slow_pwm` | `255.4 s` | `44.60 (3)` | `[2.472, 4.147, 0.157, 0.561]` (`best`) | `6.55 (0)` | `1.231 [1.208, 1.254]` (`6`) | `109.81` (**`25`**, `5.6%`) | **`0.0167`** |
| `fixed_tick_phase` | `260.8 s` | `116.98 (9)` | `[2.040, 4.586, 0.139, 0.500]` (`best`) | `70.34 (5)` | `0.952 [0.942, 0.963]` (`9`) | `82.22` (`18`, `4.0%`) | `0.0574` |
| `coupling_play_only` | `261.4 s` | `46.19 (3)` | `[2.909, 4.820, 0.224, 0.587]` (`best`) | `29.00 (1)` | **`2.796 [2.551, 2.994]`** (`5`) | `93.22` (`17`, `3.8%`) | `0.2478` |
| `gearbox_play_only` | `261.0 s` | `85.03 (6)` | `[2.154, 6.260, 0.144, 0.413]` (`mean`) | `75.15 (5)` | `1.004 [0.999, 1.011]` (`8`) | `64.23` (`13`, `2.9%`) | `0.5488` |
| `exact_feedforward` | `260.7 s` | `22.10 (1)` | `[2.374, 3.696, 0.179, 0.544]` (`best`) | `26.26 (1)` | **`1.411 [1.381, 1.450]`** (`1`) | `50.13` (`9`, `2.0%`) | `1.0000` |
| `zero_pitch_offset` | `261.1 s` | `61.73 (4)` | `[2.358, 7.215, 0.150, 0.425]` (`mean`) | `48.72 (3)` | `1.086 [1.069, 1.094]` (`5`) | `52.36` (`10`, `2.2%`) | `1.0000` |
| `wider_play_bounds` (`I1`/`J5`) | `261.6 s` | `98.87 (7)` | `[2.115, 8.037, 0.161, 0.441]` (`mean`) | `99.70 (7)` | `1.124 [1.110, 1.143]` (`5`) | `52.90` (`10`, `2.2%`) | `1.0000` |
| `wider_latency_10_15ms` (`H4`) | `261.3 s` | `83.07 (6)` | `[2.374, 6.791, 0.156, 0.456]` (`best`) | `25.64 (1)` | `1.122 [1.106, 1.138]` (**`0`**) | `50.51` (`10`, `2.2%`) | `1.0000` |

*(Reference baselines cross-evaluated identically on the true full model: `hand-tuned` gave `dev150_60s` ratio `1.873 [1.772, 1.983]` / `13` falls, `dev450_20s` `129.22` / `28` falls / `p = 0.0029`; `model-based` gave `dev150_60s` ratio `1.009 [1.002, 1.026]` / `6` falls, `dev450_20s` `54.77` / `11` falls / `p = 1.0000`.)*

### Ranking of Models for Phase E2

*(Statistical & selection-bias caveat (`K2`, `L5`): each Step 2 variant used `1` optimizer seed (`seed = 0`) and picked between its CMA-ES `mean` and `best_sample` on `dev 5000–5049`; the `full_model` control therefore picked on the full model using those **same** `5000–5049` seeds, which are then reused inside the cross-evaluation (`dev150_60s`, and `150` of the `450` scenarios in `dev450_20s`), slightly favoring `full_model`. Moreover, McNemar's test conditions on fixed gains without accounting for optimizer-seed variance. Across `11` comparisons against `full_model`, Bonferroni-adjusted $\alpha \approx 0.0045$: the first three Tier-1 models (`ideal_pitch_sensor`, `continuous_velocity`, `no_backlash`) clear Bonferroni by many orders of magnitude and survive easily, whereas PDM ($p = 0.0167$) and free tick phase ($p = 0.0574$) do not clear `0.0045` — they are kept on physical grounds with suggestive ablation evidence.)*

- **Tier 1 — Essential (keep in E2):**
  1. **IMU specific-force + complementary filter (`ideal_pitch_sensor`, $p = 2.5 \times 10^{-57}$):** Omitting collapses inner gains to `[1.01, 2.30, 0.084, 0.325]`, causing **`199/450` falls (`44.2%`)** when evaluated with the real accelerometer tilt lag.
  2. **360-count / 100 Hz / 5 ms delayed encoder (`continuous_velocity`, $p = 3.1 \times 10^{-17}$):** Omitting drives `kv_p` up to `0.792` (nearly `2×` full-model `0.432`), causing **`73/450` falls (`16.2%`)** and `+23.7%` median cost on the real encoder.
  3. **Drivetrain backlash (`no_backlash`, $p = 2.6 \times 10^{-13}$):** Omitting softens inner gains to `[1.41, 4.03, 0.115, 0.435]`, causing **`64/450` falls (`14.2%`)**. Specifically, **upstream motor-to-encoder gearbox play (across which motor motion is unseen by the encoder)** is the dominant driver (`K6`): `gearbox_play_only` converges to nearly identical gains (`[2.15, 6.26, 0.144, 0.413]`) and matches `full_model` (`dev150_60s` ratio `1.004 [0.999, 1.011]`, `dev450` `p = 0.549`), whereas training on **downstream shaft-to-wheel coupling play** alone (`coupling_play_only`) over-tunes `kp = 2.91, kd = 0.224` and raises full-model `dev150_60s` median cost by **`2.80×` (`[2.551, 2.994]`)** (`2.33×` on `dev450`).
  4. **10 ms slow-PWM pulse-density modulation (`averaged_slow_pwm`, $p = 0.0167$) & free tick phase (`fixed_tick_phase`, $p = 0.0574$):** Directionally clear / suggestive at 1 optimizer seed (`25/450` falls and `+29.4%` median cost when PDM is omitted; `18/450` falls when tick phase is held fixed at `0.25`) and kept on direct physical-fidelity grounds (Bench B1 showed ticks do not lock). Re-running with `2–3` tuning seeds would sharpen the exact fall-rate delta.
- **Tier 2 — Moderate tracking / robustness refinements (keep for E2):**
  - **Feedforward constant mismatch spreads (`exact_feedforward`):** Omitting drives gains to higher `kd = 0.179` / lower `ki = 3.70`, inflating `dev150_60s` median cost by **`+41.1%` (`[1.381, 1.450]`)** (`+35.9%` on `dev450`).
  - **Wider sample-to-HAT latency `(0.010, 0.015)` (`wider_latency_10_15ms`, `H4`):** On selection seeds `5000–5049` (`dev150_60s`), `best_sample` scored `0/150` falls (`mean = 31.57`, `+12.2%` median cost `[1.106, 1.138]`), **but** on `dev450_20s` it matched `full_model` (`10/450`, `mean = 50.51`, `p = 1.000`) and on fresh `5050–5149` 60 s seeds it fell `10/300` (Section 3, `K1`). Worth testing across multiple seeds in E2, not proven superior from one seed.
  - **Pitch-bias spread (`zero_pitch_offset`):** Omitting adds **`+8.6%`** median cost (`[1.069, 1.094]`) with identical fall rate (`10/450`, `p = 1.000`).
- **Two-way `wider_play_bounds` sensitivity check (`gearbox ±1.0–2.0°, coupling 0–4.35°`; `I1`/`J5`/`K3`):**
  - *Wider-tuned -> nominal full model:* identical fall counts (`5/150` on `dev150_60s`, `10/450` on `dev450_20s`, `mean = 52.90` vs `52.41`, McNemar `p = 1.0000`, `+10.7–12.4%` median cost from higher `ki = 8.04`).
  - *Reverse direction (nominal-tuned -> `wider_play_bounds` `dev450_20s`, `5000–5149`):* `full_model`-tuned gains scored **`mean = 65.46`, `median = 12.11`, `13/450` falls**, vs `wider_play_bounds`-tuned (`85.14` / `13.38` / `18/450`) and `model-based` (`84.18` / `18/450`) — showing no degradation beyond chaotic noise (`±5–7/450` falls). Wider play itself raises falls modestly (`10/450 -> 13/450` for `full_model`-tuned, `11/450 -> 18/450` for `model-based`), so a quick wheel-on visual play check remains worthwhile alongside Bench B3/C1/C3.

---

## 3. Step 3 — Formal Dev Comparison & Wide Latency Sweep (`7.5–40 ms`)

Artifacts: `results/legobot/comparison/e1-dev-sweep/comparison.json`, `latency_sweep.csv`, and archived staged gain JSONs in `results/legobot/comparison/e1_staged_gains/` (`compare_controllers.py --robot legobot --tier dev --latency-sweep`, `150` dev scenarios `5000–5049 × 60 s`, NumPy float64, `263.0 s`).

### Headline Dev Comparison & Side Metrics (`150 × 60 s`, Selection Seeds `5000–5049`, Reference = `PID, hand-tuned`)

| Method (runs) | Headline median ratio `[95% CI]` | Falls (`/150`, min–max) | McNemar $p$ vs hand-tuned | Median `dU/tick` (N m) | Median `TV/s` (N m/s) | Median `U rms` (N m) | Worst sat frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PID, hand-tuned` (1) | `1.000` | `13` (`8.7%`) | `1.000` | `0.172` | `17.54` | `0.232` | `7.13%` |
| `PID, model-based` (1) | `0.543` | `6` (`4.0%`) | `0.092` | `0.113` | `11.31` | `0.149` | `45.20%` |
| `PID, DE` (2) | `0.556 [0.523, 0.590]` | `6–7` (`4.0–4.7%`) | `0.118–0.210` | `0.099–0.101` | `9.89–10.21` | `0.135–0.142` | `41.3–46.7%` |
| **`PID, CMA-mid`** (2) | **`0.519 [0.496, 0.546]`** | `5–6` (`3.3–4.0%`) | `0.057–0.143` | `0.101–0.102` | `10.03–10.25` | `0.137–0.139` | `48.4–49.6%` |
| `PID, CMA-mid (mean)` (2) | `0.539 [0.516, 0.567]` | `5–5` (`3.3%`) | `0.077` | `0.105–0.107` | `10.39–10.52` | `0.142–0.145` | `49.6–50.0%` |
| `PID, CMA-warm` (2) | `0.539 [0.515, 0.567]` | `5–7` (`3.3–4.7%`) | `0.077–0.146` | `0.098–0.112` | `10.05–11.17` | `0.135–0.147` | `40.1–43.8%` |
| `PID, CMA-warm (mean)` (2) | `0.586 [0.534, 0.627]` | `4–5` (`2.7–3.3%`) | `0.022–0.057` | `0.105–0.130` | `10.45–12.95` | `0.143–0.166` | `43.4–48.4%` |
| `PID, Random` (2) | `0.681 [0.549, 0.810]` | `3–6` (`2.0–4.0%`) | `0.021–0.092` | `0.100–0.138` | `10.18–14.14` | `0.138–0.190` | `7.6–36.6%` |
| `PID, CMA-lat10-15` (1)* | `0.589` | `0` (`0.0%`, CI `0–2.4%`)* | `0.00024`* | `0.120` | `12.19` | `0.156` | `0.02%`* |

*\*Selection-bias caveat (`K1`): `CMA-lat10-15`'s `best_sample` was selected over its `mean` **on `dev 5000–5049`**, so its `0/150` (`p = 0.00024`, `worst_cost = 47.61`, `peak_pitch <= 31.2°`) and `15.0 ms` threshold on `5000–5049` carry dev-selection bias (just like `Random / seed0`'s `3/150`). See the fresh out-of-sample `5050–5149` table below. Median torque saturation is `0.0%` for every tuned PID; high worst-scenario saturation occurs only on falling trajectories.*

### Key Paired Method-vs-Method Bootstrap 95% CIs (Ratio `< 1` favors first method)
- `PID, model-based` vs `PID, CMA-mid`: **`1.046 [1.006, 1.084]`** (holds on selection seeds `5000–5049` only; out of sample on `5050–5149` no Step 1 candidate beats `model-based` — see below).
- `PID, DE` vs `PID, CMA-mid`: **`1.072 [1.016, 1.123]`** (`CMA-mid` outperforms `DE` by ~6.7%).
- `PID, CMA-mid` vs `PID, CMA-mid (mean)`: `0.963 [0.921, 0.999]`.
- `PID, CMA-mid` vs `PID, CMA-lat10-15`: **`0.881 [0.850, 0.910]`** (`CMA-mid` has ~12% lower typical tracking cost).
- `PID, CMA-warm (mean)` vs `PID, CMA-lat10-15`: `0.995 [0.917, 1.069]` (indistinguishable in typical cost).

### Wide Latency Sweep Summary (`7.5–40.0 ms`, All 15 Controllers on `150 × 60 s` Dev `5000–5049`)

Falls out of `150` at each fixed sample-to-HAT latency (note: free 10 ms tick phase still adds 0–5 ms closed-loop wait on top):

| Controller | `7.5` | `10.0` | `12.5` | `15.0` | `17.5` | `20.0` | `25.0` | `30.0` | `35.0` | `40.0 ms` | First `>=4` falls / `>=1.5×` cost / loses ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `PID, hand-tuned` | `8` | `14` | `14` | `9` | `11` | `18` | `34` | `37` | `39` | `68` | `7.5 / 40.0 / —` |
| `PID, model-based` | `4` | `6` | `8` | `6` | `7` | `15` | `27` | `67` | `108` | `143` | `7.5 / 30.0 / 35.0 ms` |
| `PID, DE / seed0` | `4` | `5` | `9` | `12` | `18` | `22` | `37` | `97` | `134` | `150` | `7.5 / 30.0 / 30.0 ms` |
| `PID, DE / seed1` | `4` | `6` | `7` | `7` | `8` | `13` | `24` | `45` | `82` | `138` | `7.5 / 30.0 / 40.0 ms` |
| `PID, CMA-mid / seed0` | `4` | `4` | `6` | `9` | `11` | `12` | `25` | `73` | `120` | `147` | `7.5 / 30.0 / 35.0 ms` |
| `PID, CMA-mid / seed1` | `3` | `4` | `6` | `9` | `11` | `13` | `36` | `92` | `133` | `149` | `10.0 / 30.0 / 30.0 ms` |
| `PID, CMA-mid (mean) / s0` | `4` | `4` | `5` | `6` | `10` | `15` | `25` | `48` | `103` | `138` | `7.5 / 30.0 / 35.0 ms` |
| `PID, CMA-mid (mean) / s1` | `5` | `5` | `5` | `6` | `9` | `15` | `28` | `56` | `116` | `143` | `7.5 / 30.0 / 35.0 ms` |
| `PID, CMA-warm / seed0` | `4` | `8` | `8` | `9` | `14` | `15` | `40` | `100` | `140` | `150` | `7.5 / 25.0 / 30.0 ms` |
| `PID, CMA-warm / seed1` | `4` | `4` | `7` | `7` | `10` | `13` | `20` | `27` | `57` | `104` | `7.5 / 35.0 / 40.0 ms` |
| `PID, CMA-warm (mean) / s0` | `3` | `4` | `5` | `7` | `8` | `9` | `29` | `65` | `114` | `144` | `10.0 / 30.0 / 35.0 ms` |
| `PID, CMA-warm (mean) / s1` | `4` | `4` | `6` | `6` | `9` | `13` | `18` | `31` | `53` | `75` | `7.5 / 40.0 / —` (`0.968` at `40 ms`) |
| `PID, Random / seed0` | `3` | `3` | `4` | `6` | `8` | `11` | `15` | `22` | `44` | `97` | `12.5 / 35.0 / 35.0 ms` |
| `PID, Random / seed1` | `5` | `6` | `9` | `11` | `16` | `18` | `28` | `36` | `82` | `115` | `7.5 / 35.0 / 35.0 ms` |
| `PID, CMA-lat10-15 / s0`* | `2`* | `1`* | `2`* | `7` | `10` | `12` | `25` | `56` | `125` | `148` | `15.0* / 30.0 / 35.0 ms` |

### Fresh Out-of-Sample Confirmation (`5050–5149`, `300 × 60 s` NumPy Float64; `K1`)

To remove selection bias on `5000–5049`, the key candidates were re-scored on **`300` fresh `60 s` dev runs (`5050–5149`)** (`results/legobot/comparison/e1-fresh-5050-5149/`):

| Controller | Mean (`60 s`) | Falls (`/300`, McNemar $p$ vs `model-based`) | Median (`60 s`) | Paired ratio vs `model-based` `[95% CI]` | Fixed `7.5 ms` | Fixed `10.0 ms` | Fixed `12.5 ms` | Fixed `15.0 ms` |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| `PID, model-based` | **`60.11`** | **`5/300`** (`1.000`) | `27.80` | `1.000` | `8` | `6` | **`9`** | **`13`** |
| `PID, CMA-mid / seed0` | `72.01` (`72.2`*) | `7/300` (`0.727`) | **`27.33`** (`27.59`*) | `0.985 [0.976, 0.993]` | `5` | `7` | `14` | `19` |
| `PID, CMA-warm (mean) / seed1` | `77.01` (`89.5`*) | `7/300` (`9/300`*; `0.727`) | `31.69` (`31.71`*) | `1.137 [1.125, 1.145]` | **`4`** | **`5`** | `11` | **`13`** |
| `PID, CMA-mid (mean) / seed0` | `79.55` | `8/300` (`0.508`) | `28.45` | `1.023 [1.016, 1.031]` | `8` | `11` | `13` | `14` |
| `PID, CMA-lat10-15 / seed0` | `94.20` | `10/300` (`0.227`) | `30.89` | `1.105 [1.098, 1.115]` | `6` | `10` | `11` | `14` |

*\*Values in parentheses use the 4-decimal rounded gains printed in the Step 1 table (which flip 2 borderline chaotic falls on `CMA-warm (mean) / seed1` from `7/300` to `9/300`); unparenthesized values use the full-precision JSON gains in `e1-fresh-5050-5149`.*

**Takeaways (`K1`, `K7`, `L1`):**
- **No Step 1 tuned PID beats the untuned `model-based` PID out of sample:** Across all Step 1 full-budget candidates on `5050–5149` (`300 × 60 s`), typical tracking cost is equal within noise (2-run `model-based` vs `CMA-mid (mean)` paired ratio `0.991 [0.972, 1.010]`; vs `CMA-warm (mean)` `0.931 [0.875, 0.994]`; single-run `CMA-mid / seed0` `0.985 [0.976, 0.993]` with `mean = 72.01` vs `60.11`) and falls are no better (`5–13/300` vs `5/300` for `model-based`, every McNemar $p \ge 0.057$). `CMA-mid`'s `1.046 [1.006, 1.084]` edge holds on the selection seeds `5000–5049` only.
- **Selection-bias warning confirmed:** Off its `5000–5049` selection seeds, `CMA-lat10-15 / seed0` did **not** dominate the `7.5–15.0 ms` operating band (`10/300` randomized falls, `6 / 10 / 11 / 14` fixed-latency falls, median `30.89`), confirming that its `0/150` (`p = 0.00024`) and `15.0 ms` threshold on `5000–5049` were selection-seed artifacts.
- **Outer-loop / integral trade-off at extreme delay:** High-`ki` `CMA-mid` (`ki = 6.0–7.4, kd = 0.142–0.146`) keeps typical tracking cost near `model-based` (`0.519` vs `hand-tuned` on `5000–5049`, `med = 27.33` vs `27.80` on `5050–5149`), while individual runs with either lower `ki` (`CMA-warm (mean) / seed1`, `ki = 3.95, kd = 0.164`: `53` falls at `35 ms`, `75` at `40 ms`) or lower outer-loop gain (`Random / seed0`, `kv_p = 0.36` despite high `kp = 2.99, ki = 8.22`: `15 / 22 / 44` falls at `25 / 30 / 35 ms`) retain much better extreme-latency survival. Notice from `CMA-warm / seed0` (`40 / 100 / 140`) and `CMA-warm (mean) / seed0` (`29 / 65 / 114`) that extreme-latency survival belongs to those specific gain basins rather than every seed of `CMA-warm`.

### Bryson Weight Sensitivity (`K8`)
Across all 9 weight perturbations (`pitch ×0.5/×2`, `velocity ×0.5/×2`, `torque ×0.5/×2`, `effort off`, `thermal 0.788 N m`), `falls_rank_below_every_non_fall` stayed **`true`** (`max_non_fall_cost = 139.59` configured, `558.03` worst perturbation). `CMA-mid` leads on `configured` (`0.519`), `pitch ×0.5/×2`, `velocity ×2` (`0.420`), and `torque ×0.5` (`0.420`), whereas **`model-based`** leads on `effort off` (`0.772` vs `0.827` `CMA-mid` and `0.825` `CMA-warm (mean)`), `thermal 0.788` (`0.764` vs `0.816`), `velocity ×0.5` (`0.662` vs `0.676`), and `torque ×2` (`0.663` vs `0.677`). That split is physical: `model-based` has the lowest velocity RMS error (`0.0945 m/s`, vs `0.0953–0.110 m/s` across tuned PIDs and `0.1076 m/s` for `hand-tuned`), whereas `DE`, `CMA-mid`, and `CMA-warm` reduce RMS torque below both `hand-tuned` (`0.232 N m`) and `model-based` (`0.149 N m`, down to `0.135–0.147 N m`; `CMA-lat10-15` `0.156`, `CMA-warm (mean)/s1` `0.166`, `Random/s0` `0.190`).

---

## 4. Post-E1 Dev Roadmap Preview (`LEARNING` Scales & Float64 Policy Preview)

### `LEARNING` Observation Scales Check (`results/legobot/e1_learning_scales_check.json`)

Closed-loop observation quantiles across 150 dev scenarios (`0/150` falls on the check set) comparing `hand-tuned` and a representative `ki ≈ 6.8` tuned PID (`CMA-lat10-15 / seed0`, `[2.3739, 6.7913, 0.1558, 0.4559]`) against `legobot_config.LEARNING`:

| Channel | Configured scale | `hand-tuned` `p99` (`max`) | Tuned PID (`ki = 6.79`) `p99` (`max`) | Status for E2 |
|---|---:|---:|---:|---|
| `v_meas` (`speed_scale`) | `0.40` | `0.529` (`0.815`) | `0.500` (`0.816`) | OK |
| `goal_minus_v` (`velocity_error_scale`) | `0.25` | `0.289` (`0.684`) | `0.280` (`0.816`) | OK |
| `theta_meas` (`pitch_scale`) | `0.30` | `0.120` (`0.275`) | `0.118` (`0.356`) | OK |
| `theta_dot_meas` (`pitch_rate_scale`) | `3.40` | `2.525` (`4.280`) | `2.277` (`4.110`) | OK |
| `cmd_1..4` (`command_scale`) | `0.40` | `0.574` (`0.900`) | `0.415` (`0.822`) | OK |
| `pid_setpoint` (`residual_setpoint_scale`) | `0.25` | `0.145` (`0.342`) | `0.127` (`0.342`) | OK |
| **`int_v` (`int_v_scale`)** | **`0.40`** | `0.694` (`0.731`) | **`0.784`** (`0.830`) | **Update (`~0.80`)** — ~`2×` too small for both baselines and tuned PIDs |
| **`int_theta` (`int_theta_scale`)** | **`0.20`** | `0.330` (`0.347`) | **`0.353`** (`0.374`) | **Update (`~0.35–0.40`)** — ~`1.8×` too small for both baselines and tuned PIDs |
| **`pid_int_theta` (`residual_int_theta_scale`)** | **`0.06`** | `0.0535` (`0.0917`) | **`0.0155`** (`0.0385`) | Re-measure once E2 picks its reference PID (scales as `~1 / ki`; reference PID choice stays open for E2, with `model-based` at `ki = 5.24` as a finalist) |

### Dev-Only Float64 Learned-Controller Preview (`--profile quick`)

Artifacts: `results/legobot/e1_policy_dev_preview/preview_manifest.json` and `results/legobot/comparison/e1-dev-with-policies-preview/comparison.json` (`30` generations, `seed 0`, float64, `float32_float64_disagree: false` on all 3 runs):

| Controller | Params / evals | Wall clock | Selected gen | Dev 60 s mean (`5000–5049`) | Dev 60 s falls (`/150`) | Median ratio vs hand-tuned | McNemar $p$ | Median `U rms` (sat frac) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `PID, hand-tuned` | `4` / — | — | — | `217.91` | `13` (`8.7%`) | `1.000` | `1.000` | `0.232` (`0.0%` / `7.1%`) |
| `PID, model-based` | `4` / — | — | — | `105.31` | `6` (`4.0%`) | `0.543` | `0.092` | `0.149` (`0.0%` / `45.2%`) |
| `PID, CMA-lat10-15` *(labeled `E1-best` in JSON)*\* | `4` / `2,400` | `261 s` | — | `31.57`* | `0`* (`0.0%`) | `0.589` | `0.00024`* | `0.156` (`0.0%` / `0.02%`) |
| `Policy, Linear (quick)` | `10` / `7,681` | `385.4 s` | gen 6 (`best`) | `719.69` | `27` (`18.0%`) | `10.508` | `0.029` | `0.766` (`60.1%` / `89.3%`) |
| `Policy, MLP (quick)` | `481` / `39,961` | `815.4 s` | gen 30 (`mean`) | `647.05` | `5` (`3.3%`) | `11.335` | `0.057` | `0.886` (`95.4%` / `100%`) |
| `Policy, Residual (quick)`\* | `513` / `39,961` | `1,346.8 s` | gen 3 (`best`) | `53.71`* | `1`* (`0.67%`) | `0.776` | `0.00049`* | `0.191` (`0.0%` / `4.6%`) |

*\*Preview caveat (`K1`, `N5`): single seed (`seed = 0`), `30` quick generations; `Policy, Residual (quick)` selected its `best` checkpoint on the `smoke` subset (`5000–5009`, overlapping `30` of the `150` `dev` scenarios scored here), and `CMA-lat10-15` was picked on `5000–5049`. Treat `Residual`'s `1/150` (`0.776` median ratio, `0.0%` median saturation vs `60–95%` for scratch Linear/MLP) as a suggestive smoke preview rather than an out-of-sample ranking.*

---

## 5. Concrete Recommendations for Phase E2 (After Bench B3–B4 & C1–C5 Finish)

*(Overall Step 1 outcome (`L1`): no Step 1 tuned PID beat the untuned `model-based` PID out of sample on `5050–5149` — typical tracking cost was equal within noise and falls were no better (`5–13/300` vs `5/300`). Accordingly, E1 does **not** freeze a tuned reference PID; the reference PID choice remains open for E2, with `model-based` kept as a finalist alongside E2's tuned candidates.)*

1. **Keep all four Tier-1 models + Tier-2 spreads in E2, and cleanly re-run the PDM / tick-phase ablations (`K2`, `K6`, `L5`):** IMU specific-force + complementary filter ($p = 2.5 \times 10^{-57}$), 360-count / 100 Hz / 5 ms encoder ($p = 3.1 \times 10^{-17}$), and drivetrain backlash — especially upstream motor-to-encoder gearbox play, across which motor motion is unseen by the encoder ($p = 2.6 \times 10^{-13}$) — are overwhelmingly significant; 10 ms slow-PWM PDM ($p = 0.0167$) and free tick phase ($p = 0.0574$) are kept on physical grounds (ablation evidence suggestive at `1` seed under `0.0045` Bonferroni). In E2, re-run the PDM and tick-phase ablations with `2–3` tuning seeds, picking `mean` vs `best_sample` on training seeds (or a separate held-out slice) and evaluating strictly on scenarios not used for picking. Keep feedforward mismatch (`+41%` median cost when omitted) and pitch-bias spreads (`+8.6%`) for tracking robustness.
2. **Test `latency_range_s = (0.010, 0.015)` vs `(0.0075, 0.0125)` properly before changing the default (`K1`):** Because `CMA-lat10-15`'s `0/150` result on `5000–5049` did not hold out-of-sample on `5050–5149` (`10/300` falls vs `5–8/300` for `model-based` and `CMA-mid`), evaluate `(0.010, 0.015)` against `(0.0075, 0.0125)` on a **fresh** dev slice across `>= 2–3` optimizer seeds before modifying `legobot_config.py`'s default latency range (and keep `catch_latencies` anchored to the physical `12.5 ms` sample-to-HAT ceiling unless bench timing widens).
3. **Address fall-luck dominance in E2's training objective (`L1`):** Under PDM + backlash, the current 150-scenario robust `mean_cost` objective is dominated by `~5–13` chaotic falls (`~1,000–2,000` each, `~7–13` points on the mean vs `~10–45` non-fall cost), so optimizers largely fit random fall luck on seeds `1–50`. Options to evaluate in E2 (proposed, not run here):
   - train on substantially more training seeds per evaluation so fall-rate noise averages down, and/or
   - optimize a **median or trimmed-mean** tracking objective with falls counted (or lexicographically/smoothly penalized) separately;
   - **always re-score the untuned `model-based` PID beside the tuned finalists** on held-out scenarios before choosing E2's reference PID.
4. **Standardize E2 optimizer & selection protocol (`K1`, `K5`):**
   - Run float64 GPU CMA-ES with both `--cma-x0 midpoint` and `--cma-x0 model-based` across multiple seeds, evaluating both `cma-es (mean)` (`xfavorite`, lower across-seed dev variance) and best-sample finalists alongside `model-based`.
   - Select finalists on `dev` (`5000–5049`), confirm on scenarios not used for selection (`5050–5149`), never re-report selection-seed scores as out-of-sample headlines, compare with paired per-scenario bootstrap CIs + McNemar tests, and touch the frozen `validation` (`3000–3099`) and `fall-rate` (`6000–6999`) sets **once** at the end.
5. **Bench C prioritization & play check (`K3`, `N1`):** Two-way `wider_play_bounds` testing (`±1.0–2.0°` gearbox, `0–4.35°` coupling) shows nominal-tuned gains transfer with no degradation beyond noise (`13/450` vs `18/450` for wider-tuned and `model-based`), while wider play itself still raises falls modestly (`10/450 -> 13/450` for nominal-tuned, `11/450 -> 18/450` for `model-based`). Prioritize Bench B3 (motor rail at 8.4 V), C1 (right motor torque constant `a`), and C3 (right drivetrain friction), and include a quick wheel-on visual check of coupling play alongside them.
6. **Re-measure `LEARNING` scales after E2 chooses its reference PID:** Re-run `python hardware/legobot/legobot_learning_scales.py` once E2 selects its reference PID (whether `model-based` or an E2-tuned finalist; do not pre-commit to a `ki ≈ 6–7` reference PID from E1).
