# External LEGO Review, Round 2 — Findings & Reconciliation (2026-11-24)

- **Branch reviewed:** `claude/legobot-tick-timing-574623` (`d252bff`, PR #15 before Phase E1)
- **Prompt:** `docs/legobot-review/review-prompt-v2.md`
- **Environment verified:** `C:\venvs\balancebot\Scripts\python.exe` (Python 3.12, NumPy, SciPy, `cma`, PyTorch CUDA on RTX 5070 Ti)
  - Pololu suite (`python -m pytest`): **655 passed**
  - LEGO suite (`ROBOT=legobot python -m pytest`): **521 passed, 13 skipped**
  - Robot suite (`python -m pytest hardware/legobot/robot/test_control.py hardware/legobot/robot/measure/test_fit.py`): **38 passed**

---

## Verdict

**Not ready for E1 on `d252bff` — ready after implementing the reconciled fixes below (`H1`–`H9`, `H11`–`H13`).**

Two blockers would directly corrupt or waste the E1 GPU tuning runs: `SEARCH.ki_max = 5.0` clips out the untuned LEGO model-based baseline (`ki = 5.24`) before optimization starts (`H1`), and `cma_es_search` on GPU float64 starts at a high-`kd` unit-cube midpoint (`kd = 0.50, kv_p = 1.00`) on an objective swamped by chaotic fall flips (`24/26` near-baseline candidates disagree in fall count between CUDA float64 and NumPy float64, shifting the objective by up to `101.7` points when `sensitivity_weight = 1.0` doubles fall variance), losing to the untuned baseline (`H2`). In addition, `gearbox_backlash` conflated a `2.58°` peak-to-peak cluster separation and unbraked `set 0` rotor slip with half-widths (`H3`), step-quantized `latency_range_s` omits the `15.0–17.5 ms` hardware loop-delay band (`H4`), `legobot_scenario.py` had unverified push-onset rotor-speed assumptions and a float step-accumulation bug (`H5`), the lifted-robot `17.0 rad/s` runaway trip has only `~5%` margin at the `6.8 V` battery cutoff (`H6`), `SimWorld` bypasses `pdm_update` (`H7`), and `compare_controllers.py --gain-runs` crashes on saved PID JSONs (`H8`).

The core 3-DOF backlash impulse/momentum physics and three-engine equivalence (`balance_sim`, `batch_sim`, `torch_sim.py`) are sound.

### Summary & Reconciled Triage Table

| # | Severity | Area | Finding | Reconciled Action |
|---|---|---|---|---|
| **H1** | blocker | 9. Readiness for E1 | `SEARCH.ki_max = 5.0` clips out `model_based_gains()` (`ki = 5.24`) | Set `ki_max = 15.0` in `legobot_config.SEARCH`; test baseline inclusion in `test_config.py` |
| **H2** | blocker | 5. Robust scoring & noise | Chaos twin penalty doubles random fall flips (`92–290` pts) & `cma_es_search` cold-starts at `kd = 0.5`, chasing GPU bit-noise | Set `sensitivity_weight = 0.0` in `legobot_config.ROBUSTNESS` (halves E1 GPU time); warm-start/inject `model_based_gains()` and re-score `xfavorite` + top-$K$ GPU elites in NumPy float64 |
| **H3** | major | 1. Backlash physics / fit | `gearbox_backlash` (`±2.6–3.0°`) used peak-to-peak cluster sep (`2.58°`) + unbraked `set 0` rotor slip as half-widths | Set `gearbox_backlash` nominal `1.3°`, range `±1.1–1.6°`; keep `coupling_backlash` `±0–3.0°` (nominal `1.5°`); update docs & `measure/backlash.py` |
| **H4** | major | 3. Tick timing after B1 | `latency_range_s = (0.0075, 0.0125)` on `2.5 ms` grid gives `7.5–15.0 ms` loop delay (`0%` in `15.0–17.5 ms` hardware band) | Shift `latency_range_s` to `(0.010, 0.015)` (or evaluate explicitly whether `12.5 ms` stays with wider sweep) |
| **H5** | major | 4. Scenario & catch bound | Push catchability under opposing PDM rotor speeds + `t += dt` float off-by-one + oracle `motor_speed` in `_Play.step` | **Settled by simulation (Section 4 below):** with `H3` (`±1.1–1.6°`), resting far-side bound re-derives to **`0.74 N`** and `100%` of settled dev push states are caught by full torque; fix `k * dt < latency - 1e-12` and pass encoder speed in `_Play.step` |
| **H6** | major | 6. Robot code since round 1 | `RUNAWAY_SPEED = 17.0 rad/s` has thin/negative margin at `6.8 V` cutoff; `calibrate_gyro` uncaught `OSError`; unjoined watchdog | **Reconciled (Section 4 below):** scale `runaway_speed = 0.85 * w_nl(vin)` (`15.1–18.9 rad/s` over `6.8–8.4 V`, `0/150` dev false trips); wrap `calibrate_gyro` cleanup; `watchdog.join()` on stop |
| **H7** | major | 6. Robot code since round 1 | `SimWorld` (`balance --sim`, `test_control.py`) omits `pdm_update`, running smooth averaged PWM instead of `10 ms` pulses | Wire `pdm_update` into `SimWorld.advance` and `specific_force`; pass `disturbance` in `_reference_falls` |
| **H8** | major | 7. Config & test integrity | Plain LEGO tuning freezes `pwm_tick_phase = 0`; `compare_controllers.py --gain-runs` crashes on `dict` gains & `--existing` skips `scoring_differences` | Fix `dict` gain parsing in `gain_runs`, check `data["config"]` in `_run_warnings` and `existing_controllers`, and draw/warn on plain-mode `pwm_tick_phase` |
| **H9** | minor | 1. Backlash physics | IMU `accelerometer_tilt` at sample steps uses un-released `contact` before `begin_step` (up to `38.1°` raw / `0.38°` filtered spike on `2%` of ticks) | Call `backlash_release` / `begin_step` before `derivatives` at IMU sample ticks in all 3 engines + `sim_hw.py` |
| **H10** | minor | 8. Model-based baseline | `linearize()` uses standstill Coulomb slope (`0.042 N m s/rad`); at `0.40 m/s` cruise (`slope ~ 0`), `0.3`-ratio PID goes unstable at `~26 ms` (`1.016` at `35 ms`, not `0.9975`) | Keep `design_integral_ratio = 0.3` (optimal at both standstill and `15 ms` cruise); update comment in `legobot_config.py:248` and check both `P` and `P.with_changes(coulomb_friction=0.0)` in `test_pid_design.py` |
| **H11** | minor | 3. Tick timing after B1 | `pdm_update` uses inclusive `>= 0.5` and `<= -0.5`, causing `[-1, +1]` full-voltage chatter on `duty = 0.0` if `total = ±0.5` | Change to strict `> 0.5` and `< -0.5` in `actuators.py:278` and `physics.py:240-241` |
| **H12** | minor | 7. Config & test integrity | No short-window (`0.2 s`) tight `torch` vs `NumPy` full-loop test; 4 skipped LEGO test groups should run | Add `80`-step `atol=1e-12` LEGO `torch` vs `NumPy` test; enable `test_velocity`, multi-seed `test_latency_hurts`, `test_compare_controllers` smoke under LEGO |
| **H13** | minor | 7. Config & test integrity | `E1_HANDOFF.md` omits backlash from Step 2 ablations; stale numbers in `BENCH_CHECKS.md`, `legobot_config.py`, `PHYSICS.md`, `DATASHEET.md` | Add backlash to `E1_HANDOFF.md` Step 2 ablations and clean up stale doc lines |

---

## Detailed Findings (`H1`–`H13`)

### H1 — `SEARCH.ki_max = 5.0` clips out the LEGO model-based baseline (`ki = 5.24`)
- **Severity:** `blocker`
- **Area:** 9. Readiness for E1
- **Location:** `legobot_config.py:248`, `legobot_config.py:313`, `config.py:485`, `optimization/tests/test_config.py:91-95`
- **Claim tested:** `SEARCH` in `legobot_config.py` contains both baselines in its interior and gives E1 room to tune `ki`.
- **What is wrong:** Commit `3321229` raised `CONTROLLER.design_integral_ratio` from `0.1` to `0.3`, tripling `LEGOBOT_CONFIG.model_based_gains()[1]` from `1.75` to `5.2360`, while `SEARCH = SearchConfig(kv_p_max=2.0)` still inherits Pololu's `ki_max = 5.0` (`config.py:485`). Every optimizer clips unit-cube coordinates to `[0, 1]`, making `ki > 5.0` unreachable. `test_config.py:91` (`test_hand_tuned_gains_match_the_search_space`) only checks `len(gains) == 4`, never that `hand_tuned_gains` and `model_based_gains()` lie inside `SEARCH`.
- **Concrete failure scenario:** `space = LEGOBOT_CONFIG.build_search_space(); g = LEGOBOT_CONFIG.model_based_gains()` returns `[1.9797, 5.2360, 0.1509, 0.5153]`. Encoding and decoding through `space` clips `ki` to `5.0` (`space.at_bounds(...) == ['ki']`), worsening NumPy float64 training score on seeds `1–50` from `mean_cost = 72.99` (`5/150` falls, `objective = 130.19`) to `mean_cost = 105.96` (`8/150` falls, `objective = 195.34`).
- **Fix and test:** Set `ki_max=15.0` in `legobot_config.py:313`. In `optimization/tests/test_config.py:91`, assert `not space.at_bounds(g)` and `np.allclose(space.decode(np.clip(space.encode(g), 0, 1)), g)` for both `CONFIG.controller.hand_tuned_gains` and `CONFIG.model_based_gains()`.
- **Basis:** Confirmed by running `C:\venvs\balancebot\Scripts\python.exe` on `LEGOBOT_CONFIG.model_based_gains()` and `BatchEvaluator` over seeds `1–50`. Confidence: `high`.

---

### H2 — Chaotic fall flips + `sensitivity_weight = 1.0` + midpoint cold-start cause GPU `cma_es_search` to chase noise and lose to the untuned baseline
- **Severity:** `blocker`
- **Area:** 5. Robust scoring and noise
- **Location:** `optimization/batch_tuning/optimizers.py:224-264`, `optimization/batch_tuning/evaluation.py:242-254`, `optimization/run_batch_tuning.py:194-205`, `optimization/run_batch_tuning.py:254`
- **Claim tested:** CMA-ES on the 150-scenario + 150-twin objective in GPU float64 converges to robustly better gains, and re-scoring each search's single winner in NumPy float64 prevents noise selection bias.
- **What is wrong:**
  1. **Chaos penalty doubles binary fall noise:** Non-falling runs cost `~10–20`, whereas a borderline fall costs `~1500–1950` (`+10` to `+13` on the 150-run mean). Under 100 Hz PDM pulses + backlash, the `1e-4 rad` twin run decorrelates within `0.5 s` and flips independent borderline falls, making `sensitivity = mean |cost - twin_cost|` equal to `57–294` (comparable to or larger than `mean_cost`) while doubling GPU wall-clock cost.
  2. **Single-sample winner's curse in `cma_es_search`:** `optimizers.py:257` keeps the single lowest noisy evaluation `costs[i] < best_cost` across thousands of GPU evaluations rather than evaluating `es.result.xfavorite` (the distribution mean) or a top-$K$ archive. Re-scoring only `s.gains` once in NumPy float64 (`run_batch_tuning.py:198`) observes regression to the mean without recovering discarded good candidates.
  3. **Hostile midpoint cold-start:** `optimizers.py:242` hardcodes `x0 = [0.5] * dim` (`kp = 3.11, ki = 2.50, kd = 0.50, kv_p = 1.00`), deep inside the `kd >= 0.3` pulse-amplification fall regime (`A4_BASELINES.md:324`).
- **Concrete failure scenario:**
  - Across `26` gain vectors within `±15%` of `model_based_gains()` on the `150` training scenarios, `torch` CUDA float64 and `numpy` float64 disagreed in fall count on **`24/26` candidates** (up to `6` falls apart on identical gains; e.g., hand-tuned: NumPy `170.87` / `12` falls / `obj = 453.48` vs CUDA `237.53` / `18` falls / `obj = 519.06`; candidate #6: NumPy `obj = 307.64` vs CUDA `obj = 205.98`, a **`101.7`-point** gap).
  - Running `cma_es_search(ev_gpu, max_iter=12, popsize=64, seed=0)` (`768` GPU float64 evaluations) returned `[1.8726, 3.6352, 0.1303, 0.3404]` with GPU train `obj = 175.46` (`11/150` falls) and NumPy dev `obj = 185.07` (`12/150` falls) — **`45` points and `6` falls worse** than the untuned model-based baseline (`obj = 130.19`, `5/150` train falls).
- **Fix and test:**
  1. Set `sensitivity_weight=0.0` in `legobot_config.ROBUSTNESS` (reconciled with builder; halves E1 GPU runtime and removes double-counted random fall flips; optionally compute sensitivity once on finalists for reporting).
  2. In `cma_es_search`, evaluate `es.result.xfavorite` alongside the top-$K$ (`K = 16`) distinct GPU elites in NumPy float64 when picking `s.gains`, and warm-start or inject `model_based_gains()` into the initial population.
  3. Require paired McNemar / paired bootstrap CIs (or `> 2` falls / `> 20` points on 150 scenarios) before calling one candidate better than another.
- **Basis:** Confirmed by running `26 × 300` CUDA vs NumPy float64 evaluations and a `12 × 64` GPU float64 `cma_es_search` run. Confidence: `high`.

---

### H3 — `gearbox_backlash` treats a `2.58°` peak-to-peak cluster separation and unbraked `set 0` rotor slip as half-widths (`5.2–6.0°` peak-to-peak)
- **Severity:** `major`
- **Area:** 1. The backlash event model's physics
- **Location:** `legobot_config.py:61-67`, `legobot_config.py:191-192`, `hardware/legobot/robot/measure/backlash.py:5-10`, `hardware/legobot/robot/measure/backlash.py:30`, `hardware/legobot/bench_data/20261123_073624_backlash_fit.json:3-18`, `hardware/legobot/DATASHEET.md:800-806`
- **Claim tested:** `gearbox_backlash` in `legobot_config.py` (nominal `2.8°`, randomized `2.6–3.0°` half-width = `5.2–6.0°` peak-to-peak) matches `bench_data/20261123_073624_backlash.csv`.
- **What is wrong:**
  1. `cluster_sep = c_hi - c_lo = 2.58°` (`backlash_cluster_deg@left` in `20261123_073624_backlash_fit.json`) is the **peak-to-peak** separation between the two end-stop dwell clusters (`±1.29°` half-width), yet `legobot_config.py:191` used `2.6°` as the **lower half-width** (`5.2°` peak-to-peak).
  2. `measure/backlash.py:5` assumed `set_pwm(0, 0)` in `pwmparams 1 0.01` holds the motor stiff (*"NOT coast"*), which Bench B2(b) (`20261123_195004_coast_vs_brake_summary.txt`) disproved — `set 0` in slow PWM coasts. In `20261123_073624_backlash.csv`, hand rocking at `t = 1..2 s` turned the unbraked rotor by `+2°` (`pos_deg` `+2`, `apos_deg` `90`), inflating `ptp` to `5–6°`, whereas steady rocking over `t = 3..15 s` stayed in two clusters `2.58°` apart (`3°` peak-to-peak, `±1.3°–1.5°` half-width). Also `backlash.py:30` hardcoded `WHEEL_DIAMETER_MM = 69.0` while the `96 mm` lever beam was mounted, writing a corrupt `backlash_visual_deg@left = 31.55°` into `_fit.json`.
- **Concrete failure scenario:** `ROBUSTNESS` never draws `gearbox_backlash` below `±2.6°` (`5.2°` peak-to-peak), ~`2×` the real unseen gearbox play (`±1.3°`), inflating baseline falls (`17/150` hand-tuned, `9/150` model-based on dev) and shrinking the scenario push bound (`0.67 N`).
- **Fix and test (reconciled):** Set `ROBOT.gearbox_backlash = math.radians(1.3)` and `randomize_range` `("gearbox_backlash", math.radians(1.1), math.radians(1.6))` (allowing for `1°` encoder counts and the unmeasured right motor while excluding rotor slip); keep `coupling_backlash` at `±0–3.0°` (nominal `1.5°`). Update `legobot_config.py:60-67, 189-192`, `DATASHEET.md` 8.7, `SOFTWARE_PLAN.md` B5, and `measure/backlash.py`; add a unit test in `test_fit.py` asserting `cluster_sep_deg` on a `±delta` square wave equals `2 * delta`.
- **Basis:** Confirmed from raw `20261123_073624_backlash.csv`, `_fit.json`, and `20261123_195004_coast_vs_brake_summary.txt`. Confidence: `high`.

---

### H4 — Step-quantized `latency_range_s` + static shared `pwm_tick_phase` omits the `15.0–17.5 ms` hardware delay band and within-run phase drift
- **Severity:** `major`
- **Area:** 3. The tick timing after B1
- **Location:** `legobot_config.py:90-98`, `legobot_config.py:197-206`, `optimization/batch_tuning/robust.py:178-183`, `simulation/balance_sim/actuators.py:291-304`, `hardware/legobot/DATASHEET.md:784-790`
- **Claim tested:** `latency_range_s = (0.0075, 0.0125)` plus per-scenario `pwm_tick_phase` covers the post-B1 hardware loop delay (`9.7–12.0 ms` sample-to-HAT + wandering `10 ms` ticks).
- **What is wrong:**
  1. On the `dt = 2.5 ms` step grid, `sample_delay_steps` draws `{3, 4, 5}` (`7.5, 10.0, 12.5 ms`), and because commands arrive every `2` steps (`5.0 ms`) while ticks fire every `4` steps (`10.0 ms`), the newest command's wait to a tick is only `0 ms` (`50%`) or `2.5 ms` (`50%`). Effective steady-state loop delay in training is `{7.5 (16.7%), 10.0 (33.3%), 12.5 (33.3%), 15.0 ms (16.7%)}` (mean `11.25 ms`), with **`0%` weight in `15.0–17.5 ms`** where hardware spends up to half its ticks (`9.7–12.0 ms + 0–5.0 ms = 9.7–17.0 ms`, `legobot_config.py:247`).
  2. Each scenario holds `pwm_tick_phase` fixed for the entire `20–60 s` run, whereas Bench B1 and `20261123_193152_balance_summary.txt` (`-75 -> -592 ppm` in `4.7 s`) show within-run tick-phase drift.
  3. Sharing one tick phase across both motors is pessimistic on pitch ripple/backlash chatter (`2×` synchronous pulse torque vs staggered pulses `~2 ms` apart in `DATASHEET.md:789`) and optimistic on yaw (unmodelled in 2D).
- **Concrete failure scenario:** Gains tuned on `(0.0075, 0.0125)` never experience `delay_steps = 6` (`15.0` or `17.5 ms` effective loop delay) during training, yet encounter `15–17 ms` routinely on the Pi.
- **Fix and test:** Widen `latency_range_s` to `(0.010, 0.015)` (`delay_steps` in `{4, 5, 6}` -> effective loop delays `{10.0, 12.5, 15.0, 17.5 ms}`, mean `13.75 ms`, keeping `catch_latencies` anchored to the physical `12.5 ms` sample-to-HAT ceiling), and add a timing-distribution test in `test_slow_pwm.py`.
- **Basis:** Traced in `robust.py:178-183` and `actuators.py:291-304`. Confidence: `high`.

---

### H5 — `legobot_scenario.py` push-onset state assumptions, oracle `drive()` speed in `_Play.step`, and `t += dt` float off-by-one
- **Severity:** `major`
- **Area:** 4. The scenario and catch bound with the play
- **Location:** `hardware/legobot/legobot_scenario.py:130-191`, `hardware/legobot/legobot_scenario.py:208-212`, `hardware/legobot/legobot_scenario.py:276-291`
- **Claim tested:** `caught()`'s resting far-side state (`closed_side = -direction`, `motor_speed = wheel_speed`) guarantees every scenario push is catchable in the `2.5 ms` closed-loop simulator, and `_Play.step` / `caught()` step the same actuator/timing rules as the engines.
- **What is wrong:**
  1. **Pre-push opposing motor speed:** During `0.40 m/s` cruise under PDM + backlash, a prior braking pulse leaves signed `motor_speed - wheel_speed` in `[-10.45, +8.73] rad/s` (`p10 = -3.09 rad/s`). Under the old `±2.6–3.0°` gearbox play (`0.67 N` push), `scen=88` (`seed=5029, tilt=-0.15`, `th = -1.74°, d_w = -9.79 rad/s`) fell after the `5.5 s` push even if full duty was commanded immediately (`caughtA = False`). **However, once `H3` (`±1.1–1.6°` gearbox play) is applied (`0.74 N` re-derived push), `100%` of dev push falls starting from a settled state (`|th| < 5°`, down to `d_w = -4.73 rad/s`) ARE caught by full torque** (see Section 4 below).
  2. **Oracle speed in `_Play.step`:** `legobot_scenario.py:209` passes true `motor_speed` (`s[5]`) to `drive()` instead of encoder shaft speed (inconsistent with all three engines, though harmless when `duty` saturates at `1.0` on the `6.8 V` worst corner).
  3. **Float step-accumulation off-by-one:** `catch_latencies` computes `lp = 0.0025 * 5 + 0.005 + 0.010 = 0.027500000000000004`, while `t += 0.0025` in `caught()` (`legobot_scenario.py:176, 182`) equals `0.0275` at step `11`, so `t < latency` holds `torque = 0` for **`12` steps (`30.0 ms`) instead of `11` (`27.5 ms`)** at `dt = 0.0025` (`12/8192` corners fail `0.67 N` due to that extra step).
- **Concrete failure scenario:** Calling `caught(R, speed=0.40, impulse=0.0335, push_duration=0.05, latency=lp, dt=0.0025)` applies `30.0 ms` of zero torque instead of `27.5 ms`.
- **Fix and test (reconciled):** Keep the resting far-side catch test and re-derive pushes together with `H3` -> **`0.74 N × 50 ms`** (`RECORDED_CATCH` updated). Replace `t += dt; t < latency` and `t < push_duration` in `legobot_scenario.py:172-183` with integer step indexing `k * dt < latency - 1e-12` and `k * dt < push_duration - 1e-12`, and pass encoder shaft speed to `drive()` in `_Play.step`. Add a unit test verifying `caught(..., dt=0.0025)` holds `torque == 0` for `round(latency / dt)` steps.
- **Basis:** Confirmed by running exact push-onset state traces on all dev falls before and after `H3` (Section 4). Confidence: `high`.

---

### H6 — Lifted-robot `17.0 rad/s` runaway margin at `6.8 V` battery cutoff, uncaught `OSError` in `calibrate_gyro`, and unjoined `watchdog.stop()`
- **Severity:** `major`
- **Area:** 6. Robot code since round 1
- **Location:** `hardware/legobot/robot/balance.py:114-115`, `hardware/legobot/robot/balance.py:465`, `hardware/legobot/robot/balance.py:470-475`, `hardware/legobot/robot/balance.py:512-519`, `hardware/legobot/robot/control.py:366-372`, `hardware/legobot/robot/loop_config.py:66-67`
- **Claim tested:** The smoothed runaway trip stops a lifted robot reliably down to `VIN_CUTOFF = 6.8 V` without false-tripping during the scenario, and startup/shutdown error handling is leak- and race-free.
- **What is wrong:**
  1. **Thin/negative margin at `6.8 V` cutoff:** *(Note: the initial review's duty-cap calculation treated off-ticks as braking rather than slow-PWM coasting; in slow PWM, $d$ cancels in $V/k$ and only scales friction drop $\tau_c/(d a k)$, so `duty 0.6` at `7.3–8.0 V` still reaches `~18–19.4 rad/s`, and floor runs do not use `--duty-cap < 1.0`.)* The real hazard is low battery voltage: at `VIN_CUTOFF = 6.8 V`, nominal free-wheel speed is `17.80 rad/s` (`17.44 rad/s` on the higher-friction right wheel, and `16.69 rad/s < 17.0 rad/s` on a `-7% a, +3% k` motor corner), leaving almost zero or negative margin above the fixed `17.0 rad/s` 1.5 s mean threshold.
  2. **Uncaught `OSError` in `calibrate_gyro`:** `balance.py:470-475` performs `400` IMU reads over `2 s` outside `run()`'s `OSError` bridge and before the main `try...finally`, catching only `ValueError`. A single I²C NACK (`OSError`) aborts without stopping `watchdog` or closing `hat` and `imu`.
  3. **Unjoined watchdog thread:** `balance.py:513` calls `watchdog.stop()` (`_stop.set()`) without `join()`. If the watchdog woke just before `stop()`, its lock-free `emergency_coast()` write can interleave with `coast_check(hat)` or `hat.close()`.
- **Concrete failure scenario:** Lifting the robot near the end of a pack (`6.8–6.9 V`) on the slightly slower right drivetrain (`tau_c = 0.027 N m`) spins at `~16.7–17.4 rad/s`, either never crossing `17.0 rad/s` or taking many seconds for the `1.5 s` rolling mean to trip.
- **Fix and test (reconciled):**
  1. Scale the runaway threshold dynamically with measured `vin` as `0.85 * w_nl(vin)` (`15.13 rad/s` at `6.8 V`, `17.01 rad/s` at `7.6 V`, `18.89 rad/s` at `8.4 V` — verified `0 / 150` false trips across all H3 dev runs in Section 4).
  2. Move `calibrate_gyro` inside the hardware `try...finally` (or close `hat`/`imu` and stop `watchdog` on any exception, bridging single transient `OSError` reads).
  3. Call `self.join(timeout=0.2)` in `Watchdog.stop()`. Add unit tests in `robot/test_control.py`.
- **Basis:** Traced in `balance.py` / `control.py` and verified across 150 dev runs. Confidence: `high`.

---

### H7 — `SimWorld` (`balance --sim` and `robot/test_control.py`) omits `pdm_update`, testing robot code under continuous averaged PWM instead of `10 ms` pulses
- **Severity:** `major`
- **Area:** 6. Robot code since round 1
- **Location:** `hardware/legobot/robot/sim_hw.py:39`, `hardware/legobot/robot/sim_hw.py:65-78`, `hardware/legobot/robot/test_control.py:194-200`
- **Claim tested:** `python -m robot.balance --sim` and `robot/test_control.py` test the robot loop against the configured LEGO slow-PWM pulse-density and backlash physics.
- **What is wrong:** `sim_hw.py:39` comments `# the HAT's own clock: data lines (and slow-PWM ticks) on it`, but `SimWorld` never calls `pdm_update`: `sim_hw.py:75` passes continuous `self.drive` directly to `self.dyn.step(self.state, self.drive, self.h)`. Also `_reference_falls` (`test_control.py:197`) omits `disturbance=cfg.build_disturbance()`.
- **Concrete failure scenario:** Running `balance --sim` or `pytest hardware/legobot/robot/test_control.py` tests `Brain`, `PitchEstimator`, `Feedforward`, and `Safety` with smooth continuous duties and zero `100 Hz` full-voltage pulse chatter across the backlash dead zone.
- **Fix and test:** Add `pdm_total` and `applied` to `SimWorld` when `self.dyn.actuator.pwm_tick > 0`, stepping `pdm_update` every `pwm_steps_per_tick` physics steps and passing `self.applied` to `self.dyn.step(...)` and `specific_force()`. Pass `disturbance` in `_reference_falls` and assert `SimWorld` emits `{-1.0, 0.0, +1.0}` pulse duties under LEGO config.
- **Basis:** Traced in `sim_hw.py` and `actuators.py`. Confidence: `high`.

---

### H8 — Plain LEGO tuning freezes `pwm_tick_phase = 0.0`, while `compare_controllers.py` crashes on saved PID `.json` files and skips config-drift checks on `--existing`
- **Severity:** `major`
- **Area:** 7. Config and test integrity
- **Location:** `config.py:725-734`, `config.py:871`, `config.py:882-907`, `optimization/run_tuning.py:67`, `optimization/batch_tuning/evaluation.py:216-225`, `optimization/compare_controllers.py:306-328`, `optimization/compare_controllers.py:349-358`, `optimization/tuning/reporting.py:46`
- **Claim tested:** No LEGO tuning path runs silently with fixed tick phase, and `compare_controllers.py` loads tuned PID runs and flags old pre-backlash / tick-locked LEGO configs.
- **What is wrong:**
  1. `run_tuning.py --robot legobot` and `run_batch_tuning.py --robot legobot --plain` (`robustness=None`) set `pwm_tick_phase = CONFIG.pwm_tick_phase() = 0.0` on every seed, locking ticks to steps `0, 4, 8, ...` against arrivals `4, 6, 8, ...`.
  2. In `compare_controllers.py:355`, `gain_runs` calls `np.asarray(data["gains"], float)` on `<method>.json` files where `save_result` (`reporting.py:46`) wrote `"gains": result.gains_dict()` (a `dict`), crashing with `TypeError: float() argument must be a string or a real number, not 'dict'`.
  3. `_run_warnings` (`compare_controllers.py:306`) checks only `path.parent / "snapshot.json"` (never `data.get("config")` inside `<method>.json`), and `existing_controllers(path)` (`compare_controllers.py:326-328`) never calls `CONFIG.scoring_differences` on `comparison.json["config"]` at all (`warnings` defaults to `[]`).
- **Concrete failure scenario:** Passing `--gain-runs "CMA=results/legobot/batch_tuning/*/cma_es.json"` crashes immediately with `TypeError`; passing `--existing <old_legobot_comparison.json>` silently compares pre-B5 or tick-locked gains with zero warnings.
- **Fix and test:** Parse `dict` gains in `gain_runs` via `[data["gains"][n] for n in CONFIG.build_search_space().names]`, check `data.get("config")` in `_run_warnings` and `existing_controllers(path)`, draw or warn on `pwm_tick_phase` in `--plain` mode, and add tests in `test_compare_controllers.py`.
- **Basis:** Traced in `config.py`, `reporting.py`, and `compare_controllers.py`. Confidence: `high`.

---

### H9 — IMU specific-force acceleration at control ticks is computed before `begin_step` (`backlash_release`), injecting up to `38.1°` raw tilt spikes
- **Severity:** `minor`
- **Area:** 1. The backlash event model's physics
- **Location:** `simulation/balance_sim/simulator.py:120`, `simulation/batch_sim/simulate.py:188`, `simulation/batch_sim/torch_sim.py:220`, `hardware/legobot/robot/sim_hw.py:90`
- **Claim tested:** IMU `accelerometer_tilt` reads `(x_ddot, theta_ddot)` consistent with unilateral gear-tooth contact (no tensile tooth force).
- **What is wrong:** `end_step` at step `k - 1` sets `contact = sign(gap)` whenever `|gap| >= play`, relying on `begin_step` (`backlash_release`) at step `k` to unlock when `contact * transmitted < 0`. At control ticks (`k % spc == 0`), all engines call `derivatives` / `backlash_terms` **before** `begin_step`, evaluating accelerations with tensile tooth forces when `applied` is `0` or reversed.
- **Concrete failure scenario:** Across `60` LEGO dev runs (`223,579` control ticks), `4,512` ticks (`2.01%`) had `contact * transmitted < 0` at IMU readout, causing raw `accelerometer_tilt` errors of `1.51°` mean, `2.79°` p95, and **`38.08°` (`0.665 rad`) max** (`0.38°` single-step error after the `tau = 0.5 s` complementary filter).
- **Fix and test:** Apply `backlash_release` / `begin_step` to the contact state passed into the IMU `derivatives` call in all three engines and `sim_hw.py`; test in `test_backlash.py`.
- **Basis:** Confirmed by instrumenting `60` reference runs. Confidence: `high`.

---

### H10 — `spectral_radius` uses standstill Coulomb slope (`0.042 N m s/rad`), overstating the `0.3`-ratio baseline's margin at `0.40 m/s` cruise (`1.016` unstable at `35 ms`)
- **Severity:** `minor`
- **Area:** 8. The model-based baseline
- **Location:** `simulation/balance_sim/planar2d.py:117`, `legobot_config.py:246-250`, `hardware/legobot/A4_BASELINES.md:467-475`, `simulation/tests/test_pid_design.py:23-37`
- **Claim tested:** `design_integral_ratio = 0.3` is non-cherry-picked, and `legobot_config.py:248`'s `spectral_radius = 0.9975 at 35 ms` claim holds at cruise.
- **What is wrong:** Selecting `0.3` (smallest `0.1` step stable at standstill) is sound and also minimizes `15 ms` cruise spectral radius (`0.9908` vs `0.9947` at `0.1` and `0.9966` at `0.5`). However, `planar2d.py:117` linearizes $\tau_c\tanh(\dot{\phi}/\omega_c)$ at $\dot{\phi}=0$, adding `0.042 N m s/rad` of viscous damping (`25×` `wheel_friction = 0.0017`) that vanishes at `0.40 m/s` cruise ($\operatorname{sech}^2(11.6) = 8.4 \times 10^{-11}$) and inside the backlash dead zone.
- **Concrete failure scenario:** With `coulomb_friction = 0.0` (tangent slope at `0.40 m/s` cruise), `spectral_radius` for `ratio = 0.3` is `0.9908` at `15 ms`, `0.9994` at `25 ms`, and **`1.0163` (unstable) at `35 ms`**.
- **Fix and test:** Update the comment at `legobot_config.py:248` and check `spectral_radius < 1.0` over `0–20 ms` for both `P` and `P.with_changes(coulomb_friction=0.0)` in `test_pid_design.py`.
- **Basis:** Confirmed by running `stability.spectral_radius` across `0..35 ms`. Confidence: `high`.

---

### H11 — Inclusive `>= 0.5` and `<= -0.5` thresholds in `pdm_update` create a `[-1, +1]` limit cycle during `duty = 0.0` coast
- **Severity:** `minor`
- **Area:** 3. The tick timing after B1
- **Location:** `simulation/balance_sim/actuators.py:277-279`, `simulation/batch_sim/physics.py:240-242`
- **Claim tested:** `pdm_update(total, 0.0)` outputs `0.0` (coast).
- **What is wrong:** Inclusive thresholds on both `+0.5` and `-0.5` cause `total = ±0.5` followed by `duty = 0.0` to alternate `-1.0, +1.0, -1.0, +1.0` forever instead of coasting (`0 / 111,788` noisy float64 ticks hit exact `±0.5`, but deterministic `duty = 0.5` then `0.0` triggers it immediately).
- **Concrete failure scenario:** `tot, p1 = pdm_update(0.0, 0.5)` -> `(-0.5, +1.0)`; subsequent `pdm_update(tot, 0.0)` ticks emit `[-1.0, +1.0, -1.0, +1.0, ...]`.
- **Fix and test:** Use strict `> 0.5` and `< -0.5` in `actuators.py:278` and `physics.py:240-241`; add a test in `test_slow_pwm.py`.
- **Basis:** Confirmed by running `pdm_update` directly. Confidence: `high`.

---

### H12 — Missing short-window tight `torch` vs `NumPy` full-loop test and 4 overly broad LEGO test skips
- **Severity:** `minor`
- **Area:** 7. Config and test integrity
- **Location:** `simulation/tests/test_torch.py:42, 257`, `simulation/tests/test_velocity.py:111-145, 235`, `simulation/tests/test_conditions.py:157`, `optimization/tests/test_compare_controllers.py:95`
- **Claim tested:** Loosened `TOL = 1e-7` / `1e-4` tolerances and 13 LEGO skips hide no real LEGO bugs.
- **What is wrong:** No test compares full-loop `TorchSimulator.run()` against `simulate_batch()` tightly (`atol=1e-12`) over an `80`-step (`0.20 s`) window before chaotic impact/pulse divergence; and skipping `test_velocity.py` (`3` tests), `test_conditions.py:157`, and `test_compare_controllers.py:95` hid outer-loop coverage and `H8`'s `gain_runs` crash.
- **Fix and test:** Add an `80`-step `atol=1e-12` full-loop LEGO equivalence test in `test_torch.py`, enable `test_velocity.py`, multi-seed `test_latency_hurts_more_than_none`, and a `compare_controllers` smoke test with `--gain-runs` under `ROBOT=legobot`.
- **Basis:** Confirmed via `ROBOT=legobot pytest -rs`. Confidence: `high`.

---

### H13 — Stale documentation and missing backlash entry in `E1_HANDOFF.md` Step 2 ablations
- **Severity:** `minor`
- **Area:** 7. Config and test integrity
- **Location:** `hardware/legobot/E1_HANDOFF.md:3, 29, 55-63`, `hardware/legobot/BENCH_CHECKS.md:12-13`, `legobot_config.py:186`, `PHYSICS.md:567-571, 1010, 1071`, `hardware/legobot/DATASHEET.md:102-104`
- **Claim tested:** Current notebook sections and config comments match the code on the branch.
- **What is wrong:** `E1_HANDOFF.md` Step 2 (`lines 55-63`) omits gearbox and coupling backlash from the ablation list and still describes Part 2 as unported; `BENCH_CHECKS.md:12-13` still says *"no arming or running without the line lock"*; `legobot_config.py:186` and `PHYSICS.md:1010` cite `l = 90 mm` (`0.011 rad/mm`) instead of `60 mm` (`0.017 rad/mm`); `PHYSICS.md:567, 571, 1071` cite stale sensor/push numbers (`0.005 rad`, `0.0135 rad`, `0.70 N`); `DATASHEET.md:102-104` says *"Not yet in legobot_config.py"*.
- **Fix and test:** Add gearbox and coupling backlash to `E1_HANDOFF.md` Step 2 ablations and update the stale lines above.
- **Basis:** Traced across docs and `legobot_config.py`. Confidence: `high`.

---

## Round-1 Regressions & Checked-and-Sound Summary

### Round-1 Regressions
- **`G1` + `3321229` (model-based PID fix):** Raising `design_integral_ratio` to `0.3` moved `model_based_gains()` to `ki = 5.24 > SEARCH.ki_max = 5.0` (`legobot_config.py:248, 313`; `H1`).
- **`G11` (runaway trip):** Fixed `17.0 rad/s` threshold was not scaled with `vin` down to `VIN_CUTOFF = 6.8 V` (`balance.py:465`; `H6`).
- **`G2` (IMU `OSError` + watchdog):** `calibrate_gyro()` (`400` boot reads) sits outside `OSError` handling and hardware `finally` cleanup, and `watchdog.stop()` omits `join()` (`balance.py:470-475, 513`; `H6`).
- **`G10` (LEGO test coverage):** `SimWorld` (`sim_hw.py:65-78`) still omits `pdm_update`, leaving `robot/test_control.py` on smooth averaged PWM (`H7`).

### Checked and Sound
- **3-DOF backlash impulse & mass-matrix coupling** (`planar2d.py:259-277`, `physics.py:130-156`, `backlash_check.py`): Conserves coupled wheel-rotor-body generalized momenta to `< 1e-15`, strictly dissipates kinetic energy, and matches `0.1 ms` stiff contact.
- **Transmitted-torque unlock sign & step-event bounds** (`planar2d.py:240-257`): `contact * (tau_m - J_r * phi_w_ddot) < 0` has the right sign on both stops, bounds chatter to `<= 1` impact per step, and produced `0 / 10,214` tensile impacts across 60 robust runs.
- **Unlocked friction, motor back-EMF speed, and encoder/feedforward shaft** (`planar2d.py:194-237`, `actuators.py:247-268`, `sim_hw.py:56-58`): Friction and back-EMF act on `MOTOR_SPEED` while wheel/body coast torque-free; `SHAFT` (`state[6]`) sits strictly between gearbox and coupling play across all engines.
- **Three-engine event equivalence, twin interleaving, and chunking** (`simulate.py`, `torch_sim.py`, `evaluation.py:92-120`): Reference and batched NumPy are bit-identical in float64; `torch` matches single-step kernels to `1e-13`.
- **Arrival-before-tick ordering & no `pwm_tick_locked` leakage** (`simulator.py:126-131`, `simulate.py:239-243`, `torch_sim.py:272-276`, `evaluation.py:216-225`): Commands arriving on a tick step pop before `pdm_update`, and `ACTUATOR.pwm_tick_locked` is `False` everywhere on LEGO.
- **8,192 corner-cube composition, non-fall cost headroom, optional line lock, seed-tier disjointness, and snapshot fields**: All verified sound (`max` non-fall cost `45.3` at `20 s`, `154.4` at `60 s` vs `fall_penalty = 1000`).

---

## Builder Response & Numerical Reconciliation (`H3`, `H5`, `H2`, `H6`)

After reviewing Questions 1–4 from Round 2, the builder proposed concrete resolutions for `H3`, `H2`, and `H6`, and requested an empirical simulation check to settle `H5` before editing the repository. All four items were verified numerically against `d252bff` using `C:\venvs\balancebot\Scripts\python.exe`:

### 1. `H3` — Gearbox Play (`±1.1–1.6°`, nominal `1.3°`) + Re-Derived Push Bound
- **Builder recommendation:** Set `gearbox_backlash` to `±1.1–1.6°` (nominal `1.3°`), excluding the `+2°` unbraked rotor slip at `t = 1–2 s` while allowing margin for `1°` encoder counts and the unmeasured right motor; keep `coupling_backlash` at `±0–3.0°` (nominal `1.5°`).
- **Numerical result:** Across all `8,192` corners with `gearbox_backlash` in `[1.1°, 1.6°]` and `coupling_backlash` in `[0°, 3.0°]` (worst total play `±4.6°` on corner `2511`, down from `±6.0°`), `largest_catchable_impulse` at `0.40 m/s` and `27.5 ms` reaction rises from `0.03423 N s` to **`0.03772 N s`**, re-deriving the `98%` scenario push force from `0.67 N` to **`0.74 N`** (`round(0.98 * 0.03772 / 0.05, 2) = 0.74 N`).

### 2. `H5` — Exact Push-Onset Full-Torque Catchability Test Across All Dev Falls
- **Builder test executed:** On all `150` dev scenarios (`seeds 5000–5049 × 3 start tilts`) for both `hand-tuned` and `model-based` baselines, every run that fell within `2.5 s` after a push (`t = 5.5 s` or `12.5 s`) was re-simulated from its **exact closed-loop state at push onset** (`state`, `pdm_total`, `applied`, `pending` latency queue, and `pwm_tick_phase`), commanding full recovery duty (`±1.0`) starting at the first sample after push onset (`Test A`, actual scenario latency/PDM pipeline) and at the `27.5 ms` worst-case reaction (`Test B`):
  - **Before `H3` (`±2.6–3.0°` gearbox play, `0.67 N` push):**
    - `hand-tuned` (`17/150` dev falls, `4` post-push): `3` failed catch (`scen=88`, `seed=5029, tilt=-0.15` started upright/backward at `th = -1.74°`, `th_d = -0.39 rad/s`, **but had `d_omega = -9.79 rad/s` at `gap = -0.68`**, confirming that an opposing pulse inside the oversized `±6°` gap genuinely made `0.67 N` uncatchable; the other `2` entered `5.5 s` with unsettled forward velocity/pitch from the `2.0 s` step).
    - `model-based` (`9/150` dev falls, `4` post-push): `3/4` caught in actual pipeline; `1` (`scen=28`, `th = +4.89°, th_d = +0.59 rad/s`) entered `5.5 s` already falling forward.
  - **After `H3` (`±1.1–1.6°` gearbox play, re-derived `0.74 N` push):**
    - `hand-tuned` (`10/150` dev falls, `3` post-push): **every settled push state was caught**, including opposing-rotor-speed cases `scen=49` (`d_omega = -2.88 rad/s, gap = -0.63` -> `caught = True`) and `scen=121` (`d_omega = -4.73 rad/s, gap = +0.37` -> `caught = True`). The sole uncaught case (`scen=82`, `d_omega = 0.00`) entered `t = 5.5 s` already at **`th = +11.34°`** from failing to settle the `t = 2.0 s` acceleration step.
    - `model-based` (`5/150` dev falls, `2` post-push): `scen=125` (`th = -0.39°, d_omega = +0.09` -> `caught = True`); the sole uncaught case (`scen=76`, `d_omega = 0.00`) entered `t = 5.5 s` already leaning **`th = +7.80°`** from velocity-step overshoot.
  - **Why worst-pulse `d_omega` should NOT be stacked onto worst corner `2511`:** Adding a `10 ms` braking pulse's `d_omega = -8.0` to `-12.5 rad/s` onto corner `2511` (`6.8 V`, max friction, `±4.6°` play) collapses the `98%` push bound from `0.74 N` to `0.13 N -> 0.00 N`. Because `0.74 N` under `H3` already catches `100%` of settled push states on dev seeds, **keep the resting far-side catch rule (`0.74 N`)** and apply only the `k * dt < latency - 1e-12` and encoder-speed fixes in `legobot_scenario.py`.

### 3. `H2` — Disabling `sensitivity_weight` on LEGO + Updated Baseline Scores
- **Builder recommendation:** Set `sensitivity_weight = 0.0` in `legobot_config.ROBUSTNESS` (halving E1 GPU time and removing `92–294` points of random fall-flip penalty).
- **Updated NumPy float64 baseline table (with `H3` `±1.1–1.6°` gearbox play + `H2` `sensitivity_weight = 0`):**

| Configuration | Controller | Train `mean_cost` (`falls / 150`) | Train `objective` | Dev `mean_cost` (`falls / 150`) | Dev `objective` |
|---|---|---:|---:|---:|---:|
| Branch `d252bff` (`2.6–3.0°`, `0.67 N`, `sens=1`) | Hand-tuned | `170.87` (`12`) | `453.49` | `226.51` (`17`) | `520.85` |
| Branch `d252bff` (`2.6–3.0°`, `0.67 N`, `sens=1`) | Model-based | `72.99` (`5`) | `130.19` | `120.88` (`9`) | `212.72` |
| `H3 + H2` at `0.67 N` (`1.1–1.6°`, `sens=0`) | Hand-tuned | `104.76` (`7`) | `104.76` | `136.01` (`10`) | `136.01` |
| `H3 + H2` at `0.67 N` (`1.1–1.6°`, `sens=0`) | Model-based | `35.71` (`2`) | `35.71` | `61.15` (`4`) | `61.15` |
| **`H3 + H2` re-derived `0.74 N` (`1.1–1.6°`, `sens=0`)** | **Hand-tuned** | **`104.83` (`7`)** | **`104.83`** | **`136.31` (`10`)** | **`136.31`** |
| **`H3 + H2` re-derived `0.74 N` (`1.1–1.6°`, `sens=0`)** | **Model-based** | **`46.89` (`3`)** | **`46.89`** | **`72.60` (`5`)** | **`72.60`** |

### 4. `H6` — Voltage-Scaled Runaway Threshold (`0.85 * w_nl(vin)`)
- **Builder correction & recommendation:** Confirmed — no floor runs use `--duty-cap < 1.0` (which would also violate full-torque push sizing), and in slow PWM off-ticks coast so `duty 0.6` at `7.3–8.0 V` still reaches `~18–19.4 rad/s`. Scaling `runaway_speed = 0.85 * w_nl(vin)` using nominal `w_nl(V) = (a * V - tau_c) / (a * k + b)` gives **`15.13 rad/s` at `6.8 V`**, **`17.01 rad/s` at `7.6 V`**, and **`18.89 rad/s` at `8.4 V`**.
- **Numerical check across all `150` H3 dev scenarios (`0.74 N` pushes):** **`0 / 150` false trips**; the peak `1.5 s` sliding-window `|wheel_speed| / threshold(vin)` ratio across all non-falling runs was **`0.923`** (`7.7%` floor headroom, while maintaining `~15%` margin below lifted free-wheel speed across `6.8–8.4 V`).

---

## Final Pre-E1 Action Checklist (In Order)

1. **Search space & optimizer safeguards (`H1`, `H2`):** Set `SEARCH.ki_max = 15.0` in `legobot_config.py` (and assert baseline interiority in `test_config.py`), set `ROBUSTNESS.sensitivity_weight = 0.0`, warm-start/inject `model_based_gains()` in `cma_es_search`, and re-score `es.result.xfavorite` + top-$K$ GPU elites in NumPy float64.
2. **Gearbox play, scenario re-derivation, and latency support (`H3`, `H5`, `H4`):** Update `gearbox_backlash` to nominal `1.3°` (`±1.1–1.6°` range), fix `k * dt < latency - 1e-12` and encoder-speed `drive()` in `legobot_scenario.py`, re-derive pushes (`0.74 N` at current latency, or re-run if `latency_range_s` shifts to `(0.010, 0.015)` for `H4`), and update `A4_BASELINES.md` / `DATASHEET.md` / `SOFTWARE_PLAN.md`.
3. **Evaluation & doc integrity (`H8`, `H13`):** Fix `compare_controllers.py` (`gain_runs` dict gains, `_run_warnings` `data["config"]` check, `existing_controllers` `scoring_differences` check), handle plain-mode `pwm_tick_phase`, and add backlash to `E1_HANDOFF.md` Step 2 ablations.
4. **Robot safety & `SimWorld` fidelity (`H6`, `H7`):** Scale `Safety.runaway_speed` as `0.85 * w_nl(vin)`, guard `calibrate_gyro` with `OSError`/cleanup handling, add `watchdog.join()` on stop, and wire `pdm_update` into `sim_hw.py`.
5. **Engine polish & test coverage (`H9`, `H10`, `H11`, `H12`):** Release tensile `contact` before IMU `derivatives`, switch `pdm_update` to strict `> 0.5` / `< -0.5`, add the `80`-step `atol=1e-12` LEGO `torch` vs `NumPy` test, and un-skip the applicable LEGO velocity/comparison tests.
