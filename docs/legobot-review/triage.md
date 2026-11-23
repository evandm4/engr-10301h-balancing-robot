# Triage of the external LEGO review (2026-11-23)

The review used the prompt and repomix bundle in this folder. Each finding was checked against the code
before anything changed. The status for each one:
- **fixed:** changed and tested.
- **decide:** needs a choice from Evan.
- **bench:** needs a hardware check.
- **open:** real, not done yet.
- **rejected:** the reasoning doesn't hold.

| # | Finding | Status | What happened |
|---|---|---|---|
| G1 | `kv_p_max = 0.9` inherited from Pololu | fixed | `legobot_config.SEARCH`: `kv_p_max = 2.0`. Gemini's evidence that 0.9 binds was weak (hand tuning found 0.7 already too much), but the wider bound costs nothing |
| G2 | An IMU `OSError` crashes the loop and loses the logs | fixed | One failed read is bridged with the last sample. Two in a row trip, coast and log (`IMU_ERROR_LIMIT`). Any exception now still writes the logs, then re-raises. Cleanup coasts first, before stopping the watchdog. `coast_check` handles missing data. Tested with `--inject imu` and a forced crash. Gemini's "blocking forever" claim was wrong: `coast_check` called `hat.coast()` first |
| G3 | `drive()` above the no-load speed | fixed | Worse than reported: **any** command ≥ 0 above the no-load speed gave full forward duty, which brakes (−0.091 N·m), not just a zero command. Now duty 0 (coast) when the denominator is ≤ 0, in both engines (torch shares the NumPy code). Tests: never against the command, at any speed. Baselines re-scored: no change past the third decimal |
| G3.1 | Does duty 0 (and an off-tick) coast or brake? | bench | Spin a wheel by hand under `port 0 ; coast` and under `pwm ; set 0`, and compare. The model assumes coast |
| G3.3 | The robot clips per wheel, the sim clips on the mean speed | open, minor | Matters only when one wheel saturates. Left as is |
| G4 | The catch bound leaves out the pulse tick's wait | fixed | Confirmed: `nominal_robot()` had no `pwm_tick`, and `RECORDED_CATCH` held the 0.27 m/s numbers. Now `catch_latencies`: with locked ticks a push gets latency + one tick (25 ms), a start tilt gets the latency (15 ms). The scenario re-derives unchanged (1.02 N, the same tilts): the old bound also used 25 ms, but the old sim could exceed it by up to a tick. `RECORDED_CATCH` updated |
| G5 | Missing backlash; float32 with the pulses | fixed (float32) / open (backlash) | `float32` against `float64` on LEGO: median cost difference 2.6% (Pololu < 0.1%). `run_batch_tuning` now warns on `--float32` with pulses. **Run the LEGO searches in float64** (~2.7× slower on the GPU). Backlash was already a known open item (B5) |
| G6 | Latency range plus tick wait, against the line-locked robot | fixed | `ActuatorConfig.pwm_tick_locked`: each scenario's ticks fall where its commands arrive (`balance_sim.locked_pwm_tick_phase`), so the latency is the whole sample-to-pulse delay, now 7.5–15 ms (was 5–25 ms plus up to 10 ms). The engines are unchanged; the evaluators set the phase (`BatchEvaluator` per scenario). The robot won't arm without the lock and trips if it's lost (`--inject unlock`). New tests: pulses land on the arrival step at every latency; locked configs match across engines. Robust baselines: hand-tuned train 32.2 (2 falls) → 10.7 (0), model-based dev 28.6 (2) → 6.5 (0). **Bench:** re-run `lock_sweep` after a HAT power cycle to confirm the tick offsets hold |
| G7 | Battery below the simulated supply range | fixed (cutoff) / bench (rail) | `VIN_CUTOFF` 6.6 → 6.8 V, the lowest supply robust scoring draws (widening the spread instead would change the scenario derivation). Bench: no-load speed at 8.4 V, to confirm the motor rail follows vin (all fits were run at 7.3–7.5 V) |
| G8 | Left/right asymmetry; friction spreads | bench / open | `a` was measured on the left motor only: run `lever.py --side right`. Friction on the floor (axle under load) isn't measured. Gemini's claim that the feedforward amplifies yaw is wrong: the faster wheel gets more duty so both deliver equal torque |
| G9 | The linear fit in `motor_steps.py` is meaningless in slow PWM | open, minor | Confirmed (`k_eff` 0.704, `v_dead` −7.73 V in the JSON). Nothing reads those values. The joint slow-PWM fit behind `legobot_config` should be added to `fit.py` |
| G10 | Tests never run under the LEGO config | fixed | `test_config` `KeyError` fixed. 10 tests that failed under `ROBOT=legobot` were fixed (variants that weren't what their names said under LEGO, hand-built robots that dropped the slow-PWM settings, Pololu-only assumptions). New `optimization/tests/test_legobot_pipeline.py` runs every LEGO feature together, scenario by scenario against the reference, under either robot |
| G11 | Watchdog fires once; no trip for a lifted robot; stall duty | fixed / open | The watchdog re-sends coast every 50 ms until the run ends. New runaway trip: both wheels ≥ 17 rad/s the same way for 1.5 s (no false trip in the closed-loop scenario test). `STALL_DUTY` is left at 0.5 until a bench check |

## Found while fixing (not in the review)

- **The model-based PID is linearly unstable at the 10 ms design latency on the LEGO robot** (spectral
  radius 1.003; `test_pid_design` under `ROBOT=legobot`). It still balances in the nonlinear sim. Worth
  understanding before it's used as a baseline or a starting point.
- **The IMU estimate error reaches 0.057 rad** through the scenario (`test_imu`, limit 0.05 set for Pololu).
  Check whether that is the pushes or the accelerations.
- **The linear voltage mode with the LEGO sensors differs by 1 ulp** in the last 4 steps of one falling run.
  Costs and falls agree. The LEGO robot doesn't use that mode.
- **Still failing under `ROBOT=legobot`, unchanged from before:** the ML-policy tests, and the linear-margin
  tests with recorded Pololu/LQR numbers. These are out of scope for the LEGO robot and should probably be
  marked Pololu-only. Under the default robot, the one failure (`test_policy` gate B) is also unchanged.
- **The torch engine tests didn't run here** (no torch in this Python). Run
  `ROBOT=legobot python -m pytest simulation/tests/test_torch.py optimization/tests/test_legobot_pipeline.py`
  on the GPU machine.
