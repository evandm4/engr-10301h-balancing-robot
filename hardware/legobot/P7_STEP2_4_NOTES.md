# P7 steps 2–4 notes: bench, floor and disturbance runs with `l_e2_s1` (2026-11-25)

These are bring-up steps 2–4 of [POLICY_ON_ROBOT_PLAN.md](POLICY_ON_ROBOT_PLAN.md) P7, following
[P7_STEP1_NOTES.md](P7_STEP1_NOTES.md):
- step 2 is the bench, wheels off;
- step 3 is the floor, standing still;
- step 4 is disturbances.

The policy is the **provisional** `l_e2_s1` (see E2_NOTES.md "E2.7 (provisional)"); it is not the E2.6 selection.

Raw logs are not in the repo. They are on the laptop at `C:\balancebot_training\legobot\pi_logs\e2_deploy\`
(raw/run/summary per run), and the originals are in `~/legobot/logs` on the Pi. The operator scripts are in
`C:\balancebot_training\shared\pi\`:
- `e2_post_deploy.sh`: sets the governor, checks the config, runs the golden self-test;
- `e2_tilt.sh`;
- `e2_bench.sh`;
- `e2_floor.sh`;
- `e2_list_logs.sh`.

## Setup

- **Deploy:** `6fe3e0b`, clean tree. `deploy.sh` stamp "6fe3e0b 2026-11-25 17:36". The stale `quick_l` bundle was removed.
- **CPU governor:** `e2_post_deploy.sh` sets `performance` (1.2 GHz on all 4 cores).
  - The setting **does not survive a reboot.** The Pi rebooted once (a Wi‑Fi drop when the robot moved to the test
    site), and the script was re-run before run 2.
- **Strict P2b config check plus golden self-test:** all 4 bundles have cfgdiff `[]` and error 0.
  - l_e2_s1 sha256 is `58cb3c615a43…`.
  - The fast forward pass gives l_e2_s1 a p50 of 25.8 µs and a p99 of 47.5 µs.
- **Calibration:** `calibration.json` pitch_zero is −0.0027 rad (−0.15°). Gyro bias is about +0.046–0.048 rad/s.
- **Duty cap:** 1.0 (default).

## Lying-down calibration procedure (used for all floor runs; no code change)

1. Start the run with the robot **lying on its side, still**. The gyro calibration runs then.
2. When the Pi prints "hold upright" (about 15–20 s after launch over ssh), lift the robot through vertical. The
   loop arms when the tilt is inside the arm window. Arming took 1.8–22.9 s of the 30 s window.
3. Support it lightly until it balances. The run ends at `--seconds` (the flag is `--seconds`, not `--duration`), or
   on a tilt trip.

In run 2 the robot was lifted early, before the prompt. The calibration was still clean.

## Runs

| # | Stamp | Controller, goal, length | Result |
|---|---|---|---|
| 1 | 17:41–17:43 | bench ×5 (model, l_pid, l_e2_s1, l_e2_s1 `--log-lines`, l_e2_s1 `--switch-interval-us 200`) | all done, 0 overruns, 100 % locked, vin 7.86–8.05, cmd_phase sd 0.095–0.106 ms (sw200: 0.122). Replays OK ×4 (diff 0) |
| 2 | `235553` | model PID, still, 10 s | done; armed 1.81 s; rms θ 2.6°, max 7.8°; drift +0.21 m; \|duty\| max 0.55; **cost 3.84** |
| 3 | `235826` | `policy:l_pid`, still, 10 s | done; armed 22.9 s; rms θ 2.4°, max 7.8°; drift −0.04 m; \|duty\| max 0.53; **cost 3.40**; REPLAY OK |
| 4 | `000104` | `policy:l_e2_s1`, still, 10 s | done; armed 10.4 s; rms θ 2.3°, max 5.9°; drift +0.08 m; \|duty\| max 0.38; 0 saturated; **cost 2.70**; REPLAY OK |
| 5 | `000439` | l_e2_s1, still, 20 s, 5 gentle taps | done; tap peaks −18.7, −14.2, −12.0, −15.5, +18.6°; settle 0.28–0.73 s; \|v\| ≤ 0.44 m/s; \|duty\| max 0.82; 0 saturated; REPLAY OK |
| 5b | `000828` | l_e2_s1, still, 20 s (Evan-run) | done; about 11 pushes, peaks up to ±27°; isolated taps settle in 0.3–0.6 s; saturated 2/4004; REPLAY OK |
| 5c | `000949` | l_e2_s1, still, 40 s (Evan-run) | survived 36.3 s of about 20 taps and repeated drives into an obstacle (peaks to −27.7°, saturated 0.9 %), then **Evan's deliberate grab and tilt** caused a tilt trip at 36.27 s. `score_run` labels it FELL; it was intentional (\|θ\| < 10° just before). REPLAY OK |

Every floor run had 0 overruns, every tick line-locked, and vin 7.84–7.95 V.
- Settle means \|θ\| < 4° held for 0.5 s.
- "REPLAY OK" means `replay_policy.py` on the laptop matched with max torque diff 0.
- The envelope checks are logged only:
  - θ̇ was outside the training envelope on about 2 % of ticks (worst 1.08× the max for l_pid, 0.88× for l_e2_s1);
  - the commanded torque reached 1.02× the max at tap peaks.

**Step 3 verdict:** `l_e2_s1` holds still better than both PIDs on the floor: cost 2.70, against 3.40 for l_pid and
3.84 for model. This is one 10 s run each. The rank matches sim, but the margin is not a measured statistic.

**Wobble (run 4; scratch `wobble.py`).** θ peak-to-peak per 2 s window was 7.9–11.4° (sim ≈ 9°), with the dominant
frequency at 2.4 Hz. It did not grow, and the ~3.8 Hz mode the review warned about did not appear. The PIDs measured
8.6–13.7° p-p, at 1.8 and 2.7 Hz.

**Step 4 verdict:** the policy recovers from taps of 12–19° within 0.3–0.7 s. It survives pushes up to ~27° and
repeated obstacle contact, with brief saturation (0.9 %).

## Timing under the `performance` governor

`cmd_at` p50 (tick start → command write, ms):

| Controller | Bench | Floor |
|---|---|---|
| model PID | 1.019 | 0.944 |
| l_pid | 1.019 | 0.961 |
| l_e2_s1 | 1.013–1.051 | 0.958–0.964 |

- **Policy ≈ PID in-loop.** Policy − PID was +0.00 ms on the bench and +0.014–0.020 ms on the floor. The self-test
  delta was +0.0095 ms. So at full clock there is no policy "extra" left.
- **Effect of the old (ondemand) constants.** With `COMPUTE_LEAD_S` 1.2 ms and `POLICY_LEAD_EXTRA_S` 0.16 ms, the
  writes landed early against the 6.5 ms `LINE_WRITE_PHASE_S` target: 0.18 ms for the PID and 0.33–0.41 ms for the
  policies. That is the safe side, away from the 7.2 ms cliff.
- **Worst case with the new lead.** The worst `cmd_at` p99 was 1.34 ms. With a 1.0 ms lead that puts the write at
  about 6.5 − 1.0 + 1.34 = 6.84 ms, still before the 7.2 ms left-tick cliff.
- **Code (committed after these runs, not yet on the Pi).**
  - `loop_config.COMPUTE_LEAD_PERF_S = 0.0010` and `POLICY_LEAD_EXTRA_PERF_S = 0.0`.
  - `control.lead_constants(governor)` uses this pair only when the loop core's `scaling_governor` reads exactly
    `performance` (`control.cpu_governor()`). Otherwise it keeps the ondemand pair, which is the longer, safe lead
    after a reboot.
  - `balance.py` logs `cpu governor` in the run notes.
  - **Before relying on it:** redeploy, then run one bench run plus one floor run and check that `cmd_phase` is
    centred on 6.5 ms.
- **`--switch-interval-us 200`:** no gain (cmd_phase sd 0.122 vs 0.095–0.106 ms). Don't use it.
- **P3d (sample-to-HAT latency, same formula as step 1).** For the PID and L at full clock: DLPF ~3 + `cmd_at` +
  serial 2.37 + HAT apply ~3. That gives about 9.3–9.4 ms typical (p50 0.94–1.05) and 9.7 ms at the worst p99
  (1.34). Step 1 under ondemand gave 9.5–10.5, and the simulator's `latency_range_s` is 7.5–12.5 ms.
- **Not measured yet:** the MLP policies (a_warm, b1) in-loop at full clock, and P3d for b1 with P3c.

## Still to do

- **Run 6: `l_e2_s1` with goal `scenario`, 20 s.** It needs ≥ 2.5 m of clear space, and the robot is expected to arc
  right. Stop and fall back to the model PID if a growing ~3.8 Hz wobble appears at t = 4–10 s. Neither of Evan's
  extra runs used the scenario goal, so this is still open.
- Redeploy with the governor-aware lead, then do the bench plus floor check above.
- Time a_warm/b1 in-loop under `performance` (`--no-drive` is enough) and re-measure P3d for b1.
