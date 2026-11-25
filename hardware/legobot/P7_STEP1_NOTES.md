# P7 step 1 notes: policies on the Pi, no drive (2026-11-24/25)

Bring-up step 1 of [POLICY_ON_ROBOT_PLAN.md](POLICY_ON_ROBOT_PLAN.md) P7: self-test, timing (P3a), `--no-drive` runs,
laptop replay (P4b). The follow-up code is on branch `legobot-robot-perf` (P3c, P3b, LineClock). The raw Pi logs are
not in the repo. They are on the laptop at `C:\balancebot_training\legobot\pi_logs\p7_step1\` (13 runs × raw/run/summary
plus `selftest.json`), and the originals are in `~/legobot/logs` on the Pi.

## What was run

- **Setup:** Pi 3B at `4375caf`, loop at SCHED_FIFO 50 pinned to CPU 3, watchdog FIFO 51, HAT reader FIFO 49 (not
  pinned), mlockall, line lock on.
- **Bundles:** l_pid, quick_l, a_warm_zero and b1_zero, run as bare `.npz` with `--allow-unchecked`, because
  `policy:<name>` failed to deploy (see the deploy bugs below). The golden self-test was run separately (`selftest.json`).
- **Runs:** 13 no-drive runs of 10–15 s. All ended "done" with 0 overruns and every tick locked.
  - Hands-free (robot lying still): `20261125_034903` hand, `035635` l_pid, `035653` a_warm_zero, `035710` b1_zero,
    `035728` quick_l, `035746` hand2.
  - Held by hand: `040423` hand, `040503` hand, `041124` l_pid, `041238` a_warm_zero, `041310` a_warm_zero.
  - With systemd-timesyncd stopped, hands-free: `042758` b1_zero_nontp (15 s), `042821` hand_nontp.
- **Replay:** b1_zero_nontp replayed on the laptop: REPLAY OK, 3002 ticks, max torque diff 0 (tolerance 1e-12).
  Golden self-test error was 0 for all bundles.

## Timing (ms)

Forward pass alone, golden self-test on the Pi (numpy reference path):

| Bundle | mean | p50 | p99 |
|---|---|---|---|
| l_pid | 0.09 | 0.088 | 0.11 |
| quick_l | 0.09 | 0.079 | 0.11 |
| a_warm_zero | 1.06 | 1.054 | 1.10 |
| b1_zero | 1.17 | 1.162 | 1.24 |

Self-test maxima reached 38–51 ms (one-off stalls in the 400-tick self-test, outside the loop).

In the loop, tick start → command, including the ~0.55 ms IMU read (`compute` column; ranges over the runs, from
the summaries):

| Controller | mean | p50 | p99 | max |
|---|---|---|---|---|
| hand PID (5 runs) | 1.01–1.13 | 0.96–1.20 | 1.41–1.70 | 1.58–2.06 |
| L (l_pid ×2, quick_l) | 1.37–1.41 | 1.40–1.43 | 1.63–1.69 | 1.91–1.98 |
| a_warm_zero (3 runs) | 2.11–2.18 | 2.10–2.16 | 2.69–2.76 | 3.06–3.26 |
| b1_zero (2 runs) | 2.29–2.30 | 2.29–2.30 | 2.65–2.91 | 3.64–3.79 |

## P3d verdict

Sample-to-HAT latency, excluding the pulse-tick wait. It is computed as the IMU DLPF (~3 ms, est.) + the in-loop
compute + the serial command (2.37 ms) + the HAT apply (~3 ms, est.):

| Controller | typical | worst |
|---|---|---|
| hand PID | 9.5–9.6 | 10.5 |
| L | 9.9 | 10.4 |
| a_warm_zero | 10.6 | 11.7 |
| b1_zero | 10.7–10.75 (p99 11.4) | 12.2 (15 s no-NTP run) |

Every controller is inside the simulator's `latency_range_s` of 7.5–12.5 ms. b1 has only 0.3 ms to spare at worst,
which is why P3c was needed. With P3c, the projected b1 worst case is about 11.3 ms; this needs re-measuring.

## Anomalies

1. **LineClock period drift.** The in-run estimate moves away from its start value, by end − start:
   - 7 runs drift −290 to −1187 ppm, 5 drift positive (+126 to +554), and 1 drifts −155.
   - The start fits, taken in the quiet gyro-calibration phase, cluster near 0 ppm (−25 to +204). There are 4
     negative outliers: −511 (`035653`), −285 (`035728`), −236 (`040423`) and −514 (`041310`). These are the 4
     runs whose estimates then drift positive (+554, +336, +231, +407), back toward −107 to +51 by the end.
   - Stopping timesyncd changed nothing (−290 and −467 ppm drift; ends at −289 and −448), so the NTP hypothesis and
     its "stop timesyncd" recommendation are **withdrawn**.
   - The cause is the GIL: see "LineClock" below. The drift is harmless to the lock, because t_ref re-anchors every
     100 ms.
   - Per-run values: `verify_f6.py` (agent scratch dir) over the 13 `*_summary.txt` files.
2. **Motor-data age is bimodal** (sd 2.5–2.9 ms). This is expected: with 5 ms ticks and 10 ms lines, alternate ticks
   see ages a and a + 5 ms.
3. **Occasional negative motor-data ages.** A line is stamped between the tick start and the `hat.pair()` read, while
   the IMU read has released the GIL. This is consistent with the GIL finding below.
4. **quick_l saturated on 99.7% of ticks** in the lying pose (theta −0.02 rad, no drive). This is expected.
5. **The Pi's disk was 99% full** (115 MB free) at the start; it was freed to 1.6 GB.

## Deploy bugs found (fixed on `legobot-deploy-fix`)

1. `deploy.sh` didn't ship `optimization/tuning`, which `config.snapshot()` → `build_search_space()` needs.
2. `snapshot()` → `build_lqr_gain()` needs scipy, which isn't on the Pi, so `policy:<name>` couldn't run at `4375caf`.
3. CRLF line endings in `robot/*.sh` from the Windows checkout caused a `run_safe.sh` syntax error.
4. Bundle `files_sha256` was computed over CRLF JSON, so an LF checkout would fail `_check_files_sha256`.

## Follow-up on `legobot-robot-perf` (laptop only; re-measure on the Pi)

- **P3c: fast forward pass.** `balance_sim.FastPolicyController` (`compile_forward`) is the default on the robot;
  `--reference-forward` selects the old path.
  - Torques are bit-identical to the reference on the golden vectors of all 4 bundles and on 20 000 random
    observations.
  - Laptop `update()` p50, single thread, reference → fast (µs):

    | Bundle | reference | fast |
    |---|---|---|
    | l_pid | 3.2 | 1.1 |
    | quick_l | 3.0 | 0.9 |
    | a_warm_zero | 38.1 | 8.1 |
    | b1_zero | 42.0 | 9.5 |

  - The Pi/laptop ratio of the reference p50s is ~27.5. Projected Pi fast `update()` p50: 30, 24, 224 and 263 µs, so
    a_warm and b1 drop about 0.83 and 0.90 ms per tick.
- **P3b: compute lead.** The lead is now `COMPUTE_LEAD_S + max(0, policy p50 − PID p50) + POLICY_LEAD_EXTRA_S`
  (0.16 ms, fitted on l_pid's in-loop compute).
  - The old formula placed the L-type writes 0.13–0.14 ms late, i.e. it underestimated their lead.
  - New predictions against observed p50: quick_l (held out) +0.011 ms; the reference-path MLPs 0.20–0.27 ms early.
  - The likely mechanism is the CPU clock (ondemand governor): light controllers run mostly in the slow state.
  - The new `cmd_at` raw column records the command point (tick start → write), in `--no-drive` too.
  - **Caveat: `POLICY_LEAD_EXTRA_S` was fitted before P3c,** on the reference forward pass under the ondemand
    governor. With the fast forward pass the self-test p50 shrinks, and so does the lead, but how much of the
    0.16 ms in-loop extra survives is unknown. It must be re-measured from `cmd_at` in P7 step 2 under the
    `performance` governor. Until then, expect the policy writes to land about 0.1–0.4 ms early (before
    `LINE_WRITE_PHASE_S`). That is the safe side, away from the 7.2 ms cliff.
- **LineClock.** Line stamps reconstructed from `t − age_left` reproduce the logged end-of-run period to a few ppm.
  - The reader thread isn't pinned, but it stamps with `perf_counter()` after `ser.read()` returns, which needs the
    GIL. Lines arriving while the loop computes are therefore stamped when the loop lets go: in the MLP runs, 70–90%
    of stamps land within 0.3 ms after the command write; in the PID/L runs, mostly inside the IMU read.
  - Even the stamps that can't have been held wander or step against the Pi clock (≈0 ppm for seconds, then ~1 ms
    steps), so the HAT period itself isn't settled.
  - A lower-envelope period fit (`LineClock(slope="envelope")`) beats least squares on synthetic load-dependent late
    stamps (12 vs 235 ppm mean). On these logs it was not clearly better (RMS window-slope error 460 vs 523 ppm, worse
    on the PID/L runs), so the default stays `"lsq"` (`loop_config.LINE_SLOPE`) and envelope is opt-in via
    `--line-slope envelope`.
  - Evidence scripts, in the agent's scratch dir: `lineclock_replay.py`, `stamp_resid.py`, `lineclock_synthetic.py`,
    `calibrate_lead.py`, `bench_forward.py`.

## Recommendations for P7 step 2

- Re-measure `POLICY_LEAD_EXTRA_S` from `cmd_at` with the fast forward pass, under the `performance` cpufreq
  governor (or at least log `scaling_cur_freq`) to test the CPU-state explanation. It was fitted before P3c, so
  until then the writes land ~0.1–0.4 ms early (see P3b above).
- Check that `cmd_phase` holds the target (sd < 0.1 ms) for each policy now that writes are sent.
- Run once with `--log-lines`, and A/B `--switch-interval-us 200`, which lets the reader take the GIL sooner. Then
  re-check the line period and C3's −880 ppm from the raw stamps.
- Re-measure the self-test and in-loop timing with P3c, and P3d for b1.
