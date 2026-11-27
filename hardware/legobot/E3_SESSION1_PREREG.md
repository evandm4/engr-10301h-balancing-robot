# E3 session 1 pre-registration: drift, repeats and scenario runs on the LEGO robot

- **Drafted:** 2026-11-25 (evening, after the first floor session). The commit that adds this file is the
  timestamp. Nothing below was changed after any session-1 run started; later edits are logged as deviations.
- **Robot code:** deploy `24b9b46` (Pi `DEPLOYED` stamp "24b9b46 2026-11-25 18:59"). Since then, `robot/` has only gained
  the laptop-side `sim_vs_real.py` and its test (`git diff --stat 24b9b46 HEAD`), so the Pi loop, `control.py` and the
  bundles are unchanged. A redeploy before the session is logged as a deviation, with its stamp.
- **Controllers:**
  - **M**, `--controller model`: the frozen E2 reference, the model-based PID.
  - **P**, `policy:l_pid`: the L-PID equivalent of M (sha256 `00897c7e8e3e…`; gate (b) in sim: within 1 % of M where both stay up, E2_NOTES E2.4).
  - **L**, `policy:l_e2_s1`: provisional, sha256 `58cb3c615a43…`. Results are robot validation, **not** E2.6 selection evidence (E2_NOTES E2.7).
- **Rule:** the analysis follows this document. Any change of order, N, metric, threshold or exclusion goes in
  the session notes under **Deviations**, with the reason, the time, and whether it was decided before or after seeing data.

Sources are cited as [P7] = `P7_STEP2_4_NOTES.md`, [REV] = `C:\balancebot\REVIEW_l_e2_s1.md`,
[SVR] = `C:\balancebot\reports\sim_vs_real\README.md`, [TR] = `C:\balancebot\reports\training_report.md`, and
[LOG] = recomputed from the 2026-11-25 logs in `C:\balancebot_training\legobot\pi_logs\e2_deploy\` (scratch scripts
`prereg_stats.py`, `prereg_scen.py`, `prereg_arc.py` and `prereg_power.py` in this conversation's scratch folder).
**Guess** marks a number that is my own judgment call, with no source behind it.

## 0. Scope of charge 1 (2026-11-25 night), fixed before the first run

- The full plan is ~30 runs; one 600 mAh charge gives ~12–15 [P7]. Charge 1 runs, in this order:
  1. **Block A** complete: A1, trim, A2, A3+, A3−, A4, A5 (6 runs).
  2. **Block B group 1**: M P L L P M (6 runs).
  3. **Block C, positions 1–3 of L M L L M L (L, M, L)**: run only while the start vin is ≥ 7.6 V; whatever
     doesn't fit moves to charge 2. Evan agreed to the M condition on 2026-11-25 at 22:26, before any run, so
     H-C5 is active.
- Charge 2+: Block B groups 2–3, the rest of Block C. H-B1/B2 are tested only after all 18 Block B runs; after
  charge 1 the Block B numbers are reported as interim descriptives, with no test.
- **Trim method:** as in Block A (calibration.json via `merge_save`), using `C:\balancebot_training\shared\pi\e3_set_pitch_zero.sh <rad> <note>`,
  which backs up the file first. A3± use `--pitch-zero` and put the value in the label (`e3_A3p_pz<value>`).
- **Metrics script:** `hardware/legobot/robot/session_metrics.py` must be committed before Block B starts. Block A's
  metrics (θ̄, v̄, the OLS slope) are fully defined below; if the script lands after Block A, that is logged as a
  deviation (decided before data).
- **Lying orientation:** Evan picks the one he used last session and records it in the run sheet before A1; it is
  held fixed all session.
- **Superseded:** a shorter draft (`P7_SESSION2_PREREG.md`, never committed) was deleted before any run; this
  document is the only pre-registration.
- **Run labels:** `e3_<block><run>_<ctrl>` (e.g. `e3_A1_L`, `e3_B1_M`, `e3_C1_L`); the run sheet maps stamps to labels.

## 1. Fixed conditions (every run)

| Condition | Rule |
|---|---|
| Battery | Start each run at vin ≥ 7.6 V (the summary's `vin at start`). Stop the block and recharge if a start reads < 7.6 V, or if the in-run minimum (raw `vin`) drops below ~7.3 V. Last session's lowest was 7.27 V [P7]. The loop's own cutoff (6.8 V) is a trip, not a test condition. |
| Battery mount | Strapped tight, so the < 1.5 mm play is gone [P7]. Keep it at or below the 60 mm COM mark: +12 mm of COM takes L from 0/90 to 9/90 falls in sim [REV 2]. Photograph the mount before the first run. |
| Governor | Run `e2_post_deploy.sh` after every Pi boot; the setting does not survive a reboot [P7]. Each summary must show `cpu governor: performance` and `compute lead` ≈ 1.0 ms. If not, the run is void (logged reason), re-run the script, repeat the run. |
| Pi temperature | `vcgencmd measure_temp` and `vcgencmd get_throttled` over ssh just before launch and just after the run ends. Record both in the run sheet. This reads the SoC temperature only: `imu.py` does not read the ICM-20948's own temperature sensor. |
| Floor | The same spot for every run, wheel positions marked with tape. Record the robot's heading against a fixed room feature. Floor slope was ruled out by Evan [P7], so heading is a recorded constant, not a variable. |
| Calibration | The lying-down procedure [P7]: launch with the robot lying still. The gyro bias is taken over 2 s (sd < 0.05 rad/s). At "hold the robot upright", lift through vertical and support it lightly until "armed" (\|θ\| < 0.05 rad, rate < 0.3 rad/s for 0.5 s; 30 s timeout). Use one lying orientation all session and record it. [P7] says "on its side"; the `e2_floor.sh` comment says face down/back, "not on its side". |
| Launch | `e2_floor.sh <ctrl> <goal> <seconds> <label> [flags]`: duty cap 1.0 (the script refuses `--duty-cap`), envelope trip off. Pull the logs with `pi_logs.py stamp <STAMP>`. |

## 2. Blocks and hypotheses

Notation: θ̄ is the mean of `theta` over the armed run (`_run.csv`). v̄ = net Δx / duration, with x and v from the
robot's own estimates. "Cost" is `score_run.py`'s `cost` field (the simulator's cost on the run CSV).

### Block A: balance point and drift (L only)

| Run | What | Flags |
|---|---|---|
| A1 | still, 30 s, current `pitch_zero` (−0.0027 rad) | — |
| — | **Trim:** pitch_zero_new = −0.0027 + θ̄(A1). This works because θ = accel tilt − pitch_zero (`control.PitchEstimator`), so raising pitch_zero by θ̄ moves the mean estimate to 0. Back up `~/legobot/calibration.json`, then write the new value with `robot.calibrate.merge_save(file, {"pitch_zero": …, "pitch_zero_note": "E3 s1 trim from A1"})`. **Don't use `--pitch-zero` for the trim:** the run meta logs the calibration *file*, so an override is visible only in stdout. It can be recovered as atan2(−acc_fwd, acc_up) − `tilt` from `_raw.csv`, but not from the meta. | — |
| A2 | still, 20 s, trimmed ("pre") | — |
| A3+ / A3− | still, 10 s each, deliberate offset (my addition: it gives H-A3 a controlled test). This is the trimmed value ± 0.015 rad, the sim's pitch_bias range; L tolerated +0.06 / −0.04 rad in sim [REV 5]. | `--pitch-zero <new+0.015>`, then `<new−0.015>`. Write the value in the run sheet (the meta won't show it) and check it from `_raw.csv`. |
| A4 | 20 s, about 10 hard pushes peaking at 15–27°, like `000828`; no obstacle hits | — |
| A5 | still, 20 s, trimmed ("post") | — |

| H | Prediction | Source | Metric | Pass / fail |
|---|---|---|---|---|
| H-A1 | θ̄(A1) = +0.012 ± 0.005 rad, so the robot creeps backward at about 1 cm/s | The three post-push runs: +0.011, +0.011, +0.014 [P7]. The ± is a guess (it covers the strap removing ≤ 0.003 rad [P7]). | θ̄ from `_run.csv` | Pass if θ̄ is in [+0.007, +0.017]. Outside that range, the structure moved again since `010508`: report it, then trim anyway. |
| H-A2 | After the trim, \|v̄\| < 0.005 m/s and \|θ̄\| < 0.004 rad in A2 and A5 (A5 only if H-A4 shows no shift) | 0.85 × ~0.006 rad, the largest half-to-half change in θ̄ within a 10 s still run (`000104`) [LOG]. Before the trim: 1.3–1.6 cm/s [P7]. | v̄; θ̄ | Pass if both hold in A2. Fail if \|v̄\| ≥ 0.005 m/s. |
| H-A3 | v̄ = −0.85 · θ̄ across L still runs. A3+ creeps forward and A3− backward, each at about 0.013 m/s. | Fit over 5 L runs: −0.849 (m/s)/rad, r = −0.992, intercept −0.0007 m/s, residual sd 0.0015 m/s [LOG; P7 says −0.86]. Sims: −0.71 [P7], −0.93 (−0.28 m / 20 s at +0.015 rad) [REV 5]. | OLS of v̄ on θ̄ over A1, A2, A3±, A5 | Pass if the slope is in [−1.10, −0.60] (the band spans both sims and the all-controller fit, −1.06 [LOG]; the band is a guess) **and** A3± have the predicted signs. |
| H-A4 | If impacts shift the structure, θ̄(A5) − θ̄(A2) > +0.01 rad, with a forward sign like the +0.022 step last session | Step between `000439` and `000828` [P7] | Δθ̄ between the two 20 s still runs | \|Δ\| > 0.01: impacts shift the zero (supported). \|Δ\| < 0.005: not supported. In between: inconclusive. The 0.005 is a guess, about 2× the typical half-run noise in θ̄ (0.0005–0.006 [LOG]). |
| H-A5 | Pi temperature does not explain the drift | Accelerometer offset tempco ~0.8 mg/°C, so 0.022 rad needs ~27 °C, and warm-up drift would ramp, not step [P7] | ΔT_SoC across Block A; θ̄ vs T at the start | Supported if 0.0008 rad/°C × the largest ΔT_SoC < ½ \|any θ̄ change being explained\|. **Caveat:** the SoC temperature is only a proxy for the IMU's. |

### Block B: repeated still runs (M, P, L)

- **Design:** 10 s still, trimmed calibration. Order **M P L L P M**, three times (18 runs). That is 6 per
  controller, each with mean run index 9.5, so a linear battery sag cancels.
  - Recharge only between six-run groups. If a start reads < 7.6 V mid-group, pause, recharge, and resume the same
    group (pre-specified, so not a deviation). Record vin at the start of every run.
- **Why N = 6:**
  - The only run-to-run spread available is L's two still costs, 2.70 and 2.88 [P7]. That gives sd ≈ 0.13, from
    n = 2, which is nearly meaningless.
  - A rough power check for an exact one-sided Mann–Whitney test at α = 0.05 (normal costs; this is my
    simulation, ad-hoc seed 7,320,000) [LOG]:

    | Gap | sd | n = 5 | n = 6 |
    |---|---|---|---|
    | 0.70 | ≤ 0.25 | 0.99 | 1.00 |
    | 0.70 | 0.40 | 0.78 | 0.84 |
    | 0.44 | 0.40 | 0.45 | 0.51 |

  - So N = 6 can detect L vs the PIDs if the real sd is ≤ ~0.4. It **cannot** detect, or establish, a P-vs-M
    difference. N = 6 is also a feasibility choice: 18 runs at about 1 min each, within one or two charges.
    The drain per run is a guess: about 0.02–0.04 V per still run, from the vin starts in [LOG].

| H | Prediction | Source | Metric | Pass / fail |
|---|---|---|---|---|
| H-B1 | Cost: L < P. L median 2.6–3.0, P about 3.4. | Robot: 2.70 / 2.88 vs 3.40 [P7]. Sim: L 30 % lower median than M on dev [TR]. | `score_run.py` `cost` | Exact one-sided Mann–Whitney (`scipy.stats.mannwhitneyu(..., alternative="less", method="exact")`), Holm over H-B1 and H-B2 at α = 0.05. Fail if not significant **or** median(L) ≥ median(P). |
| H-B2 | Cost: L < M. M about 3.8. | Robot: 3.84 [P7] | same | same |
| H-B3 | P ≈ M: the median cost ratio P/M is in [0.85, 1.15] | Sim gate (b), < 1 % [E2_NOTES E2.4]. The robot single runs gave 0.88 [P7]. The band is a guess. | median(P) / median(M) | **Descriptive only**: no equivalence test, given the power above. |
| H-B4 | Wobble: L's median θ p-p per 2 s window < both PIDs' | L 7.9–11.6°, PIDs 8.6–13.7° [P7, LOG] | θ p-p in consecutive 2 s windows | Descriptive |

Also report each controller's run-to-run sd: it sets N for the next pre-registration.

### Block C: scenario repeats

- **Order:** **L M L L M L** (4 L, 2 M; both have mean index 3.5). Goal `scenario`, 20 s, open floor radius ≥ 2.5 m [REV].
- **M condition:** M is included only if Evan agrees on the day. It has never run the scenario on the robot.
  In sim it has twice L's linearized cruise delay margin (25.25 vs 12.25 ms) [REV 2]. Without M: L × 4, and H-C5 is dropped (logged).
- **Profile** [LOG, run 9]: goal 0 for 0–2 s; ramp to +0.40, plateau 3.25–9.5 s; ramp to −0.15, plateau
  11.9–14.0 s; ramp to 0, plateau 16.0–20 s.

| H | Prediction | Source | Metric | Pass / fail |
|---|---|---|---|---|
| H-C1 | L tracks every plateau. Cruise is slightly under the goal (≈ −0.03 m/s), the reverse plateau ≈ +0.02, the final 0 plateau ≈ +0.015. | Run 9: −0.031, +0.017, +0.015 [LOG; P7: +0.370, −0.134, +0.014] | Plateau = each maximal run of constant `velocity_goal` longer than 0.5 s. Error = mean `x_dot` − goal over it. | Pass if \|error\| ≤ 0.05 m/s on every plateau of every L repeat (the 0.05 is a guess, 0.03 observed + margin). |
| H-C2 | L arcs right: the right wheel covers ≈ 0.8× the left | Run 9: Δpos_right / Δpos_left = 0.81 over the run, 0.73 on the cruise [LOG; TR "19 % less"]. Right motor −19.5 % at 0.10 duty [REV]. | Ratio of `pos_right` to `pos_left` change over the armed run (`_raw.csv`, forward-positive) | Pass if the ratio is < 1 in every L repeat **and** Evan sees a right curve |
| H-C3 | No growing ~3.8 Hz mode | Linear margin 12.25 ms vs ~9.3–9.7 ms measured [REV 2, P7]. Run 9 band fraction 0.026–0.103 [LOG] (3–11 % [P7]). | θ power fraction in 3.3–4.3 Hz per 4 s window (mean removed, Hann, total > 0.25 Hz). Dominant frequency per window. | Pass if the fraction is ≤ 0.20 in every window (a guess, ~2× the observed max) **and** no cruise window (4–8, 8–12 s) peaks in 3.3–4.3 Hz |
| H-C4 | Real wobble = 1.3–1.7× the free nominal sim's | Runs 9 / 8: 1.30 / 1.65 [SVR] | `sim_vs_real.py <stamp> --no-anim`: `wobble_ratio_real_over_free.theta` in `<stamp>_metrics_T1.json` | Pass if the median over L repeats is in [1.3, 1.7]. Weak support if it is in [1.0, 2.0]. Fail otherwise. Also run it on Block B's L runs (descriptive). |
| H-C5 | M completes the scenario with no fall. Its cost is higher than L's. | Sim medians: L/M 0.70 [TR]; robust scenarios, not this profile | `score_run.py` `cost`, `fell` | Fall-free: pass/fail. The cost comparison is descriptive (n = 2). |

### Block D: delay injection — **PLACEHOLDER**

> [!IMPORTANT]
> **PLACEHOLDER.** A verifier is recomputing the delay margins with the sensor dynamics included. Its numbers replace
> the TBDs below, and this block is committed **before** any delay run. Until then, Block D does not run.

- **There is no latency-injection option on the robot.** `balance.py --inject {loop, hang, stale, vin, fault, imu,
  unlock, ood}` is the D3 fault-injection set. Each injection fires **once**, `--inject-at` seconds after arming,
  to test a trip:
  - `loop` is one 50 ms tick;
  - `hang` blocks the loop for 0.5 s;
  - `ood` offsets the measured velocity.

  The choice is recorded as `"inject"` in the meta JSON (the `# {...}` header line of `_run.csv`), not in
  `_summary.txt`.
  - `--compute-lead` and `--line-phase` move the command write against the HAT's line grid. Past the left motor's
    7.2–8.3 ms cliff, the write waits for the next HAT tick (`loop_config.LINE_WRITE_PHASE_S`). That is one coarse
    step, on the left motor only. It is not a validated delay injector.
  - A clean added delay needs new code, e.g. a FIFO of commands in `Brain.act`. **Not implemented here.**
  - An integer-tick FIFO gives only +5 ms steps, which goes from ~9.5 to ~14.5 ms in one step, past L's 12.25 ms margin.
- **Structure (to fill in):**
  - Added latency steps: TBD (verifier margins).
  - Each step is one 20 s scenario run for L and one for M, alternating.
  - Prediction: L's onset delay is in [TBD]; M shows no onset up to TBD.
  - Metric: H-C3's band fraction and dominant frequency, plus cruise θ rms.
- **Stop rule:** at the first step with a visibly growing wobble, a band fraction > TBD, or \|θ\| > 20° during
  the cruise:
  - catch the robot or abort;
  - make no further steps for L;
  - finish that step with M only if it is not itself unstable.
- **Pass/fail form:** "L's onset step is within ±1 step of the verifier's predicted margin, and M's is later than L's."

## 3. Safety

- Evan at the robot, hand ready to catch, ssh Ctrl-C ready (`run_safe.sh` coasts the motors).
- Catch zone clear: ≥ 1 m for still runs and ≥ 2.5 m radius for the scenario, with no people or pets in it.
- **Duty cap stays at 1.0 for L.** Below 0.70, the clamp desynchronizes L's command history: 28/30 falls at 0.30 in sim [REV 3].
  `e2_floor.sh` refuses `--duty-cap`.
- L has no lean cap: the only tilt trip is 0.70 rad (40°) [REV 3].
- Envelope trip off: at ≤ 1.5 it false-trips [REV 4].
- **Abort the run (catch, Ctrl-C) if any of these happens:**
  - a growing wobble you can see or hear;
  - \|θ\| > 25° outside a deliberate push;
  - a wheel near an edge or obstacle;
  - any trip message.
- **Abort the block** on:
  - in-run vin < ~7.3 V;
  - SoC temperature ≥ 80 °C, or `get_throttled` ≠ 0x0 (throttling invalidates the lead timing; 80 °C is the Pi's documented throttling threshold, not a project number);
  - a failed self-test or config check;
  - two tilt trips in a row without contact.
- Pushes in A4: ≤ ~27°, as last session. Duty saturates from ≥ 24° pushes, which settle in 2.6–4.5 s [P7].

## 4. Analysis plan

1. Pull every run's logs with `pi_logs.py stamp <STAMP>`. Keep a run sheet: stamp, block/H, controller, flags,
   vin start, temperature before/after, heading, operator notes.
2. On the laptop:
   - `python hardware/legobot/robot/score_run.py <runs> --json <file>` for every run;
   - `replay_policy.py <log>` for every P and L run, which must give max diff 0, as last session;
   - `sim_vs_real.py <stamp> --no-anim` for the L runs in Blocks B and C.
   - The defined metrics above (θ̄, v̄, plateau error, wheel ratio, band fraction, θ p-p) go in one script,
     written and committed **before** the session. The scratch `prereg_*.py` scripts are reference implementations.
3. Block B statistics as in H-B1/B2; everything else is descriptive or uses the thresholds above.
4. **Exclusions:** a run is excluded only for one of these pre-listed reasons:
   - the governor is not `performance`;
   - vin at start < 7.6 V;
   - operator contact during a still run (other than a deliberate push);
   - an arm timeout (no data);
   - a replay mismatch, which also stops the session for diagnosis.

   Each exclusion is logged with its reason, and the run is repeated at the end of its group. A fall without
   contact is **not** excluded: it is scored with `score_run`'s fall penalty and ranks worst. **Nothing is
   excluded post hoc without a logged reason.**
5. **Notes:** write a new `hardware/legobot/E3_SESSION1_NOTES.md` containing:
   - a table with one row per H#: prediction, observed value, verdict (pass / fail / inconclusive / not run);
   - the run sheet;
   - a **Deviations** section;
   - the run-to-run sd per controller (it feeds the next pre-registration).

   Failed predictions are reported as prominently as passed ones.

## 5. Clarifications fixed before any E3 data (2026-11-25, ~22:30)

`session_metrics.py` was written against this document and turned up these ambiguities. They are settled here,
**before the first session-1 run**, so they aren't deviations:

1. **H-A3 source:** the "5 L runs" behind the −0.849 slope are 000104 and 010508 (still), 000439 (taps), and
   000828 and 000949 (pushes, 000949 cut at 35 s). The session-1 fit itself uses only the still runs A1, A2, A3±, A5.
2. **H-C1 plateaus:** the opening goal-0 hold (0–2 s) counts as a plateau. That is what the written definition
   says, and it gets the same ±0.05 m/s rule (run 9: −0.007).
3. **H-C3 spectrum:** fs is the nominal 200 Hz, so 4 s windows have 0.25 Hz bins. "Total > 0.25 Hz" starts at
   the 0.5 Hz bin. "Peak in 3.3–4.3 Hz" means the 3.5–4.25 Hz bins.
4. **H-A2:** |v̄| < 0.005 m/s with |θ̄| ≥ 0.004 rad is **inconclusive**.
5. **H-A4:** the verdict uses |Δθ̄|, as the pass rule states. The sign is reported separately: forward (+) matches
   the prediction's direction.
6. **H-A5:** supported if 0.0008 rad/°C × (max − min of all Block A SoC readings) < ½ |Δθ̄(H-A4)|. If H-A4 is
   "not supported", H-A5 is **not applicable**: there is no shift to explain.
7. **H-B4:** pool every 2 s window across a controller's runs and take the median.
8. **`t_max` crops** the θ̄, v̄, window, plateau and wheel-ratio metrics, but not `score_run`'s `cost`, which is
   always the whole run.
9. **Armed span** = rows with `phase == 1`, as in `balance.write_logs`.

## 6. Session 2 addendum (2026-11-25, ~23:10), fixed before any session-2 run

**Why there is a session 2:** session 1 (see `E3_SESSION1_NOTES.md`) found that the Pi/IMU mount moved. The balance
point jumped ~+0.030 rad between A2 and A3+, and the lying tilt moved 1.594 → 1.659 rad. After that, Evan stopped
because of Pi wobble. At ~23:00, Evan added **three small, very low-mass rubber pieces that hold the Pi in place**. He
describes the result as "more rigid, not perfect". Nothing else changed: same robot code (`24b9b46` deployed), battery
mount, floor spot and lying orientation (front face down).

- **This is a new build condition.** Session-2 numbers are reported separately from session 1 and are never pooled
  with session-1 numbers for any H#. Session-1 verdicts stand as written.
- **Battery:** charge 1 continues (5 runs used; last start vin 7.94 V). The §1 vin rules apply unchanged.
- **Labels:** `e3s2_<run>_<ctrl>`. Run sheet: `C:\balancebot_training\legobot\e3_s2_runsheet.csv`.
- **Tape check (new, descriptive):** after every still run, Evan reports the tape-measured net displacement and
  rotation if he can. The ratio of robot Δx to tape Δx is reported as descriptive only (session 1: robot +13 cm vs
  ≤ 5 cm by eye).

### Block V: mount verification (L only). Runs first, and gates everything after it.

| Step | What |
|---|---|
| V0 | Lying tilt ×3 with `e2_tilt.sh` (no motors). Lift and re-lay the robot between readings. |
| V0p | *Optional:* `e3_tilt_stream.sh 20` (no motors) while Evan gently presses the Pi fore/aft and releases. This script has never run; if it errors, the step is skipped, which is not a deviation. |
| V1 | still, 30 s, calibration.json `pitch_zero` = +0.0213 (the re-trim from the end of session 1) |
| trim | pitch_zero_new = 0.0213 + θ̄(V1), written with `e3_set_pitch_zero.sh` (same method as §0) |
| V2, V3 | still, 20 s each, trimmed |
| V4 | Lying tilt ×3, as in V0 |

| H | Prediction | Metric | Pass / fail |
|---|---|---|---|
| H-V1 | Lying tilt is repeatable after re-laying the robot | Range of the 3 V0 readings | Pass if the range is ≤ 0.004 rad (session 1's post-run triple had a range of 0.002) |
| H-V2 | The trim holds: the robot balances where it was trimmed | θ̄ and v̄ of V2 and V3 | Pass if \|θ̄\| < 0.004 rad and \|v̄\| < 0.005 m/s in **both** (the H-A2 rule) |
| H-V3 | The balance point repeats run to run | \|θ̄(V3) − θ̄(V2)\| | Pass if ≤ 0.004 rad |
| H-V4 | Driving does not move the IMU relative to the chassis | mean(V4) − mean(V0) | Pass if \|Δ\| ≤ 0.005 rad (session 1 moved by 0.065 rad) |

V0p is descriptive: it reports the peak deflection under a press and how far the reading is from its pre-press
baseline 2 s after release.

**Gate:** comparative blocks run only if H-V2, H-V3 and H-V4 all pass. If H-V1 fails, the lying tilt can't serve as a
check, so H-V4 is not applicable and the gate uses H-V2 and H-V3 alone (logged). If the gate fails, the session ends
with no Block A/B/C runs. The result is reported as "mount still not stable enough for comparisons", and at most two
more still runs (20 s) are allowed, for diagnosis only.

### After the gate (order fixed; stop wherever the §1 vin rule says)

1. **A4:** 20 s, about 10 hard pushes (≤ 27°), as in §2.
2. **A5:** still, 20 s. H-A4 is tested as θ̄(A5) − θ̄(V3), with the §2 thresholds.
3. **Lying tilt ×3** (V5). If H-A4 is "supported" (\|Δ\| > 0.01), or \|mean(V5) − mean(V0)\| > 0.005 rad: re-trim from
   A5 (pitch_zero += θ̄(A5)), then do one 20 s still confirmation run, which must pass the H-V2 rule. If it fails, stop
   before Block B: impacts move the structure. All of this is pre-specified, not a deviation.
4. **Block B group 1:** M P L L P M, 10 s still, as in §2.
   - **Tripwire:** if any still run has \|θ̄\| > 0.010 rad, do a lying tilt ×3 before the next run. If it moved
     > 0.005 rad from V0, Block B stops, and the run that tripped it is reported but flagged.
5. **A3+ / A3−:** 10 s each, at the current trim ± 0.015 rad (`--pitch-zero`, value in the label). This repeats the
   session-1 H-A3 sign test, which drift confounded. The H-A3 rule applies, with the OLS over the session-2 L still
   runs V1, V2, V3, A3±, A5. A3 sits after Block B so that Block B gets the battery first.
6. **Block C positions 1–3 (L, M, L):** as in §2, only while the start vin is ≥ 7.6 V.

H-A1 is not re-tested: its prediction was for the old mount. V1 plays A1's role (the trim) without a prediction.
The §3 safety rules and §4 analysis apply unchanged. `session_metrics.py --block A` is used for the V and A runs; the
H-V metrics are computed from its θ̄/v̄ output and the tilt readings.

## 7. Session 3 addendum (2026-11-26, ~19:10), fixed before any session-3 motor run

**Why there is a session 3:** session 2's gate failed (`E3_SESSION2_NOTES.md`). The trim came from the run in which the
mount settled, and two falls after data collection (FUN60, PUSH60) loosened pins. Before session 3, Evan reseated the
connections and rubber pieces. `TODO(Evan): what was reseated, and anything else changed`.

- **New build condition.** Session-3 numbers are reported separately and never pooled with sessions 1–2. Earlier
  verdicts stand as written.
- **Robot code: deploy `8609b46`** (Pi `DEPLOYED` stamp "8609b46 2026-11-26 19:00"), strict post-deploy self-test
  ALL BUNDLES PASS. Since `24b9b46` the Pi loop gained Block D's `--added-delay-ms` FIFO (`1b525e6`). It is not used
  here: at the default 0 the issued duty is written unchanged (`test_added_delay.py`). Block D stays a placeholder and
  does not run.
- **Controllers:** M and P unchanged. **L is now `policy:l_e2`** (sha256 `321cda5ceb9e…`), the E2.6 selection
  (E2_NOTES E2.7). `l_e2_s1` is not run this session. The two have weights within ~0.01, but no session-3 number is
  compared against a session-1/2 `l_e2_s1` number as if they were the same controller.
- **Battery:** one freshly charged pack for the whole session. §1's rules apply, and the in-run rule is **enforced
  live**: after every run, the orchestrator reads the raw `vin` minimum before the next launch. A run whose minimum is
  < 7.3 V is flagged, and the block stops for a recharge.
- **Labels:** `e3s3_<run>_<ctrl>`. Run sheet: `C:\balancebot_training\legobot\e3_s3_runsheet.csv`.
- **Pushes:** within §3's ~27°. No push-to-failure runs.

### Calibration readings taken before this addendum (disclosed, descriptive)

Lying tilt (front face down, no motors), taken 19:06–19:09 after the reseat, **before** this text was written:
1.5953, 1.5938, 1.5903, 1.5915 rad (each a 3 s mean, sd 0.0015–0.0016). Mean **1.5927**, range 0.0050. The first three
fell monotonically; Evan OK'd the fourth after seeing them. Session 2's V4 was 1.5979 (not pooled).
No test is applied to these four. They serve as session 3's **V0** reference for H-S3-3 below.

### Block S: seating, trim and gate (L only). Runs first, and gates everything after it.

| Step | What |
|---|---|
| S0 | Seating, no motors: a firm press-and-release on the Pi fore and aft, 3 times each. Then lying tilt ×1 (descriptive). |
| S1, S2 | Still, 20 s, calibration.json `pitch_zero` = −0.0203 (current). Seating/trim runs; no hypothesis. |
| S3 | Only if \|θ̄(S2) − θ̄(S1)\| > 0.004: one more 20 s still run, and the trim uses S2 and S3 instead. At most one S3. |
| trim | pitch_zero_new = −0.0203 + mean of the two θ̄ used, written with `e3_set_pitch_zero.sh` (note gives the stamps). |
| G1, G2 | Still, 20 s each, trimmed. The gate runs. |
| VL | Lying tilt ×3, lift and re-lay between readings. |

| H | Prediction | Metric | Pass / fail |
|---|---|---|---|
| H-S3-1 | The trim holds | θ̄ and v̄ of G1 and G2 | Pass if \|θ̄\| < 0.004 rad and \|v̄\| < 0.005 m/s in **both** (the H-A2 rule, unchanged) |
| H-S3-2 | The balance point repeats run to run | \|θ̄(G2) − θ̄(G1)\| | Pass if ≤ 0.004 rad (unchanged from H-V3) |
| H-S3-3 | Driving does not move the IMU relative to the chassis | mean(VL) − 1.5927 | Pass if \|Δ\| ≤ 0.005 rad (unchanged from H-V4) |

**Gate:** comparative blocks run only if H-S3-1, H-S3-2 and H-S3-3 all pass. If the gate fails, the session ends with no
comparative runs, reported as "mount still not stable enough for comparisons", and at most two more 20 s still runs are
allowed, for diagnosis only. The 0.004 / 0.005 limits are deliberately unchanged from session 2.

### After the gate

§6's "After the gate" list applies unchanged with L = `l_e2` and labels `e3s3_*`, except:

- **Block B does not start until §10/§11 (cost-duration rule, H-B1 power and n) are committed** (session-2 TODO 9).
  If they are not committed by then, skip Block B and go to A3± (§6 item 5).
- §6's references to V0 and V3 mean session 3's V0 (above) and G2.
