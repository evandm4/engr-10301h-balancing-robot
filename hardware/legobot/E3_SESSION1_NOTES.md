# E3 session 1 notes (2026-11-25, 22:30–22:51 CST): stopped after Block A, 5 runs

- **Pre-registration:** `E3_SESSION1_PREREG.md` (commits `1159edc` and `7a93cc3`, pushed before the first run).
- **Robot code:** deploy `24b9b46`. Governor `performance` on every run. Compute lead 1.009 ms.
- **Pi temperature:** 41.9–45.1 °C. `get_throttled` = 0x0 before and after every run.
- **Lying orientation:** front face down.
- **Battery:** strapped tight and photographed (Evan). vin at start 8.06 → 7.91 V.
- **Metrics:** `robot/session_metrics.py`. Run sheet: `C:\balancebot_training\legobot\e3_s1_runsheet.csv`.
  Logs: `C:\balancebot_training\legobot\pi_logs\e2_deploy\20261126_04*`.

**Why the session stopped:** Evan stopped it before A4. He saw an unacceptable amount of wobble in the Pi mount, and the
data independently show the structure moving (below). Block A is incomplete: A4 and A5 were not run. Blocks B and C
were not run.

## Run sheet

| Stamp | Run | pitch_zero | θ̄ (rad) | v̄ (m/s) | Cost | vin | Evan's observation |
|---|---|---|---|---|---|---|---|
| 043156 | A1, 30 s | −0.0027 | −0.0059 | +0.0044 | 9.73 | 8.06 | Slightly forward (≤ 5 cm by eye; the robot's x says +13 cm). Rotated 3–5° left. |
| 043524 | A2, 20 s | −0.0086 (trim) | −0.0002 | +0.0002 | 5.41 | 7.95 | — |
| 043746 | A3+, 10 s | +0.0064 (override) | +0.0172 | −0.0159 | 2.97 | 8.01 | Clearly backward. Hit a tall carpet near the end. |
| 044028 | A3−, 10 s | −0.0236 (override) | +0.0448 | −0.0396 | 3.60 | 7.91 | Backward. Strong CCW rotation, ~90° over the run. |
| 044404 | A2b, 20 s (deviation) | −0.0086 | +0.0299 | −0.0249 | 6.66 | 7.94 | Backward, ~45° CCW. |

**After A2b:** with the robot lying face down, the IMU tilt was read three times in a row: 1.6589, 1.6569 and 1.6588 rad.
The run was then re-trimmed to +0.0213. No run used that value.

## Hypotheses

| H | Prediction | Observed | Verdict |
|---|---|---|---|
| H-A1 | θ̄(A1) in [+0.007, +0.017] | −0.0059 | **Fail.** The balance point moved back toward session 1's first run (000104: −0.0092). The +0.022 step seen after the pushes last session had reversed. |
| H-A2 | After the trim, \|v̄\| < 0.005 and \|θ̄\| < 0.004 in A2 | v̄ +0.0002, θ̄ −0.0002 | **Pass.** A5 was not run. |
| H-A3 | Slope in [−1.10, −0.60], **and** A3+ forward / A3− backward | Slope −0.861 (r −0.999, n = 5, includes A2b)\*. A3+ and A3− both went backward. | **Inconclusive.** The slope is in band. The sign test is confounded: the balance point moved by +0.03 between A2 and A3+, which is larger than the ±0.015 offsets. |
| H-A4 | Impacts shift the zero (A5 − A2) | A4 and A5 not run | **Not run.** See the finding below: the zero moved **without** deliberate pushes. |
| H-A5 | Temperature does not explain the drift | ΔT_SoC across Block A ≤ 3.2 °C, so 0.0008 × 3.2 ≈ 0.0026 rad, against a 0.030 rad shift | **Supported** (descriptive; the SoC temperature is only a proxy for the IMU's). |
| H-B*, H-C* | — | — | **Not run** |

\*The pre-registered H-A3 set is A1, A2, A3±, A5. Without A5 and without A2b the fit uses n = 4: slope −0.874, r −1.000.
Both values are within the band.

## Main finding: the balance point jumps because the structure moves

1. **The balance point jumped about +0.030 rad between A2 and A3+.** A3+ already leaned +0.024 in its second second,
   before the carpet. After subtracting each run's deliberate offset, A3+, A3− and A2b each show the same extra
   +0.030–0.032 rad. A2b (no offset) confirmed it: the prediction was θ̄ ≈ +0.03 with ~2.5 cm/s backward creep, and the
   run gave +0.0299 and −0.025.
2. **The IMU's angle to the chassis changed too.** With the robot lying face down, the tilt read 1.594 rad (logged
   while waiting to arm before A2). It then read 1.637 before A3+, 1.642 before A3−, 1.657 before A2b, and 1.658 three
   times after A2b. Laying the robot down repeats to within 0.002 rad, so this is a real rotation of about 3.7° between
   the IMU and the face the robot rests on. It doesn't move one-for-one with the balance point (+0.064 vs +0.030), so
   more than one joint is probably moving.
3. **Evan:** the Pi, which carries the IMU, is not rigidly mounted. He saw an unacceptable wobble and stopped the session.
4. **Implication:** last session's "+0.022 rad step after hard pushes", and tonight's −0.018 reversal from it, are best
   explained by the loose Pi/IMU mount, not by the controller, sensor noise or temperature. This contradicts the earlier
   [P7] judgment that "the IMU mount is rigid". The run-to-run θ̄ spread is a build problem. **Before any comparative
   floor statistics (Block B), the Pi/IMU needs a rigid mount**, followed by a re-trim and repeatability check
   (lying-tilt ×3, still run ×2).

## Other observations
- **v̄ = −0.86 θ̄ holds tightly** across five runs and four different balance-point conditions (residual sd 0.0009 m/s).
  The robot's creep is set entirely by its average measured lean.
- **The robot's x overstates real travel.** On A1, x was +13 cm; Evan saw ≤ 5 cm. Not yet explained: the candidates are
  wheel slip, backlash or the wheel radius. Ground-truth displacement should be marked on tape in future still runs.
- **Yaw:** when the robot creeps backward it turns CCW. Δright/Δleft was 0.73–0.85 in A3+, A3− and A2b. This is
  consistent with the known weaker right motor (−19.5 % at 0.10 duty [REV]).
- **No 3.8 Hz mode.** Dominant θ frequencies in 4 s windows were 1.0–3.0 Hz. A2b had one window (8–12 s) at 3.5 Hz with
  band fraction 0.34. This is descriptive only: H-C3's 0.20 threshold applies to scenario runs.
- **Timing was clean throughout:** 0 overruns, all ticks locked to the HAT's lines, and command point p99 ≤ 1.12 ms.

## Deviations
1. **22:26 (before data):** M added to Block C (agreed by Evan). Committed in `7a93cc3`.
2. **22:41 (after data):** inserted A2b, a repeat of A2, to measure the suspected zero shift. Evan approved.
3. **22:46 (after data):** lying-tilt repeatability check (3 × 3 s, no motors).
4. **22:49 (after data):** re-trimmed `pitch_zero` to +0.0213, from A2b. No run used it.
5. **22:51:** session stopped by Evan (Pi mount wobble). A4, A5 and Blocks B and C were not run. The Pi was shut down cleanly.

## Next
- Mount the Pi/IMU rigidly. Then: lying-tilt ×3, the A1-style trim, and two still runs to confirm θ̄ repeats within
  ~0.004 rad, **before** Block B. Re-use this pre-registration's Block A–C hypotheses. Log the changed build as a new
  condition.
- `calibration.json` on the Pi now holds +0.0213. Re-trim after the remount anyway.
