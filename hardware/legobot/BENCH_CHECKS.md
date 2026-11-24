# Bench checks still open (after the 2026-11-23 review)

These come out of the external review (`docs/legobot-review/triage.md`) and the tick-lock change (commit
5bac32a). Every command runs on the Pi from `~/legobot` after `bash hardware/legobot/deploy.sh`.

Section A is about safety. Section B can change the simulation model, and B1 and B2 should run before E1
spends GPU hours. Section C fills gaps in the measured spreads; it gates E2 (the final tuning), not E1.

## A. The new robot code, on hardware (safety; before any floor test)

The robot code changed: IMU-error handling, logs on a crash, coast before the watchdog stops, a watchdog
that keeps coasting, a voltage-scaled runaway trip (`0.85 * w_nl(vin)`, ~15.1–18.9 rad/s over 6.8–8.4 V;
Round-2 `H6`), the vin cutoff now 6.8 V, and an optional line lock that falls back to the Pi's clock if lost
(changed after Bench B1; see A3 and A6 below). Robot upside down on the stand, wheels clear. Use
`--pitch-zero` set to the stand's tilt, as in the D1 bench runs.

| # | Check | Command (add `--pitch-zero <stand tilt>` to each) | Pass |
|---|---|---|---|
| A1 | Deploy, then confirm the stamp | `bash hardware/legobot/deploy.sh`, then `cat ~/legobot/DEPLOYED` | Shows 5bac32a or later, without `-dirty` |
| A2 | IMU error trip | `robot/run_safe.sh robot.balance --bench --seconds 5 --inject imu` | Stops with "IMU read failed 2 times in a row"; wheels at 0 deg/s 1 s later; logs written |
| A3 | Loss of the line lock | `... --bench --seconds 5 --inject unlock` | *Changed after B1:* the run keeps going on the Pi's clock and prints "line lock lost"; the summary records it. (It used to stop: passed in that form 2026-11-23.) |
| A4 | Watchdog keeps coasting | `... --bench --seconds 5 --inject hang` | Watchdog fires within ~100 ms; wheels stay stopped for the whole 0.5 s hang |
| A5 | Runaway trip (a lifted robot) | `... --duty-cap 0.6 --seconds 10` (the velocity loop runs away to the cap on the stand, as in D1) | Stops with "wheels running away" ~1.5 s after the smoothed wheel speed passes `0.85 * w_nl(vin)` (~15.1 rad/s at 6.8 V, ~17.0 at 7.6 V, ~18.9 at 8.4 V; was fixed 17 rad/s). If duty 0.6 doesn't trip, try 0.7 |
| A6 | ~~Arming waits for the lock~~ | Dropped after B1: arming no longer waits for the lock | The startup lines still say whether it locked |

## B. Checks that can change the simulation model

| # | Check | How | Why, and what a result changes |
|---|---|---|---|
| **B1** | **Do the tick offsets survive a power cycle?** | Power the HAT off and on (battery out, and wait for the firmware to reload), then `robot/run_safe.sh robot.lock_sweep --wheels-off` for the left wheel and `--side right` for the right. Do this after two separate power-ups | The sim's 7.5–15 ms range assumes a write-to-tick wait of 1.8 ms (left) and 3.8 ms (right), with the left motor's tick at 8.3 ms and the right's at 10.3 ms after the line, measured in one session. **If a tick moves by more than ~1 ms,** either the robot measures the offset at startup or the range must widen. If they move, stop and report before E1 |
| **B2** | **Does duty 0 coast or brake?** | (a) The lowering test: `robot/run_safe.sh robot.measure.load --side left --lift-dir +1` (setup in its docstring; `--lower-duties` starts at 0.0). (b) A quick check by hand: spin a wheel by hand under `port 0 ; coast`, then under `port 0 ; pwm ; set 0`, and compare how fast it stops | `SlowPwmActuator` assumes duty 0, and every pulse-density off-tick, gives zero torque (coast). `backlash.py` assumes `set 0` is stiff, which suggests braking. **If `set 0` brakes,** the off-ticks add drag in the model, which is a change to all three engines. If it brakes, stop and report before E1 |
| B3 | Does the motor rail follow vin up to 8.4 V? | `robot/run_safe.sh robot.measure.discharge --wheels-off` from a full charge (an hour or more; it also fills in the missing discharge log). Quick version: `robot.measure.motor_steps --wheels-off --side left --duties 1.0 0.5` on a fresh 8.4 V pack and again near 7.0 V | Every fit was run at 7.3–7.5 V. The sim and the feedforward use V = vin. **If the speed stops scaling with vin** (a regulated rail), the feedforward must use the rail voltage, and the supply spread changes |
| B4 | Re-measure the slow-PWM ripple at low duty, with the robot secured | `robot/run_safe.sh robot.measure.pwm_probe --wheels-off --side left` (and `--side right`) | Open from B1b. It confirms the pulse-density model at the duties balancing uses (0.05–0.2) |

## C. Spreads and calibration (before E2)

| # | Check | How | What it changes |
|---|---|---|---|
| C1 | The right motor's torque scale `a` | `robot/run_safe.sh robot.measure.lever --side right --arm-mm 96 --pwmparams 1 0.01` (setup in its docstring) | Only the left motor was measured (a = 0.0597 ± 0.0019 N·m/V). If the right differs by more than ~3%, widen `resistance` / `gear_efficiency` in `legobot_config.ROBUSTNESS` (now ±5% / ±2%) |
| C2 | The balance point | `~/legobot-venv/bin/python -m robot.calibrate balance --save` (hold it at the balance point, still) | The current `pitch_zero` (−0.010 ± 0.010 rad) came from C3's release test. The sim assumes ±0.015 rad remains after calibration (B6); check that the trials agree within that |
| C3 | The right drivetrain's extra friction | Look and feel: spin each wheel by hand and look for beam rub or a pinched axle on the right. Then re-run `robot.measure.motor_steps --wheels-off --side both --duties 0.1 0.15 0.2` | At duty 0.1 the right wheel runs 29% slower (5.9 against 8.3 rad/s). If it's mechanical, fix it. If not, the `coulomb_friction` range must cover it |
| C4 | Friction with the wheels on the floor | No script yet. A possible test: a coast-down on the floor, with the robot held upright on a light trolley | Every friction value was measured with the wheels in the air (no axle load, no rolling resistance). Optional; it decides whether to widen the friction ranges |
| C5 | Is a stall below duty 0.5 safe? | A wheel held at duty 0.35–0.45 for 10 s (`robot.bench_log --pwm 0.4` with a hand on the wheel), checking the motor's temperature by touch | The stall trip only watches \|duty\| ≥ 0.5. If 0.4 heats the motor noticeably, lower `loop_config.STALL_DUTY` |

## Where results go

- **CSVs and `_fit.json` files:** in `hardware/legobot/bench_data/`, copied from `~/legobot/logs/`.
- **Results:** a short entry per check in DATASHEET 8.7.
- **Status:** the matching row in `docs/legobot-review/triage.md`.
- **Config changes:** change a spread or constant in `legobot_config.py` with a comment citing the
  measurement. A model change (B2 braking, B3 rail) is its own task: record it and stop.

---

## Handoff prompt (paste into a fresh window, started in this repository's root)

I need to run the remaining bench checks on my LEGO balancing robot, which is a Raspberry Pi 3B with a
Build HAT and an ICM-20948 IMU. This session owns the bench work. Simulation and tuning happen elsewhere.

**Read first, in this order:**
1. `hardware/legobot/BENCH_CHECKS.md`: the checklist. Sections A, B and C, the commands, and what each
   result changes.
2. `docs/legobot-review/triage.md`: why each check exists.
3. `hardware/legobot/SOFTWARE_PLAN.md`: the "Phase D as built" notes and the "External review" section.
4. `hardware/legobot/DATASHEET.md` 8.7: past bench results, and the format for new entries.
5. The docstring of each script before you have me run it (setups and safety notes are there).

**How we work:**
- I run every command on the Pi myself and paste you the output. You can't reach the Pi.
- Deploy first (A1) with `bash hardware/legobot/deploy.sh`, which you can run from this machine.
- Do section A before anything else. The robot code changed, and a floor test waits until A passes.
- Then B1 and B2, which can change the simulation model. **If either result disagrees with the model's
  assumption, stop and tell me.** Don't change the engines in this session.
- Then whatever of B3–B4 and C1–C5 I have time for. Tell me the setup before each one, and say which ones
  need the wheels off the ground or the robot clamped.
- After each check:
  - copy its logs into `hardware/legobot/bench_data/` (I'll `scp` them; tell me the paths);
  - add a dated entry to DATASHEET 8.7;
  - update the row in `docs/legobot-review/triage.md`.

  A measured spread or constant can change in `legobot_config.py`, with a comment citing the measurement.
- Commit at the end, on the current branch, after asking me.
