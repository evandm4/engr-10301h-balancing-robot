"""Settings for the balancing loop (SOFTWARE_PLAN D1-D5), separate from hw_config's I/O constants.

Everything the simulation also models (filter time constant, control period, motor constants used by the
feedforward, torque and lean caps, integral limits) is NOT here: the loop reads it from legobot_config.py
through config.load_config("legobot"), so robot and simulator share one copy (control.py).
"""

# ---- loop timing (D2) ----
SPIN_S = 0.0004            # sleep until this long before the deadline, then busy-wait (sleep wakes late by ~0.1 ms)
OVERRUN_S = 0.0075         # a tick that starts this late (1.5 periods) is logged as an overrun
STALL_LOOP_S = 0.025       # a tick this late trips the loop-stall guard (5 periods)
RT_PRIORITY = 50           # SCHED_FIFO priority asked for; falls back to nice -10, then to normal (logged)
CPU = 3                    # pin the loop (and the threads it starts) to this core; None to leave it

# ---- lock to the Build HAT's data-line grid (D2) ----
# In slow PWM the HAT's 10 ms pulse ticks are locked to its 100 Hz data lines (C3, measure/tick_lock.py),
# so the loop runs two ticks per line period at a fixed phase of that grid (control.LineClock, Pacer) and
# each command reaches the HAT at the same point before a pulse tick. --no-line-lock runs on the Pi's clock.
# Bench B1 (2026-11-23) then showed the pulse ticks wander against the lines, so the lock doesn't fix the
# wait to a pulse (the simulator draws it at random). It still steadies the motor data's age, so the loop
# uses it when it has it and falls back to the Pi's clock when it doesn't; arming doesn't wait for it.
LINE_LOCK = True
LINE_WRITE_PHASE_S = 0.0065  # s after the grid's line (lower envelope of the left port's arrivals) at which the
                           # command write starts. robot.lock_sweep, 2026-11-23 (duty 1.0 steps, sawtooth fits): the
                           # left motor's tick cliff is at 8.3 ms (90% 7.2-8.3), the right's at 10.3 ms (10.2-11.3).
                           # 6.5 ms is before the earlier one with margin: waits of about 1.8 ms (left) and 3.8 ms
                           # (right) for the tick, against 0-10 ms each at a random phase. Used mod half a period
COMPUTE_LEAD_S = 0.0012    # tick start to command write: IMU read + control (1.07-1.30 ms mean, bench 2026-11-23)
LINE_UPDATE_TICKS = 20     # re-estimate the line grid every 20 ticks (100 ms)

# ---- startup and calibration (D4) ----
CALIBRATION_FILE = "calibration.json"   # in ~/legobot; written by robot.calibrate
GYRO_CAL_S = 2.0           # robot still for this long at startup: gyro bias = mean pitch rate
GYRO_CAL_TIMEOUT_S = 20.0  # keep retrying 2 s windows until one is still, for this long
GYRO_CAL_MAX_SD = 0.05     # rad/s: refuse the bias (robot moved) above this sd; the sim assumes 0.02 of noise, a hand adds tremor
GYRO_CAL_MAX_BIAS = 0.10   # rad/s: refuse a larger bias. The ICM-20948's zero-rate offset is spec'd at +-5 deg/s
                           # (0.087 rad/s); this chip reads 0.044-0.047 at rest (2026-11-23). Motion is caught by the sd
ACCEL_G_TOLERANCE = 0.05   # |specific force| must be within 5% of g during calibration (robot still)

# ---- arming (D3) ----
ARM_TILT = 0.05            # rad: arm only within this of the balance point...
ARM_RATE = 0.3             # rad/s: ...turning slower than this...
ARM_HOLD_S = 0.5           # ...for this long (the robot held upright and still)
ARM_TIMEOUT_S = 30.0       # give up if not armed by then

# ---- safety trips (D3); each one coasts both motors and ends the run ----
TILT_LIMIT = 0.70          # rad (40 deg): past the lean cap (25 deg) and the scenario's largest start tilt
                           # (0.30 rad) with margin; well short of the simulator's 60 deg fall angle
DUTY_CAP = 1.0             # |duty| clamp in the balancing run (the Build HAT's full scale)
BENCH_DUTY_CAP = 0.3       # --bench: wheels off the ground, as hw_config.BENCH_DUTY_CAP
STALL_DUTY = 0.5           # a wheel commanded at least this duty...
STALL_SPEED = 3.0          # ...averaging less than this (rad/s, in the duty's direction, relative to the body)...
                           # A free wheel at duty 0.24 does about 12 rad/s; a held one creeps at -2 to +2
STALL_S = 1.0              # ...for this long is stalled (the HAT has no current limit to protect it)
VIN_CUTOFF = 6.8           # V under load, 3.4 V per cell: the lowest supply robust scoring draws (legobot_config
                           # supply_voltage 7.6 V -10.5%), so the robot never runs on a battery no tuned gain saw.
                           # Was 6.6 V (3.3 V per cell, DATASHEET 3.6); review finding G7, 2026-11-23
VIN_LOW_COUNT = 3          # consecutive readings below the cutoff before tripping (load transients)
VIN_PERIOD_S = 0.5         # how often the loop asks the HAT for vin (a 4-byte command; reply read in the background)
VIN_STALE_S = 2.0          # trip if no vin reply for this long
MOTOR_STALE_S = 0.05       # trip if either motor's newest data line is older than this (5 lines missed)
WATCHDOG_S = 0.10          # a separate thread coasts the motors if the loop has not ticked for this long
WATCHDOG_REPEAT_S = 0.05   # ...and keeps re-sending the coast at this interval until the run ends, so a command
                           # the stuck loop finishes writing after the first coast can't leave a motor driven
IMU_ERROR_LIMIT = 2        # consecutive failed IMU reads (I2C errors) before tripping; one is bridged with the last sample
RUNAWAY_SPEED = 17.0       # rad/s: both wheels' mean speed at least this, the same way (no-load is ~19.6 at 7.4 V;
RUNAWAY_S = 1.5            # the scenario cruises at 11.6) over this window: lifted while armed, or running away

# ---- logging (D5) ----
LOG_DIR = "logs"
