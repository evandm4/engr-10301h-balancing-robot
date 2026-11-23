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

# ---- startup and calibration (D4) ----
CALIBRATION_FILE = "calibration.json"   # in ~/legobot; written by robot.calibrate
GYRO_CAL_S = 2.0           # robot still for this long at startup: gyro bias = mean pitch rate
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
TILT_LIMIT = 0.70          # rad (40 deg): past the lean cap (15 deg) and the scenario's largest start tilt
                           # (0.18 rad) with margin; well short of the simulator's 60 deg fall angle
DUTY_CAP = 1.0             # |duty| clamp in the balancing run (the Build HAT's full scale)
BENCH_DUTY_CAP = 0.3       # --bench: wheels off the ground, as hw_config.BENCH_DUTY_CAP
STALL_DUTY = 0.5           # a wheel commanded at least this duty...
STALL_SPEED = 1.0          # ...turning slower than this (rad/s, relative to the body)...
STALL_S = 1.0              # ...for this long is stalled (the HAT has no current limit to protect it)
VIN_CUTOFF = 6.6           # V under load: 2S LiPo, 3.3 V per cell (DATASHEET 3.6)
VIN_LOW_COUNT = 3          # consecutive readings below the cutoff before tripping (load transients)
VIN_PERIOD_S = 0.5         # how often the loop asks the HAT for vin (a 4-byte command; reply read in the background)
VIN_STALE_S = 2.0          # trip if no vin reply for this long
MOTOR_STALE_S = 0.05       # trip if either motor's newest data line is older than this (5 lines missed)
WATCHDOG_S = 0.10          # a separate thread coasts the motors if the loop has not ticked for this long

# ---- logging (D5) ----
LOG_DIR = "logs"
