"""Hardware constants for the LEGO robot's Pi I/O layer, in one place.

Sign conventions follow PHYSICS.md: pitch theta is positive leaning forward, and a positive
wheel command drives the robot forward. Values marked "bench check" are set by the scripts in
this folder (DATASHEET 3.3, 3.4) and must be re-checked if the build changes.
"""

# ---- Build HAT serial link (DATASHEET 3.3) ----
SERIAL_PORT = "/dev/serial0"
BAUD = 115200

LEFT_PORT = 0           # Build HAT port A
RIGHT_PORT = 1          # Build HAT port B
# Multiply a "forward" duty by this to get the HAT's PWM for that port, and a HAT speed or
# position by it to get the forward-positive value. One motor is mirrored. Checked 2026-11-22
# (wheels off the ground): HAT +duty on port A drives the left wheel forward, -duty on port B the right.
PORT_SIGN = {LEFT_PORT: +1, RIGHT_PORT: -1}

# port_plimit is NOT a usable bench power cap: at or below about 0.4 it stops the drive entirely
# for duties in the slow-PWM range, and above that it scales the duty (DATASHEET 3.3). So the
# bench keeps it at 1 and caps the duty in software instead (BENCH_DUTY_CAP, bench_log.py).
BENCH_PLIMIT = 1.0
RUN_PLIMIT = 1.0
BENCH_DUTY_CAP = 0.3    # |duty| refused above this in the bench scripts (wheels off the ground)
# pwmparams <pwmthresh> <minpwm> (firmware defaults 0 0; the buildhat library sends 0.65 0.01):
# below pwmthresh the firmware uses its "slow" PWM, which changes the duty-to-speed map a lot
# (DATASHEET 3.3 sweep). Kept at the library's values until C3/B1 choose.
PWMPARAMS = (0.65, 0.01)
SELRATE_MS = 10         # data line period per port (100 Hz)
DUTY_CAP = 1.0          # clamp on |duty| in set_pwm, before the sign

# ---- ICM-20948 on the SparkFun Auto pHAT (DATASHEET 3.4) ----
I2C_BUS = 1
IMU_ADDR = 0x69          # 0x5D and 0x73 are the pHAT's unused motor driver and encoder reader
GYRO_FS_SEL = 1          # 0..3 = +-250, 500, 1000, 2000 deg/s
ACCEL_FS_SEL = 1         # 0..3 = +-2, 4, 8, 16 g
GYRO_DLPFCFG = 3         # 51.2 Hz 3 dB
ACCEL_DLPFCFG = 3        # 50.4 Hz 3 dB
GYRO_SMPLRT_DIV = 0      # ODR = 1125/(1+div) Hz with the DLPF on; 1.125 kHz keeps the newest
ACCEL_SMPLRT_DIV = 0     # sample under 1 ms old when the 200 Hz loop reads it

IMU_HEIGHT = 0.100       # m above the axle
IMU_FORWARD = 0.011      # m forward of the vertical through the axle

# Axis mapping, chip frame -> robot frame. Each entry is (chip axis index 0/1/2 = x/y/z, sign).
# Robot frame: "forward" (direction of travel), "up" (along the body, axle to head), and
# pitch rate positive when leaning forward. Board: vertical, component side forward, upside
# down (header at the bottom). Measured 2026-11-22 with imu_axes.py (35 deg forward lean: gyro x
# integrated +0.56 rad against 0.60 rad from the accelerometer). Right-handed: up (y) x forward (z)
# = +x, so a forward lean is a positive rotation about chip x.
IMU_PITCH_RATE = (0, +1)     # gyro x, along the axle
IMU_ACCEL_FORWARD = (2, +1)  # accelerometer z, out of the component side
IMU_ACCEL_UP = (1, +1)       # accelerometer y, up the body (reads +g upright)
