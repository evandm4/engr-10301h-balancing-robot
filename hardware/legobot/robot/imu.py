"""ICM-20948 driver over I2C with smbus2 (SOFTWARE_PLAN C2, DATASHEET 3.4).

Registers are written directly: the SparkFun library can set the DLPF but not the sample-rate
divider, and its getAgmt() also reads the magnetometer through the chip's I2C master. One 12-byte
block read (ACCEL_XOUT_H .. GYRO_ZOUT_L) gives accelerometer and gyro.

    imu = Imu()                      # resets and configures from hw_config
    r = imu.read()                   # ImuReading(t, pitch_rate, acc_fwd, acc_up, raw)
    theta = pitch_from_accel(r.acc_fwd, r.acc_up)
"""
import math
import time
from collections import namedtuple

from smbus2 import SMBus, i2c_msg

from . import hw_config as cfg

# Bank 0
WHO_AM_I, USER_CTRL, LP_CONFIG, PWR_MGMT_1, PWR_MGMT_2 = 0x00, 0x03, 0x05, 0x06, 0x07
ACCEL_XOUT_H = 0x2D            # .. GYRO_ZOUT_L 0x38
REG_BANK_SEL = 0x7F
# Bank 2
GYRO_SMPLRT_DIV, GYRO_CONFIG_1 = 0x00, 0x01
ACCEL_SMPLRT_DIV_1, ACCEL_SMPLRT_DIV_2, ACCEL_CONFIG = 0x10, 0x11, 0x14

G = 9.80665
GYRO_LSB_PER_DPS = (131.0, 65.5, 32.8, 16.4)
ACCEL_LSB_PER_G = (16384.0, 8192.0, 4096.0, 2048.0)

# raw: (ax, ay, az) m/s^2 and (gx, gy, gz) rad/s in the chip frame
ImuReading = namedtuple("ImuReading", "t pitch_rate acc_fwd acc_up acc gyro read_s")


def pitch_from_accel(acc_fwd, acc_up):
    """Tilt from the specific force, theta > 0 leaning forward (PHYSICS.md).

    At rest the accelerometer reads +g world-up; in the body frame that is g cos(theta) along
    "up" and -g sin(theta) along "forward"."""
    return math.atan2(-acc_fwd, acc_up)


class Imu:
    def __init__(self, bus=cfg.I2C_BUS, addr=cfg.IMU_ADDR):
        self.bus = SMBus(bus)
        self.addr = addr
        self._msgs = (i2c_msg.write(addr, [ACCEL_XOUT_H]), )
        self.gyro_scale = math.radians(1.0) / GYRO_LSB_PER_DPS[cfg.GYRO_FS_SEL]
        self.accel_scale = G / ACCEL_LSB_PER_G[cfg.ACCEL_FS_SEL]
        self._configure()

    def _w(self, reg, val):
        self.bus.write_byte_data(self.addr, reg, val)

    def _bank(self, b):
        self._w(REG_BANK_SEL, b << 4)

    def _configure(self):
        self._bank(0)
        who = self.bus.read_byte_data(self.addr, WHO_AM_I)
        if who != 0xEA:
            raise RuntimeError(f"ICM-20948 WHO_AM_I = {who:#x}, expected 0xea")
        self._w(PWR_MGMT_1, 0x80)               # reset
        time.sleep(0.1)
        self._bank(0)
        self._w(PWR_MGMT_1, 0x01)               # awake, auto clock (PLL)
        time.sleep(0.05)
        self._w(PWR_MGMT_2, 0x00)               # accelerometer and gyro on
        self._w(LP_CONFIG, 0x00)                # continuous, not duty-cycled
        self._w(USER_CTRL, 0x00)                # no DMP, FIFO or I2C master
        self._bank(2)
        self._w(GYRO_SMPLRT_DIV, cfg.GYRO_SMPLRT_DIV)
        self._w(GYRO_CONFIG_1, (cfg.GYRO_DLPFCFG << 3) | (cfg.GYRO_FS_SEL << 1) | 1)
        self._w(ACCEL_SMPLRT_DIV_1, (cfg.ACCEL_SMPLRT_DIV >> 8) & 0x0F)
        self._w(ACCEL_SMPLRT_DIV_2, cfg.ACCEL_SMPLRT_DIV & 0xFF)
        self._w(ACCEL_CONFIG, (cfg.ACCEL_DLPFCFG << 3) | (cfg.ACCEL_FS_SEL << 1) | 1)
        self.config_readback = {
            "GYRO_CONFIG_1": self.bus.read_byte_data(self.addr, GYRO_CONFIG_1),
            "ACCEL_CONFIG": self.bus.read_byte_data(self.addr, ACCEL_CONFIG),
            "GYRO_SMPLRT_DIV": self.bus.read_byte_data(self.addr, GYRO_SMPLRT_DIV),
        }
        self._bank(0)
        time.sleep(0.1)                          # let the filters settle

    def read_raw(self):
        """12 bytes in one combined transaction; returns (t, accel m/s^2, gyro rad/s, seconds taken)."""
        t0 = time.perf_counter()
        rd = i2c_msg.read(self.addr, 12)
        self.bus.i2c_rdwr(self._msgs[0], rd)
        t1 = time.perf_counter()
        b = bytes(rd)
        v = [int.from_bytes(b[i:i + 2], "big", signed=True) for i in range(0, 12, 2)]
        acc = tuple(x * self.accel_scale for x in v[:3])
        gyro = tuple(x * self.gyro_scale for x in v[3:])
        return t1, acc, gyro, t1 - t0

    def read(self):
        t, acc, gyro, dt = self.read_raw()
        ia, sa = cfg.IMU_PITCH_RATE
        jf, sf = cfg.IMU_ACCEL_FORWARD
        ju, su = cfg.IMU_ACCEL_UP
        return ImuReading(t, sa * gyro[ia], sf * acc[jf], su * acc[ju], acc, gyro, dt)

    def close(self):
        self.bus.close()
