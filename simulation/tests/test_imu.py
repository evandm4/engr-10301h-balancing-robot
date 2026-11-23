"""The simulated IMU and the robot code's complementary filter (SOFTWARE_PLAN B2)."""

import math
from dataclasses import replace

import numpy as np
import pytest

from config import CONFIG
from balance_sim import ComplementaryFilter, IMUSensor, accelerometer_tilt, simulate

G = 9.81
DT = 0.005


def quiet_imu(**changes):
    """An IMU with every noise and bias off."""
    base = dict(imu_height=0.10, filter_time_constant=0.5, control_dt=DT, gravity=G)
    base.update(changes)
    return IMUSensor(**base)


def test_accelerometer_reads_the_tilt_when_still():
    for theta in (-0.4, -0.05, 0.0, 0.2, 0.7):
        assert accelerometer_tilt(theta, 0.0, 0.0, 0.0, 0.10, G) == pytest.approx(theta, abs=1e-15)


def test_accelerometer_reads_forward_acceleration_as_a_backward_lean():
    """Specific force: accelerating forward at a reads like leaning back by atan(a / g)."""
    assert accelerometer_tilt(0.0, 0.0, 1.0, 0.0, 0.10, G) == pytest.approx(-math.atan(1.0 / G))
    # The mount height adds the tangential d theta_ddot.
    assert accelerometer_tilt(0.0, 0.0, 0.0, 5.0, 0.10, G) == pytest.approx(-math.atan(0.5 / G))


def test_estimate_converges_to_the_truth_when_still():
    imu = quiet_imu()
    state = np.array([0.0, 0.0, 0.12, 0.0])
    for _ in range(400):
        measured = imu.measure(state, (0.0, 0.0))
    assert measured[2] == pytest.approx(0.12, abs=1e-12)
    assert measured[3] == 0.0


def test_gyro_bias_becomes_a_pitch_offset_of_bias_times_tau():
    imu = quiet_imu(pitch_rate_bias=0.01)
    state = np.array([0.0, 0.0, 0.0, 0.0])
    for _ in range(int(10 * 0.5 / DT)):   # ten time constants
        measured = imu.measure(state, (0.0, 0.0))
    assert measured[2] == pytest.approx(0.01 * 0.5, rel=1e-3)


def test_acceleration_error_has_the_size_datasheet_predicts():
    """DATASHEET 3.4: 1 m/s^2 held for 0.5 s with tau = 0.5 s leaves about 0.06 rad of error,
    the first-order low-pass of the accelerometer's -atan(a/g)."""
    imu = quiet_imu()
    state = np.array([0.0, 0.0, 0.0, 0.0])
    imu.measure(state, (0.0, 0.0))        # starts still: the filter starts from a true reading
    for _ in range(int(0.5 / DT)):
        measured = imu.measure(state, (1.0, 0.0))
    predicted = -math.atan(1.0 / G) * (1 - math.exp(-0.5 / 0.5))
    assert measured[2] == pytest.approx(predicted, rel=0.02)
    assert abs(measured[2]) == pytest.approx(0.06, abs=0.006)


def test_filter_matches_its_formula_and_starts_from_the_accelerometer():
    f = ComplementaryFilter(0.5, DT)
    assert f.update(gyro=3.0, tilt=0.2) == 0.2
    alpha = 0.5 / (0.5 + DT)
    assert f.update(gyro=1.0, tilt=0.1) == pytest.approx(alpha * (0.2 + DT * 1.0) + (1 - alpha) * 0.1)
    f.reset()
    assert f.update(gyro=0.0, tilt=-0.3) == -0.3


def test_same_noise_draws_as_sensor_noise():
    """Four standard normals per tick in SensorNoise's order, so batched noise tables apply."""
    imu = quiet_imu(position_std=1.0, velocity_std=1.0, pitch_rate_std=1.0, seed=7)
    draws = np.random.default_rng(7).standard_normal(4)
    measured = imu.measure(np.zeros(4), (0.0, 0.0))
    assert measured[0] == draws[0] and measured[1] == draws[1] and measured[3] == draws[3]


def test_seed_replace_gives_a_fresh_filter():
    imu = quiet_imu(seed=1)
    imu.measure(np.array([0.0, 0.0, 0.3, 0.0]))
    fresh = replace(imu, seed=2)
    assert fresh._filter.estimate is None and fresh.seed == 2


def test_simulator_rejects_a_filter_at_another_rate():
    dyn = CONFIG.build_dynamics()
    with pytest.raises(ValueError):
        simulate(dyn, CONFIG.build_controller(), duration=0.1, dt=0.0025, control_dt=0.0025,
                 sensor=quiet_imu(control_dt=0.005))


def test_balances_through_the_scenario_with_the_imu():
    """The configured robot and its hand-tuned gains, noise off, IMU on: no fall, and the pitch
    estimate's error stays at the size the accelerations explain. With noise and biases off, the filter's
    error is a low-pass of the raw accelerometer tilt's error (the gyro path is exact but for its Euler
    step), so it can't exceed that error's peak. A fixed 0.05 rad limit, set for the 0.27 m/s scenario,
    failed at 0.063 rad once the LEGO cruise became 0.40 m/s (peaks in the speed changes, not the pushes)."""
    cfg = replace(CONFIG, sensor=replace(CONFIG.sensor, model="imu", pitch_std=0.0, pitch_bias=0.0,
                                         pitch_rate_std=0.0, pitch_rate_bias=0.0, position_std=0.0,
                                         velocity_std=0.0))
    dyn = cfg.build_dynamics()
    errors, raw = [], []

    class Recording(IMUSensor):
        def measure(self, state, accelerations=(0.0, 0.0)):
            out = super().measure(state, accelerations)
            errors.append(out[2] - state[2])
            raw.append(accelerometer_tilt(state[2], state[3], *accelerations, self.imu_height, self.gravity)
                       - state[2])
            return out

    s = cfg.build_sensor()
    sensor = Recording(**{k: getattr(s, k) for k in ("pitch_std", "pitch_bias", "pitch_rate_std", "pitch_rate_bias",
                                                     "position_std", "velocity_std", "seed", "imu_height",
                                                     "filter_time_constant", "control_dt", "gravity")})
    result = simulate(dyn, cfg.build_controller(), duration=cfg.scenario.duration_s, dt=cfg.timing.dt_s,
                      initial_state=dyn.initial_state(cfg.scenario.start_pitches_rad[0]),
                      control_dt=cfg.timing.control_dt_s, sensor=sensor, latency=cfg.timing.latency_s,
                      disturbance=cfg.build_disturbance())
    if CONFIG.name == "legobot":   # sized for this robot's gentle scenario
        assert not result.fell
        assert 0.0 < np.max(np.abs(errors)) <= np.max(np.abs(raw)) + 0.005     # + the gyro's Euler step
