"""The encoder model (balance_sim.Encoder; SOFTWARE_PLAN B3)."""

import math

import numpy as np
import pytest

from balance_sim import Encoder, SensorNoise

R = 0.0345
DT = 0.0025


def drive_at(speed, encoder, duration, control_every=2):
    """Roll at a constant ground speed with the body upright; return (t, encoder speed) at control ticks."""
    out = []
    for k in range(int(round(duration / DT))):
        t = k * DT
        encoder.observe(t, np.array([speed * t, speed, 0.0, 0.0]))
        if k % control_every == 0:
            out.append((t, encoder.speed()))
    return np.array(out)


def test_velocity_noise_matches_datasheet():
    """DATASHEET 5.4: counts 10 ms apart at r = 34.5 mm give sigma = 0.025 m/s (q / (T sqrt 6)).
    That is an average over speeds: at one speed the error is q sqrt(p (1 - p)), p the fractional
    count step per period, so pool many speeds."""
    errors = []
    for v in np.linspace(0.05, 0.6, 111):
        enc = Encoder(360, 0.010, 0.005, R)
        readings = drive_at(v, enc, 1.0)[10:]          # skip start-up
        errors.append(R * readings[:, 1] - v)
    error = np.concatenate(errors)
    expected = (2 * math.pi * R / 360) / 0.010 / math.sqrt(6)
    assert expected == pytest.approx(0.0246, abs=5e-4)
    assert np.mean(error) == pytest.approx(0.0, abs=2e-3)
    assert np.sqrt(np.mean(error**2)) == pytest.approx(expected, rel=0.1)


def test_speed_is_held_for_a_period_and_arrives_after_the_delay():
    enc = Encoder(360, 0.010, 0.005, R)
    readings = drive_at(0.3, enc, 0.2, control_every=1)
    speeds = readings[:, 1]
    changes = readings[1:, 0][np.diff(speeds) != 0]
    # The first nonzero speed needs two samples: t = 0 and 0.010, the second available at 0.015.
    assert readings[speeds != 0][0, 0] == pytest.approx(0.015)
    # Changes happen only when a sample arrives: at 0.005 + multiples of 0.010.
    periods = (changes - 0.005) / 0.010
    assert np.allclose(periods, np.round(periods), atol=1e-9)


def test_quantized_to_whole_counts():
    enc = Encoder(360, 0.010, 0.0, R)
    enc.observe(0.0, np.array([0.0, 0.0, 0.0, 0.0]))
    enc.observe(0.010, np.array([R * math.radians(2.7), 0.0, 0.0, 0.0]))
    assert enc.angle() == pytest.approx(math.radians(2.0))
    assert enc.speed() == pytest.approx(math.radians(2.0) / 0.010)


def test_relative_angle_subtracts_the_body_pitch():
    """The motor's encoder reads the wheel relative to the body: x / r - theta."""
    enc = Encoder(360, 0.010, 0.0, R)
    enc.observe(0.0, np.array([0.0, 0.0, math.radians(5.5), 0.0]))
    assert enc.angle() == pytest.approx(math.radians(-6.0))


def test_sensor_uses_the_encoder_for_velocity_and_position():
    s = SensorNoise(pitch_rate_bias=0.01, encoder_counts=360, encoder_period=0.010, encoder_delay=0.0,
                    wheel_radius=R, velocity_std=5.0, position_std=5.0)   # stds unused with the encoder
    state0, state1 = np.array([0.0, 0.0, 0.0, 0.0]), np.array([R * math.radians(3.2), 0.4, 0.0, 0.0])
    s.observe(0.0, state0)
    s.observe(0.010, state1)
    m = s.measure(state1)
    assert m[1] == pytest.approx(R * (math.radians(3.0) / 0.010 + 0.01))
    assert m[0] == pytest.approx(R * (math.radians(3.0) + 0.0))
    s.reset()
    assert s._encoder.speed() == 0.0 and s._encoder.angle() == 0.0


def test_encoder_off_by_default():
    assert not SensorNoise().has_encoder
