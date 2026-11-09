"""The model-based PID baseline (balance_sim.pid_design)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CONFIG  # noqa: E402

from balance_sim import LinearController, critical_pitch_gain, model_based_pid_gains, spectral_radius  # noqa: E402
from balance_sim.pid_design import _plant_constants  # noqa: E402

P = CONFIG.build_robot_params()
T = CONFIG.timing.control_dt_s


def test_inner_loop_poles_are_where_they_were_placed():
    kp, ki, kd, _ = model_based_pid_gains(P, inner_speed=2.0, inner_damping=1.0, integral_ratio=0.1)
    a, b0, d, det0, c, mgl = _plant_constants(P)
    roots = np.sort_complex(np.roots([det0, c * kd, c * kp - a * mgl, c * ki]))
    w = 2.0 * math.sqrt(a * mgl / det0)
    expected = np.sort_complex(np.array([-0.1 * w, -w, -w], dtype=complex))
    assert np.allclose(roots, expected, rtol=1e-4, atol=1e-3)


def test_outer_bandwidth_is_the_chosen_fraction_of_the_rhp_zero():
    *_, kv = model_based_pid_gains(P, outer_fraction=1.0 / 3.0)
    a, b0, d, det0, c, mgl = _plant_constants(P)
    zero = math.sqrt(mgl / (d + b0 * P.wheel_radius))
    lean_gain = mgl / (b0 + a * P.wheel_radius)
    assert kv * lean_gain == pytest.approx(zero / 3.0)


def test_default_design_is_stable_at_the_configured_latency():
    gains = CONFIG.model_based_gains()
    assert gains[0] > critical_pitch_gain(P)
    loop = LinearController.cascaded_pid(gains, T)
    assert spectral_radius(P, loop, T, CONFIG.timing.latency_s) < 1.0


def test_faster_outer_loop_is_unstable_at_the_configured_latency():
    """The module docstring's claim: an outer bandwidth of z/2 fails with the 10 ms delay."""
    gains = model_based_pid_gains(P, outer_fraction=0.5)
    loop = LinearController.cascaded_pid(gains, T)
    assert spectral_radius(P, loop, T, CONFIG.timing.latency_s) > 1.0
