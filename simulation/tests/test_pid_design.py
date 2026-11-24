"""The model-based PID baseline (balance_sim.pid_design)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CONFIG  # noqa: E402

pololu_only = pytest.mark.skipif(CONFIG.name != "pololu", reason="checks numbers or gains specific to config.py's Pololu robot")

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


@pololu_only
def test_faster_outer_loop_is_unstable_at_the_configured_latency():
    """The module docstring's claim: an outer bandwidth of z/2 fails with the 10 ms delay."""
    gains = model_based_pid_gains(P, outer_fraction=0.5)
    loop = LinearController.cascaded_pid(gains, T)
    assert spectral_radius(P, loop, T, CONFIG.timing.latency_s) > 1.0


@pytest.mark.skipif(CONFIG.name != "legobot", reason="the claim in legobot_config.py's design_integral_ratio comment")
def test_model_based_design_is_stable_over_the_latency_range_standstill_and_cruise():
    """H10: at standstill (Coulomb friction's zero-crossing slope) the design is stable for any delay up to 35 ms,
    and at cruise (Coulomb saturated: no friction slope) up to 20 ms; it crosses 1.0 near 25 ms."""
    loop = LinearController.cascaded_pid(CONFIG.model_based_gains(), T)
    cruise = P.with_changes(coulomb_friction=0.0)
    for latency in (0.0, 0.005, 0.01, 0.02, 0.035):
        assert spectral_radius(P, loop, T, latency) < 1.0, latency
    for latency in (0.0, 0.005, 0.01, 0.02):
        assert spectral_radius(cruise, loop, T, latency) < 1.0, latency
    assert spectral_radius(cruise, loop, T, 0.03) > 1.0
