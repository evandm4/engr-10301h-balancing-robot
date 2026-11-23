"""The generalized LinearController (balance_sim.stability): several leaky integrators, and the
exact linearization of the linear policy L (docs/ml-baselines-plan.md 3.5).

  * the PID and LQR margins recorded in the paper's comparison run are reproduced exactly;
  * L with the PID-equivalent weights and plain integrators has exactly the PID's coupled modes;
  * integrators that share a leak are merged, so an unused one adds no eigenvalue;
  * the linearized loop matches a small-amplitude, noise-free simulation of the real plant.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from balance_sim import (LinearController, Planar2D, PolicyController, loop_matrix, pid_equivalent_weights,
                         simulate, spectral_radius, stable_latency_range)
from config import CONFIG

ROOT = Path(__file__).resolve().parents[2]
RESULTS = json.loads((ROOT / "results/lqr_comparison/20261108-225341/comparison.json").read_text())["results"]
GAINS = RESULTS["PID, ML-tuned"]["gains"]
T = CONFIG.timing.control_dt_s
LATENCY = CONFIG.timing.latency_s
P = CONFIG.build_robot_params()
LINEAR = CONFIG.build_policy_spec("linear")


def _eigs(controller, latency):
    return np.sort_complex(np.linalg.eigvals(loop_matrix(P, controller, T, latency)))


@pytest.mark.parametrize("name", list(RESULTS))
def test_recorded_margins_are_unchanged(name):
    """The generalization leaves the PID and LQR analysis exactly as in comparison.json."""
    entry = RESULTS[name]
    model = (LinearController.lqr(entry["gains"]) if entry["kind"] == "lqr"
             else LinearController.cascaded_pid(entry["gains"], T))
    recorded = entry["linear_stability"]
    assert spectral_radius(P, model, T, LATENCY) == recorded["spectral_radius_at_design_latency"]
    window = stable_latency_range(P, model, T, around=LATENCY, max_latency=0.08)
    assert [1000 * x for x in window] == recorded["stable_latency_range_ms"]


@pytest.mark.parametrize("latency", [0.0, 0.0075, LATENCY, 0.020])
def test_l_pid_with_plain_integrators_has_exactly_the_pids_modes(latency):
    spec = replace(LINEAR, t_leak=float("inf"))
    l_pid = LinearController.linear_policy(spec, pid_equivalent_weights(spec, GAINS), T)
    assert len(l_pid.integrators()) == 1            # the two integrators merge: u sees one combination
    np.testing.assert_allclose(_eigs(l_pid, latency), _eigs(LinearController.cascaded_pid(GAINS, T), latency),
                               rtol=1e-9, atol=1e-12)


def test_unused_and_merged_integrators_add_no_modes():
    pid = LinearController.cascaded_pid(GAINS, T)
    padded = LinearController(pid.state_gain, (pid.integral_gain, 0.0), (pid.integrand, (1.0, 0.0, 0.0)),
                              (), (1.0, 0.99))
    np.testing.assert_array_equal(loop_matrix(P, padded, T, LATENCY), loop_matrix(P, pid, T, LATENCY))
    # Different decays can't be merged: both stay.
    two = LinearController(pid.state_gain, (pid.integral_gain, 0.1), (pid.integrand, (1.0, 0.0, 0.0)),
                           (), (1.0, 0.99))
    assert loop_matrix(P, two, T, LATENCY).shape[0] == loop_matrix(P, pid, T, LATENCY).shape[0] + 1


def test_l_pid_with_leaky_integrators_is_stable_at_the_design_latency():
    l_pid = LinearController.linear_policy(LINEAR, pid_equivalent_weights(LINEAR, GAINS), T)
    radius = spectral_radius(P, l_pid, T, LATENCY)
    assert radius < 1.0
    low, high = stable_latency_range(P, l_pid, T, around=LATENCY)
    pid_high = RESULTS["PID, ML-tuned"]["linear_stability"]["stable_latency_range_ms"][1] / 1000
    assert low == 0.0 and abs(high - pid_high) < 0.003     # the leak barely moves the delay margin


@pytest.mark.parametrize("latency", [0.0, 0.0075, 0.010])
def test_linearized_loop_matches_a_small_amplitude_simulation(latency):
    """z_{k+1} = M z_k against balance_sim at 1e-5 rad (no noise, no Coulomb friction, nothing
    saturates), with random linear weights that keep the loop stable, history gains included."""
    params = replace(P, coulomb_friction=0.0)
    weights = pid_equivalent_weights(LINEAR, GAINS)
    rng = np.random.default_rng(1)
    weights = weights * (1.0 + 0.1 * rng.uniform(-1, 1, weights.size))
    for name in ("cmd_1", "cmd_2"):
        weights[LINEAR.input_names.index(name)] = 0.05 * rng.uniform(-1, 1)
    model = LinearController.linear_policy(LINEAR, weights, T)
    assert spectral_radius(params, model, T, latency) < 1.0

    theta0, n_ticks = 1e-5, 200
    dyn = Planar2D(params, fall_angle=1.0)
    controller = PolicyController(LINEAR, weights, CONFIG.robot.max_torque)
    run = simulate(dyn, controller, duration=n_ticks * T, dt=CONFIG.timing.dt_s,
                   initial_state=dyn.initial_state(theta0), control_dt=T, latency=latency)
    spc = round(T / CONFIG.timing.dt_s)
    sim = run.states[::spc, 1:][:n_ticks]

    M = loop_matrix(params, model, T, latency)
    z = np.zeros(M.shape[0])
    z[1] = theta0
    lin = []
    for _ in range(n_ticks):
        lin.append(z[:3].copy())
        z = M @ z
    lin = np.array(lin)
    scale = np.abs(lin).max(axis=0)
    np.testing.assert_allclose(sim / scale, lin / scale, atol=1e-4)
