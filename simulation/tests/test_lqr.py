"""The LQR controller: linear model, delay-aware design, reference controller,
batched (NumPy and PyTorch) versions against it, loop stability, and the
search-space helpers the comparison script tunes it with.
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from balance_sim import (
    LinearController,
    LQRController,
    LQRWeights,
    Planar2D,
    SensorNoise,
    delayed_commands,
    discrete_model,
    linearize,
    loop_matrix,
    lqr_gain,
    simulate,
    spectral_radius,
    stable_latency_range,
    step_profile,
)
from batch_sim import BatchRobot, build_schedule, lqr_update, noise_table, simulate_batch
from batch_tuning import BatchEvaluator, decode_units, spec_from_eval_config
from config import CONFIG, CostConfig, ProjectConfig
from tuning import Evaluator, LQRGainScaling, LQRWeightDesigner, lqr_weights

P = CONFIG.build_robot_params()
T = CONFIG.timing.control_dt_s
COST_MATCHED = CONFIG.cost_matched_lqr_weights()
# The cost weights LQR.md's 2026-10-14 results were produced with (before the Bryson-rule weights).
OLD_COST_WEIGHTS = LQRWeights(velocity=1.0, pitch=1.0, pitch_rate=0.0, torque=0.01)
# CONFIG holds the pitch-rate weight and the commands' gains off; this searches them too, as before 2026-10-24.
SEARCHED = replace(CONFIG, lqr=replace(CONFIG.lqr, w_pitch_fixed=None, w_velocity_fixed=None, w_pitch_rate_fixed=None,
                                        direct_commands_scale=None))
WITH_INTEGRAL = LQRWeights(velocity=1.0, pitch=1.0, pitch_rate=0.01, torque=0.05, velocity_integral=0.5)
COST_RTOL = 1e-12

# The project's scenario, shortened, plus variations (as in test_equivalence.py). Instant goal
# jumps (goal_transition=0), so some of the gain sets below fall and both paths get checked.
BASE = CONFIG.build_eval_config()
SHORT = replace(BASE, duration=2.0, goal_steps=((0.3, 0.3), (1.2, -0.2)), goal_transition=0.0,
                start_pitches=(0.1, -0.08), train_seeds=(1, 2))
CONFIGS = {
    "default": SHORT,
    "ideal_actuator": replace(SHORT, motor=None),
    "no_noise_no_latency": replace(SHORT, sensor=SensorNoise(), latency=0.0),
    "latency_between_steps": replace(SHORT, latency=0.0037),
    "commanded_reference": replace(SHORT, pitch_reference="commanded"),
    "friction_weak_motor": replace(SHORT, params=SHORT.params.with_changes(wheel_friction=0.003, max_torque=0.3),
                                   motor=replace(SHORT.motor, supply_voltage=6.0)),
}


def _controller(config, gain) -> LQRController:
    return LQRController(gain, config.params.max_torque, goal=step_profile(config.goal_steps, config.goal_transition),
                         integral_limit=config.velocity_integral_limit)


def _gain_sets(config):
    """Designed gains (with and without integral action), a randomly perturbed set, and one too weak to balance."""
    designed = lqr_gain(P, COST_MATCHED, T, config.latency)
    integral = lqr_gain(P, WITH_INTEGRAL, T, config.latency)
    perturbed = designed * np.random.default_rng(3).uniform(0.6, 1.4, len(designed))
    weak = designed * 0.05
    return np.array([designed, integral, perturbed, weak])


# ---- linear model -------------------------------------------------------------------

@pytest.mark.parametrize("friction", [0.0, 0.004])
def test_linearize_matches_derivatives(friction):
    params = P.with_changes(wheel_friction=friction, max_torque=1e9)   # unsaturated, as the linear model assumes
    dyn = Planar2D(params, fall_angle=1.0)
    A, B = linearize(params)
    h = 1e-6
    numeric = np.column_stack([(dyn.derivatives(h * e, 0.0) - dyn.derivatives(-h * e, 0.0)) / (2 * h)
                               for e in np.eye(4)])
    np.testing.assert_allclose(A, numeric, atol=1e-6)
    np.testing.assert_allclose(B[:, 0], (dyn.derivatives(np.zeros(4), h) - dyn.derivatives(np.zeros(4), -h)) / (2 * h),
                               rtol=1e-8)


def test_linearize_matches_physics_md():
    A, B = linearize(P)
    assert A[1, 2] == pytest.approx(-4.07, abs=0.01)
    assert A[3, 2] == pytest.approx(71.19, abs=0.01)
    assert B[1, 0] == pytest.approx(37.63, abs=0.01)
    assert B[3, 0] == pytest.approx(-238.85, abs=0.01)


# ---- the delay-aware design ---------------------------------------------------------

@pytest.mark.parametrize("latency, expected", [(0.0, 0), (0.0037, 1), (0.005, 1), (0.0087, 2), (0.010, 2)])
def test_delayed_commands(latency, expected):
    assert delayed_commands(0.005, latency) == expected


@pytest.mark.parametrize("latency", [0.0, 0.0037, 0.005, 0.0087, 0.010])
@pytest.mark.parametrize("integral", [False, True])
def test_discrete_model_matches_integrating_the_linear_robot(latency, integral):
    """One period of the augmented model against the continuous linear robot fed
    the commands it would actually see: each issued `age` ticks ago, arriving `latency` after its tick.
    The integral is the controller's own sum, one velocity sample per tick."""
    Ad, Bd, n = discrete_model(P, T, latency, integral)
    m = Ad.shape[0] - n
    rng = np.random.default_rng(1)
    s0, in_flight, new = rng.normal(size=n), rng.normal(size=m), rng.normal()
    commands = {m - i: u for i, u in enumerate(in_flight)}   # age -> command; oldest first in the state
    commands[0] = new

    A4, B4 = linearize(P)
    A, B = A4[1:, 1:], B4[1:, 0]

    def command_at(t):
        arrived = [age for age in commands if -age * T + latency <= t + 1e-15]
        return commands[min(arrived)] if arrived else 0.0

    breaks = sorted({0.0, T} | {-age * T + latency for age in commands if 0 < -age * T + latency < T})
    s = s0[:3].copy()
    for start, end in zip(breaks[:-1], breaks[1:]):
        s = solve_ivp(lambda t, y: A @ y + B * command_at(t), (start, end), s, rtol=1e-12, atol=1e-12).y[:, -1]
    if integral:
        s = np.append(s, s0[3] + T * s0[0])
    predicted = Ad @ np.concatenate([s0, in_flight]) + Bd[:, 0] * new
    np.testing.assert_allclose(predicted[:n], s, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(predicted[n:], np.append(in_flight, new)[1:])


@pytest.mark.parametrize("ticks", [1, 2, 3])
def test_delay_aware_gain_acts_on_the_predicted_state(ticks):
    """With a latency of whole control periods, the optimal controller is the delay-free LQR
    applied to the state predicted `ticks` ahead: K = K0 [Phi^q, Phi^(q-1) Gamma, ..., Gamma]."""
    K0 = lqr_gain(P, COST_MATCHED, T, 0.0)[:3]
    K = lqr_gain(P, COST_MATCHED, T, ticks * T)
    Phi, Gamma, _ = discrete_model(P, T, 0.0)
    expected = [K0 @ np.linalg.matrix_power(Phi, ticks)]
    expected += [K0 @ np.linalg.matrix_power(Phi, ticks - 1 - j) @ Gamma[:, 0] for j in range(ticks)]
    np.testing.assert_allclose(K[[0, 1, 2]], expected[0], rtol=1e-6)
    np.testing.assert_allclose(K[4:], expected[1:], rtol=1e-6)
    assert K[3] == 0.0


def test_gain_layout_and_signs():
    K = lqr_gain(P, COST_MATCHED, T, CONFIG.timing.latency_s)
    assert len(K) == 4 + 2           # 10 ms at 5 ms: two commands in flight
    assert K[3] == 0.0               # no integral weight, no integral gain
    # Leaning forward needs positive torque (u = -K z), as for the PID.
    assert K[1] < 0
    assert lqr_gain(P, WITH_INTEGRAL, T, 0.010)[3] != 0.0


def test_only_ratios_of_weights_matter():
    scaled = LQRWeights(**{k: 7.0 * v for k, v in vars(WITH_INTEGRAL).items()})
    np.testing.assert_allclose(lqr_gain(P, scaled, T, 0.010), lqr_gain(P, WITH_INTEGRAL, T, 0.010), rtol=1e-7)


def test_weights_are_validated():
    with pytest.raises(ValueError):
        LQRWeights(velocity=1.0, pitch=1.0, pitch_rate=0.0, torque=0.0)     # torque must be positive
    with pytest.raises(ValueError):
        LQRWeights(velocity=1.0, pitch=1.0, pitch_rate=-1.0, torque=0.01)
    LQRWeights(velocity=0.0, pitch=0.0, pitch_rate=0.0, torque=1.0, velocity_integral=1.0)   # zero state weights are allowed


# ---- loop stability ---------------------------------------------------------------

def test_delay_margins_match_physics_md():
    """PHYSICS.md 7.4: the hand-tuned cascade goes unstable at 16.0 ms, the 2026-11-08 tuned one at 40.75 ms."""
    tuned = (1.5665012042015811, 2.1008133218371503, 0.2849281098561087, 0.6448938913835949)
    for gains, unstable_from in ((CONFIG.controller.hand_tuned_gains, 0.0160), (tuned, 0.04075)):
        low, high = stable_latency_range(P, LinearController.cascaded_pid(gains, T), T)
        assert low == 0.0
        assert high + 0.00025 == pytest.approx(unstable_from)


@pytest.mark.parametrize("latency", [0.0, 0.0037, 0.005, 0.010, 0.015])
@pytest.mark.parametrize("weights", [COST_MATCHED, WITH_INTEGRAL])
def test_every_design_stabilizes_its_own_latency(latency, weights):
    gain = lqr_gain(P, weights, T, latency)
    assert spectral_radius(P, LinearController.lqr(gain), T, latency) < 1.0


def test_loop_matches_the_design_model():
    """At the design latency, the generic loop has the eigenvalues of Ad - Bd K."""
    gain = lqr_gain(P, WITH_INTEGRAL, T, 0.010)
    Ad, Bd, n = discrete_model(P, T, 0.010, integral=True)
    design = np.sort_complex(np.linalg.eigvals(Ad - Bd @ gain[None, :]))   # same layout as the gain
    loop = np.sort_complex(np.linalg.eigvals(loop_matrix(P, LinearController.lqr(gain), T, 0.010)))
    np.testing.assert_allclose(design, loop, atol=1e-6)   # a double eigenvalue at 0 splits by ~sqrt(rounding)


def test_ignoring_the_latency_is_unstable_at_10_ms():
    """The textbook design (no latency) on the robot's 10 ms latency: LQR.md's claim.

    It holds for the cost weights LQR.md was written with (OLD_COST_WEIGHTS), not for the
    current Bryson-rule weights, whose much heavier torque penalty makes the design gentle
    enough to tolerate the delay."""
    gain = lqr_gain(P, OLD_COST_WEIGHTS, T, 0.0)
    assert spectral_radius(P, LinearController.lqr(gain), T, 0.010) > 1.0
    assert stable_latency_range(P, LinearController.lqr(gain), T, around=0.0)[1] < 0.010


# ---- the reference controller -------------------------------------------------------

def test_balances_and_tracks_a_velocity_step():
    dyn = CONFIG.build_dynamics()
    controller = LQRController(lqr_gain(P, COST_MATCHED, T, 0.010), P.max_torque, goal=step_profile([(1.0, 0.4)]))
    result = simulate(dyn, controller, duration=6.0, dt=0.0025, initial_state=dyn.initial_state(0.1),
                      control_dt=T, latency=0.010)
    assert not result.fell
    assert result.states[-1, 1] == pytest.approx(0.4, abs=0.01)
    assert abs(result.states[-1, 2]) < 0.01
    np.testing.assert_array_equal(result.pitch_reference, 0.0)   # no lean command: scored from upright


def test_controller_remembers_its_clipped_commands():
    gain = np.array([0.0, -100.0, 0.0, 0.0, 0.5, 0.25])
    c = LQRController(gain, torque_limit=0.4)
    first = c.update(np.array([0.0, 0.0, 0.1, 0.0]), T)        # 10 N m asked, clipped
    assert first == 0.4
    assert list(c._in_flight) == [0.0, 0.4]
    second = c.update(np.array([0.0, 0.0, 0.0, 0.0]), T)       # only the commands in flight act now
    assert second == pytest.approx(-(0.5 * 0.0 + 0.25 * 0.4))
    c.reset()
    assert list(c._in_flight) == [0.0, 0.0]


def test_rejects_short_gains():
    with pytest.raises(ValueError):
        LQRController([1.0, 2.0, 3.0], torque_limit=0.4)


# ---- batched against the reference --------------------------------------------------

@pytest.mark.parametrize("name", list(CONFIGS))
def test_batched_costs_and_falls_match_reference(name):
    config = CONFIGS[name]
    ref = Evaluator(config, CONFIG.build_search_space())
    batched = BatchEvaluator(config, CONFIG.build_lqr_space(), controller="lqr")
    gains = _gain_sets(config)
    result = batched.evaluate_population(gains)
    for i, g in enumerate(gains):
        runs = [ref.run_controller(_controller(config, g), pitch, seed) for pitch, seed in ref.scenarios()]
        np.testing.assert_allclose(result.costs[i], [ref.score(r) for r in runs], rtol=COST_RTOL)
        assert result.n_fell[i] == sum(r.fell for r in runs)
    assert 0 < result.n_fell.sum() < result.costs.size


@pytest.mark.parametrize("name", ["default", "latency_between_steps"])
def test_batched_trajectories_match_reference_exactly(name):
    config = CONFIGS[name]
    ref = Evaluator(config, CONFIG.build_search_space())
    spec = spec_from_eval_config(config, controller="lqr")
    robot = BatchRobot.from_params(config.params, config.motor)
    gains = _gain_sets(config)[:2]
    pitch, seed = config.start_pitches[0], config.train_seeds[0]
    schedule = build_schedule(spec)
    result = simulate_batch(spec, robot, gains, pitch, noise=noise_table([seed], schedule.n_ticks),
                            seed_index=np.zeros(len(gains), dtype=int), schedule=schedule, record=True)
    for i, g in enumerate(gains):
        expected = ref.run_controller(_controller(config, g), pitch, seed).states
        np.testing.assert_array_equal(result.states[:len(expected), i], expected)


def test_lqr_update_with_nothing_in_flight():
    torque, integral = lqr_update(np.array([0.5]), np.array([0.1]), np.array([0.0]), 0.0, T,
                                  [np.array([-1.0]), np.array([-2.0]), np.array([0.0]), np.array([0.0])],
                                  np.array([0.0]), [], 0.4, 1.0)
    assert torque[0] == pytest.approx(0.4)          # 0.5 + 0.2 = 0.7, clipped
    assert integral[0] == pytest.approx(0.5 * T)


def test_spec_rejects_unknown_controller():
    with pytest.raises(ValueError):
        replace(spec_from_eval_config(SHORT), controller="mpc")


def test_wrong_gain_width_rejected():
    spec = spec_from_eval_config(SHORT, controller="lqr")
    with pytest.raises(ValueError):
        simulate_batch(spec, BatchRobot.from_params(SHORT.params, SHORT.motor), np.zeros((2, 3)), 0.1)


def test_cost_terms_add_up():
    batched = BatchEvaluator(SHORT, CONFIG.build_lqr_space(), controller="lqr")
    r = batched.evaluate_population(_gain_sets(SHORT)[:3])
    total = r.pitch_error + SHORT.effort_weight * r.effort + SHORT.velocity_weight * r.velocity_error
    np.testing.assert_allclose(total, r.costs, rtol=1e-12)    # these three don't fall


# ---- PyTorch backend ------------------------------------------------------------------

try:
    import torch
except ImportError:
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="needs PyTorch")
needs_cuda = pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="needs CUDA")
DEVICES = ["cpu"] + (["cuda"] if torch is not None and torch.cuda.is_available() else [])


@needs_torch
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("latency", [0.010, 0.0037, 0.0])
def test_torch_float64_matches_numpy(device, latency):
    config = replace(SHORT, latency=latency)
    numpy_eval = BatchEvaluator(config, CONFIG.build_lqr_space(), controller="lqr")
    torch_eval = BatchEvaluator(config, CONFIG.build_lqr_space(), controller="lqr", backend="torch", device=device)
    gains = _gain_sets(config)[:2]   # designed gains: well-behaved, so the last-bit differences of
                                     # torch's sin/cos stay at rounding level (see test_torch.py)
    np.testing.assert_allclose(torch_eval.evaluate_population(gains).costs,
                               numpy_eval.evaluate_population(gains).costs, rtol=1e-12)


@needs_cuda
def test_torch_float32_close_and_cache_keyed_by_gain_width():
    numpy_eval = BatchEvaluator(SHORT, CONFIG.build_lqr_space(), controller="lqr")
    torch_eval = BatchEvaluator(SHORT, CONFIG.build_lqr_space(), controller="lqr", backend="torch",
                                dtype=np.float32)
    gains = _gain_sets(SHORT)[:3]
    np.testing.assert_allclose(torch_eval.evaluate_population(gains).costs,
                               numpy_eval.evaluate_population(gains).costs, rtol=1e-4)
    narrow = lqr_gain(P, COST_MATCHED, T, 0.0)[None, :]          # no commands in flight: another simulator
    np.testing.assert_allclose(torch_eval.evaluate_population(narrow).costs,
                               numpy_eval.evaluate_population(narrow).costs, rtol=1e-4)
    assert len({key[2] for key in torch_eval._torch_sims}) == 2


# ---- what the tuner searches ------------------------------------------------------

def test_weight_designer_through_the_evaluator():
    space, designer = CONFIG.build_lqr_space(), CONFIG.build_lqr_designer()
    evaluator = BatchEvaluator(SHORT, space, controller="lqr", to_gains=designer)
    units = np.random.default_rng(5).random((3, space.dim))
    expected = evaluator.evaluate_population(designer(decode_units(space, units))).objective
    np.testing.assert_array_equal(evaluator.evaluate_units(units), expected)
    weights = lqr_weights(space.decode(units[0]), CONFIG.lqr.fixed_weights)
    np.testing.assert_allclose(designer(space.decode(units[0])[None, :])[0],
                               lqr_gain(P, weights, T, CONFIG.timing.latency_s))


def test_weight_designer_scores_a_failed_design_as_zero_gains(monkeypatch):
    import balance_sim

    def fail(*args):
        raise np.linalg.LinAlgError("no solution")

    monkeypatch.setattr(balance_sim, "lqr_gain", fail)
    designer = LQRWeightDesigner(P, T, 0.010)
    np.testing.assert_array_equal(designer(np.ones((2, 4))), np.zeros((2, 6)))


@pytest.mark.parametrize("cfg", [CONFIG, SEARCHED], ids=["commands_off", "commands_searched"])
def test_gain_scaling_keeps_the_loop_stable(cfg):
    space, scaling = cfg.build_direct_lqr_space(), cfg.build_direct_lqr_designer()
    off = cfg.lqr.direct_commands_scale is not None
    reference = cfg.build_lqr_gain().copy()
    if off:
        reference[4:] *= cfg.lqr.direct_commands_scale          # the design with the commands' gains held
    np.testing.assert_allclose(scaling.reference, reference)
    assert space.names[-1] == "k_integral" and scaling.integral
    # Within a factor of about 1.8 of the reference: some of these stabilize the loop, some don't.
    points = decode_units(space, 0.5 + 0.1 * np.random.default_rng(0).uniform(-1, 1, (40, space.dim)))
    gains = scaling(points)
    radii = np.array([spectral_radius(P, LinearController.lqr(g), T, CONFIG.timing.latency_s) for g in gains])
    with_integral = gains[:, 3] != 0.0
    assert with_integral.any()
    assert radii.max() < 1.0
    assert radii[with_integral].max() < cfg.lqr.direct_max_spectral_radius      # the margin holds with integral action
    assert np.all(np.sign(gains[with_integral, 3]) == np.sign(reference[0]))   # the velocity gain's sign
    if off:                                                     # held gains stay held, even when pulled back
        np.testing.assert_array_equal(gains[:, 4:], np.broadcast_to(reference[4:], gains[:, 4:].shape))
    # The unstable ones were pulled back, so not every result is a pure scaling.
    searched = [0, 1, 2] if off else [0, 1, 2, 4, 5]
    pure = [np.allclose(g[searched] / reference[searched], u[:len(searched)]) and
            np.isclose(abs(g[3]), u[-1]) for g, u in zip(gains, points)]
    assert 0 < sum(pure) < len(pure)


def test_gain_scaling_rejects_a_slow_integral_mode():
    """A tiny integral gain leaves a mode just inside the unit circle (a slow runaway in
    practice); the margin rejects it and falls back to the reference, with no integral."""
    scaling = CONFIG.build_direct_lqr_designer()
    tiny = np.ones(len(scaling.scaled) + 1)
    tiny[-1] = 1e-6
    gain = scaling(tiny[None, :])[0]
    np.testing.assert_allclose(gain, scaling.reference)
    no_integral = replace(CONFIG, lqr=replace(CONFIG.lqr, direct_integral_bounds=None))
    assert no_integral.build_direct_lqr_space().names == ["s_velocity", "s_pitch", "s_pitch_rate"]
    assert not no_integral.build_direct_lqr_designer().integral


def test_held_off_commands_give_a_stable_design_at_the_configured_latency():
    gain = CONFIG.build_direct_lqr_designer().reference
    np.testing.assert_array_equal(gain[4:], 0.0)
    assert spectral_radius(P, LinearController.lqr(gain), T, CONFIG.timing.latency_s) < 1.0


def test_gain_scaling_needs_a_stable_reference():
    with pytest.raises(ValueError):
        LQRGainScaling(lqr_gain(P, OLD_COST_WEIGHTS, T, 0.0), P, T, 0.010)   # unstable at 10 ms


# ---- config ------------------------------------------------------------------------

def test_cost_matched_weights_follow_cost_config():
    cfg = ProjectConfig(cost=CostConfig(pitch_limit_rad=0.6, velocity_limit=0.3, torque_limit=0.3))   # weights 4 and 4
    w = cfg.cost_matched_lqr_weights()
    assert (w.pitch, w.pitch_rate, w.velocity_integral) == (1.0, 0.0, 0.0)
    assert (w.velocity, w.torque) == pytest.approx((4.0, 4.0))
    assert (w.velocity, w.torque) == (cfg.cost.velocity_weight, cfg.cost.effort_weight)


def test_config_builds_the_lqr_controller():
    c = CONFIG.build_lqr_controller()
    np.testing.assert_array_equal(c.K, CONFIG.build_lqr_gain())
    assert c.torque_limit == CONFIG.robot.max_torque
    assert c.integral_limit == CONFIG.controller.velocity_integral_limit
    assert c.goal_velocity(2.5) == CONFIG.build_goal()(2.5)
    assert len(CONFIG.build_lqr_controller(design_latency=0.0).K) == 4
    assert CONFIG.snapshot()["derived"]["cost_matched_lqr_gain"] == pytest.approx(list(CONFIG.build_lqr_gain()))


def test_search_spaces_drop_the_parameters_held_off():
    assert CONFIG.build_lqr_space().names == ["w_integral"]
    assert CONFIG.build_direct_lqr_space().names == ["s_velocity", "s_pitch", "s_pitch_rate", "k_integral"]
    assert SEARCHED.build_lqr_space().names == ["w_pitch", "w_velocity", "w_pitch_rate", "w_integral"]
    assert SEARCHED.build_direct_lqr_space().dim == 6
    # Fixed weights reach the design, in the right places; the torque weight is always 1.
    w = lqr_weights([30.0], CONFIG.lqr.fixed_weights)
    assert (w.pitch, w.velocity, w.pitch_rate, w.torque, w.velocity_integral) == (0.0, 0.0, 0.0, 1.0, 30.0)
    w = lqr_weights([0.1, 0.5, 0.0, 2.0], SEARCHED.lqr.fixed_weights)
    assert (w.pitch, w.velocity, w.pitch_rate, w.torque, w.velocity_integral) == (0.1, 0.5, 0.0, 1.0, 2.0)
    with pytest.raises(ValueError):
        lqr_weights([1.0, 2.0], CONFIG.lqr.fixed_weights)       # wrong number of searched weights
    with pytest.raises(ValueError):
        lqr_weights([1.0] * 4, {"w_nonsense": 0.0})


def test_torque_normalized_weights_give_the_same_gain_as_any_scaling():
    """Only ratios matter: the cost-matched design, written per unit of torque weight, is the same gain."""
    from balance_sim import LQRWeights

    w = CONFIG.cost_matched_lqr_weights()
    per_torque = LQRWeights(velocity=w.velocity / w.torque, pitch=w.pitch / w.torque, pitch_rate=0.0, torque=1.0)
    np.testing.assert_allclose(lqr_gain(P, per_torque, T, 0.010), CONFIG.build_lqr_gain(), rtol=1e-8)


def test_zero_state_weights_converge_to_the_limit_design():
    """Pitch and velocity weights of exactly 0 are the limit of small ones (what the search runs into)."""
    from balance_sim import LQRWeights

    limit = lqr_gain(P, LQRWeights(velocity=0.0, pitch=0.0, pitch_rate=0.0, torque=1.0, velocity_integral=4.0), T, 0.010)
    near = lqr_gain(P, LQRWeights(velocity=1e-9, pitch=1e-9, pitch_rate=0.0, torque=1.0, velocity_integral=4.0), T, 0.010)
    np.testing.assert_allclose(near, limit, rtol=1e-6)
    assert spectral_radius(P, LinearController.lqr(limit), T, 0.010) < 1.0
    with pytest.raises(ValueError):
        LQRWeights(velocity=1.0, pitch=1.0, pitch_rate=0.0, torque=0.0)
    with pytest.raises(ValueError):
        LQRWeights(velocity=1.0, pitch=-1.0, pitch_rate=0.0, torque=1.0)
