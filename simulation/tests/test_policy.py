"""Learned policies (balance_sim.policy, batch_sim.policy): the Phase 0 checks of docs/ml-baselines-plan.md.

  * shapes and parameter counts;
  * the batched NumPy policy against the one-robot reference, bit for bit in float64;
  * many different networks in one batch, against each on its own;
  * the gate test: (a) the linear policy with the PID-equivalent weights against an
    independent "cascaded PID with the policy's integrators", and (b) that variant within 1%
    of the real reference PID on the training objective and the 60 s development score;
  * the output clip, saturation freezing the integrators, save/load;
  * the seed tiers;
  * the policy search;
  * the residual policy (B1): zero output is the reference PID bit for bit;
  * the GPU against NumPy (skipped without torch).
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from balance_sim import (Controller, PolicyController, PolicySpec, SensorNoise, load_policy, pack,
                         pid_equivalent_weights, save_policy, step_profile, unpack)
from balance_sim.policy import OBSERVATIONS, RESIDUAL_INPUTS
from batch_sim import BatchRobot, SimSpec, build_schedule, noise_table, simulate_batch
from batch_tuning import BatchEvaluator, RobustnessConfig, cma_es_policy_search, spec_from_eval_config
from config import CONFIG
from tuning import Evaluator

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = json.loads((ROOT / "results/lqr_comparison/20261108-225341/comparison.json").read_text())
REFERENCE_PID = REFERENCE["results"]["PID, ML-tuned"]
GAINS = REFERENCE_PID["gains"]                     # kp, ki, kd, kv_p every ratio in the paper is against

SPACE = CONFIG.build_search_space()
BASE = CONFIG.build_eval_config()
SHORT = replace(BASE, duration=2.0, goal_steps=((0.3, 0.3), (1.2, -0.2)), start_pitches=(0.1, -0.08, 0.3),
                train_seeds=(1, 2))
CONFIGS = {
    "default": SHORT,
    "ideal_actuator": replace(SHORT, motor=None),
    "no_noise_no_latency": replace(SHORT, sensor=SensorNoise(), latency=0.0),
    "latency_between_steps": replace(SHORT, latency=0.0037),
}
LINEAR, MLP, WARM, RESIDUAL = (CONFIG.build_policy_spec(k) for k in ("linear", "mlp", "warm", "residual"))
MAX_TORQUE = BASE.params.max_torque


def _goal(config):
    return step_profile(config.goal_steps, config.goal_transition)


def _pid_linear_path(spec):
    """The PID-equivalent weights laid out on spec's linear path (zeros where it has more inputs)."""
    full = CONFIG.build_policy_spec("linear")
    w = dict(zip(full.inputs, pid_equivalent_weights(full, GAINS)))
    return np.array([w.get(i, 0.0) for i in spec.inputs])


def _near_pid(spec, n, spread, seed=0):
    """Policies that balance: the PID-equivalent linear path plus a small random MLP."""
    rng = np.random.default_rng(seed)
    params = rng.normal(0.0, spread, (n, spec.n_params))
    if spec.linear_path:
        params[:, :spec.n_inputs] = _pid_linear_path(spec)
    return params


def _controller(config, spec, params):
    """The reference PolicyController with config's goal and the PID limits the batch uses."""
    sim = spec_from_eval_config(config, "policy", spec)
    return PolicyController(spec, params, config.params.max_torque, goal=_goal(config),
                            max_pitch_command=sim.max_pitch_command, pitch_integral_limit=sim.pitch_integral_limit,
                            velocity_integral_limit=sim.velocity_integral_limit)


def _zero_residual(n=1, spread=0.3, seed=0):
    """Residual parameters with random hidden layers but a zero output layer: exactly the PID."""
    params = np.random.default_rng(seed).normal(0.0, spread, (n, RESIDUAL.n_params))
    params[:, -17:] = 0.0                     # output layer: 16 weights and the bias
    return params


def _flat_scenarios(config, P):
    ev = Evaluator(config, SPACE)
    scen = ev.scenarios()
    seeds = tuple(config.train_seeds)
    return scen, np.tile([p for p, _ in scen], P), np.tile([seeds.index(s) for _, s in scen], P), seeds


def _batched(config, spec, params, record=False):
    sim = spec_from_eval_config(config, "policy", spec)
    scen, pitch, col, seeds = _flat_scenarios(config, len(params))
    schedule = build_schedule(sim)
    return simulate_batch(sim, BatchRobot.from_params(config.params, config.motor), params, pitch,
                          noise=noise_table(seeds, schedule.n_ticks), seed_index=col, schedule=schedule,
                          record=record, scenarios_per_candidate=len(scen)), scen


# ---- shapes ---------------------------------------------------------------------------------

def test_parameter_counts_match_the_plan():
    assert LINEAR.n_params == 10 and "goal_minus_v" not in LINEAR.input_names
    assert MLP.n_params == 481
    assert WARM.n_params == 11 + 481
    assert CONFIG.build_policy_spec("mlp", hidden=(32, 32)).n_params == 1473
    assert RESIDUAL.n_params == 513 and RESIDUAL.input_names[-2:] == RESIDUAL_INPUTS
    assert RESIDUAL.residual_gains == tuple(GAINS)
    np.testing.assert_allclose(RESIDUAL.output_scale, 0.25 * MAX_TORQUE)


def test_pack_unpack_round_trip():
    params = np.random.default_rng(0).normal(size=WARM.n_params)
    linear, layers = unpack(WARM, params)
    assert linear.shape == (11,) and [W.shape for W, _ in layers] == [(16, 11), (16, 16), (1, 16)]
    np.testing.assert_array_equal(pack(WARM, linear, layers), params)


def test_spec_and_simspec_validation():
    with pytest.raises(ValueError):
        PolicySpec(inputs=(0,), hidden=(), linear_path=False, scales=LINEAR.scales, output_scale=0.1,
                   t_leak=10.0, integral_limits=(1.0, 1.0))
    with pytest.raises(ValueError):
        replace(spec_from_eval_config(SHORT), controller="policy")          # no PolicySpec
    with pytest.raises(ValueError):
        spec_from_eval_config(SHORT, "cascaded", LINEAR)                    # a PolicySpec it won't use


# ---- batched against the reference -------------------------------------------------------------

@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("kind", ["linear", "mlp", "warm", "residual"])
def test_batched_trajectories_match_reference_exactly(name, kind):
    config, spec = CONFIGS[name], CONFIG.build_policy_spec(kind)
    rng = np.random.default_rng(3)
    # Two that balance, one that falls for sure.
    params = np.vstack([_near_pid(spec, 2, 0.05, seed=3) if spec.linear_path else rng.normal(0, 0.1, (2, spec.n_params)),
                        rng.normal(0, 3.0, (1, spec.n_params))])
    result, scen = _batched(config, spec, params, record=True)
    ref = Evaluator(config, SPACE)
    S = len(scen)
    for p in range(len(params)):
        for s, (pitch, seed) in enumerate(scen):
            run = ref.run_controller(_controller(config, spec, params[p]), pitch, seed)
            row = p * S + s
            np.testing.assert_array_equal(result.states[:len(run.states), row], run.states)
            assert bool(result.fell[row]) == run.fell
            np.testing.assert_allclose(result.cost[row], ref.score(run), rtol=1e-12)
    if spec.linear_path:   # the near-PID policies stay up, the random one falls
        assert not result.fell[:2 * S].any() and result.fell[2 * S:].all()


def test_different_networks_in_one_batch_match_each_alone():
    params = np.vstack([_near_pid(WARM, 3, 0.05, seed=5), np.random.default_rng(5).normal(0, 1.0, (2, WARM.n_params))])
    together, _ = _batched(SHORT, WARM, params)
    for p in range(len(params)):
        alone, scen = _batched(SHORT, WARM, params[p:p + 1])
        S = len(scen)
        np.testing.assert_array_equal(together.cost[p * S:(p + 1) * S], alone.cost)


# ---- the gate ---------------------------------------------------------------------------------

class PIDWithPolicyIntegrators(Controller):
    """The cascaded PID written directly, with the policy's integrators in place of its own.

    torque = kp (theta - kv (goal - v)) + kd theta_dot + ki (int_theta - kv int_v), with both
    integrals leaky and frozen while the previous command was saturated. No lean cap. Written
    without balance_sim.policy, so the gate's part (a) compares two independent codes.
    """

    def __init__(self, gains, t_leak, max_torque, goal):
        self.kp, self.ki, self.kd, self.kv = gains
        self.leak, self.max_torque, self.goal = 1.0 / t_leak, max_torque, goal
        self.reset()

    def goal_velocity(self, t):
        return self.goal(t)

    def reset(self):
        self.int_v = self.int_theta = 0.0
        self.saturated = False
        self.time = 0.0

    def update(self, state, dt):
        v, theta, theta_dot = float(state[1]), float(state[2]), float(state[3])
        error_v = self.goal(self.time) - v
        if not self.saturated:
            self.int_v += dt * (error_v - self.leak * self.int_v)
            self.int_theta += dt * (theta - self.leak * self.int_theta)
        torque = (self.kp * (theta - self.kv * error_v) + self.kd * theta_dot
                  + self.ki * (self.int_theta - self.kv * self.int_v))
        self.saturated = abs(torque) > self.max_torque
        self.time += dt
        return min(max(torque, -self.max_torque), self.max_torque)


@pytest.mark.parametrize("config", [SHORT, replace(BASE, train_seeds=(1, 2))], ids=["short", "full_20s"])
def test_gate_a_linear_policy_is_exactly_the_pid_with_policy_integrators(config):
    weights = pid_equivalent_weights(LINEAR, GAINS)
    batched, scen = _batched(config, LINEAR, weights[None, :])
    ref = Evaluator(config, SPACE)
    for s, (pitch, seed) in enumerate(scen):
        direct = ref.run_controller(PIDWithPolicyIntegrators(GAINS, LINEAR.t_leak, config.params.max_torque,
                                                             _goal(config)), pitch, seed)
        assert not direct.fell
        np.testing.assert_allclose(batched.cost[s], ref.score(direct), rtol=1e-9)


def test_gate_b_policy_integrators_stay_within_one_percent_of_the_real_pid():
    """Bound stated in the plan before measuring here: 1%. Measured 2026-11-09: +0.68% on the
    training objective, -0.66% on the 60 s development score. This gap is reported in the paper."""
    robust = CONFIG.build_robustness()
    weights = pid_equivalent_weights(LINEAR, GAINS)
    for config, seeds in ((CONFIG.build_eval_config(), None),
                          (CONFIG.build_eval_config(validation=True), robust.dev_seeds)):
        pid = BatchEvaluator(config, SPACE, robustness=robust).evaluate_population([GAINS], seeds)
        lin = BatchEvaluator(config, SPACE, robustness=robust, controller="policy",
                             policy=LINEAR).evaluate_policies([weights], seeds)
        assert pid.n_fell[0] == 0 and lin.n_fell[0] == 0
        assert abs(lin.objective[0] / pid.objective[0] - 1.0) < 0.01
    # The training objective of the reference PID itself is reproduced, so the comparison is
    # against the same controller as in the paper.
    train = BatchEvaluator(CONFIG.build_eval_config(), SPACE, robustness=robust).evaluate_population([GAINS])
    np.testing.assert_allclose(train.objective[0], REFERENCE_PID["train_objective"], rtol=1e-9)


# ---- the residual policy (B1) ------------------------------------------------------------------

@pytest.mark.parametrize("name", list(CONFIGS))
def test_zero_residual_is_the_reference_pid_bit_for_bit(name):
    """Training starts exactly at the reference PID: same trajectories, one robot and batched."""
    config = CONFIGS[name]
    params = _zero_residual(2)
    ref = Evaluator(config, SPACE)
    pid_runs = {sc: ref.run(GAINS, *sc) for sc in ref.scenarios()}
    for sc, pid in pid_runs.items():
        res = ref.run_controller(_controller(config, RESIDUAL, params[0]), *sc)
        np.testing.assert_array_equal(res.states, pid.states)
        np.testing.assert_array_equal(res.torques, pid.torques)
    batched, scen = _batched(config, RESIDUAL, params, record=True)
    pid_sim = spec_from_eval_config(config)
    _, pitch, col, seeds = _flat_scenarios(config, 2)
    schedule = build_schedule(pid_sim)
    pid_batch = simulate_batch(pid_sim, BatchRobot.from_params(config.params, config.motor),
                               np.tile(GAINS, (len(pitch), 1)), pitch, noise=noise_table(seeds, schedule.n_ticks),
                               seed_index=col, schedule=schedule, record=True)
    np.testing.assert_array_equal(batched.states, pid_batch.states)
    np.testing.assert_array_equal(batched.cost, pid_batch.cost)


def test_zero_residual_scores_exactly_the_pid_under_robust_scoring():
    robust = replace(CONFIG.build_robustness(), train_seeds=(1, 2, 3))
    pid = BatchEvaluator(SHORT, SPACE, robustness=robust).evaluate_population([GAINS])
    res = BatchEvaluator(SHORT, SPACE, robustness=robust, controller="policy",
                         policy=RESIDUAL).evaluate_policies(_zero_residual(1))
    np.testing.assert_array_equal(res.costs, pid.costs)
    assert res.objective[0] == pid.objective[0]


def test_residual_is_bounded_by_its_authority():
    params = np.zeros(RESIDUAL.n_params)
    params[-1] = 50.0                                   # output bias: tanh saturates at +1
    c = _controller(SHORT, RESIDUAL, params)
    c.update(np.array([0.0, 0.0, 0.0, 0.0]), 0.005)
    np.testing.assert_allclose(c.last_residual, RESIDUAL.output_scale)
    with pytest.raises(ValueError):
        replace(spec_from_eval_config(SHORT, "policy", RESIDUAL), pitch_reference="commanded")


# ---- output, saturation, files -----------------------------------------------------------------

def test_output_reaches_exactly_the_torque_limit_and_saturation_freezes_the_integrators():
    params = np.zeros(LINEAR.n_params)
    params[LINEAR.input_names.index("theta_meas")] = 1e3       # any tilt saturates
    c = PolicyController(LINEAR, params, MAX_TORQUE)
    state = np.array([0.0, 0.0, 0.1, 0.0])
    assert c.update(state, 0.005) == MAX_TORQUE
    frozen = c._int_theta
    assert c.update(state, 0.005) == MAX_TORQUE
    assert c._int_theta == frozen                                 # frozen while saturated
    assert c.update(-state, 0.005) == -MAX_TORQUE
    c2 = PolicyController(LINEAR, np.zeros(LINEAR.n_params), MAX_TORQUE)
    c2.update(state, 0.005)
    c2.update(state, 0.005)
    assert c2._int_theta > 0.0                                   # not saturated: it integrates


def test_save_and_load_give_identical_torques(tmp_path):
    params = _near_pid(WARM, 1, 0.05, seed=9)[0]
    path = tmp_path / "policy.npz"
    save_policy(path, WARM, params, MAX_TORQUE, base_gains=np.array(GAINS))
    spec, loaded, max_torque = load_policy(path)
    assert spec == WARM and max_torque == MAX_TORQUE
    np.testing.assert_array_equal(loaded, params)
    a, b = PolicyController(WARM, params, MAX_TORQUE), PolicyController(spec, loaded, max_torque)
    rng = np.random.default_rng(0)
    for _ in range(50):
        state = rng.normal(0, 0.2, 4)
        assert a.update(state, 0.005) == b.update(state, 0.005)


# ---- seed tiers -------------------------------------------------------------------------------

def test_seed_tiers_are_configured_and_checked():
    r = CONFIG.build_robustness()
    assert set(r.smoke_seeds) <= set(r.dev_seeds) and len(r.fall_rate_seeds) == 1000
    with pytest.raises(ValueError, match="overlap"):
        replace(r, dev_seeds=r.dev_seeds + (r.fall_rate_seeds[0],))
    with pytest.raises(ValueError, match="subset"):
        replace(r, smoke_seeds=(1,))
    with pytest.raises(ValueError, match="retired"):
        replace(r, fall_rate_seeds=(1000,))


# ---- the search -------------------------------------------------------------------------------

def test_policy_search_starts_from_and_never_ends_worse_than_x0():
    ev = BatchEvaluator(SHORT, SPACE, controller="policy", policy=LINEAR)
    x0 = pid_equivalent_weights(LINEAR, GAINS)
    start = ev.evaluate_policies(x0[None, :]).objective[0]
    result = cma_es_policy_search(ev, x0, sigma0=0.05, popsize=8, max_iter=3, seed=0)
    assert result.method == "cma_es" and result.n_evals == 1 + 3 * 8
    assert result.history[0] == (1, start) and result.best_cost <= start
    assert result.robot_seconds == result.n_evals * ev.simulations_per_evaluation() * SHORT.duration
    np.testing.assert_allclose(ev.evaluate_policies(result.best_params[None, :]).objective[0], result.best_cost)


def test_policy_search_goes_diagonal_above_the_threshold():
    ev = BatchEvaluator(replace(SHORT, duration=0.2), SPACE, controller="policy", policy=MLP)
    result = cma_es_policy_search(ev, np.zeros(MLP.n_params), sigma0=0.1, popsize=4, max_iter=1,
                                  full_cma_max_params=100)
    assert result.method == "sep_cma_es"


def test_evaluator_rejects_mismatched_policies():
    with pytest.raises(ValueError):
        BatchEvaluator(SHORT, SPACE).evaluate_policies(np.zeros((1, LINEAR.n_params)))
    with pytest.raises(ValueError):
        BatchEvaluator(SHORT, SPACE, controller="policy", policy=LINEAR).evaluate_policies(np.zeros((1, 3)))


# ---- GPU --------------------------------------------------------------------------------------

def test_torch_matches_numpy_including_chunking_and_padding():
    torch = pytest.importorskip("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    robust = replace(CONFIG.build_robustness(), train_seeds=(1, 2, 3))
    params = _near_pid(WARM, 5, 0.03, seed=2)
    a = BatchEvaluator(SHORT, SPACE, robustness=robust, controller="policy", policy=WARM).evaluate_policies(params)
    n_total = BatchEvaluator(SHORT, SPACE, robustness=robust).simulations_per_evaluation()
    for max_batch in (None, 2 * n_total):          # one chunk; three chunks of 2, 2 and 1 candidates
        b = BatchEvaluator(SHORT, SPACE, robustness=robust, controller="policy", policy=WARM, backend="torch",
                           device=device, max_batch=max_batch).evaluate_policies(params)
        np.testing.assert_allclose(b.costs, a.costs, rtol=1e-9)
        np.testing.assert_array_equal(b.n_fell, a.n_fell)
    assert torch.get_float32_matmul_precision() == "highest"


def test_torch_residual_matches_numpy():
    torch = pytest.importorskip("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    robust = replace(CONFIG.build_robustness(), train_seeds=(1, 2))
    params = np.vstack([_zero_residual(1), np.random.default_rng(4).normal(0, 0.3, (2, RESIDUAL.n_params))])
    a = BatchEvaluator(SHORT, SPACE, robustness=robust, controller="policy", policy=RESIDUAL).evaluate_policies(params)
    b = BatchEvaluator(SHORT, SPACE, robustness=robust, controller="policy", policy=RESIDUAL, backend="torch",
                       device=device).evaluate_policies(params)
    np.testing.assert_allclose(b.costs, a.costs, rtol=1e-9)
