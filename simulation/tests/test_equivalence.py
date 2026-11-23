"""Whole runs: the batched simulator against the reference simulator, run by run.

In float64 the two do the same arithmetic in the same order, so trajectories,
fall times and final states match exactly. Costs agree to rounding error; the
only difference is that the reference code sums each integral with np.sum
while the batched code adds it up step by step.
"""

from dataclasses import replace

import numpy as np
import pytest

from balance_sim import IMUSensor, SensorNoise
from batch_sim import BatchRobot, simulate_batch
from batch_tuning import BatchEvaluator, decode_units, robot_from_eval_config
from config import CONFIG
from tuning import Evaluator

HAND_TUNED = list(CONFIG.controller.hand_tuned_gains)   # kp, ki, kd, kv_p
COST_RTOL = 1e-12
SPACE = CONFIG.build_search_space()

# The project's scenario, shortened so the tests run quickly, plus variations of it.
BASE = CONFIG.build_eval_config()
SHORT = replace(BASE, duration=2.0, goal_steps=((0.3, 0.3), (1.2, -0.2)), start_pitches=(0.1, -0.08),
                train_seeds=(1, 2))
CONFIGS = {
    "default": SHORT,
    "ideal_actuator": replace(SHORT, motor=None),
    "no_noise_no_latency": replace(SHORT, sensor=SensorNoise(), latency=0.0),
    "biases_without_noise": replace(SHORT, sensor=SensorNoise(pitch_bias=0.02, pitch_rate_bias=0.01)),
    "no_rotor_lossless_gearbox": replace(SHORT, params=SHORT.params.with_changes(rotor_inertia=0.0),
                                         motor=replace(SHORT.motor, gear_efficiency=1.0)),
    "latency_between_steps": replace(SHORT, latency=0.0037),
    "control_every_step": replace(SHORT, control_dt=0.0025),
    # CONFIG scores pitch from upright, so the default case covers that branch; this covers the other.
    "commanded_reference": replace(SHORT, pitch_reference="commanded"),
    "friction_weak_motor": replace(SHORT, params=SHORT.params.with_changes(wheel_friction=0.003, max_torque=0.3),
                                   motor=replace(SHORT.motor, supply_voltage=6.0)),
    "coulomb_friction": replace(SHORT, params=SHORT.params.with_changes(coulomb_friction=0.02,
                                                                        coulomb_friction_speed=0.5)),
}


def _voltage_config(feedforward=True, base=SHORT, gain=1.0):
    """Voltage-mode driver whose constants (resistance x 0.9, torque constant x 1.1) differ from the true motor."""
    estimate = replace(base.motor, resistance=base.motor.resistance * 0.9,
                       torque_constant=base.motor.torque_constant * 1.1)
    return replace(base, voltage_estimate=estimate, voltage_feedforward=feedforward,
                   voltage_feedforward_gain=gain)


VOLTAGE_CONFIGS = {
    "feedforward": _voltage_config(True),
    "no_feedforward": _voltage_config(False),
    "exact_constants": replace(SHORT, voltage_estimate=SHORT.motor),
    "latency_between_steps": _voltage_config(True, replace(SHORT, latency=0.0037)),
    "gain_1p2": _voltage_config(True, gain=1.2),
    "gain_0p9": _voltage_config(True, gain=0.9),
    "current_limited": _voltage_config(True, replace(SHORT, motor=replace(SHORT.motor, current_limit=0.8))),
}


def _candidates(n=24, seed=0):
    """Random gains plus corner cases: every gain at its lower / upper bound, the hand-tuned set,
    and one too weak to balance (Kp below the critical gain), so the falling path is always covered."""
    units = np.random.default_rng(seed).random((n, SPACE.dim))
    units[0], units[1] = 0.0, 1.0
    gains = decode_units(SPACE, units)
    gains[2] = HAND_TUNED
    gains[3] = [0.1, 0.0, 0.05, 0.0]
    return gains


@pytest.mark.parametrize("name", list(CONFIGS))
def test_costs_and_falls_match_reference(name):
    config = CONFIGS[name]
    ref, batched = Evaluator(config, SPACE), BatchEvaluator(config, SPACE)
    gains = _candidates()

    result = batched.evaluate_population(gains)
    for i, g in enumerate(gains):
        expected = ref.evaluate_gains(g)
        np.testing.assert_allclose(result.costs[i], expected.costs, rtol=COST_RTOL)
        assert result.n_fell[i] == expected.n_fell
    # The candidates should cover both outcomes, or the test isn't checking much.
    assert 0 < result.n_fell.sum() < result.costs.size


@pytest.mark.parametrize("name", list(VOLTAGE_CONFIGS))
def test_voltage_mode_costs_and_falls_match_reference(name):
    config = VOLTAGE_CONFIGS[name]
    ref, batched = Evaluator(config, SPACE), BatchEvaluator(config, SPACE)
    assert batched.robot.voltage_mode
    gains = _candidates()

    result = batched.evaluate_population(gains)
    for i, g in enumerate(gains):
        expected = ref.evaluate_gains(g)
        np.testing.assert_allclose(result.costs[i], expected.costs, rtol=COST_RTOL)
        assert result.n_fell[i] == expected.n_fell
    assert 0 < result.n_fell.sum() < result.costs.size


@pytest.mark.parametrize("name", ["feedforward", "no_feedforward", "latency_between_steps", "gain_1p2", "gain_0p9"])
def test_voltage_mode_trajectories_match_reference_exactly(name):
    config = VOLTAGE_CONFIGS[name]
    ref, batched = Evaluator(config, SPACE), BatchEvaluator(config, SPACE)
    gains = _candidates(n=12, seed=1)
    pitch, seed = ref.scenarios()[1]

    run = simulate_batch(batched.spec, batched.robot, gains, pitch, noise=batched.noise_for((seed,)),
                         seed_index=np.zeros(len(gains), dtype=int), schedule=batched.schedule, record=True)
    for i, g in enumerate(gains):
        expected = ref.run(g, pitch, seed)
        n = len(expected.t)
        np.testing.assert_array_equal(run.states[:n, i], expected.states)
        assert run.fell[i] == expected.fell


def test_voltage_mode_differs_from_current_mode():
    """The new path actually runs: voltage mode with mismatched constants scores differently."""
    gains = _candidates(n=6, seed=5)
    current = BatchEvaluator(SHORT, SPACE).evaluate_population(gains).costs
    voltage = BatchEvaluator(VOLTAGE_CONFIGS["no_feedforward"], SPACE).evaluate_population(gains).costs
    assert not np.allclose(current, voltage, rtol=1e-6)


def test_voltage_feedforward_gain_changes_the_result():
    gains = _candidates(n=6, seed=5)
    base = BatchEvaluator(VOLTAGE_CONFIGS["feedforward"], SPACE).evaluate_population(gains).costs
    scaled = BatchEvaluator(VOLTAGE_CONFIGS["gain_1p2"], SPACE).evaluate_population(gains).costs
    assert not np.allclose(base, scaled, rtol=1e-6)


@pytest.mark.parametrize("feedforward", [True, False])
def test_voltage_mode_per_robot_parameters_match_separate_reference_runs(feedforward):
    """True motor constants, mass and supply voltage differ per robot; the robot code's constants don't."""
    config = _voltage_config(feedforward)
    rng = np.random.default_rng(6)
    n = 10
    masses = rng.uniform(0.6, 1.0, n)
    resistance = config.motor.resistance * rng.uniform(0.85, 1.15, n)
    constant = config.motor.torque_constant * rng.uniform(0.9, 1.1, n)
    supply = config.motor.supply_voltage * rng.uniform(0.8, 1.0, n)
    batched = BatchEvaluator(config, SPACE)
    robot = BatchRobot.from_params(config.params, config.motor, voltage_estimate=config.voltage_estimate,
                                   voltage_feedforward=feedforward, body_mass=masses, resistance=resistance,
                                   torque_constant=constant, supply_voltage=supply)
    pitch, seed = batched.scenarios()[0]
    run = simulate_batch(batched.spec, robot, np.tile(HAND_TUNED, (n, 1)), pitch,
                         noise=batched.noise_for((seed,)), seed_index=np.zeros(n, dtype=int),
                         schedule=batched.schedule)
    for i in range(n):
        motor = replace(config.motor, resistance=resistance[i], torque_constant=constant[i],
                        supply_voltage=supply[i])
        ref = Evaluator(replace(config, params=config.params.with_changes(body_mass=masses[i]), motor=motor), SPACE)
        expected = ref.run(HAND_TUNED, pitch, seed)
        np.testing.assert_allclose(run.cost[i], ref.score(expected), rtol=COST_RTOL)
        np.testing.assert_array_equal(run.final_state[i], expected.states[-1])


def test_voltage_mode_per_simulation_latency_matches_shared_latency():
    config = VOLTAGE_CONFIGS["feedforward"]
    batched = BatchEvaluator(config, SPACE)
    gains = _candidates(n=8, seed=9)
    pitch, seed = batched.scenarios()[0]
    steps = int(round(config.latency / config.dt))
    kwargs = dict(noise=batched.noise_for((seed,)), seed_index=np.zeros(len(gains), dtype=int))
    shared = simulate_batch(batched.spec, batched.robot, gains, pitch, schedule=batched.schedule, **kwargs)
    per_sim = simulate_batch(batched.spec, batched.robot, gains, pitch, delay_steps=np.full(len(gains), steps),
                             **kwargs)
    np.testing.assert_array_equal(shared.cost, per_sim.cost)
    np.testing.assert_array_equal(shared.final_state, per_sim.final_state)


def test_five_gain_sets_with_a_velocity_integral_still_match():
    """The project dropped kv_i, but both simulators still support it."""
    ref, batched = Evaluator(SHORT, SPACE), BatchEvaluator(SHORT, SPACE)
    rng = np.random.default_rng(8)
    gains = np.column_stack([_candidates(n=10, seed=8), rng.uniform(0.0, 0.3, 10)])
    gains[0, 4] = 0.03
    result = batched.evaluate_population(gains)
    for i, g in enumerate(gains):
        np.testing.assert_allclose(result.costs[i], ref.evaluate_gains(g).costs, rtol=COST_RTOL)
    # A zero kv_i is the same as leaving it out.
    four = batched.evaluate_population(gains[:, :4])
    zero = batched.evaluate_population(np.column_stack([gains[:, :4], np.zeros(10)]))
    np.testing.assert_array_equal(four.costs, zero.costs)


@pytest.mark.parametrize("name", ["default", "latency_between_steps"])
def test_trajectories_match_reference_exactly(name):
    config = CONFIGS[name]
    ref, batched = Evaluator(config, SPACE), BatchEvaluator(config, SPACE)
    gains = _candidates(n=12, seed=1)
    pitch, seed = ref.scenarios()[1]

    run = simulate_batch(batched.spec, batched.robot, gains, pitch, noise=batched.noise_for((seed,)),
                         seed_index=np.zeros(len(gains), dtype=int), schedule=batched.schedule, record=True)
    for i, g in enumerate(gains):
        expected = ref.run(g, pitch, seed)
        n = len(expected.t)
        np.testing.assert_array_equal(run.states[:n, i], expected.states)
        np.testing.assert_array_equal(run.final_state[i], expected.states[-1])
        assert run.fell[i] == expected.fell
        if expected.fell:
            assert run.fall_time[i] == expected.fall_time
            assert np.all(run.states[n:, i] == expected.states[-1])   # frozen after the fall
        else:
            assert np.isnan(run.fall_time[i])


def test_per_robot_parameters_match_separate_reference_runs():
    """A batch where every robot has its own mass and COM height (domain randomization)."""
    config = SHORT
    rng = np.random.default_rng(3)
    n = 10
    masses, heights = rng.uniform(0.6, 1.0, n), rng.uniform(0.07, 0.13, n)
    batched = BatchEvaluator(config, SPACE)
    robot = robot_from_eval_config(config, body_mass=masses, body_com_height=heights)
    pitch, seed = batched.scenarios()[0]
    run = simulate_batch(batched.spec, robot, np.tile(HAND_TUNED, (n, 1)), pitch,
                         noise=batched.noise_for((seed,)), seed_index=np.zeros(n, dtype=int),
                         schedule=batched.schedule)
    for i in range(n):
        params = config.params.with_changes(body_mass=masses[i], body_com_height=heights[i])
        ref = Evaluator(replace(config, params=params), SPACE)
        expected = ref.run(HAND_TUNED, pitch, seed)
        np.testing.assert_allclose(run.cost[i], ref.score(expected), rtol=COST_RTOL)
        np.testing.assert_array_equal(run.final_state[i], expected.states[-1])


def test_full_default_scenario_matches_reference():
    """The real tuning setup from config.py (6 s runs, 9 scenarios), on a few candidates."""
    ref, batched = Evaluator(BASE, SPACE), BatchEvaluator(BASE, SPACE)
    gains = _candidates(n=4, seed=2)
    result = batched.evaluate_population(gains)
    for i, g in enumerate(gains):
        np.testing.assert_allclose(result.costs[i], ref.evaluate_gains(g).costs, rtol=COST_RTOL)


def test_float32_stays_close_to_float64():
    """Preview of GPU precision: float32 costs for stable gains within 0.1% of float64.

    Gains near the edge of falling can legitimately diverge in float32 (a tiny
    difference decides fall / no fall), so this checks the hand-tuned gains and
    small perturbations of them.
    """
    gains = np.array(HAND_TUNED) * np.random.default_rng(4).uniform(0.8, 1.2, (16, SPACE.dim))
    r64 = BatchEvaluator(SHORT, SPACE, dtype=np.float64).evaluate_population(gains)
    r32 = BatchEvaluator(SHORT, SPACE, dtype=np.float32).evaluate_population(gains)
    # With the rotor inertia modelled, a few of these perturbations fall in float64 too; those
    # are the edge cases the docstring excludes. Check the rest.
    stable = r64.n_fell == 0
    assert stable.sum() >= 12
    assert (r32.n_fell[stable] == 0).all()
    # A quantized encoder (B3) turns a last-bit difference into a whole count now and then, so
    # float32 drifts further from float64 there: 0.5%, against 0.1% with smooth sensing.
    rtol = 5e-3 if SHORT.sensor.has_encoder else 1e-3
    np.testing.assert_allclose(r32.costs[stable], r64.costs[stable], rtol=rtol)


# ---- IMU sensor: gyro + accelerometer through the complementary filter (SOFTWARE_PLAN B2) ----

def _imu_config(voltage=False, base=SHORT):
    """Noisy, biased IMU with the accelerometer above the axle, and pushes so the force path runs."""
    sensor = IMUSensor(pitch_std=0.004, pitch_bias=0.01, pitch_rate_std=0.02, pitch_rate_bias=0.015,
                       velocity_std=0.01, imu_height=0.1, filter_time_constant=0.5,
                       control_dt=base.control_dt, gravity=base.params.gravity, seed=0)
    config = replace(base, sensor=sensor, pushes=((0.5, 0.1, 1.5), (1.4, 0.05, -2.0)))
    # voltage=False forces current mode even when the robot's config is already voltage mode (legobot)
    return _voltage_config(True, config) if voltage else replace(config, voltage_estimate=None)


IMU_CONFIGS = {
    "current": _imu_config(False),
    "voltage": _imu_config(True),
    "current_latency_between_steps": _imu_config(False, replace(SHORT, latency=0.0037)),
}


@pytest.mark.parametrize("name", list(IMU_CONFIGS))
def test_imu_costs_and_falls_match_reference(name):
    config = IMU_CONFIGS[name]
    ref, batched = Evaluator(config, SPACE), BatchEvaluator(config, SPACE)
    assert batched.spec.imu and batched.robot.voltage_mode == (name == "voltage")
    gains = _candidates()
    result = batched.evaluate_population(gains)
    for i, g in enumerate(gains):
        expected = ref.evaluate_gains(g)
        np.testing.assert_allclose(result.costs[i], expected.costs, rtol=COST_RTOL)
        assert result.n_fell[i] == expected.n_fell
    assert 0 < result.n_fell.sum() < result.costs.size


@pytest.mark.parametrize("name", list(IMU_CONFIGS))
def test_imu_trajectories_match_reference_exactly(name):
    config = IMU_CONFIGS[name]
    ref, batched = Evaluator(config, SPACE), BatchEvaluator(config, SPACE)
    gains = _candidates(n=12, seed=1)
    pitch, seed = ref.scenarios()[1]
    run = simulate_batch(batched.spec, batched.robot, gains, pitch, noise=batched.noise_for((seed,)),
                         seed_index=np.zeros(len(gains), dtype=int), schedule=batched.schedule, record=True)
    for i, g in enumerate(gains):
        expected = ref.run(g, pitch, seed)
        n = len(expected.t)
        np.testing.assert_array_equal(run.states[:n, i], expected.states)
        assert run.fell[i] == expected.fell


def test_imu_changes_the_run_and_voltage_estimate_differs_from_true_motor():
    gains = _candidates(n=6, seed=5)
    plain = BatchEvaluator(replace(IMU_CONFIGS["current"], sensor=SensorNoise(pitch_std=0.004, pitch_bias=0.01,
                           pitch_rate_std=0.02, pitch_rate_bias=0.015, velocity_std=0.01, seed=0)), SPACE)
    imu = BatchEvaluator(IMU_CONFIGS["current"], SPACE)
    assert not np.allclose(plain.evaluate_population(gains).costs, imu.evaluate_population(gains).costs, rtol=1e-6)


def test_imu_without_noise_table_applies_biases_like_the_reference():
    """A zero-noise IMU with biases: the no-table branch must match the reference (noise 0, biases kept)."""
    config = replace(IMU_CONFIGS["current"], sensor=IMUSensor(
        pitch_bias=0.01, pitch_rate_bias=0.015, imu_height=0.1, control_dt=SHORT.control_dt,
        gravity=SHORT.params.gravity, seed=0))
    ref, batched = Evaluator(config, SPACE), BatchEvaluator(config, SPACE)
    gains = _candidates(n=8, seed=3)
    pitch, seed = ref.scenarios()[0]
    run = simulate_batch(batched.spec, batched.robot, gains, pitch, schedule=batched.schedule, record=True)
    for i, g in enumerate(gains):
        expected = ref.run(g, pitch, seed)
        np.testing.assert_array_equal(run.states[:len(expected.t), i], expected.states)


@pytest.mark.parametrize("voltage", [False, True])
def test_imu_per_simulation_delay_matches_reference(voltage):
    config = _imu_config(voltage)
    batched = BatchEvaluator(config, SPACE)
    gains = _candidates(n=6, seed=2)
    pitch, seed = batched.scenarios()[0]
    delays = np.array([0, 1, 3, 4, 7, 12])
    run = simulate_batch(batched.spec, batched.robot, gains, pitch, noise=batched.noise_for((seed,)),
                         seed_index=np.zeros(len(gains), dtype=int), delay_steps=delays, record=True)
    for i, (g, d) in enumerate(zip(gains, delays)):
        ref = Evaluator(replace(config, latency=d * config.dt), SPACE)
        expected = ref.run(g, pitch, seed)
        np.testing.assert_array_equal(run.states[:len(expected.t), i], expected.states)


def test_imu_filter_runs_at_the_eval_configs_control_period():
    """Both evaluators run the filter at EvalConfig.control_dt, not the sensor's own control_dt."""
    config = replace(IMU_CONFIGS["current"], sensor=replace(IMU_CONFIGS["current"].sensor, control_dt=0.01))
    batched = BatchEvaluator(config, SPACE).evaluate_gains(HAND_TUNED)
    expected = BatchEvaluator(IMU_CONFIGS["current"], SPACE).evaluate_gains(HAND_TUNED)
    np.testing.assert_array_equal(batched.costs, expected.costs)
    ref = Evaluator(config, SPACE).evaluate_gains(HAND_TUNED)
    np.testing.assert_allclose(batched.costs, ref.costs, rtol=COST_RTOL)


# ---- Encoder: quantized, sampled, delayed wheel angle (SOFTWARE_PLAN B3) ----

def _encoder_config(imu=True, voltage=False, delay=0.005, base=SHORT, latency=None):
    """Noisy sensor (IMU or noise model) with the encoder on, built explicitly, plus pushes."""
    common = dict(pitch_std=0.004, pitch_bias=0.01, pitch_rate_std=0.02, pitch_rate_bias=0.015, velocity_std=0.01,
                  seed=0, encoder_counts=360, encoder_period=0.010, encoder_delay=delay,
                  wheel_radius=base.params.wheel_radius)
    if imu:
        sensor = IMUSensor(imu_height=0.1, filter_time_constant=0.5, control_dt=base.control_dt,
                           gravity=base.params.gravity, **common)
    else:
        sensor = SensorNoise(**common)
    config = replace(base, sensor=sensor, pushes=((0.5, 0.1, 1.5), (1.4, 0.05, -2.0)))
    if latency is not None:
        config = replace(config, latency=latency)
    return _voltage_config(True, config) if voltage else replace(config, voltage_estimate=None)


ENCODER_CONFIGS = {
    "noise_model": _encoder_config(imu=False),
    "imu": _encoder_config(imu=True),
    "voltage_imu": _encoder_config(imu=True, voltage=True),
    "imu_no_delay": _encoder_config(imu=True, delay=0.0),
    "imu_latency_between_steps": _encoder_config(imu=True, latency=0.0037),
}


@pytest.mark.parametrize("name", list(ENCODER_CONFIGS))
def test_encoder_costs_and_falls_match_reference(name):
    config = ENCODER_CONFIGS[name]
    ref, batched = Evaluator(config, SPACE), BatchEvaluator(config, SPACE)
    assert batched.spec.encoder_counts == 360
    gains = _candidates()
    result = batched.evaluate_population(gains)
    for i, g in enumerate(gains):
        expected = ref.evaluate_gains(g)
        np.testing.assert_allclose(result.costs[i], expected.costs, rtol=COST_RTOL)
        assert result.n_fell[i] == expected.n_fell
    assert 0 < result.n_fell.sum() < result.costs.size


@pytest.mark.parametrize("name", list(ENCODER_CONFIGS))
def test_encoder_trajectories_match_reference_exactly(name):
    config = ENCODER_CONFIGS[name]
    ref, batched = Evaluator(config, SPACE), BatchEvaluator(config, SPACE)
    gains = _candidates(n=12, seed=1)
    pitch, seed = ref.scenarios()[1]
    run = simulate_batch(batched.spec, batched.robot, gains, pitch, noise=batched.noise_for((seed,)),
                         seed_index=np.zeros(len(gains), dtype=int), schedule=batched.schedule, record=True)
    for i, g in enumerate(gains):
        expected = ref.run(g, pitch, seed)
        np.testing.assert_array_equal(run.states[:len(expected.t), i], expected.states)
        assert run.fell[i] == expected.fell


def test_encoder_changes_the_run():
    gains = _candidates(n=6, seed=5)
    config = ENCODER_CONFIGS["imu"]
    off = replace(config, sensor=replace(config.sensor, encoder_counts=0))
    assert not np.allclose(BatchEvaluator(off, SPACE).evaluate_population(gains).costs,
                           BatchEvaluator(config, SPACE).evaluate_population(gains).costs, rtol=1e-6)


@pytest.mark.parametrize("voltage", [False, True])
def test_encoder_per_simulation_delay_matches_reference(voltage):
    config = _encoder_config(imu=True, voltage=voltage)
    batched = BatchEvaluator(config, SPACE)
    gains = _candidates(n=6, seed=2)
    pitch, seed = batched.scenarios()[0]
    delays = np.array([0, 1, 3, 4, 7, 12])
    run = simulate_batch(batched.spec, batched.robot, gains, pitch, noise=batched.noise_for((seed,)),
                         seed_index=np.zeros(len(gains), dtype=int), delay_steps=delays, record=True)
    for i, (g, d) in enumerate(zip(gains, delays)):
        ref = Evaluator(replace(config, latency=d * config.dt), SPACE)
        expected = ref.run(g, pitch, seed)
        np.testing.assert_array_equal(run.states[:len(expected.t), i], expected.states)


@pytest.mark.parametrize("field, value", [("encoder_period", 0.0123), ("encoder_delay", 0.0037)])
def test_encoder_timing_must_be_a_multiple_of_dt(field, value):
    sensor = replace(ENCODER_CONFIGS["imu"].sensor, **{field: value})
    with pytest.raises(ValueError, match=field):
        BatchEvaluator(replace(ENCODER_CONFIGS["imu"], sensor=sensor), SPACE)
