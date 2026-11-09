"""External pushes (a horizontal force on the body) and per-change goal blends.

The push enters Planar2D as the generalized forces of a horizontal force F at
the body's center of mass: Q_x = F, Q_theta = F l cos(theta). The batched
engines must apply it identically: bit for bit in NumPy float64, and to
rounding error on the GPU.
"""

from dataclasses import replace

import numpy as np
import pytest

from balance_sim import Planar2D, push_profile, simulate, step_profile
from batch_sim import simulate_batch
from batch_tuning import BatchEvaluator, decode_units
from config import CONFIG
from tuning import Evaluator

ROBOT = CONFIG.build_robot_params()
SPACE = CONFIG.build_search_space()
HAND_TUNED = np.array(CONFIG.controller.hand_tuned_gains)
# A short run with a push each way, one of them not a whole number of physics steps long.
SHORT = replace(CONFIG.build_eval_config(), duration=2.0, goal_steps=((0.3, 0.3, 0.6), (1.2, -0.2)),
                pushes=((0.5, 0.05, 6.5), (1.3, 0.0337, -9.0)), start_pitches=(0.1, -0.08), train_seeds=(1, 2))


# ---- profiles -------------------------------------------------------------------

def test_push_profile_timing_and_overlap():
    f = push_profile([(1.0, 0.05, 4.0), (1.02, 0.1, -1.0)])
    assert f(0.999) == 0.0
    assert f(1.0) == 4.0
    assert f(1.01) == 4.0
    assert f(1.03) == 3.0            # overlapping pushes add
    assert f(1.05) == -1.0           # the first has ended (the interval is half-open)
    assert f(1.12) == 0.0
    assert push_profile([])(1.0) == 0.0
    with pytest.raises(ValueError):
        push_profile([(1.0, -0.1, 2.0)])
    with pytest.raises(ValueError):
        push_profile([(1.0, 2.0)])


def test_per_change_blend_times():
    prof = step_profile([(1.0, 1.0, 0.5), (3.0, 0.0)], transition=2.0)
    assert prof(0.9) == 0.0
    assert prof(1.25) == pytest.approx(0.5)      # halfway through its own 0.5 s blend
    assert prof(1.5) == 1.0
    assert prof(4.0) == pytest.approx(0.5)       # the second change uses the 2 s default
    assert prof(5.0) == 0.0
    # Pairs and triples with the default blend time give the same profile.
    a, b = step_profile([(1.0, 0.3), (2.0, -0.2)], 0.8), step_profile([(1.0, 0.3, 0.8), (2.0, -0.2, 0.8)])
    assert all(a(t) == b(t) for t in np.linspace(0, 4, 401))
    with pytest.raises(ValueError):
        step_profile([(1.0, 0.3, -0.1)])


def test_blend_peak_acceleration_is_as_stated():
    """step_profile's blend peaks at 2.1875 x change / blend time (config.py quotes it)."""
    prof = step_profile([(0.0, 1.3, 1.6)])
    t = np.linspace(0, 1.6, 16001)
    v = np.array([prof(x) for x in t])
    assert np.max(np.diff(v) / np.diff(t)) == pytest.approx(2.1875 * 1.3 / 1.6, rel=1e-4)


# ---- physics ----------------------------------------------------------------------

def test_push_generalized_forces_match_the_derivation():
    """derivatives() with a push solves M q'' = f + (F, F l cos theta)."""
    p = ROBOT
    dyn = Planar2D(p, fall_angle=10.0)
    state = np.array([0.0, 0.7, 0.3, -1.1])
    tau, F = 0.12, 5.0
    _, x_dot, theta, theta_dot = state
    J, r, ml = p.rotor_inertia, p.wheel_radius, p.body_mass * p.body_com_height
    M = np.array([
        [p.body_mass + p.wheel_mass + p.wheel_inertia / r**2 + J / r**2, ml * np.cos(theta) - J / r],
        [ml * np.cos(theta) - J / r, p.body_mass * p.body_com_height**2 + p.body_inertia + J],
    ])
    f = np.array([tau / r + ml * np.sin(theta) * theta_dot**2 + F,
                  -tau + ml * p.gravity * np.sin(theta) + F * p.body_com_height * np.cos(theta)])
    np.testing.assert_allclose(dyn.derivatives(state, tau, F)[[1, 3]], np.linalg.solve(M, f), rtol=1e-12)
    np.testing.assert_array_equal(dyn.derivatives(state, tau, 0.0), dyn.derivatives(state, tau))


def test_push_does_the_work_of_a_force_at_the_center_of_mass():
    """No torque or friction: the energy gained is the integral of F times the COM's horizontal speed."""
    dyn = Planar2D(ROBOT, fall_angle=10.0)
    state, F, h, l = np.array([0.0, 0.2, 0.1, 0.3]), 3.0, 1e-4, ROBOT.body_com_height
    e0, work = dyn.energy(state), 0.0
    com_speed = lambda s: s[1] + l * np.cos(s[2]) * s[3]
    for _ in range(1000):
        k1 = dyn.derivatives(state, 0.0, F)
        k2 = dyn.derivatives(state + 0.5 * h * k1, 0.0, F)
        k3 = dyn.derivatives(state + 0.5 * h * k2, 0.0, F)
        k4 = dyn.derivatives(state + h * k3, 0.0, F)
        new = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        work += F * 0.5 * (com_speed(state) + com_speed(new)) * h
        state = new
    assert dyn.energy(state) - e0 == pytest.approx(work, rel=1e-6)


def test_short_push_is_an_impulse_on_the_linear_model():
    """A brief push changes the speeds by M^-1 (J, J l) (PHYSICS.md 3.1's mass matrix at upright)."""
    dyn = Planar2D(ROBOT, fall_angle=10.0)
    p, impulse, width = ROBOT, 0.01, 0.002
    r, l = p.wheel_radius, p.body_com_height
    a = p.body_mass + p.wheel_mass + (p.wheel_inertia + p.rotor_inertia) / r**2
    b0 = p.body_mass * l - p.rotor_inertia / r
    d = p.body_mass * l**2 + p.body_inertia + p.rotor_inertia
    expected = np.linalg.solve([[a, b0], [b0, d]], [impulse, impulse * l])

    class Hold:   # a controller that commands nothing
        def reset(self): pass
        def update(self, measured, dt): return 0.0
        def pitch_reference(self): return 0.0
        def goal_velocity(self, t): return 0.0

    res = simulate(dyn, Hold(), duration=width, dt=width / 200, initial_state=np.zeros(4),
                   disturbance=push_profile([(0.0, width, impulse / width)]))
    np.testing.assert_allclose(res.states[-1, [1, 3]], expected, rtol=0.02)


def test_zero_push_changes_nothing():
    """No pushes builds no profile; a zero-force push is evaluated every step and must change nothing."""
    plain = Evaluator(replace(SHORT, pushes=()), SPACE).run(HAND_TUNED, 0.1, 1)
    zeroed = Evaluator(replace(SHORT, pushes=((0.5, 0.5, 0.0),)), SPACE).run(HAND_TUNED, 0.1, 1)
    np.testing.assert_array_equal(zeroed.states, plain.states)


def test_push_moves_the_robot_the_right_way():
    ev = Evaluator(replace(SHORT, goal_steps=(), pushes=()), SPACE)
    still = ev.run(HAND_TUNED, 0.0, 1)
    for force, sign in ((6.5, 1), (-6.5, -1)):
        pushed = Evaluator(replace(ev.config, pushes=((0.5, 0.05, force),)), SPACE).run(HAND_TUNED, 0.0, 1)
        k = int(round(0.51 / ev.config.dt))   # 10 ms in: before the controller's response arrives
        assert sign * (pushed.states[k, 3] - still.states[k, 3]) > 0.1     # body pitches the way it is pushed
        k = int(round(0.8 / ev.config.dt))
        assert sign * (pushed.states[k, 1] - still.states[k, 1]) > 0.05    # and the robot moves that way
        assert not pushed.fell


# ---- the batched engines reproduce the reference ------------------------------------------

def _candidates(n=16, seed=0):
    units = np.random.default_rng(seed).random((n, SPACE.dim))
    gains = decode_units(SPACE, units)
    gains[0] = HAND_TUNED
    gains[1] = [0.1, 0.0, 0.05, 0.0]   # falls
    return gains


def test_batch_trajectories_with_pushes_match_reference_exactly():
    ref, batched = Evaluator(SHORT, SPACE), BatchEvaluator(SHORT, SPACE)
    assert np.count_nonzero(batched.schedule.force_per_step) > 0
    gains = _candidates()
    pitch, seed = ref.scenarios()[1]
    run = simulate_batch(batched.spec, batched.robot, gains, pitch, noise=batched.noise_for((seed,)),
                         seed_index=np.zeros(len(gains), dtype=int), schedule=batched.schedule, record=True)
    for i, g in enumerate(gains):
        expected = ref.run(g, pitch, seed)
        n = len(expected.t)
        np.testing.assert_array_equal(run.states[:n, i], expected.states)
        assert run.fell[i] == expected.fell


def test_batch_costs_with_pushes_match_reference():
    ref, batched = Evaluator(SHORT, SPACE), BatchEvaluator(SHORT, SPACE)
    gains = _candidates(seed=1)
    result = batched.evaluate_population(gains)
    for i, g in enumerate(gains):
        expected = ref.evaluate_gains(g)
        np.testing.assert_allclose(result.costs[i], expected.costs, rtol=1e-12)
        assert result.n_fell[i] == expected.n_fell


def test_schedule_force_matches_profile():
    ev = BatchEvaluator(SHORT, SPACE)
    f = push_profile(SHORT.pushes)
    np.testing.assert_array_equal(ev.schedule.force_per_step, [f(k * SHORT.dt) for k in range(ev.schedule.n_steps)])
    assert not BatchEvaluator(replace(SHORT, pushes=()), SPACE).schedule.force_per_step.any()


torch = pytest.importorskip("torch")
DEVICES = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]


@pytest.mark.parametrize("device", DEVICES)
def test_torch_with_pushes_matches_numpy(device):
    from batch_sim.torch_sim import TorchSimulator

    ev = BatchEvaluator(SHORT, SPACE)
    gains = HAND_TUNED * np.random.default_rng(4).uniform(0.8, 1.2, (24, 4))
    scen = ev.scenarios()
    G = np.repeat(gains, len(scen), axis=0)
    pitch = np.tile([p for p, _ in scen], len(gains))
    seed_index = np.tile([SHORT.train_seeds.index(s) for _, s in scen], len(gains))
    noise = ev.noise_for(SHORT.train_seeds)
    ref = simulate_batch(ev.spec, ev.robot, G, pitch, noise, seed_index, ev.schedule, record=True)
    got = TorchSimulator(ev.spec, ev.robot, len(G), noise, ev.schedule, device=device,
                         dtype=torch.float64, record=True).run(G, pitch, seed_index)
    up = ~ref.fell
    assert up.sum() >= 0.75 * len(G)
    assert not got.fell[up].any()
    np.testing.assert_allclose(got.cost[up], ref.cost[up], rtol=1e-9)
    np.testing.assert_allclose(got.states[:, up], ref.states[:, up], rtol=0, atol=1e-9)
