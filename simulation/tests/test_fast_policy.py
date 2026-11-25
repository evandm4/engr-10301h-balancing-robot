"""FastPolicyController (POLICY_ON_ROBOT_PLAN P3c) against the reference PolicyController: the same float64
bits, not just close. Every spec kind (linear, mlp, warm, residual) of the active robot's config, random
weights small and large (so saturation and the integrator freeze happen), random states, both goal kinds.

    python -m pytest simulation/tests/test_fast_policy.py -q
"""
import math
import zlib

import numpy as np
import pytest

from balance_sim import FastPolicyController, PolicyController, compile_forward, step_profile
from config import CONFIG

KINDS = ("linear", "mlp", "warm", "residual")
SPECS = {k: CONFIG.build_policy_spec(k) for k in KINDS}
MAX_TORQUE = CONFIG.robot.max_torque
DT = CONFIG.timing.control_dt_s


def _same(a, b):
    a, b = float(a), float(b)
    return a == b or (math.isnan(a) and math.isnan(b))


def _pair(spec, params, goal):
    kw = dict(max_pitch_command=0.2, pitch_integral_limit=0.5, velocity_integral_limit=0.8)
    return (PolicyController(spec, params, MAX_TORQUE, goal=goal, **kw),
            FastPolicyController(spec, params, MAX_TORQUE, goal=goal, **kw))


def _assert_same_tick(ref, fast, u_ref, u_fast, k):
    assert _same(u_ref, u_fast), f"tick {k}: torque {u_ref!r} vs {u_fast!r}"
    assert _same(ref.last_pre_clip, fast.last_pre_clip), k
    assert _same(ref.last_residual, fast.last_residual), k
    assert ref.last_saturated == fast.last_saturated and ref.last_integrators_frozen == fast.last_integrators_frozen
    assert len(ref.last_raw) == len(fast.last_raw)
    assert all(_same(a, b) for a, b in zip(ref.last_raw, fast.last_raw)), k
    assert _same(ref._int_v, fast._int_v) and _same(ref._int_theta, fast._int_theta)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("spread", [0.05, 0.5, 3.0])
@pytest.mark.parametrize("goal_kind", ["float", "profile"])
def test_fast_controller_is_bit_identical_to_the_reference(kind, spread, goal_kind):
    spec = SPECS[kind]
    rng = np.random.default_rng(zlib.crc32(f"{kind} {spread} {goal_kind}".encode()))
    goal = 0.17 if goal_kind == "float" else step_profile(((0.5, 0.3), (2.0, -0.2)), 0.4)
    for trial in range(3):
        params = rng.normal(0.0, spread, spec.n_params)
        ref, fast = _pair(spec, params, goal)
        n_sat = 0
        for k in range(600):
            state = rng.normal(0.0, [0.1, 0.6, 0.25, 2.0]) * (10.0 if k % 97 == 0 else 1.0)
            u_ref, u_fast = ref.update(state, DT), fast.update(state, DT)
            _assert_same_tick(ref, fast, u_ref, u_fast, k)
            n_sat += fast.last_saturated
        ref.reset()
        fast.reset()
        assert fast.last_raw == [0.0] * len(ref.last_raw) and fast._time == 0.0
    if spread == 3.0:
        assert n_sat > 0                     # the freeze path was exercised


@pytest.mark.parametrize("kind", KINDS)
def test_compiled_forward_matches_on_wide_random_observations(kind):
    spec = SPECS[kind]
    rng = np.random.default_rng(7)
    params = rng.normal(0.0, 1.0, spec.n_params)
    ref, fast = _pair(spec, params, 0.0)
    n_raw = len(ref.last_raw)
    for _ in range(3000):
        raw = (rng.normal(0.0, 1.0, n_raw) * 10.0 ** rng.integers(-6, 7, n_raw)).tolist()
        assert _same(PolicyController.forward(ref, np.array(raw)), fast.forward(raw))


def test_special_weights_compile_exactly():
    """-0.0, inf, nan, subnormal and the largest double survive as literals, and zero terms are kept
    (0 * inf = nan, as in the reference)."""
    spec = SPECS["warm"]
    params = np.random.default_rng(3).normal(0.0, 0.3, spec.n_params)
    specials = [-0.0, 5e-324, -1.7976931348623157e308, 1e-310, 0.0]
    params[:len(specials)] = specials
    ref, fast = _pair(spec, params, 0.0)
    for raw in ([1.0] * len(ref.last_raw), [math.inf] + [0.5] * (len(ref.last_raw) - 1),
                [-0.0] * len(ref.last_raw)):
        z_ref, z_fast = PolicyController.forward(ref, np.array(raw)), fast.forward(raw)
        assert _same(z_ref, z_fast) and math.copysign(1.0, z_ref) == math.copysign(1.0, z_fast)
    for bad in (math.inf, -math.inf, math.nan):
        p = params.copy()
        p[-1] = bad                                   # the output bias
        ref, fast = _pair(spec, p, 0.0)
        raw = [0.3] * len(ref.last_raw)
        assert _same(PolicyController.forward(ref, np.array(raw)), fast.forward(raw))
        assert _same(ref.update(np.array([0.0, 0.1, 0.05, 0.0]), DT), fast.update(np.array([0.0, 0.1, 0.05, 0.0]), DT))


def test_generated_code_keeps_the_reference_order():
    """No builtin sum (compensated since Python 3.12), no numpy dot; one left-to-right chain per unit."""
    spec = SPECS["residual"]
    c = FastPolicyController(spec, np.ones(spec.n_params), MAX_TORQUE)
    _, src = compile_forward(spec, c.linear, c.layers)
    assert src == c.forward_source
    assert "sum(" not in src and "dot" not in src and "@" not in src
    assert src.count("_tanh(") == len(spec.hidden)
    assert src.count("\n    h") == sum(spec.hidden) + 1


def test_weights_changed_in_place_are_recompiled_at_reset():
    spec = SPECS["linear"]
    params = np.random.default_rng(1).normal(0.0, 0.5, spec.n_params)
    _, fast = _pair(spec, params, 0.0)
    state = np.array([0.1, 0.2, 0.05, -0.3])
    u0 = fast.update(state, DT)
    fast.linear[0] += 1e-2
    fast.reset()
    ref, _ = _pair(spec, fast.linear.copy(), 0.0)
    u1, u_ref = fast.update(state, DT), ref.update(state, DT)
    assert u1 == u_ref and u1 != u0
    fast.linear[0] = math.nan
    fast.reset()
    assert math.isnan(fast.forward([0.1] * len(fast.last_raw)))


@pytest.mark.parametrize("kind", KINDS)
def test_float32_dt_state_and_raw_stay_bit_identical(kind):
    """Review F3: under NumPy 2's scalar promotion (NEP 50) a numpy float32 dt made the integrators float32, and
    the fast forward then multiplied float32 raw entries by its float64 literals in float32 (the reference
    float()s every entry first). Both controllers now take dt as a Python float and the fast public forward()
    float()s its raw entries, so float32 inputs give the same float64 bits as float64 ones."""
    spec = SPECS[kind]
    rng = np.random.default_rng(zlib.crc32(f"float32 {kind}".encode()))
    params = rng.normal(0.0, 0.5, spec.n_params)
    ref, fast = _pair(spec, params, 0.17)
    ref64, _ = _pair(spec, params, 0.17)
    dt32 = np.float32(DT)
    for k in range(400):
        state = rng.normal(0.0, [0.1, 0.6, 0.25, 2.0]).astype(np.float32)
        u_ref, u_fast = ref.update(state, dt32), fast.update(state, dt32)
        _assert_same_tick(ref, fast, u_ref, u_fast, k)
        assert u_fast == u_ref and fast.last_raw == ref.last_raw.tolist()
        assert type(u_fast) is float and type(fast._int_v) is float and type(fast._int_theta) is float
        assert type(ref._int_v) is float and type(ref._time) is float and type(fast._time) is float
        # the same as a float64 dt of the float32 value (dt is coerced, not computed with in float32)
        assert ref64.update(state.astype(float), float(dt32)) == u_ref
    n_raw = len(ref.last_raw)
    for _ in range(300):
        raw = rng.normal(0.0, 1.0, n_raw).astype(np.float32)
        z_fast = fast.forward(raw)
        assert type(z_fast) is float and z_fast == PolicyController.forward(ref, raw)
        assert z_fast == fast.forward(raw.astype(float).tolist())
