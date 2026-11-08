"""Checks that the project-wide config.py builds valid, consistent objects.

These aren't tests of the physics or the tuner (those live in their own test
files); they just guard against config.py drifting out of sync with the
packages it wires together.
"""

import numpy as np
import pytest

from balance_sim import DCMotorActuator, IdealActuator, Planar2D
from config import CONFIG, ActuatorConfig, ProjectConfig


def test_default_config_builds_a_working_dynamics_model():
    dyn = CONFIG.build_dynamics()
    assert isinstance(dyn, Planar2D)
    assert isinstance(dyn.actuator, DCMotorActuator)   # default actuator.model is "dc_motor"
    assert dyn.fall_angle == pytest.approx(CONFIG.timing.fall_angle_rad)


def test_actuator_model_switch():
    ideal_dyn = ProjectConfig(actuator=ActuatorConfig(model="ideal")).build_dynamics()
    assert isinstance(ideal_dyn.actuator, IdealActuator)

    with pytest.raises(ValueError):
        ProjectConfig(actuator=ActuatorConfig(model="warp_drive")).build_dynamics()


def test_eval_config_matches_project_config():
    ec = CONFIG.build_eval_config()
    assert ec.latency == CONFIG.timing.latency_s
    assert ec.duration == CONFIG.scenario.duration_s
    assert ec.max_pitch_command == pytest.approx(CONFIG.controller.max_lean_rad)
    assert ec.fall_angle == pytest.approx(CONFIG.timing.fall_angle_rad)
    assert ec.motor is not None   # dc_motor is the default actuator model
    assert np.allclose(ec.start_pitches, CONFIG.scenario.start_pitches_rad)
    assert ec.pushes == CONFIG.scenario.pushes


def test_validation_scenario_is_the_training_one_run_longer():
    from dataclasses import replace

    train, val = CONFIG.build_eval_config(), CONFIG.build_eval_config(validation=True)
    assert val.duration == CONFIG.scenario.validation_duration_s > train.duration
    assert replace(val, duration=train.duration) == replace(train, sensor=val.sensor)
    # Every goal change and push happens within the training run, so the extra time is a hold.
    s = CONFIG.scenario
    assert all(step[0] + (step[2] if len(step) == 3 else s.goal_transition_s) <= s.duration_s for step in s.goal_steps)
    assert all(start + width <= s.duration_s for start, width, _ in s.pushes)


def test_cost_limits_are_consistent_with_the_scenario():
    """CostConfig's comment: the pitch limit is above the lean the goal needs (PHYSICS.md 2.8)
    and below the lean cap, and no start tilt exceeds it."""
    import math

    p, c = CONFIG.build_robot_params(), CONFIG.cost
    goal, dt = CONFIG.build_goal(), 1e-3
    v = np.array([goal(k * dt) for k in range(int(CONFIG.scenario.duration_s / dt))])
    peak_accel = np.max(np.abs(np.diff(v))) / dt
    ml = p.body_mass * p.body_com_height
    a = p.body_mass + p.wheel_mass + (p.wheel_inertia + p.rotor_inertia) / p.wheel_radius**2
    lean = lambda th: ml * p.gravity * math.sin(th) / (ml * math.cos(th) - p.rotor_inertia / p.wheel_radius
                                                        + a * p.wheel_radius)
    lean_needed = next(th for th in np.arange(0, 0.6, 1e-4) if lean(th) >= peak_accel)
    assert lean_needed < c.pitch_limit_rad < CONFIG.controller.max_lean_rad
    assert max(abs(t) for t in CONFIG.scenario.start_pitches_rad) <= c.pitch_limit_rad + 1e-6


def test_eval_config_has_no_motor_when_actuator_is_ideal():
    ec = ProjectConfig(actuator=ActuatorConfig(model="ideal")).build_eval_config()
    assert ec.motor is None


def test_search_space_uses_configured_bounds():
    from config import SearchConfig

    cfg = ProjectConfig(search=SearchConfig(kp_max=7.0, kv_p_max=0.05))
    assert cfg.build_search_space().names == ["kp", "ki", "kd", "kv_p"]
    space = cfg.build_search_space()
    by_name = {p.name: p for p in space.parameters}
    assert by_name["kp"].high == pytest.approx(7.0)
    assert by_name["kv_p"].high == pytest.approx(0.05)


def test_hand_tuned_gains_match_the_search_space():
    gains = CONFIG.controller.hand_tuned_gains
    assert len(gains) == CONFIG.build_search_space().dim == 4   # kp, ki, kd, kv_p
    assert all(isinstance(g, (int, float)) for g in gains)


def test_integral_limits_reach_every_consumer():
    from batch_tuning import spec_from_eval_config

    ec = CONFIG.build_eval_config()
    assert ec.pitch_integral_limit == CONFIG.controller.pitch_integral_limit
    assert ec.velocity_integral_limit == CONFIG.controller.velocity_integral_limit
    spec = spec_from_eval_config(ec)
    assert spec.pitch_integral_limit == CONFIG.controller.pitch_integral_limit
    assert spec.velocity_integral_limit == CONFIG.controller.velocity_integral_limit
    assert CONFIG.build_pitch_controller().integral_limit == CONFIG.controller.pitch_integral_limit


def test_config_objects_are_immutable():
    with pytest.raises(Exception):
        CONFIG.timing.latency_s = 0.5


def test_snapshot_is_json_serializable_and_complete():
    import json

    snap = CONFIG.snapshot()
    json.dumps(snap)   # must not raise

    # Every section of the config is present, with its values.
    assert set(snap) >= {"robot", "actuator", "sensor", "timing", "controller",
                         "cost", "scenario", "search", "robustness", "derived"}
    assert snap["robot"]["body_mass"] == CONFIG.robot.body_mass
    assert snap["cost"]["velocity_weight"] == CONFIG.cost.velocity_weight
    assert snap["actuator"]["model"] == CONFIG.actuator.model
    assert tuple(snap["scenario"]["validation_seeds"]) == CONFIG.scenario.validation_seeds

    # Derived values are recorded too.
    derived = snap["derived"]
    assert derived["max_lean_rad"] == pytest.approx(CONFIG.controller.max_lean_rad)
    assert derived["critical_pitch_gain"] > 0
    assert set(derived["search_bounds"]) == {"kp", "ki", "kd", "kv_p"}
    assert "motor_stall_torque_limit" in derived


def test_dc_motor_respects_the_robot_torque_limit():
    from config import RobotConfig

    # Below the motor's own stall limit, the robot's max_torque is what binds.
    low = ProjectConfig(robot=RobotConfig(max_torque=0.2)).build_dynamics()
    assert low.actuator.torque(10.0, 0.0) == pytest.approx(0.2)
    # Above it, the motor physics bind instead.
    high = ProjectConfig(robot=RobotConfig(max_torque=5.0)).build_dynamics()
    assert high.actuator.torque(10.0, 0.0) == pytest.approx(CONFIG.actuator.build_motor_params().stall_torque_limit)


def test_eval_config_carries_pitch_reference():
    from config import CostConfig

    assert CONFIG.build_eval_config().pitch_reference == CONFIG.cost.pitch_reference
    ec = ProjectConfig(cost=CostConfig(pitch_reference="commanded")).build_eval_config()   # CONFIG's is "upright"
    assert ec.pitch_reference == "commanded"
    with pytest.raises(ValueError):
        CostConfig(pitch_reference="sideways")


def test_cost_weights_follow_bryson_limits():
    from config import CostConfig

    c = CostConfig(pitch_limit_rad=0.3, velocity_limit=0.2, torque_limit=0.1)
    assert c.velocity_weight == pytest.approx(2.25) and c.effort_weight == pytest.approx(9.0)
    # Loosening a limit lowers that term's weight; an infinite limit switches it off.
    assert CostConfig(velocity_limit=0.4).velocity_weight < c.velocity_weight
    assert CostConfig(torque_limit=float("inf")).effort_weight == 0.0
    for bad in ({"pitch_limit_rad": 0.0}, {"velocity_limit": -1.0}, {"torque_limit": 0.0}):
        with pytest.raises(ValueError):
            CostConfig(**bad)


def test_snapshot_records_limits_and_derived_weights():
    snap = CONFIG.snapshot()["cost"]
    assert snap["pitch_limit_rad"] == CONFIG.cost.pitch_limit_rad
    assert snap["velocity_weight"] == CONFIG.cost.velocity_weight
    assert snap["effort_weight"] == CONFIG.cost.effort_weight


def test_build_controller_matches_the_evaluator():
    from tuning import Evaluator

    ours = CONFIG.build_controller()
    theirs = Evaluator(CONFIG.build_eval_config(), CONFIG.build_search_space()).controller(
        CONFIG.controller.hand_tuned_gains)
    assert ours.pitch_pid.output_limit == theirs.pitch_pid.output_limit == CONFIG.robot.max_torque
    assert ours.pitch_pid.integral_limit == theirs.pitch_pid.integral_limit
    assert ours.integral_limit == theirs.integral_limit
    assert ours.max_pitch_command == pytest.approx(theirs.max_pitch_command)
    assert ours.goal_velocity(1.0) == theirs.goal_velocity(1.0)


def test_snapshot_reflects_overrides():
    snap = ProjectConfig(actuator=ActuatorConfig(model="ideal")).snapshot()
    assert snap["actuator"]["model"] == "ideal"
    assert "motor_stall_torque_limit" not in snap["derived"]   # only meaningful for the DC motor model


# ---- guards added 2026-11-06 ------------------------------------------------------------

def test_train_and_validation_seeds_must_not_overlap():
    from dataclasses import replace

    from batch_tuning import RobustnessConfig
    from config import ScenarioConfig

    with pytest.raises(ValueError, match="overlap"):
        ScenarioConfig(train_seeds=(1, 2, 100), validation_seeds=(100, 101))
    with pytest.raises(ValueError, match="overlap"):
        replace(CONFIG.build_robustness(), train_seeds=tuple(range(1, 1001)))   # --seeds 1000 reaches seed 1000


def test_control_period_must_be_a_whole_number_of_physics_steps():
    from dataclasses import replace

    from balance_sim import simulate, steps_per_control
    from batch_tuning import spec_from_eval_config

    assert steps_per_control(0.005, 0.0025) == 2
    with pytest.raises(ValueError, match="whole multiple"):
        steps_per_control(0.005, 0.003)
    with pytest.raises(ValueError, match="whole multiple"):
        spec_from_eval_config(replace(CONFIG.build_eval_config(), control_dt=0.006))
    with pytest.raises(ValueError, match="whole multiple"):
        simulate(CONFIG.build_dynamics(), CONFIG.build_controller(), duration=0.1, dt=0.0025, control_dt=0.004)


def test_results_record_their_provenance():
    from tuning import provenance

    info = provenance()
    assert set(info) >= {"git_commit", "git_dirty", "python", "numpy"}


def test_fall_penalty_exceeds_any_non_fall_cost():
    """PHYSICS.md 8.3: a run that stays up can't cost more than the fall penalty.

    Bound each term over a whole run: pitch by the fall angle, torque by the torque
    limit, and speed by the DC motor's no-load speed on the fastest randomized robot
    (supply voltage high, torque constant low) plus the largest goal speed.
    """
    c, T = CONFIG.cost, CONFIG.scenario.duration_s
    spreads = dict(CONFIG.robustness.randomize)
    a = CONFIG.actuator
    volts = a.supply_voltage * (1 + spreads.get("supply_voltage", 0.0))
    k_t = a.torque_constant * (1 - spreads.get("torque_constant", 0.0))
    v_max = volts / (k_t * a.gear_ratio) * CONFIG.robot.wheel_radius
    goal_max = max(abs(step[1]) for step in CONFIG.scenario.goal_steps)
    bound = (CONFIG.timing.fall_angle_rad**2 * T
             + c.effort_weight * CONFIG.robot.max_torque**2 * T
             + c.velocity_weight * (v_max + goal_max) ** 2 * T)
    assert CONFIG.actuator.model == "dc_motor", "with the ideal actuator nothing bounds the speed"
    assert bound < c.fall_penalty, f"a non-fall run could cost up to {bound:.0f}, over the fall penalty"


def test_kp_floor_covers_every_randomized_robot():
    """No scenario's robot needs more Kp than the robot the search floor is set from."""
    from balance_sim import critical_pitch_gain
    from batch_tuning import BatchEvaluator

    floor_robot = CONFIG.kp_floor_robot()
    ev = BatchEvaluator(CONFIG.build_eval_config(), CONFIG.build_search_space(), robustness=CONFIG.build_robustness())
    robots = ev.robots(CONFIG.robustness.validation_seeds)
    needed = robots.critical_pitch_gain()
    assert np.max(needed) <= critical_pitch_gain(floor_robot) + 1e-12
    assert critical_pitch_gain(floor_robot) > critical_pitch_gain(CONFIG.build_robot_params())
    kp_low = CONFIG.build_search_space().parameters[0].low
    assert kp_low == pytest.approx(CONFIG.search.kp_min_factor * critical_pitch_gain(floor_robot))
