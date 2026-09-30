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
    ec = ProjectConfig(cost=CostConfig(pitch_reference="upright")).build_eval_config()
    assert ec.pitch_reference == "upright"
    with pytest.raises(ValueError):
        CostConfig(pitch_reference="sideways")


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
