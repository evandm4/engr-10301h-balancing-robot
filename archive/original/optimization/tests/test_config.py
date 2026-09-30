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
    space = cfg.build_search_space()
    by_name = {p.name: p for p in space.parameters}
    assert by_name["kp"].high == pytest.approx(7.0)
    assert by_name["kv_p"].high == pytest.approx(0.05)


def test_hand_tuned_gains_are_a_valid_five_gain_set():
    gains = CONFIG.controller.hand_tuned_gains
    assert len(gains) == 5
    assert all(isinstance(g, (int, float)) for g in gains)


def test_config_objects_are_immutable():
    with pytest.raises(Exception):
        CONFIG.timing.latency_s = 0.5


def test_snapshot_is_json_serializable_and_complete():
    import json

    snap = CONFIG.snapshot()
    json.dumps(snap)   # must not raise

    # Every section of the config is present, with its values.
    assert set(snap) >= {"robot", "actuator", "sensor", "timing", "controller",
                         "cost", "scenario", "search", "derived"}
    assert snap["robot"]["body_mass"] == CONFIG.robot.body_mass
    assert snap["cost"]["velocity_weight"] == CONFIG.cost.velocity_weight
    assert snap["actuator"]["model"] == CONFIG.actuator.model
    assert tuple(snap["scenario"]["validation_seeds"]) == CONFIG.scenario.validation_seeds

    # Derived values are recorded too.
    derived = snap["derived"]
    assert derived["max_lean_rad"] == pytest.approx(CONFIG.controller.max_lean_rad)
    assert derived["critical_pitch_gain"] > 0
    assert set(derived["search_bounds"]) == {"kp", "ki", "kd", "kv_p", "kv_i"}
    assert "motor_stall_torque_limit" in derived


def test_snapshot_reflects_overrides():
    snap = ProjectConfig(actuator=ActuatorConfig(model="ideal")).snapshot()
    assert snap["actuator"]["model"] == "ideal"
    assert "motor_stall_torque_limit" not in snap["derived"]   # only meaningful for the DC motor model
