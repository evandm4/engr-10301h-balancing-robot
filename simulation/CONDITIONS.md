# Simulation Conditions

Running list of the real-world effects and test scenarios the simulation includes, and candidates not yet added. Updated each time one is added.

Last updated: 2026-11-08

## Included

| # | Condition | How to enable | Default | Notes |
|---|---|---|---|---|
| 1 | Initial pitch offset | `initial_state(pitch=...)` | 0 rad | The starting disturbance. |
| 2 | Torque saturation | `RobotParams.max_torque` (both actuator models) | 0.34 N m total | Commanded torque is clipped. With the DC motor, back-EMF and the current limit can lower it further. The controller also knows this limit (`output_limit`) so its integrators don't wind up against it. |
| 3 | Wheel/motor friction, viscous and Coulomb | `RobotParams.wheel_friction`, `coulomb_friction` | 0 (randomized in batched tuning) | Acts on wheel speed relative to the body. Ranges from the motors' free-run current (DATASHEET.md 5.2). |
| 4 | Finite control rate (sample and hold) | `simulate(control_dt=...)` | same as physics step | E.g. `0.005` for a 200 Hz loop; torque is held between updates. |
| 5 | Fall detection | `Planar2D(fall_angle=...)` | 60 degrees | Run ends and the cost function penalizes it. |
| 6 | Sensor noise + gyro bias | `simulate(sensor=SensorNoise(...))` | off | Gaussian noise on pitch, pitch rate, position, velocity; constant gyro bias and constant pitch-estimate offset (`pitch_bias`). Seeded and reproducible. The project's levels are in `config.py` (`SensorConfig`; DATASHEET.md 2.3), built with `CONFIG.build_sensor(seed)`. |
| 7 | Control latency | `simulate(latency=seconds)` | 0 | Delay from the controller computing a torque to it reaching the actuator, on top of the hold in #4. |
| 8 | DC motor model | `Planar2D(params, actuator=DCMotorActuator(MotorParams(...)))` | on in `config.py` | Back-EMF, supply voltage, current limit, gear ratio, gearbox efficiency (`gear_efficiency`: less torque at the current limit, back-EMF unchanged). Available torque drops as the wheels spin faster. Constants from the Pololu 25D HP 6V 20.4:1 gearmotor (DATASHEET.md). |
| 9 | Goal velocity profile | `CascadedPIDController(..., goal=step_profile([(t, v), ...], transition=s))` | 0 m/s | Test scenario rather than a physical effect. Constant or time-varying forward velocity the robot must track. `transition` > 0 blends each change smoothly (continuous acceleration and jerk) so the goal can be tracked exactly; 0 gives instant jumps. The project uses `ScenarioConfig.goal_transition_s`. The cost function scores tracking error (`velocity_weight`). |
| 10 | Parameter uncertainty (randomized robots) | `config.py` `RobustnessSettings.randomize`; used by `run_batch_tuning.py` | on in batched tuning; off in `run_tuning.py` | Every scenario gets its own robot: mass, COM height, inertias, torque constant, winding resistance, battery voltage and gearbox efficiency scaled within ±10–25%, rotor inertia within ±50%; friction and latency drawn from ranges. A fixed, varying battery voltage, but not sag during a run. Batched code only (`batch_sim` accepts per-robot parameter arrays). |
| 11 | Reflected motor rotor inertia | `RobotParams.rotor_inertia` (`config.py`: `ActuatorConfig.rotor_inertia` per motor, reflected as n N² J_r) | on, 4.2e-4 kg m² | The rotors spin relative to the body, so they enter the mass matrix in three places (PHYSICS.md 10.3), not as extra wheel inertia. Estimated, not published (DATASHEET.md 5.3). |

### What we've seen so far

With PID gains (4.0, 0.5, 0.15), DC motors, and typical sensor noise:

- Balances with up to about 5 ms latency with almost no change.
- Around 10 to 20 ms, it takes much longer to settle or oscillates.
- Around 30 ms and up, it falls.

That gap is the tuning problem the ML loop needs to solve: gains that work with no latency don't survive realistic delay.

The DC motor model changes nothing at small angles, because the wheels never spin fast enough for back-EMF to matter. It should matter once disturbances or larger speed commands push the wheels harder.

Gyro bias (#6) makes a pitch-only PID drift steadily. In a 10 s test with a 0.05 rad/s bias, the pitch-only controller ended 0.39 m from where it started and still moving. The velocity loop (#9, goal 0) held that to 0.03 m.

The velocity loop is touchy: with the inner loop unchanged, outer gains much above Kv_p of about 0.25 oscillate, and under 10 ms latency plus noise the usable range shrinks further. Another job for the tuner.

## Assumptions (always on, not switchable)

- Wheels roll without slipping.
- Rigid body, flat level ground.
- Planar motion only (pitch and forward travel).
- Motor torque is applied as one combined value across both wheels.

## Candidates not yet added

- External pushes on the body (impulse or random disturbance)
- Winding inductance (motor current lag)
- Gearbox backlash
- Sensor latency separate from actuator latency
- Sensor quantization and sample rate (IMU, encoder resolution)
- Battery voltage sag during a run (#10 varies the voltage between runs only)
- Wheel slip and rolling resistance
- Slopes and uneven ground
- Position hold (a third loop that turns position error into a goal velocity)
- Steering (yaw) and a 3D model

## Change log

- 2026-09-18: Initial list. Items 1 to 5 already existed in the core. Added 6 (sensor noise), 7 (latency), 8 (DC motor model).
- 2026-09-19: Added 9 (goal velocity profile) with the cascaded velocity controller and a velocity term in the cost function.
- 2026-09-28 (corrected version): #2 now applies to the DC motor too (it used to ignore `max_torque`). Both controller loops got anti-windup. The cost's pitch term now defaults to measuring error from the commanded lean instead of from upright (`CostConfig.pitch_reference`).
- 2026-09-28: Added 10 (parameter uncertainty), as part of the batched tuner's robust scoring.
- 2026-11-08: Added 11 (reflected rotor inertia), gearbox efficiency (#8) and a pitch-estimate offset (#6). Every physical constant now comes from DATASHEET.md's reference build. Runs before this change used the earlier robot and are not comparable.
- 2026-10-24: The cost now scores pitch from upright again (`pitch_reference` default), with weights derived from acceptable-value limits by Bryson's rule (velocity 2.25, effort 9; fall penalty 500). The goal profile is smooth (`goal_transition_s`). Runs before this change used the old weights and are not comparable.
