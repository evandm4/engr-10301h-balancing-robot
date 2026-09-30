# Archive: the original code, and what changed from it

`archive/original/` holds the project as it was before 2026-09-28: the first `config.py`, `simulation/`, `optimization/`, `installedsoftware.py`, and the tuning results they produced (`results/tuning/`, 2026-09-21 to 2026-09-28). None of it is used any more, and nothing imports it. It's kept so those early results can still be traced to the code that made them. Delete it once you no longer need that.

On 2026-09-28 a corrected copy was made (then called `CPU/`). On the same day it became the project's main code at the top level, with the batched engine alongside it. The corrections are listed below; the file names refer to the current `simulation/` and `optimization/`.

## What changed from the original, and why

1. **The DC motor now respects `RobotConfig.max_torque`.** `DCMotorActuator` used to ignore it, which went unnoticed only because the motor's stall limit also happens to be 0.40 N m. The command is now capped at `max_torque` before the motor physics, so the effective limit is whichever is lower. (`actuators.py`, `config.py`, `evaluation.py`)

2. **Anti-windup on both controller loops.** The pitch integral was capped at 1.0, so with Ki up to 5 the integral term alone could ask for 5 N m from a 0.4 N m actuator. That banked-up torque kept pushing the wrong way after the error reversed. Both loops now use conditional integration: the integrator stops growing once the output hits its limit, still unwinds freely when the error reverses, and the integral term alone is capped at the limit. The inner loop's limit is the torque limit (`PIDController(output_limit=...)`, or `CascadedPIDController.from_gains(..., torque_limit=...)`); the outer loop's is the lean cap. The pitch command is also clipped to the torque limit, so the cost's effort term now counts deliverable torque rather than impossible commands. (`controllers.py`)

3. **Pitch cost measured against the commanded lean.** The original cost penalized pitch away from vertical, but following a velocity goal *requires* leaning, so the cost fought the task. That's a likely reason the tuned Kp values came out far below the hand-tuned 4.0. The simulator now records the lean the controller asked for (`SimResult.pitch_reference`), and the pitch term is measured against it by default. Set `CostConfig.pitch_reference = "upright"` to reproduce the original cost. `compute_metrics` reports both `rms_pitch` and `rms_pitch_tracking_error`. (`simulator.py`, `metrics.py`, `config.py`)

4. **Every start tilt is tried with every noise seed.** Before, seed *i* was only paired with tilt *i*, so training used just 3 runs per score. Now it uses 3 tilts × 3 seeds = 9, and validation uses 3 × 10 = 30. Scoring is about 3× slower per evaluation, which the GPU version is meant to absorb. (`evaluation.py`)

5. **Fair method comparison.** `run_tuning.py` now:
   - prints each method's evaluation budget
   - warns when *any* method's result sits on a search bound, not just the winner's
   - adds an equal-budget table giving each method's best cost within the smallest budget any method used (Bayesian optimization's, by default)
   - saves all of this in `comparison.json`

   Differential evolution's history now records scipy's real evaluation count (`nfev`) instead of estimating it. DE only reports once per generation, so it has no entry before its first generation finishes. (`optimizers.py`, `run_tuning.py`)

6. **One requirements file with correct minimums.** It includes `scikit-optimize` and `scipy>=1.12`, the first version whose `differential_evolution` callback accepts `intermediate_result`. The per-folder requirements files were removed. (`requirements.txt`)

7. **Repeatable benchmark.** Past tuning times for identical work varied from 68 s to 185 s. `optimization/benchmark.py` repeats each measurement, reports the median and spread with machine and library versions, and saves them, so the GPU version has a fair baseline to beat.

Baseline on 2026-09-28: one 6 s run (2,400 steps) takes a median of 30.5 ms, and one 9-run evaluation 272 ms. That's about 79k physics steps/s on one core.

## Results aren't comparable with the original's

Fixes 2–4 change what the cost measures and how many runs make up a score, so costs in `archive/original/results/tuning/` can't be compared directly with anything in the current `results/`. Each run's saved config snapshot records which setup produced it.
