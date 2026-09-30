# The batched engine (NumPy and GPU)

Details of `batch_sim` and `batch_tuning`: how they work, how they were verified, and how fast they are. For the project layout and the order everything runs in, see `README.md`.

`batch_sim` is the balance simulation rewritten to step many robots at once, with two backends:

- **NumPy** on the CPU: bit-identical to the reference `balance_sim` in float64.
- **PyTorch** on the RTX 5070 Ti Laptop GPU (12 GB, compute capability 12.0), using CUDA graphs. It runs the same physics and controller code through a small shim.

`balance_sim` stays the readable, tested reference. The NumPy backend is tested against it, and the PyTorch backend is tested against the NumPy one.

```
simulation/batch_sim/     the simulator, as functions over (B,) arrays
  robot.py                  BatchRobot: robot + motor params, scalar or one per simulation
  physics.py                actuator, derivatives, RK4    (balance_sim: planar2d.py, actuators.py)
  controller.py             anti-windup integrator, cascaded PID tick   (balance_sim: controllers.py)
  xp.py                     the few array ops physics/controller use, for NumPy or PyTorch
  schedule.py               SimSpec, precomputed latency/goal schedule, noise tables
  simulate.py               simulate_batch: the NumPy loop  (balance_sim: simulator.py + metrics.cost)
  torch_sim.py              TorchSimulator: the same loop on the GPU, with CUDA graphs
optimization/batch_tuning/  BatchEvaluator (backend="numpy" or "torch") + random search,
                            vectorized DE, CMA-ES
  robust.py                 RobustnessConfig: many seeds, randomized robots, chaos penalty
optimization/benchmark_batch.py   throughput vs batch size, compared with benchmark.py's baseline
optimization/run_batch_tuning.py  batched searches with large budgets (robust by default)
simulation/tests/test_{components,equivalence,torch}.py    batch_sim vs balance_sim, PyTorch vs NumPy
optimization/tests/test_{batch_tuning,robust}.py           batch_tuning vs tuning, robust scoring
```

All settings, including the robustness test, come from `config.py`.

## Running it

From the project folder, with `C:\venvs\balancebot` active:

```
python -m pytest                                                   # every suite, ~1 min
python optimization/benchmark_batch.py --backend torch --float32
python optimization/run_batch_tuning.py --backend torch --float32  # robust tuning, ~70 s
python optimization/run_batch_tuning.py --plain --backend torch --float32 --de-pop 2000 --cma-pop 10000
```

In code:

```python
evaluator = BatchEvaluator(CONFIG.build_eval_config(), CONFIG.build_search_space(),
                           backend="torch", dtype=np.float32)
costs = evaluator.evaluate_units(units)          # (P, 5) unit-cube points -> (P,) mean costs
```

## How it differs from the reference loop

| Reference simulator (`balance_sim`) | Batched (`batch_sim`) |
|---|---|
| One `Controller` / `Actuator` / `Dynamics` object per robot | Plain functions over (B,) arrays. State is a tuple of four (B,) arrays; the controller integrators are arrays passed in and returned |
| `break` when the robot falls | `alive` mask plus a recorded fall time. Fallen robots are frozen and stop adding cost. The loop stops once all have fallen (checked every 100 steps on the GPU) |
| Latency `deque` | Ring buffer of 3 in-flight commands, with the active command for each step precomputed by replaying the deque's own float comparisons |
| `SensorNoise` with `default_rng(seed)` | The same draws, precomputed on the CPU as a table of shape (ticks, 4, seeds), uploaded once, gathered per simulation |
| Goal velocity function called every tick | Precomputed per control tick (on the controller's own accumulated clock) and per physics step |
| Stores the trajectory, then `cost()` | Cost integrals added up as the run goes. The trajectory is only stored with `record=True` |

### How the GPU version is driven

A Python loop of 2,400 steps that launches ~200 small GPU kernels per step would spend its time on launch overhead. `TorchSimulator` avoids that:

- All state lives in fixed GPU buffers that are updated in place. The step and tick counters are GPU tensors, and per-step decisions (which command has reached the motor, the goal, the fall time) are lookups into GPU tables. Every step therefore does the same thing.
- Two steps are recorded once as **CUDA graphs**: a physics step, and a controller tick followed by a physics step. They are replayed 2,400 times. A replay costs one launch instead of ~200.
- `torch.compile` would also fuse the kernels, but on Windows it needs Triton, which isn't available here. CUDA graphs need nothing beyond PyTorch.
- `BatchEvaluator` keeps a few captured simulators and pads batch sizes up to multiples of 1,024, so DE and CMA-ES generations reuse the same graphs.

## Verified agreement

**NumPy float64 vs the reference code:**
- Trajectories, final states and fall times are bit-identical to `balance_sim` runs. Tested with the DC motor and ideal actuator, zero latency, a latency between physics steps, control every step, the upright pitch reference, friction with a weak motor, and per-robot parameters.
- Costs agree to about 1e-15; only the summation order differs.
- Random search and deferred DE pick the same candidates as the `tuning` searches.

**PyTorch vs NumPy float64:**
- The controller and anti-windup integrator match exactly on CPU and CUDA.
- The physics derivatives are bit-identical for about 90% of inputs. The rest differ in the last bit, because the GPU's `sin`/`cos` round differently.
- CUDA-graph replay is bit-identical to running the steps one by one, and repeated runs are identical.
- With well-behaved gains, float64 GPU costs match NumPy to `rtol=1e-9` and trajectories to 1e-9. That holds for the full 6 s tuning scenario, for batches where every robot falls (including fall times and early stop), and with no noise, zero latency and per-robot masses.
- float32 on the GPU is within 0.1% of float64.

**Chaos, and why some random-gain runs disagree.** With random gains, roughly a third of runs don't reproduce between the GPU and NumPy. Those same runs don't reproduce between two *NumPy* runs whose start tilt differs by 1e-15: 647 of 651 overlap. Torque saturation and the integrator switching amplify last-bit differences in these gain sets until the runs have nothing in common. Consequences:
- This affects the reference code too. Its score for such gains depends on rounding noise.
- The test for this checks that almost every GPU disagreement is a run NumPy can't reproduce either.
- A float32 search can over- or under-rate chaotic candidates. `run_batch_tuning.py` re-scores every finalist in float64 with NumPy. The winners are in the well-behaved region, where all backends agree.

**One PyTorch pitfall this uncovered.** PyTorch computes `number / tensor` (and, on CUDA, `tensor / number`) as a reciprocal followed by a multiply. That rounds twice and can be one bit off. Divisions that mix a tensor and a number therefore go through `xp.divide`, which divides by a cached 0-d GPU tensor instead and is correctly rounded.

## Speed (2026-09-28)

Real tuning scenario: 6 s runs, 2,400 physics steps each. Baseline (`benchmark.py`, the reference code one run at a time): 30.5 ms per simulation.

| sims per batch | NumPy f64 | NumPy f32 | GPU f64 | GPU f32 |
|---:|---:|---:|---:|---:|
| 1,200 | 0.45 ms (68×) | 0.41 ms (74×) | 0.78 ms (39×) | 0.62 ms (49×) |
| 12,000 | 0.21 ms (148×) | 0.14 ms (217×) | 0.082 ms (372×) | 0.067 ms (452×) |
| 120,000 | — | — | 16 µs (1,870×) | 8.1 µs (3,750×) |
| 400,000 | — | — | 14 µs (2,150×) | **5.3 µs (5,800×)** |

- **Below ~100k simulations, a GPU batch takes a flat ~0.7 s.** That's ~200 kernels per step × 2,400 steps. Use the GPU for big populations and NumPy for small ones.
- float64 on the GPU is only ~2.7× slower than float32 here, not 64×, because the kernels are limited by memory traffic and launches rather than arithmetic.

Tuning with `--backend torch --float32 --de-pop 2000 --cma-pop 10000 --cma-iters 30` took:
- CMA-ES: 2.7 million simulations in 34 s
- DE: 2.8 million in 71 s

At 30 ms each, the reference code would need about a day for either. Both reach the same optimum as the small NumPy runs (training cost 0.1069, validation 0.1077, no falls). It's worth noting:
- The optimum is robust.
- `kv_i` = 0 (the velocity integral) consistently comes out best, so it was later dropped from the gain set (the runs in this section still had 5 gains).

## Robust scoring

The reference tuner (`run_tuning.py`) scores gains on 3 tilts × 3 noise seeds with the nominal robot. `run_batch_tuning.py` defaults to a harsher test, set in `config.py` (`RobustnessSettings`) and implemented in `batch_tuning/robust.py`:

1. **50 training seeds**, so each candidate is scored on 150 scenarios instead of 9.
2. **A randomized robot in every scenario.** Body mass and COM height vary by ±15%, body inertia by ±25%, wheel inertia by ±20%, torque constant by ±10%, winding resistance by ±15% and battery voltage by ±10%. These spreads are guesses; replace them with measured tolerances. Each scenario's robot is seeded by its noise seed and tilt, so every candidate faces the same robots and the score is repeatable. Unseen validation seeds bring unseen robots.
3. **A chaos penalty.** Every scenario runs again with the start tilt nudged by 1e-4 rad (0.006°), keeping the same noise and robot. The objective is `mean cost + 1.0 × mean |cost − nudged cost|`.

Each candidate then costs 300 simulations, and finalists are validated on 100 unseen seeds with unseen robots (another 300). `--plain` switches this off, `--seeds N`, `--no-randomize` and `--sensitivity-weight W` adjust it, and passing no robustness config to `BatchEvaluator` scores exactly like the reference code.

Latest run, with the 4-gain controller (`results/batch_tuning/20260928-145838/`):

| gains | reference-style validation | robust validation (300 unseen) | worst scenario | chaos |
|---|---:|---:|---:|---:|
| hand-tuned (4.0, 0.5, 0.15, 0.1) | 0.243 | 0.293 | 0.928 | 4e-5 |
| CMA-ES (robust) | 0.108 | **0.108** | **0.116** | 7e-6 |
| DE (robust) | 0.108 | 0.108 | 0.116 | 7e-6 |

The best gains are kp ≈ 1.03, ki ≈ 0.20, kd ≈ 0.10, kv_p ≈ 0.33.

- **The tuned gains are robust.** Their cost barely moves across randomized robots, and their worst scenario is within 8% of the average. The hand-tuned gains get 21% worse on randomized robots, with a tail up to 0.93. `robustness_summary.png` shows both distributions.
- **Robust tuning found essentially the same gains as the plain tuning.** So the optimum isn't an artifact of 3 seeds and one robot. That's the main value of this run: the answer is now backed by far more evidence.
- **Dropping kv_i changed nothing for the tuned gains** (the 5-gain runs had already put it at 0, with the same cost). It made the hand-tuned baseline better: its old kv_i of 0.03 cost it 0.277 → 0.243 on reference validation. The earlier 5-gain run is in `results/batch_tuning/20260928-140800/`.
- **The chaos penalty was never large for the finalists** (they're well-behaved, with nudge effects of about 0.01% of the cost). It matters during the search, where it keeps chaotic candidates with a lucky score from winning.

## Where to go from here

- **Wider randomization.** Increase the spreads, or add the sensor noise levels and latency, to find where the tuned gains start to break down.
- **A fused kernel for higher throughput.** One GPU thread per simulation, with the 4-number state kept in registers, would remove the ~0.7 s floor. NVIDIA Warp (`pip install warp-lang`) compiles kernels at runtime without MSVC or `nvcc`, so it works on this laptop. The NumPy float64 version is its reference.
- **Gradients.** The same physics and controller functions work with PyTorch autograd (the NeurIPS 2018 paper's idea: gradients of the cost with respect to the gains, through the physics). `TorchSimulator` updates its buffers in place for CUDA graphs, which autograd doesn't support, so gradients would need a separate, out-of-place driver loop. The clipping and the fall mask also make the gradient zero or undefined in places.
