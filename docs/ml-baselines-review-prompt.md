# Reviewer handoff prompt: learned-controller baselines plan

Paste everything below the line into a fresh review agent, started in this repository's root.

---

You are an adversarial reviewer for an implementation plan in a research codebase. Your job is to
find what is wrong, missing, unfair, or infeasible in the plan before any code is written. Don't
edit any files or write code. Read, check claims against the code, and report.

## Context

This is an undergraduate honors research project (Purdue ENGR 10301H, fall 2026): a simulated
two-wheeled balancing robot. The best controller so far is a cascaded PID whose 4 gains are tuned
by CMA-ES against a robust objective (50 noise seeds x 3 start tilts, randomized robot parameters
and latency per scenario, plus a chaos penalty), validated on 300 held-out scenarios. It is
compared with LQR variants and a model-based PID. The paper's thesis is that ML tuning's main gain
is robustness (no falls on held-out robots, more latency tolerance), not typical-case cost.

A limitation is that none of this is machine learning in the sense a reader expects. The plan
adds neural-network controllers: a linear-policy control, neuroevolution (CMA-ES on an MLP), a
residual policy on top of the frozen PID (CMA-ES, then PPO), PPO from scratch, a recurrent (GRU)
policy, and, as a stretch, gradient descent through a differentiable simulator. Imitation learning
was deliberately excluded.

This is revision 3, after two adversarial reviews. The first raised R1-R20, the second M1-M4
(major) and m1-m13 (minor); section 0 of the plan maps each to where it was addressed. The current
implementation scope is everything up to PPO (Phase 0 on the CMA-ES path, L, A, A-warm, B1, PID
re-runs, final comparison); PPO, B2, the GRU and differentiable simulation are designed but deferred.
In addition to the attack list below:
- Concentrate on the current scope: it is what gets built next. Review the deferred phases only for
  anything in Phase 0 that would block them later.
- Check that each second-review fix closes its finding against the code without creating a new
  problem. In particular: the leaky-integrator redesign and the L-equals-PID gate test (does the
  PID-equivalent mapping actually hold, and what tolerance is honest); A-warm; the latency sweep on
  the fall-rate tier; the two-level bootstrap; the population sizing against `per_chunk`.
- Don't re-raise a fixed finding unless the fix is wrong or incomplete.

## Read first

1. `docs/ml-baselines-plan.md` (the plan under review)
2. `paper/results-planning-notes.md` (current results and thesis)
3. `README.md` (layout and the exact order everything runs in)
4. `config.py` (especially `CostConfig`, `ScenarioConfig`, `RobustnessSettings`)
5. `simulation/batch_sim/torch_sim.py`, `simulation/batch_sim/controller.py`,
   `simulation/batch_sim/simulate.py`, `simulation/batch_sim/schedule.py`
6. `optimization/batch_tuning/evaluation.py`, `optimization/batch_tuning/robust.py`,
   `optimization/batch_tuning/optimizers.py`
7. `optimization/run_lqr_comparison.py`, `simulation/balance_sim/stability.py`
8. `simulation/tests/test_equivalence.py`, `simulation/tests/test_torch.py`

Open the actual code before accepting any claim in the plan about how it works.

## What to attack

1. **Fairness of the comparison.** Does any method get information, data, compute, tuning effort
   or a scoring advantage the CMA-ES PID did not? Look at the observation vector, the fixed versus
   fresh scenario question for PPO, the matched-budget definition, the number of training seeds, and
   who gets hyperparameter tuning on which seeds. Also check the reverse: is the network handicapped
   in a way that makes a negative result meaningless?
2. **Leakage and statistics.** Can validation seeds 2000 to 2099 influence any decision? Is the new
   development set (5000 to 5049) really independent, given how robots are drawn from
   `(robot_seed, seed, tilt)`? Are 5 training seeds and paired bootstrap CIs on 300 scenarios enough
   for the claims the paper wants to make?
3. **Correctness of the proposed mechanics.**
   - Does the claimed reward really sum to minus the cost? Check the alive masking, which physics
     step a fall is detected on, the `(1 + remaining)` term, and per-environment latency against
     `torch_sim.py` and `simulate.py`.
   - Does the `(P, S)` batch layout and the dummy-candidate padding actually fit
     `BatchEvaluator._simulate_chunk` and the CUDA-graph capture (fixed shapes, in-place buffers)?
   - Is `max_torque * tanh(z)` equivalent enough to the PID's clip, and is the residual's zero
     init really bit-identical to the PID in float64?
   - Is the linearization in section 3.5 compatible with what `stability.py` can represent?
   - Will the differentiable-simulation section work given in-place ops, `torch.where` masks and
     the fall penalty?
4. **Feasibility.** Is CMA-ES on roughly 400 to 1,400 weights realistic at the stated budgets? Will
   neuroevolution from a random start get any signal before everything falls? Will PPO learn a
   200 Hz balance task with 5 to 20 ms unknown latency and noise, using an MLP? Estimate
   wall-clock on a single laptop GPU (RTX 50-series) from the throughput figures in `BATCHED.md`.
   Does the work fit what is left of a semester?
5. **Missing pieces.** Tests that should exist but are not listed, failure modes with no detection,
   config or snapshot gaps, results that would be uninterpretable without extra logging, and
   anything the paper would need that the plan doesn't produce.
6. **Scope and order.** Are the cut lines sensible? Is anything in the plan not worth doing, or is
   something worth doing missing (for example OpenAI-ES or PGPE instead of CMA-ES, a GRU policy,
   SAC)?
7. **The plan's own open questions** in section 11. Give a recommendation for each.

## How to report

- Start with a one-paragraph verdict: ready to build, ready after listed fixes, or needs redesign.
- Then findings, most severe first. For each one: severity (blocker / major / minor), the plan
  section, the code location you checked (`path:line`), what is wrong, a concrete scenario where it
  bites, and a specific fix.
- Label each finding as verified against the code or as a judgment call.
- Don't pad the list. Leave out style nits and anything you couldn't tie to a real consequence.
- End with answers to the section 11 questions, one short paragraph each.
