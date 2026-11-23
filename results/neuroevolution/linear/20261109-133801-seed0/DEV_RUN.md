# Development run, not a final result

The first full-profile run of the linear policy L (run seed 0), made on 2026-11-09 while developing.
It is kept here because docs/ml-baselines-plan.md, ARCHITECTURE.md and paper/results-planning-notes.md
quote its numbers.

- Trained with uncommitted changes (`snapshot.json`: `git_dirty: true`). Those changes were the
  training-output folder and the per-generation power logging only; the training and scoring code
  was that of commit dc66b36.
- The final L result is 10 run seeds made from a clean commit (plan 2.7). Its seed-0 run replaces
  this one in the paper's table.
- Trained and scored under an earlier config: latency drawn from 5-20 ms (2.5-20 ms since PR #9) and
  the earlier reference PID (`results/lqr_comparison/20261108-225341`). Not to be pooled with runs
  made since; `compare_controllers.py` flags it. Re-scored on dev under the current config by the
  PR #10 review: still 0.870 against the earlier PID, and the re-tuned PID itself 0.999.
- Its dev comparison (including the weight sensitivity and side metrics) and the knock-out script
  are in `results/comparison/20261109-dev-L-seed0/`.
