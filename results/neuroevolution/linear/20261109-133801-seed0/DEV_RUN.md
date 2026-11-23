# Development run, not a final result

The first full-profile run of the linear policy L (run seed 0), made on 2026-11-09 while developing.
It is kept here because docs/ml-baselines-plan.md, ARCHITECTURE.md and paper/results-planning-notes.md
quote its numbers.

- Trained with uncommitted changes (`snapshot.json`: `git_dirty: true`). Those changes were the
  training-output folder and the per-generation power logging only; the training and scoring code
  was that of commit dc66b36.
- The final L result is 10 run seeds made from a clean commit (plan 2.7). Its seed-0 run replaces
  this one in the paper's table.
- Its dev comparison (including the weight sensitivity and side metrics) and the knock-out script
  are in `results/comparison/20261109-dev-L-seed0/`.
