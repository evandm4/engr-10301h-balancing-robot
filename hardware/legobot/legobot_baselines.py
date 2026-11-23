"""
Baseline scores on the LEGO robot (SOFTWARE_PLAN A4): the hand-tuned and
model-based cascaded PIDs, neither tuned by search. The cost-matched LQR is
dropped for this robot (decided 2026-11-23: it fell in most robust scenarios
under the measured slow-PWM drive at every cruise speed tried; A4_BASELINES.md). Re-run after the config changes (e.g. after the bench measurements).

    python hardware/legobot/legobot_baselines.py

Scored three ways: the plain scenario (nominal robot, train seeds 1-3, as
run_tuning.py), robust on the training seeds (1-50, what the batched tuner
minimizes), and robust on the development seeds (5000-5049). The frozen
validation set (3000-3099) is not touched here.
"""
import os
import sys
from pathlib import Path

os.environ["ROBOT"] = "legobot"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from config import CONFIG as C
from batch_tuning import BatchEvaluator
assert C.name == "legobot"
ec = C.build_eval_config(); rb = C.build_robustness()
pid_space = C.build_search_space()
DEV = tuple(range(5000, 5050))
rows = [("hand-tuned PID", "cascaded", pid_space, np.array(C.controller.hand_tuned_gains)),
        ("model-based PID", "cascaded", pid_space, np.array(C.model_based_gains()))]
print(f"{'baseline':<18}{'plain (seeds 1-3)':>20}{'robust train (1-50)':>22}{'robust dev (5000-5049)':>26}{'dev worst':>11}{'dev chaos':>11}")
for name, kind, space, g in rows:
    plain = BatchEvaluator(ec, space, controller=kind).evaluate_population([g])
    ev = BatchEvaluator(ec, space, controller=kind, robustness=rb)
    tr = ev.evaluate_population([g])
    dv = ev.evaluate_population([g], seeds=DEV)
    f = lambda r: f"{r.mean_cost[0]:.3f} ({r.n_fell[0]}/{r.costs.shape[1]} fell)"
    print(f"{name:<18}{f(plain):>20}{f(tr):>22}{f(dv):>26}{dv.costs[0].max():>11.3f}{dv.sensitivity[0]:>11.1e}")
    print("   gains", np.round(g, 4), " train objective", round(float(tr.objective[0]), 4))
