"""batch_tuning: gain searches that score whole populations with batch_sim.

Builds on the CPU tuning package (EvalConfig, SearchSpace, TuningResult,
save_result) in the same folder; see BATCHED.md.
"""

from .robust import RobustnessConfig, robot_at, sample_robots
from .evaluation import BatchEvalResult, BatchEvaluator, decode_units, spec_from_eval_config
from .optimizers import cma_es_search, differential_evolution_search, random_search

__all__ = [
    "RobustnessConfig",
    "sample_robots",
    "robot_at",
    "BatchEvaluator",
    "BatchEvalResult",
    "decode_units",
    "spec_from_eval_config",
    "random_search",
    "differential_evolution_search",
    "cma_es_search",
]
