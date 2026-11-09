#!/bin/bash
# One run seed per array task: the 10 final runs of a method on a SLURM cluster.
# Only calls the normal entry point; everything else comes from config.py (docs/ml-baselines-plan.md 9).
#
#   sbatch --array=0-9 scripts/slurm_neuroevolution.sh mlp
#   sbatch --array=0-9 scripts/slurm_neuroevolution.sh warm --warm-from <training folder>/neuroevolution/linear/<run>/final.npz
#
# A task that runs out of time can be resubmitted with the same array index: it resumes from the
# run folder's state.pkl.
#SBATCH --job-name=neuroevolution
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
METHOD="$1"; shift
SEED="${SLURM_ARRAY_TASK_ID:-0}"
RUN="${BALANCEBOT_TRAINING_DIR:-results}/neuroevolution/${METHOD}/cluster-seed${SEED}"   # set BALANCEBOT_TRAINING_DIR to the scratch disk
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
if [ -f "${RUN}/state.pkl" ]; then
    python optimization/run_neuroevolution.py --resume "${RUN}"
else
    python optimization/run_neuroevolution.py --method "${METHOD}" --profile full --run-seed "${SEED}" \
        --out "${RUN}" "$@"
fi
