#!/bin/bash
#SBATCH --account=kmpardo_1126
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --array=1-3
#SBATCH --job-name=dmsl
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-type=end
#SBATCH --mail-user=sarnaaik@usc.edu

# Usage:  sbatch slurm_run.sh MODEL SIGMA [extra run_cluster.py options]
#   sbatch slurm_run.sh CDM 0.1                      full run, seeds 1-3 (one array task per seed)
#   sbatch --time=00:30:00 --array=1 slurm_run.sh CDM 0.1 --nchains 8 --ntune 1 --nsamples 2   smoke test
# One node only: multiprocessing.Pool cannot use more than one node. The Pool size comes from --cpus-per-task.
MODEL=${1:-CDM}; SIGMA=${2:-0.1}; shift 2

module load gcc/11.3.0 texlive
eval "$(conda shell.bash hook)"
conda activate dmsl-new
cd "$SLURM_SUBMIT_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python -c "import dmsl.sampler as s; print('dmsl from', s.__file__)"
python drivers/run_cluster.py --model "$MODEL" --sigma "$SIGMA" --seed "${SLURM_ARRAY_TASK_ID:-1}" "$@"
