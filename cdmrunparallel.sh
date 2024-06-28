#!/bin/bash
#SBATCH --account=kmpardo_1126
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --output=cdmrunparallel3.out
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=sarnaaik@usc.edu

#This run is a test, 64 chains, 10,000 samples, with parallel processing and no convergence checks!
# 1 was 4 nodes, 8 nodes per task, 2 was 1 nodes 64 tasks, 3 is 2 nodes 32 tasks per node
module load gcc/11.3.0 texlive
eval "$(conda shell.bash hook)"
conda activate dmsl
python drivers/run_analysis.py