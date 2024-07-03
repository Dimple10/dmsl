#!/bin/bash
#SBATCH --account=kmpardo_1126
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --output=cdmfinal.out
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=sarnaaik@usc.edu

#This is the final cdm run, 128 chains, 10,000 samples, with parallel processing and no convergence checks!
# 1 is 8 nodes, 16 tasks per; 2 is 16 nodes, 8 tasks per
# cdmfinal.out is the last run!! updated/checked on local

module load gcc/11.3.0 texlive
eval "$(conda shell.bash hook)"
conda activate dmsl
python drivers/run_analysis.py