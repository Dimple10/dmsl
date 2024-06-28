#!/bin/bash
#SBATCH --account=kmpardo_1126
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --output=cdmrun.out
#SBATCH --mem=6G
#SBATCH --time=48:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=sarnaaik@usc.edu

#This run is a test, 64 chains, 10,000 samples, with no parallel processing and no convergence checks!
module load gcc/11.3.0 texlive
eval "$(conda shell.bash hook)"
conda activate dmsl
python drivers/run_analysis.py