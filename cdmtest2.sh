#!/bin/bash
#SBATCH --account=kmpardo_1126
#SBATCH --nodes=4
#SBATCH --ntasks=8
#SBATCH --output=cdmMultiTest3.out
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=sarnaaik@usc.edu

module load gcc/11.3.0 texlive
eval "$(conda shell.bash hook)"
conda activate dmsl
python drivers/run_analysis.py