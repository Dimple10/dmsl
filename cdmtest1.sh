#!/bin/bash
#SBATCH --account=kmpardo_1126
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --output=cdmtest2.out
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=sarnaaik@usc.edu

##Test 1 is 16 nodes, test 2 is 2 node, 8 tasks

module load gcc/11.3.0 texlive
eval "$(conda shell.bash hook)"
conda activate dmsl
python drivers/run_analysis.py