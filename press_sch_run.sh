#!/bin/bash
#SBATCH --account=kmpardo_1126
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --output=press_s_final.out
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=sarnaaik@usc.edu

##Press_schechter run

module load gcc/11.3.0 texlive
eval "$(conda shell.bash hook)"
conda activate dmsl
python drivers/run_analysis.py