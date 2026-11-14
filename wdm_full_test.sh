#!/bin/bash
#SBATCH --account=kmpardo_1126
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --job-name=wdm_full
#SBATCH --output=wdm_nolens_full_test.out
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=sarnaaik@usc.edu

module load gcc/13.3.0 texlive
eval "$(conda shell.bash hook)"
conda activate dmsl

echo "Starting full-prior WDM run on $(hostname)"
python run_wdm_nolens.py --chains 168 --samples 8000 --tune 1000 --procs 32
echo "Run completed"