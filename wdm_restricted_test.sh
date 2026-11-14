#!/bin/bash
#SBATCH --account=kmpardo_1126
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --job-name=wdm_restr
#SBATCH --output=wdm_nolens_restricted_test.out
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=sarnaaik@usc.edu

module load gcc/13.3.0 texlive
eval "$(conda shell.bash hook)"
conda activate dmsl

echo "Starting restricted-prior WDM run on $(hostname)"
python run_wdm_nolens.py --chains 64 --samples 5000 --tune 500 --lmin 2.5 --lmax 4 --procs 32
echo "Run completed"