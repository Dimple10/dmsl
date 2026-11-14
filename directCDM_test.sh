#!/bin/bash
#SBATCH --account=kmpardo_1126
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --output=directCDM_4kpc_GSH_noAddedLens.out
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=sarnaaik@usc.edu

module load gcc/13.3.0 texlive
eval "$(conda shell.bash hook)"
conda activate dmsl
echo "Starting Python process..."

python << 'EOF'
import dmsl.sampler as sm
import dmsl.mass_function as mf
import dmsl.mass_profile as mp
import dmsl.survey as survey
import numpy as np
import astropy.units as u

# Setup survey
r = survey.Roman(
    alphasigma=0.1 * u.uas / u.yr**2 * (1000 / 1.e8)**(1. / 2.),
    fov_deg=np.sqrt(2) * u.deg,
    maxdlens=4 * u.kpc
)

# Setup mass profile and function
nfw = mp.NFW(Ml=1.e5 * u.Msun, c200=13)
cdm = mf.CDM_Test(m_l=np.logspace(0, 6, 100), sur=r)

# Initialize sampler
s = sm.Sampler(
    nstars=1000,
    nsamples=13000,
    survey=r,
    nchains=128,
    MassFunction=cdm,
    MassProfile=nfw
)

print("Sampler initialized:", s)
EOF

echo "Python process completed"
# python drivers/run_analysis_CDMTest.py
