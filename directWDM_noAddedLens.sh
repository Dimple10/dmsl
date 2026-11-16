#!/bin/bash
#SBATCH --account=kmpardo_1126
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --output=directWDM_noAddedLens_%j.out
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=sarnaaik@usc.edu

module load gcc/13.3.0 texlive
eval "$(conda shell.bash hook)"
conda activate dmsl
# cd ~/Repo/Kris/dmsl_test          # dmsl and the .pkl files are found from this folder
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "Starting Python process..."

python << 'EOF'
import os, subprocess
import numpy as np
import astropy.units as u
import dmsl.sampler as sm
import dmsl.mass_function as mf
import dmsl.mass_profile as mp
import dmsl.survey as survey
import dmsl.prior as prior
from dmsl.paths import RESULTSDIR

print('dmsl from', sm.__file__)
assert 'dmsl_test' in sm.__file__ and hasattr(prior, 'sample_beta'), 'wrong dmsl imported'

# Setup survey
r = survey.Roman(
    alphasigma=0.1 * u.uas / u.yr**2 * (1000 / 1.e8)**(1. / 2.),
    fov_deg=np.sqrt(2) * u.deg,
    maxdlens=4 * u.kpc
)

# Setup mass profile and function
nfw = mp.NFW(Ml=1.e5 * u.Msun, c200=13)
wdm = mf.WDM_stream(m_l=np.logspace(0, 6, 100), sur=r)

# Initialize sampler (runs the MCMC); 128 walkers (was 168) so it fits in 48 h
s = sm.Sampler(
    nstars=1000,
    nsamples=12000,
    survey=r,
    nchains=128,
    MassFunction=wdm,
    MassProfile=nfw
)

# Unpruned chain and log-probabilities (burn-in included), named by job ID
tag = f"WDM_noAddedLens_{os.environ.get('SLURM_JOB_ID', 'local')}"
np.save(os.path.join(RESULTSDIR, tag + '_chain.npy'), s.sampler.get_chain())
np.save(os.path.join(RESULTSDIR, tag + '_logprob.npy'), s.sampler.get_log_prob())
commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
print('git commit', commit, '| saved', tag)
EOF

echo "Python process completed"
