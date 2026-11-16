#!/bin/bash
#SBATCH --account=kmpardo_1126
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=oneweek
#SBATCH --output=directTinkerMishra_WFIRST_noAddedLens_%j.out
#SBATCH --mem=0
#SBATCH --time=167:00:00
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

# Setup survey: 5% of the sky (square of side sqrt(0.05 * 4 pi) rad), 1 kpc deep,
# sigma rescaled from the 5e9 survey stars to the 1000 simulated stars, as for Roman
wf = survey.WFIRSTLike(
    alphasigma=0.1 * u.uas / u.yr**2 * (1000 / 5.e9)**(1. / 2.),
    fov_deg=np.sqrt(0.05 * 4 * np.pi) * u.rad,
    maxdlens=1 * u.kpc
)

# Setup mass profile and function
nfw = mp.NFW(Ml=1.e5 * u.Msun, c200=13)
tinker = mf.Tinker_Mishra(m_l=np.logspace(0, 6, 100), sur=wf)

# Initialize sampler (runs the MCMC)
s = sm.Sampler(
    nstars=1000,
    nsamples=13000,
    survey=wf,
    nchains=140,
    MassFunction=tinker,
    MassProfile=nfw
)

# Unpruned chain and log-probabilities (burn-in included), named by job ID
tag = f"TinkerMishra_WFIRST_noAddedLens_{os.environ.get('SLURM_JOB_ID', 'local')}"
np.save(os.path.join(RESULTSDIR, tag + '_chain.npy'), s.sampler.get_chain())
np.save(os.path.join(RESULTSDIR, tag + '_logprob.npy'), s.sampler.get_log_prob())
commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
print('git commit', commit, '| saved', tag)
EOF

echo "Python process completed"
