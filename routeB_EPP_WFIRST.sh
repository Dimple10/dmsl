#!/bin/bash
#SBATCH --account=kmpardo_1126
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --output=routeB_EPP_WFIRST_%j.out
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=sarnaaik@usc.edu

# Route B (likelihood='expected') EPP run. Submit with sbatch (not srun) from ~/Repo/Kris/dmsl_routeB.
module load gcc/13.3.0 texlive
eval "$(conda shell.bash hook)"
conda activate dmsl
cd ~/Repo/Kris/dmsl_routeB        # dmsl and the .pkl files are found from this folder
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "Starting Python process..."

python << 'EOF'
import os, json, time, subprocess
import numpy as np
import astropy.units as u
import dmsl.sampler as sm
import dmsl.mass_function as mf
import dmsl.mass_profile as mp
import dmsl.survey as survey
from dmsl.paths import RESULTSDIR

print('dmsl from', sm.__file__)
assert 'dmsl_routeB' in sm.__file__ and hasattr(sm.Sampler, 'lnlike_expected'), 'wrong dmsl imported'

SIGMA = 0.1                 # per-star alpha precision [uas/yr^2] for the full survey (5.19 from 1.1 mas)
LIKELIHOOD = 'expected'     # or 'detections'
NCHAINS, NSAMPLES, NTUNE = 128, 13000, 1000

# WFIRST-like: 5% of the sky (square of side sqrt(0.05 * 4 pi) rad), lenses to 1 kpc,
# sigma rescaled from 5e9 survey stars to the 1000 simulated stars
sur = survey.WFIRSTLike(alphasigma=SIGMA * u.uas / u.yr**2 * (1000 / 5.e9)**0.5,
                        fov_deg=np.sqrt(0.05 * 4 * np.pi) * u.rad, maxdlens=1 * u.kpc)
nfw = mp.NFW(Ml=1.e5 * u.Msun, c200=13)
tinker = mf.Tinker_Mishra(m_l=np.logspace(0, 6, 100), sur=sur)

start = time.strftime('%Y-%m-%d %H:%M:%S'); t0 = time.time()
s = sm.Sampler(nstars=1000, nsamples=NSAMPLES, ntune=NTUNE, nchains=NCHAINS, survey=sur,
               MassFunction=tinker, MassProfile=nfw, likelihood=LIKELIHOOD)

# Full chain (burn-in included): columns [log c200, logkb, nb]
tag = f"EPP_WFIRST_routeB_{LIKELIHOOD}_sig{SIGMA:g}_{os.environ.get('SLURM_JOB_ID', 'local')}"
np.save(os.path.join(RESULTSDIR, tag + '_chain.npy'), s.sampler.get_chain())
np.save(os.path.join(RESULTSDIR, tag + '_logprob.npy'), s.sampler.get_log_prob())
commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
record = dict(tag=tag, survey=sur.name, sigma_per_star=SIGMA, alpha_eff=float(sur.alphasigma.value),
              likelihood=LIKELIHOOD, K_exp=s.K_exp, R_exp=s.R_exp, nchains=NCHAINS, nsamples=NSAMPLES, ntune=NTUNE,
              git_commit=commit, start=start, hours=round((time.time() - t0) / 3600, 2),
              acceptance=float(np.mean(s.sampler.acceptance_fraction)))
with open(os.path.join(RESULTSDIR, tag + '_record.json'), 'w') as f:
    json.dump(record, f, indent=1)
print(json.dumps(record, indent=1))
EOF

echo "Python process completed"
