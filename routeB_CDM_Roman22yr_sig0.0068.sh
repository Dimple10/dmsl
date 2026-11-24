#!/bin/bash
#SBATCH --account=kmpardo_1126
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --output=routeB_CDM_Roman22yr_sig0.0068_%j.out
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=sarnaaik@usc.edu

# Route B (likelihood='expected') CDM run, Roman 5 yr + one season ending at 22 yr.
# Submit with sbatch (not srun) from ~/Repo/Kris/dmsl_routeB.
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

SIGMA = 0.0068              # per-star alpha precision [uas/yr^2]: the 0.1 case plus the year-22 season (statistical best case)
BASELINE_YR = 22.          # b_min = max(R_E, v T) with T = 22 yr, to match SIGMA
LIKELIHOOD = 'expected'    # or 'detections'
NCHAINS, NSAMPLES, NTUNE = 128, 13000, 1000

# Roman GBTDS: 2 deg^2, lenses to 4 kpc, sigma rescaled from 1e8 survey stars to the 1000 simulated stars
sur = survey.Roman(alphasigma=SIGMA * u.uas / u.yr**2 * (1000 / 1.e8)**0.5,
                   fov_deg=np.sqrt(2) * u.deg, maxdlens=4 * u.kpc)
nfw = mp.NFW(Ml=1.e5 * u.Msun, c200=13)
cdm = mf.CDM_Test(m_l=np.logspace(0, 6, 100), sur=sur)

start = time.strftime('%Y-%m-%d %H:%M:%S'); t0 = time.time()
s = sm.Sampler(nstars=1000, nsamples=NSAMPLES, ntune=NTUNE, nchains=NCHAINS, survey=sur,
               MassFunction=cdm, MassProfile=nfw, likelihood=LIKELIHOOD, baseline_yr=BASELINE_YR)

# Full chain (burn-in included): columns [log c200, slope b]
tag = f"CDM_Roman22yr_routeB_{LIKELIHOOD}_sig{SIGMA:g}_{os.environ.get('SLURM_JOB_ID', 'local')}"
np.save(os.path.join(RESULTSDIR, tag + '_chain.npy'), s.sampler.get_chain())
np.save(os.path.join(RESULTSDIR, tag + '_logprob.npy'), s.sampler.get_log_prob())
commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
record = dict(tag=tag, survey=sur.name, sigma_per_star=SIGMA, alpha_eff=float(sur.alphasigma.value),
              baseline_yr=s.baseline_yr, likelihood=LIKELIHOOD, K_exp=s.K_exp, R_exp=s.R_exp,
              nchains=NCHAINS, nsamples=NSAMPLES, ntune=NTUNE, git_commit=commit, start=start,
              hours=round((time.time() - t0) / 3600, 2), acceptance=float(np.mean(s.sampler.acceptance_fraction)))
with open(os.path.join(RESULTSDIR, tag + '_record.json'), 'w') as f:
    json.dump(record, f, indent=1)
print(json.dumps(record, indent=1))
EOF

echo "Python process completed"
