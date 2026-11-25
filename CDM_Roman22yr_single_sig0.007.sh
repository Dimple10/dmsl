#!/bin/bash
#SBATCH --account=kmpardo_1126
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --output=CDM_Roman22yr_single_sig0.007_%j.out
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=sarnaaik@usc.edu

# Future survey, CDM + NFW, Roman, SINGLE-DRAW likelihood (branch noAddedLens, NOT Route B).
# sigma = 0.007 uas/yr^2 per star: the paper 0.1 uas/yr^2 case plus the year-22 season (14.7x gain; 0.1/14.7 = 0.0068, rounded to 0.007).
# baseline_yr = 22: b_min = max(R_E, v T) with T = 22 yr, to match sigma.
# Submit with sbatch (not srun) from ~/Repo/Kris/dmsl_test.
module load gcc/13.3.0 texlive
eval "$(conda shell.bash hook)"
conda activate dmsl
cd ~/Repo/Kris/dmsl_test          # dmsl and the .pkl files are found from this folder
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
import dmsl.prior as prior
from dmsl.paths import RESULTSDIR

print('dmsl from', sm.__file__)
assert 'dmsl_test' in sm.__file__ and hasattr(prior, 'sample_beta'), 'wrong dmsl imported'

SIGMA = 0.007            # per-star alpha precision [uas/yr^2]
BASELINE_YR = 22.          # b_min = max(R_E, v T), T = 22 yr
NCHAINS, NSAMPLES, NTUNE = 128, 13000, 1000

# Roman GBTDS: 2 deg^2, lenses to 4 kpc, sigma rescaled from 1e8 survey stars to the 1000 simulated stars
r = survey.Roman(alphasigma=SIGMA * u.uas / u.yr**2 * (1000 / 1.e8)**0.5,
                 fov_deg=np.sqrt(2) * u.deg, maxdlens=4 * u.kpc)
nfw = mp.NFW(Ml=1.e5 * u.Msun, c200=13)
cdm = mf.CDM_Test(m_l=np.logspace(0, 6, 100), sur=r)

start = time.strftime('%Y-%m-%d %H:%M:%S'); t0 = time.time()
s = sm.Sampler(nstars=1000, nsamples=NSAMPLES, ntune=NTUNE, nchains=NCHAINS, survey=r,
               MassFunction=cdm, MassProfile=nfw, baseline_yr=BASELINE_YR)   # likelihood is the single draw (no 'likelihood' option on this branch)
hours = (time.time() - t0) / 3600

# Full chain and log-prob (burn-in included): chain columns [log10 c200, slope b]; the job ID is in the tag
tag = f"CDM_Roman22yr_single_sig{SIGMA:g}_{os.environ.get('SLURM_JOB_ID', 'local')}"
chain, logprob = s.sampler.get_chain(), s.sampler.get_log_prob()
np.save(os.path.join(RESULTSDIR, tag + '_chain.npy'), chain)
np.save(os.path.join(RESULTSDIR, tag + '_logprob.npy'), logprob)

# Diagnostics for the record (the test list reads these)
sig = r.alphasigma.value
lnl_null = -0.5 * np.sum(s.data**2 / sig**2 - np.log(2 * np.pi * sig**2))      # no-lensing lnL on the fixed mock data
lp = logprob[NTUNE:]
acc = s.sampler.acceptance_fraction
try:
    tau = [float(t) for t in s.sampler.get_autocorr_time(discard=NTUNE, tol=0, quiet=True)]
except Exception as e:
    tau = repr(e)
try:
    kept = len(s.prune_chains()[0]) / NSAMPLES
except Exception as e:
    kept = repr(e)
commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True).stdout.strip().splitlines()
record = dict(tag=tag, survey=r.name, likelihood='single', sigma_per_star=SIGMA, alpha_eff=float(sig), baseline_yr=s.baseline_yr,
              nstars=1000, nchains=NCHAINS, nsamples=NSAMPLES, ntune=NTUNE, git_commit=commit, git_modified_files=dirty, start=start,
              hours=round(hours, 2), acceptance=float(np.mean(acc)), acceptance_min=float(np.min(acc)), acceptance_max=float(np.max(acc)),
              autocorr_steps=tau, chains_after_pruning=kept, lnl_null=float(lnl_null),
              frac_logprob_neginf=float(np.mean(~np.isfinite(lp))),
              frac_logprob_nolens=float(np.mean(np.isclose(lp, lnl_null, rtol=0, atol=1e-9))),   # samples where the draw had no lens (lnL = null exactly)
              frac_logc_above_6=float(np.mean(chain[NTUNE:, :, 0] > 6)))
with open(os.path.join(RESULTSDIR, tag + '_record.json'), 'w') as f:
    json.dump(record, f, indent=1)
print(json.dumps(record, indent=1))
print('saved', tag)
EOF

echo "Python process completed"
