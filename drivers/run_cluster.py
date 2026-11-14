'''
One MCMC run for the paper. Example:
    python drivers/run_cluster.py --model CDM --sigma 0.1 --seed 1
Writes the usual Sampler outputs to results/ plus <tag>_chain.npy, <tag>_logprob.npy and <tag>_record.json.
'''
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, time, argparse, platform, subprocess
import numpy as np, scipy, astropy, emcee
import astropy.units as u
import dmsl, dmsl.prior as prior
import dmsl.sampler as sm, dmsl.mass_function as mf, dmsl.mass_profile as mp, dmsl.survey as survey
from dmsl.paths import RESULTSDIR, STARDATADIR

p = argparse.ArgumentParser()
p.add_argument('--model', choices=['CDM', 'EPP', 'WDM', 'PBH'], required=True)
p.add_argument('--sigma', type=float, default=0.1, help='per-star alpha precision [uas/yr^2] for the full 1e8-star survey')
p.add_argument('--nstars', type=int, default=1000)
p.add_argument('--nchains', type=int, default=128)   # as in the paper runs
p.add_argument('--ntune', type=int, default=1000)
p.add_argument('--nsamples', type=int, default=10000)
p.add_argument('--seed', type=int, default=1)
a = p.parse_args()

# Refuse to run on an old installed copy of dmsl.
assert hasattr(prior, 'sample_beta'), f'old dmsl imported from {dmsl.__file__}'
repo = os.path.dirname(os.path.dirname(os.path.abspath(sm.__file__)))
git = lambda *c: subprocess.run(['git', '-C', repo, *c], capture_output=True, text=True).stdout.strip()
os.makedirs(RESULTSDIR, exist_ok=True); os.makedirs(STARDATADIR, exist_ok=True)

np.random.seed(a.seed)
r = survey.Roman(alphasigma=a.sigma*u.uas/u.yr**2*(a.nstars/1.e8)**0.5, fov_deg=np.sqrt(2)*u.deg, maxdlens=4*u.kpc)
ML = np.logspace(0, 6, 100)
nfw = mp.NFW(Ml=1e5*u.Msun, c200=13)
mfun, mprof = {'CDM': lambda: (mf.CDM_Test(m_l=ML, sur=r), nfw),
               'EPP': lambda: (mf.Tinker_Mishra(m_l=ML, sur=r), nfw),
               'WDM': lambda: (mf.WDM_stream(m_l=ML, sur=r), nfw),
               'PBH': lambda: (mf.PBH_Gaussian(sur=r), mp.PointSource(Ml=1*u.Msun))}[a.model]()

tag = f"run_{a.model}_sig{a.sigma:g}_seed{a.seed}_{os.environ.get('SLURM_JOB_ID', 'local')}"
record = dict(args=vars(a), tag=tag, dmsl_path=sm.__file__, git_commit=git('rev-parse', 'HEAD'),
              git_branch=git('rev-parse', '--abbrev-ref', 'HEAD'), git_dirty=bool(git('status', '--porcelain', '--', 'dmsl')),
              alpha_eff_uas_yr2=float(r.alphasigma.value), host=platform.node(),
              slurm_job=os.environ.get('SLURM_JOB_ID'), slurm_cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
              versions=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                            astropy=astropy.__version__, emcee=emcee.__version__),
              start=time.strftime('%Y-%m-%d %H:%M:%S'))
print(json.dumps(record, indent=1), flush=True)

t0 = time.time()
s = sm.Sampler(nstars=a.nstars, ntune=a.ntune, nsamples=a.nsamples, nchains=a.nchains, ndims=2, survey=r,
               MassFunction=mfun, MassProfile=mprof, overwrite=True)

np.save(os.path.join(RESULTSDIR, tag + '_chain.npy'), s.sampler.get_chain())        # (steps, walkers, params), burn-in included
np.save(os.path.join(RESULTSDIR, tag + '_logprob.npy'), s.sampler.get_log_prob())
record.update(end=time.strftime('%Y-%m-%d %H:%M:%S'), hours=(time.time() - t0)/3600,
              acceptance=float(np.mean(s.sampler.acceptance_fraction)), param_names=list(mfun.param_names))
with open(os.path.join(RESULTSDIR, tag + '_record.json'), 'w') as f:
    json.dump(record, f, indent=1)
print('Wrote', tag, flush=True)
