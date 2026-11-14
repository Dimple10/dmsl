#!/usr/bin/env python
"""WDM Stream / NFW sampler run with the no-lens fix, for direct command-line use.

Run from the repo root (the folder that contains dmsl/), with the dmsl env active and
`module load gcc/13.3.0 texlive` done (the Sampler makes TeX plots at the end):

    python run_wdm_nolens.py                                                      # test: 32 chains x 500 samples
    python run_wdm_nolens.py --chains 168 --samples 15000 --tune 1000 --procs 32  # production size

Three runtime patches are applied here, so nothing in dmsl/ needs editing:
  1. Initial walkers are spread uniformly over each parameter's prior range.
  2. A first draw with zero lenses gets the finite no-signal likelihood instead of -inf.
  3. The worker pool is capped at --procs and every worker reseeds its random state.
Assumes the NFW + WDM Stream setup below (4 parameters: logc200, logmwdm, gamma, beta) and that
the installed sampler.py returns a bare -inf float from make_new_mass for zero lenses.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'  # must be set before numpy is imported

import argparse
import multiprocessing
import random

import numpy as np
import astropy.units as u

import dmsl.sampler as sm
import dmsl.mass_function as mf
import dmsl.mass_profile as mp
import dmsl.survey as survey


def get_args():
    ncpu = len(os.sched_getaffinity(0))
    p = argparse.ArgumentParser()
    p.add_argument('--chains', type=int, default=32)
    p.add_argument('--samples', type=int, default=500)
    p.add_argument('--tune', type=int, default=100)
    p.add_argument('--nstars', type=int, default=1000)
    p.add_argument('--procs', type=int, default=ncpu, help=f'worker processes (visible cores: {ncpu})')
    p.add_argument('--seed', type=int, default=None)
    return p.parse_args()


def _reseed():
    np.random.seed()
    random.seed()


def patch_sampler(nprocs):
    # 1. cap the worker pool (Sampler calls Pool() with no arguments) and reseed each worker
    sm.Pool = lambda *a, **k: multiprocessing.Pool(processes=nprocs, initializer=_reseed)

    # 2. spread the initial walkers over the whole prior instead of rand() + lower bound
    def initialize(self):
        mfn = self.massfunction
        p = np.empty((self.nchains, mfn.nparams))
        for i, name in enumerate(mfn.param_names):
            lo, hi = mfn.param_range[name]
            p[:, i] = np.random.uniform(lo, hi, self.nchains)
        return p
    sm.Sampler.initialize = initialize

    # 3. zero-lens draws get the finite no-signal likelihood.
    # Record whether the FIRST make_new_mass call of each likelihood evaluation was a zero-lens
    # draw (it returns a bare float); snr_check retries are ignored.
    orig_mnm = sm.Sampler.make_new_mass
    orig_lnlike = sm.Sampler.lnlike

    def make_new_mass(self, pars):
        res = orig_mnm(self, pars)
        if getattr(self, '_first_nolens', None) is None:
            self._first_nolens = isinstance(res, float)
        return res

    def lnlike(self, pars):
        self._first_nolens = None
        out = orig_lnlike(self, pars)
        if np.isfinite(out) or not self._first_nolens:
            return out
        if not np.isfinite(self.logprior(pars)):
            return -np.inf
        sig = self.survey.alphasigma.value
        return -0.5 * np.sum(self.data**2 / sig**2 - np.log(2 * np.pi * sig**2))

    sm.Sampler.make_new_mass = make_new_mass
    sm.Sampler.lnlike = lnlike


def report(s, wdm):
    sam = s.sampler
    chain = sam.get_chain()[s.ntune:]           # (steps, walkers, pars)
    lp = sam.get_log_prob()[s.ntune:]
    nwalk = chain.shape[1]

    def longest_run(x):
        edges = np.r_[-1, np.flatnonzero(np.diff(x) != 0), len(x) - 1]
        return np.diff(edges).max()

    runs = np.array([longest_run(chain[:, k, 0]) for k in range(nwalk)])
    stuck = runs >= max(1, int(0.05 * s.nsamples))      # same rule as prune_chains
    sig = s.survey.alphasigma.value
    ll0 = -0.5 * np.sum(s.data**2 / sig**2 - np.log(2 * np.pi * sig**2))
    fin = np.isfinite(lp)
    logMhm = np.log10(wdm.a) + wdm.b * chain[-1, :, 1]

    print('\n===== diagnostics =====')
    print(f'acceptance: mean {sam.acceptance_fraction.mean():.3f}, min {sam.acceptance_fraction.min():.3f}   (want mean >= 0.05)')
    print(f'stuck walkers: {stuck.sum()} of {nwalk}   (want <= ~10%)')
    print(f'non-finite log-prob fraction (post burn-in): {1 - fin.mean():.4f}   (want ~0)')
    print(f'walkers on the no-lens plateau at the end: {np.sum(np.isclose(lp[-1], ll0, atol=1e-6))} of {nwalk}')
    print(f'no-lens lnL = {ll0:.1f} | best chain lnL = {lp[fin].max():.1f} | '
          f'median |alpha|/sigma of data = {np.median(np.linalg.norm(s.data, axis=1)) / sig:.2f} (noise only: ~1.18)')
    print(f'final median log10 M_hm: {np.median(logMhm):.1f}  (grid tops out at 6)')
    print('final position quartiles (logc200, logmwdm, gamma, beta):')
    print(np.percentile(chain[-1], [0, 25, 50, 75, 100], axis=0).round(2))


if __name__ == '__main__':
    args = get_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
    patch_sampler(args.procs)

    r = survey.Roman(
        alphasigma=0.1 * u.uas / u.yr**2 * (1000 / 1.e8)**(1. / 2.),
        fov_deg=np.sqrt(2) * u.deg,
        maxdlens=4 * u.kpc,
    )
    nfw = mp.NFW(Ml=1.e5 * u.Msun, c200=13)
    wdm = mf.WDM_stream(m_l=np.logspace(0, 6, 100), sur=r)
    assert wdm.Name == 'WDM Stream' and nfw.type == 'nfw'

    print(f'chains={args.chains} samples={args.samples} tune={args.tune} procs={args.procs}')
    s = sm.Sampler(
        nstars=args.nstars,
        nsamples=args.samples,
        ntune=args.tune,
        survey=r,
        nchains=args.chains,
        MassFunction=wdm,
        MassProfile=nfw,
    )
    report(s, wdm)