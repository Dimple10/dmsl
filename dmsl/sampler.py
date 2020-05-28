'''
Defines sampler class for doing the main analysis
'''

import os
import pickle
import numpy as np
import exoplanet as xo
import pandas as pd
import pymc3 as pm
from pymc3.distributions import Interpolated

from dmsl.constants import *
from dmsl.paths import *
from dmsl.convenience import *
from dmsl.prior_sampler import PriorSampler, find_nlens_pm
from dmsl.accel_data import AccelData

import dmsl.lensing_model as lm
import dmsl.background_model as bm
import dmsl.plotting as plot


class Sampler():

    def __init__(self, nstars=1000, nbsamples=100, nsamples=2000, nchains=8,
            ncores=2, ntune=1000, ndims=1, nblog10Ml=3, minlogMl=np.log(1e3),
            maxlogMl=np.log(1e8), minlogb =
            -15., maxlogb = 1.,overwrite=True):
        self.nstars=nstars
        self.nbsamples=nbsamples
        self.nblog10Ml=nblog10Ml
        self.nsamples=nsamples
        self.nchains=nchains
        self.ncores=ncores
        self.ntune=ntune
        self.ndims=ndims
        self.minlogMl = minlogMl
        self.maxlogMl = maxlogMl

        self.load_prior()
        self.load_data()
        self.logbarray = np.linspace(minlogb, maxlogb, 1000)
        self.run_inference()
        self.make_diagnostic_plots()


    def run_inference(self):

        with pm.Model() as model:

            ## lens signal
            logMl = pm.Uniform(name='logMl', lower=self.minlogMl, upper=self.maxlogMl)
            Ml = pm.math.exp(logMl)
            #nlens = pm.Poisson(name='nlens', mu=find_nlens_pm(Ml))
            nlens = find_nlens_pm(Ml)
            logbs = Interpolated('logbs',self.logbarray,
                    self.plogb(self.logbarray))
            bs_raw = pm.math.exp(logbs)
            bs = bs_raw/(nlens)**(1./3.)
            pmprint(bs)
            vl = pm.TruncatedNormal(name='vl', mu=0., sigma=220, lower=0, upper=550.)
            if self.ndims==2:
                bstheta = pm.Uniform(name='bstheta', lower=0, upper=np.pi/2.)
                vltheta = pm.Uniform(name='vltheta', lower=0, upper=np.pi/2.)
            else:
                bstheta = None
                vltheta = None

            alphal = lm.alphal(Ml, bs, vl,
                btheta_=bstheta, vltheta_=vltheta)
            logalphal = pm.Deterministic('logabsalphal',
                    pm.math.log(pm.math.abs_(alphal)))
            pmprint(alphal)
            ## background signal
            #rmw = pm.TruncatedNormal(name='rmw', mu=AVE_RMW.value,
            #        sigma=SIGMA_RMW.value, lower=0., upper=2.)
            rmw = pm.Exponential(name='rmw', lam=1./RD_MW.value)
            if self.ndims==2:
                rmwtheta = pm.Uniform(name='rmwtheta', lower=0.5267,
                        upper=1.043)
            else:
                rmwtheta = None

            alphab = bm.alphab(rmw,
                rtheta_=rmwtheta)
            logalphab = pm.Deterministic('logabsalphab',
                    pm.math.log(pm.math.abs_(alphab)))
            pmprint(alphab)
            ## set up obs
            alpha = alphal + alphab

            ## likelihood
            obs = pm.Normal('obs', mu=alpha, sigma=WFIRST_SIGMA.value,
                    observed=self.data)

            ## run sampler
            trace = pm.sample(self.nsamples, tune=self.ntune, cores=self.ncores,
                    chains=self.nchains,
                    step=xo.get_dense_nuts_step(target_accept=0.9))
        self.trace = trace

        ## save results
        pklpath = make_file_path(RESULTSDIR, [np.log10(self.nstars), np.log10(self.nsamples), self.ndims],
                extra_string='trace', ext='.pkl')
        with open(pklpath, 'wb') as buff:
            pickle.dump(trace, buff)
        print('Wrote {}'.format(pklpath))

    def load_prior(self):
        ## loads prior or make if file not found.
        fileinds = [np.log10(self.nstars), self.nblog10Ml, np.log10(self.nbsamples)]
        pklpath = make_file_path(RESULTSDIR, fileinds, extra_string='plogb',
                ext='.pkl')
        if os.path.exists(pklpath):
            self.plogb  = pickle.load(open(pklpath, 'rb'))

        else:
            ps = PriorSampler(nstars=self.nstars, nbsamples=self.nbsamples,
                    log10Ml=self.nblog10Ml)
            ps.run_sampler()
            self.plogb = ps.plogb

    def load_data(self):
        ## loads data or makes if file not found.
        fileinds = [np.log10(self.nstars), self.ndims]
        filepath = make_file_path(STARDATADIR, fileinds, ext='.dat')
        if os.path.exists(filepath):
            self.data = pd.read_csv(filepath).to_numpy()
        else:
            AccelData(nstars=self.nstars, ndims=self.ndims)
            self.data = pd.read_csv(filepath).to_numpy(dtype=np.float64)

    def make_diagnostic_plots(self):
        outpath = make_file_path(RESULTSDIR, [np.log10(self.nstars), np.log10(self.nsamples),
            self.ndims],extra_string='traceplot', ext='.png')
        plot.plot_trace(self.trace, outpath)

        outpath = make_file_path(RESULTSDIR, [np.log10(self.nstars), np.log10(self.nsamples),
            self.ndims],extra_string='corner', ext='.png')
        plot.plot_corner(self.trace, outpath)

