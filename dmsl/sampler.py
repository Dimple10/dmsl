'''
Defines sampler class for doing the main analysis
'''

import os
import pickle
import numpy as np
import pymc3 as pm
from pymc3.distributions import Interpolated
import exoplanet as xo


from dmsl.constants import *
from dmsl.paths import *
from dmsl.convenience import *
from dmsl.prior_sampler import PriorSampler
from dmsl.accel_data import AccelData

import dmsl.lensing_model as lm
import dmsl.background_model as bm


class Sampler():

    def __init__(self, nstars=1000, nbsamples=100, nsamples=2000, nchains=8,
            ncores=2, ntune=500, ndims=1, nblog10Ml=3, minlogMl=1., maxlogMl=8., minlogb =
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


    def run_inference(self):

        with pm.Model() as model:

            ## lens signal
            logMl = pm.Uniform(name='logMl', lower=self.minlogMl, upper=self.maxlogMl)
            Ml = pm.math.exp(logMl)
            logbs = Interpolated('logbs',self.logbarray,
                    self.plogb(self.logbarray), shape=(self.ndims))
            bs = pm.math.exp(logbs)
            if self.ndims == 1:
                vl = pm.TruncatedNormal(name='vl', mu=0., sigma=220, lower=0, upper=550.)
            else:
                vl = pm.Normal(name='vl', mu=0., sigma=220, shape=(self.nbims))
            alphall = pm.Deterministic('alphal', lm.alphal(Ml, bs, vl))

            ## background signal
            rmw = pm.TruncatedNormal(name='rmw', mu=AVE_RMW.value, sigma=SIGMA_RMW.value, lower=0., upper=2.)
            alphabb = pm.Deterministic('alphab', bm.alphab(rmw))

            ## set up obs
            alpha = alphall + alphabb

            ## likelihood
            obs = pm.Normal('obs', mu=alpha, sigma=WFIRST_SIGMA.value,
                    observed=self.data)

            ## run sampler
            trace = pm.sample(self.nsamples, tune=self.ntune, cores=self.ncores,
                    chains=self.nchains,
                    step=xo.get_dense_nuts_step())

        self.trace = trace

        ## save results
        pklpath = make_file_path(RESULTSDIR, [self.nsamples, self.ndims],
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
            self.data  = np.loadtxt(filepath)

        else:
            self.data = AccelData(nstars=self.nstars, ndims=self.ndims)

