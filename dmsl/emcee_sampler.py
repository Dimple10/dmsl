import emcee
import pickle
import numpy as np
import corner
import pandas as pd
import pymc3 as pm
from scipy.spatial import distance
import exoplanet as xo

from dmsl.convenience import *
from dmsl.constants import *
from dmsl.paths import *
from dmsl.convenience import *
from dmsl.prior_sampler import PriorSampler, find_nlens_np
from dmsl.accel_data import AccelData
from dmsl.star_field import StarField
import dmsl.plotting as plot

import dmsl.lensing_model as lm
import dmsl.background_model as bm


class Sampler():

    def __init__(self, nstars=1000, nbsamples=100, nsamples=1000, nchains=8,
            ntune=1000, ndims=1, nblog10Ml=3, minlogMl=np.log10(1e0),
            maxlogMl=np.log10(1e8), minlogb =-15., maxlogb = 1.,overwrite=True):
        self.nstars=nstars
        self.nbsamples=nbsamples
        self.nblog10Ml=nblog10Ml
        self.nsamples=nsamples
        self.nchains=nchains
        self.ntune=ntune
        self.ndims=ndims
        self.minlogMl = minlogMl
        self.maxlogMl = maxlogMl

        self.load_starpos()
        self.load_data()
        self.run_inference()
        self.make_diagnostic_plots()

    def run_inference(self):
        npar, nwalkers = self.ndims, self.nchains
        #p0 = np.random.rand(nwalkers, npar)*(self.maxlogMl-self.minlogMl)+self.minlogMl
        p0 = np.random.rand(nwalkers, npar)*1.0+self.minlogMl+2.
        print(p0)

        sampler = emcee.EnsembleSampler(nwalkers, npar, self.lnlike,
                moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])
        sampler.run_mcmc(p0, self.ntune+self.nsamples, progress=True)

        samples = sampler.get_chain(discard=self.ntune)

        ## save results
        pklpath = make_file_path(RESULTSDIR, [np.log10(self.nstars), np.log10(self.nsamples), self.ndims],
                extra_string='samples', ext='.pkl')
        with open(pklpath, 'wb') as buff:
            pickle.dump(samples, buff)
        print('Wrote {}'.format(pklpath))

        ## save samples to class
        self.sampler = sampler

    def logprior(self,pars):
        log10Ml = pars
        if (log10Ml < self.minlogMl) or (log10Ml>self.maxlogMl):
            return -np.inf
        else:
            return 0.

    def lnlike(self,pars):
        log10Ml = pars
        if ~np.isfinite(self.logprior(pars)):
            return -np.inf
        Ml = 10**log10Ml
        nlens = int(find_nlens_np(Ml))
        x= pm.Triangular.dist(lower=0, upper=FOV,c=FOV/2.).random(size=nlens)
        y= pm.Triangular.dist(lower=0, upper=FOV,c=FOV/2.).random(size=nlens)
        lenspos = np.vstack([x,y]).T
        dists = distance.cdist(lenspos, self.starpos)
        beff = np.sum(dists, axis=0)
        vl = pm.TruncatedNormal.dist(mu=0., sigma=220, lower=0,
                        upper=550.).random(1)
        if self.ndims==2:
            bstheta = pm.Uniform(name='bstheta', lower=0, upper=np.pi/2.)
            vltheta = pm.Uniform(name='vltheta', lower=0, upper=np.pi/2.)
        else:
            bstheta = None
            vltheta = None

        alphal = lm.alphal_np(Ml, beff, vl,
            btheta_=bstheta, vltheta_=vltheta)
        ## background signal
        ##rmw = pm.Exponential.dist(lam=1./RD_MW.value).random()
        ##if self.ndims==2:
        ##     rmwtheta = pm.Uniform(name='rmwtheta', lower=0.299,
        ##             upper=1.272) ## see test_angles.py in tests.
        ## else:
        ##     rmwtheta = None

        #alphab = bm.alphab(rmw,rtheta_=rmwtheta)
        sigalphab = bm.sig_alphab()

        diff = alphal - self.data
        chisq = np.sum(diff**2/(WFIRST_SIGMA.value**2+sigalphab.value**2))
        return -0.5*chisq

    def load_starpos(self):
        ## loads data or makes if file not found.
        fileinds = [np.log10(self.nstars)]
        filepath = make_file_path(STARPOSDIR, fileinds, ext='.dat',
                extra_string='nstars')
        if os.path.exists(filepath):
            self.starpos = pd.read_csv(filepath).to_numpy()
        else:
            StarField(nstars=self.nstars)
            self.starpos = pd.read_csv(filepath).to_numpy(dtype=np.float64)
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
            self.ndims],extra_string='posteriorplot', ext='.png')
        ##FIXME: should also thin out samples by half the autocorr time.
        plot.plot_emcee(self.sampler.get_chain(flat=True, discard=self.ntune), outpath)
        outpath = make_file_path(RESULTSDIR, [np.log10(self.nstars), np.log10(self.nsamples),
            self.ndims],extra_string='chainsplot', ext='.png')
        plot.plot_chains(self.sampler.get_chain(), outpath)
        outpath = make_file_path(RESULTSDIR, [np.log10(self.nstars), np.log10(self.nsamples),
            self.ndims],extra_string='logprobplot', ext='.png')
        plot.plot_logprob(self.sampler.get_log_prob(), outpath)

