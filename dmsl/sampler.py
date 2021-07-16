'''
MCMC sampler
'''

import emcee
import dill
import pickle
import numpy as np
import corner
import pandas as pd
import pymc3 as pm
from scipy.spatial import distance
from scipy.stats import trim1
import scipy.stats
from scipy.interpolate import UnivariateSpline
import exoplanet as xo
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric, Galactic
from collections import Counter

from dmsl.convenience import *
from dmsl.paths import *
from dmsl.accel_data import AccelData
from dmsl.survey import *
import dmsl.plotting as plot
import dmsl.galaxy_subhalo_dist as gsh
from dmsl.prior import *

import dmsl.lensing_model as lm
import dmsl.mass_profile as mp

RHO_DM = gsh.density(8.*u.kpc)


class Sampler():

    def __init__(self, nstars=None, nsamples=1000, nchains=8, ntune=1000,
            ndims=2, minlogMl=np.log10(1e0), maxlogMl=np.log10(1e8), MassProfile=mp.PointSource(**{'Ml' :
                1.e7*u.Msun}), SNRcutoff=10., survey=None, overwrite=True,
            usefraction=False):
        self.nstars=nstars
        self.nsamples=nsamples
        self.nchains=nchains
        self.ntune=ntune
        self.ndims=ndims
        self.minlogMl = minlogMl
        self.maxlogMl = maxlogMl
        self.massprofile = MassProfile
        self.SNRcutoff = 10.
        self.logradmax = 4 #pc
        self.logradmin = -4
        self.logconcmax = 4
        self.logconcmin = 0
        self.overwrite = overwrite
        self.usefraction = usefraction
        if survey is None:
            self.survey = Roman()
        else:
            self.survey = survey

        if nstars is None:
            print("""Defaulting to number of stars from survey...this might be
            too large for your computer memory...""")
            self.nstars = survey.nstars
            print(f"Nstars set to {self.nstars}")

        ## main
        self.load_data()
        self.load_prior()
        self.run_inference()
        if self.overwrite:
            self.make_diagnostic_plots()

    def run_inference(self):
        npar, nwalkers = 1, self.nchains
        ## add number of params according to sample type
        if self.massprofile.type != 'ps':
            npar += self.massprofile.nparams - 1
        if self.usefraction:
            npar += 1
        ## initial walker position
        p0 = np.random.rand(nwalkers, npar)*2+self.minlogMl
        if self.massprofile.type == 'gaussian':
            p0[:, 1] = np.random.rand(nwalkers)*(self.logradmax-self.logradmin) + self.logradmin
        if self.massprofile.type == 'nfw':
            p0[:, 1] = np.random.rand(nwalkers)*(self.logconcmax-self.logconcmin) + self.logconcmin
        if self.usefraction:
            p0[:, -1] = np.random.rand(nwalkers)

        ## set different likelihood if noise mcmc.
        if self.massprofile.type == 'noise':
            npar = 2
            p0 = np.random.rand(nwalkers, npar)*1.e-3
            sampler = emcee.EnsembleSampler(nwalkers, npar, self.lnlike_noise)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, npar, self.lnlike)

        ## run sampler
        sampler.run_mcmc(p0, self.ntune+self.nsamples, progress=True)

        samples = sampler.get_chain(discard=self.ntune, flat=True)
        print(f"90\% upper limit on Ml: {np.percentile(samples[:,0], 90)}")
        ## save samples to class
        self.sampler = sampler

        ## save results
        if self.overwrite:
            extra_string = f'samples_{self.survey.name}_{self.massprofile.type}'
            if self.usefraction == True:
                extra_string += '_frac'
            pklpath = make_file_path(RESULTSDIR, [np.log10(self.nstars),
                np.log10(self.nsamples), self.ndims], extra_string=
                extra_string, ext='.pkl')
            with open(pklpath, 'wb') as buff:
                pickle.dump(samples, buff)
            print('Wrote {}'.format(pklpath))
            extra_string = f'loglike_{self.survey.name}_{self.massprofile.type}'
            if self.usefraction == True:
                extra_string += '_frac'
            flatchain, __ = self.prune_chains()
            loglike = sampler.get_log_prob(discard=self.ntune, flat=True)
            pklpath = make_file_path(RESULTSDIR, [np.log10(self.nstars),
                np.log10(self.nsamples), self.ndims], extra_string=
                extra_string, ext='.pkl')
            with open(pklpath, 'wb') as buff:
                pickle.dump(loglike, buff)
            print('Wrote {}'.format(pklpath))
            extra_string = f'pruned_samples_{self.survey.name}_{self.massprofile.type}'
            if self.usefraction == True:
                extra_string += '_frac'
            pklpath = make_file_path(RESULTSDIR, [np.log10(self.nstars),
                np.log10(self.nsamples), self.ndims],
                extra_string=extra_string, ext='.pkl')
            with open(pklpath, 'wb') as buff:
                pickle.dump(np.array(flatchain), buff)
            print('Wrote {}'.format(pklpath))

    def logprior(self,pars):
        log10Ml = pars[0]
        if (log10Ml < self.minlogMl) or (log10Ml>self.maxlogMl):
            return -np.inf
        if self.usefraction:
            frac = pars[-1]
            if (frac < 0) or (frac>1.):
                return -np.inf
        if self.massprofile.type == 'gaussian':
            modelpars = self.massprofile.nparams  - 1
            for i in range(0, modelpars):
                logradius = pars[i+1]
                if logradius < self.logradmin or logradius > self.logradmax:
                    return -np.inf
        if self.massprofile.type == 'nfw':
            modelpars = self.massprofile.nparams  - 1
            for i in range(0, modelpars):
                conc = pars[i+1] ##logconc
                if (conc < self.logconcmin) or (conc > self.logconcmax):
                    return -np.inf
        return 0.

    def samplealphal(self, pars):
        ## Samples p(alpha_l | M_l)
        newmassprofile = self.make_new_mass(pars)
        if self.usefraction:
            f = pars[-1]
        else:
            f = 1.
        log10Ml = pars[0]

        Ml = 10**log10Ml
        nlens = int(np.ceil(f*Sampler.find_nlens(Ml, self.survey)))
        priorpdf = pdf(self.bs, a1=self.survey.fov_rad, a2=self.survey.fov_rad,
                n=nlens)
        priorpdfspline = UnivariateSpline(np.log10(self.bs[priorpdf>0]),
                np.log10(priorpdf[priorpdf>0]), ext='zeros', s=0)

        prior = pm.distributions.continuous.Interpolated.dist(self.bs,
                priorpdf)
        dists = self.rdist.rvs(self.nstars) * u.kpc
        beff = prior.random(size=self.nstars) * dists

        vl = np.array(pm.TruncatedNormal.dist(mu=0., sigma=220, lower=0,
                        upper=550.).random(size=self.nstars))
        bvec = np.zeros((self.nstars, 2))
        vvec = np.zeros((self.nstars, 2))
        if self.ndims==2:
            btheta = np.random.rand(self.nstars)* 2. * np.pi
            vtheta = np.random.rand(self.nstars)* 2. * np.pi
            bvec[:, 0] = beff * np.cos(btheta)
            bvec[:, 1] = beff * np.sin(btheta)
            vvec[:, 0] = vl * np.cos(vtheta)
            vvec[:, 1] = vl * np.sin(vtheta)
        else:
            ## default to b perp. to v. gives larger signal for NFW halo
            bvec[:,0] = beff
            vvec[:, 1] = vl
        ## add units back in because astropy is dumb...or really probably it's
        ## me.
        bvec *= u.kpc
        vvec *= u.km / u.s
        ## get alphal given sampled other params.
        alphal = lm.alphal(newmassprofile, bvec, vvec)
        ## if only sampling in 1D, get magnitude of vec.
        if self.ndims == 1:
            alphal = np.linalg.norm(alphal, axis=1)
        return alphal

    def snr_check(self, alphal0, pars, maxiter=100):
        alphanorm = np.linalg.norm(alphal0, axis=1)
        mask = alphanorm < self.survey.alphasigma*self.SNRcutoff
        alphal = alphal0[mask, :]
        count = 0
        while len(alphal) < self.nstars:
            newalphas = self.samplealphal(pars)
            alphanorm = np.linalg.norm(newalphas, axis=1)
            mask = alphanorm < self.survey.alphasigma*self.SNRcutoff
            alphal = np.append(alphal, newalphas[mask, :], axis=0)
            count +=1
            if count > maxiter:
                return -np.inf
        alphal = alphal[:self.nstars, :]
        return alphal

    def lnlike(self,pars):
        if ~np.isfinite(self.logprior(pars)):
            return -np.inf
        alphal = self.samplealphal(pars)
        if self.massprofile.type == 'ps':
            alphal = self.snr_check(alphal, pars)
        if np.any(np.isnan(alphal)):
            return -np.inf
        diff = alphal.value - self.data
        chisq = -0.5 * np.sum((diff)**2 / self.survey.alphasigma.value**2)
        chisq += -np.log(2 * np.pi * self.survey.alphasigma.value**2)
        prange(alphal)
        print(chisq)
        return chisq

    def lnlike_noise(self,pars):
        mu, logsig = pars
        loglike = np.sum(np.log(scipy.stats.norm(loc=mu,
            scale=10**logsig).pdf(self.data)))
        return loglike

    @staticmethod
    def find_nlens(Ml_, survey):
        volume = survey.fov_rad**2 * survey.maxdlens**3 / 3.
        mass = (RHO_DM*volume).to(u.Msun)
        nlens_k = mass.value/Ml_
        return np.ceil(nlens_k)

    def load_data(self):
        print('Creating data vector')
        self.data = AccelData(self.survey, nstars=self.nstars,
                ndims=self.ndims).data.to_numpy()

    def load_prior(self):
        print('Loading prior')
        rv = gsh.initialize_dist(target=self.survey.target,
                rmax=self.survey.maxdlens.to(u.kpc).value)
        self.rdist = rv
        self.bs = np.logspace(-8, np.log10(np.sqrt(2)*self.survey.fov_rad),
                1000)
        print('Prior loaded')

    def make_diagnostic_plots(self):
        flatchain = self.sampler.get_chain(flat=True, discard=self.ntune)
        plot.plot_emcee(flatchain,
                self.nstars, self.nsamples,self.ndims, self.massprofile,
                self.survey.name, self.usefraction)
        extra_string = f'chainsplot_{self.survey.name}_{self.massprofile.type}'
        if self.usefraction == True:
            extra_string += '_frac'
        outpath = make_file_path(RESULTSDIR, [np.log10(self.nstars),
            np.log10(self.nsamples), self.ndims],
            extra_string= extra_string,
            ext='.png')
        plot.plot_chains(self.sampler.get_chain(), outpath)
        extra_string = f'logprob_{self.survey.name}_{self.massprofile.type}'
        if self.usefraction == True:
            extra_string += '_frac'
        outpath = make_file_path(RESULTSDIR, [np.log10(self.nstars),
            np.log10(self.nsamples), self.ndims],
            extra_string=extra_string,
            ext='.png')
        plot.plot_logprob(self.sampler.get_log_prob(), outpath)

    def prune_chains(self, maxlengthfrac=0.05):
        chains = []
        loglikes = []
        for i in range(self.nchains):
            chain = self.sampler.get_chain()[self.ntune:, i,:]
            counts = Counter(chain[:,0])
            most = counts.most_common(1)
            if most[0][1] < int(self.nsamples*maxlengthfrac):
                loglikes.append(self.sampler.get_log_prob()[self.ntune:, i])
                chains.append(chain)
        flatchain = [item for sublist in chains for item in sublist]
        loglike = [item for sublist in loglikes for item in sublist]
        print(f"Chains have been pruned. {len(flatchain)/self.nsamples} chains remain")
        self.flatchain = flatchain
        return flatchain, loglike

    def make_new_mass(self,pars):
        mptype = self.massprofile.type
        kwargs = self.massprofile.kwargs

        if mptype == 'ps':
            kwargs['Ml'] = 10**pars[0]*u.Msun
            newmp = mp.PointSource(**kwargs)
        elif mptype == 'gaussian':
            kwargs['Ml'] = 10**pars[0]*u.Msun
            kwargs['R0'] = 10**pars[1]*u.pc
            newmp = mp.Gaussian(**kwargs)
        elif mptype == 'nfw':
            kwargs['Ml'] = 10**pars[0]*u.Msun
            kwargs['c200'] = 10**pars[1]
            newmp = mp.NFW(**kwargs)
        else:
            raise NotImplementedError("""Need to add this mass profile to
            sampler.""")
        return newmp
