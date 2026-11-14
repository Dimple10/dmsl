'''
MCMC sampler
'''
import os
import time

import emcee
import dill
import pickle
import numpy as np
#import corner
import pandas as pd
#TEST
import pymc3 as pm
import matplotlib.cm
from scipy.spatial import distance
from scipy.stats import trim1
import scipy.stats
import scipy.interpolate
from scipy.interpolate import UnivariateSpline
import exoplanet_core as xo
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
import dmsl.mass_function as mf
from multiprocessing import Pool
import time
RHO_DM = gsh.density(8.*u.kpc)

def _seed_worker():
    '''Forked workers inherit the parent's RNG state, so every worker would draw the same random numbers.
    Reseed numpy and the stdlib random module (used in mass_function) from OS entropy.'''
    import os, random
    seed = int.from_bytes(os.urandom(4), 'little')
    np.random.seed(seed)
    random.seed(seed)

class Sampler():

    def __init__(self, nstars=None, nsamples=1000, nchains=8, ntune=1000,
            ndims=2, minlogMl=np.log10(1e0), maxlogMl=np.log10(1e6), MassProfile=mp.PointSource(**{'Ml' :
                1.e7*u.Msun}),MassFunction=None, SNRcutoff=10., survey=None, overwrite=True,
            usefraction=False, bs_upper_factor=4):
        self.nstars=nstars
        self.nsamples=nsamples
        self.nchains=nchains
        self.ntune=ntune
        self.ndims=ndims
        self.minlogMl = minlogMl
        self.maxlogMl = maxlogMl
        self.massprofile = MassProfile
        self.massfunction = MassFunction
        self.SNRcutoff = SNRcutoff
        self.logradmax = 4 #pc
        self.logradmin = -4
        self.logconcmax = 8
        self.logconcmin = 0
        self.overwrite = overwrite
        self.usefraction = usefraction # Also sameple for fractional dark matter, just multiple nlens with f_chi
        self.bs_upper_factor = bs_upper_factor
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
        #self.testing = self.make_new_mass((13,-4.49,-1.9,7.41))
        self.chisq = []
        self.beff_avg = []
        self.run_inference()
        if self.overwrite:
            self.make_diagnostic_plots()

    def run_inference(self):
        npar_mf, npar_mp, nwalkers = 0, 0, self.nchains
        ## add number of params according to sample type
        if self.massfunction:
            npar_mf += self.massfunction.nparams
            npar_mp += self.massprofile.nparams - 1 #Remove Ml
        else:
            npar_mp += self.massprofile.nparams

        if self.usefraction:
            npar_mp += 1

        npar = npar_mf + npar_mp
        print("Total pars:", npar)

        ## initial walker position
        if npar_mf == 0:
            p0 = np.random.rand(nwalkers, npar_mp)*2+self.minlogMl #-- Kris needed to not get stuck sampling
            if self.massprofile.type == 'gaussian':
                p0[:, 1] = np.random.rand(nwalkers)*(self.logradmax-self.logradmin) + self.logradmin
            elif self.massprofile.type == 'nfw':
                p0[:, 1] = np.random.rand(nwalkers)*(self.logconcmax-self.logconcmin) + self.logconcmin
            if self.usefraction:
                p0[:, -1] = np.random.rand(nwalkers)
        else:
            p0 = np.random.rand(nwalkers, npar)
            name = self.massprofile.type
            if(name == 'ps'):
                p0[:, 0:] = self.initialize()
            else:
                p0[:, 1:] = self.initialize() #FIXME Possible shape issue if use_fraction
            if name == 'gaussian':
                p0[:, 0] = np.random.rand(nwalkers) * (self.logradmax - self.logradmin) + self.logradmin
            elif name == 'nfw':
                p0[:, 0] = np.random.rand(nwalkers) * (self.logconcmax - self.logconcmin) + self.logconcmin
            if self.usefraction: #FIXME Should work just fine -- possible shape error
                p0[:, -1] = np.random.rand(nwalkers)

        moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)] #All moves are from Claude to speadup stuff
        ## set different likelihood if noise mcmc.
        if self.massprofile.type == 'noise':
            npar = 2
            p0 = np.random.rand(nwalkers, npar)*1.e-3
            sampler = emcee.EnsembleSampler(nwalkers, npar, self.lnlike_noise, moves=moves)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, npar, self.lnlike, moves=moves)

        ##Testing auto-correlation time for samples
        max_n = self.ntune+self.nsamples

        index = 0
        autocorr = np.empty(max_n)
        old_tau = np.inf
        # Now we'll sample for up to max_n steps
        print("Sampling..")
        nproc = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))   # cluster: cores given to the job, local: all cores
        nproc = max(1, min(nproc, self.nchains // 2))                         # emcee evaluates half the walkers at a time
        print(f'Pool with {nproc} processes')
        with Pool(processes=nproc, initializer=_seed_worker) as pool:
        # with Pool(initializer=_seed_worker) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, npar, self.lnlike, pool=pool, moves=moves)
            start = time.time()
            sampler.run_mcmc(p0, max_n, progress=True)
            end = time.time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))

        try:
            tau = sampler.get_autocorr_time()
            print(f"Autocorrelation times: {tau}")
            converged = np.all(max_n > 50 * tau)
            print(f"Has the chain converged? {converged}")
        except emcee.autocorr.AutocorrError:
            print(
                "Chains are too short or not well-behaved for reliable autocorrelation time estimation. Consider running for more steps.")

        samples = sampler.get_chain(discard=self.ntune, flat=True)
        print(f"95\% upper limit on c200: {np.percentile(samples[:, 0], 95)}")

        ## save samples to class
        self.sampler = sampler

        ## save results
        if self.overwrite:
            if self.massfunction:
                extra_string = f'samples_{self.survey.name}_{self.massprofile.type}_{self.massfunction.Name}'
            else:
                extra_string = f'samples_{self.survey.name}_{self.massprofile.type}'
            if self.usefraction == True:
                extra_string += '_frac'
            # print("length:", len(extra_string))
            pklpath = make_file_path(RESULTSDIR, [np.log10(self.nstars),
                np.log10(self.nsamples), self.ndims], extra_string=
                extra_string, ext='.pkl')
            with open(pklpath, 'wb') as buff:
                pickle.dump(samples, buff)
            print('Wrote {}'.format(pklpath))
            if self.massfunction:
                extra_string = f'loglike_{self.survey.name}_{self.massprofile.type}_{self.massfunction.Name}'
            else:
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
            if self.massfunction:
                extra_string = f'pruned_samples_{self.survey.name}_{self.massprofile.type}_{self.massfunction.Name}'
            else:
                extra_string = f'pruned_samples_{self.survey.name}_{self.massprofile.type}'
            if self.usefraction == True:
                extra_string += '_frac'
            pklpath = make_file_path(RESULTSDIR, [np.log10(self.nstars),
                np.log10(self.nsamples), self.ndims],
                extra_string=extra_string, ext='.pkl')
            with open(pklpath, 'wb') as buff:
                pickle.dump(np.array(flatchain), buff)
            print('Wrote {}'.format(pklpath))

    def initialize(self):
        nwalkers = self.nchains
        npar_mf = self.massfunction.nparams
        p_mf = np.empty(shape=(nwalkers, npar_mf))
        for i in range(npar_mf):
            lo, hi = self.massfunction.param_range[self.massfunction.param_names[i]]
            p_mf[:,i] = np.random.uniform(lo, hi, nwalkers)
            # p_mf[:,i] = np.random.rand(nwalkers) + self.massfunction.param_range[self.massfunction.param_names[i]][0]
        return p_mf

    def logprior(self,pars): ##Just checks if params to be sampled are within physical range
        if not self.massfunction:
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
            elif self.massprofile.type == 'nfw':
                modelpars = self.massprofile.nparams  - 1
                for i in range(0, modelpars):
                    conc = pars[i+1] ##logconc
                    if (conc < self.logconcmin) or (conc > self.logconcmax):
                        return -np.inf
        else:
            if self.usefraction:
                frac = pars[-1] #FIXME Figure out a final index for frac in mf case
                if (frac < 0) or (frac>1.):
                    return -np.inf
            if self.massprofile.type == 'gaussian':
                logradius = pars[0]
                if logradius < self.logradmin or logradius > self.logradmax:
                    return -np.inf
            elif self.massprofile.type == 'nfw':
                conc = pars[0] ##logconc
                if (conc < self.logconcmin) or (conc > self.logconcmax):
                    # print("True inside logprior nfw")
                    return -np.inf
            # #Mass Function
            for i in range(self.massfunction.nparams):
                min = self.massfunction.param_range[self.massfunction.param_names[i]][0]
                max = self.massfunction.param_range[self.massfunction.param_names[i]][1]
                if self.massprofile.type!='ps':
                    par = pars[i+1]
                else:
                    par = pars[i]
                if par < min or par > max:
                    #print("outside param range for:", self.massfunction, self.massfunction.param_names[i])
                    return -np.inf
        return 0.

    def samplealphal(self, pars):
        if self.usefraction:
            f = pars[-1]
        else:
            f = 1.
        if not self.massfunction:
            newmassprofile = self.make_new_mass(pars)
            log10Ml = pars[0]

            Ml = 10**log10Ml
            N = int(np.ceil(f*Sampler.find_nlens(Ml, self.survey)))
        else:
            result = self.make_new_mass(pars)
            if isinstance(result, (float,str)):
                return result
            newmassprofile, newmassfunction = result
            N = int(np.sum(np.ceil(f*newmassfunction.n_l)))

        dists = self.rdist.rvs(self.nstars)                          # d_L per star [kpc]
        beta = sample_beta(self.nstars, self.survey.fov_rad, N)      # beta per star [rad]
        beff = beta * dists                                          # b per star [kpc]

        vl = scipy.stats.truncnorm.rvs(a=0, b=550./220, loc=0.,scale =220, size=self.nstars)
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
            bvec[:,0] = beff
            vvec[:, 1] = vl
        bvec *= u.kpc
        vvec *= u.km / u.s
        # alphal = lm.alphal(newmassprofile, bvec, vvec)
        eff = np.clip(1. - dists / self.survey.dstars.to(u.kpc).value, 0., None)   # D_LS/D_S per star
        alphal = lm.alphal(newmassprofile, bvec, vvec, eff=eff)
        if self.ndims == 1:
            alphal = np.linalg.norm(alphal, axis=1)
        return alphal

    def snr_check(self, alphal0, pars, maxiter=100): #FIXME handle array as alphal0 -- only for ps
        if isinstance(alphal0, (float, str)) or not np.all(np.isfinite(alphal0)):
            return "error"

        alphanorm = np.linalg.norm(alphal0, axis=1)
        mask = alphanorm < self.survey.alphasigma * self.SNRcutoff
        alphal = alphal0[mask, :]
        count = 0
        while len(alphal) < self.nstars:
            count += 1
            if count > maxiter:                       # check first, so continue can't bypass it
                return "error"
            newalphas = self.samplealphal(pars)
            if isinstance(newalphas, (float, str)) or not np.all(np.isfinite(newalphas)):
                continue                           # rejected draw: retry
            alphanorm = np.linalg.norm(newalphas, axis=1)
            mask = alphanorm < self.survey.alphasigma * self.SNRcutoff
            alphal = np.append(alphal, newalphas[mask, :], axis=0)
        return alphal[:self.nstars, :]

    def lnlike(self,pars):
        if ~np.isfinite(self.logprior(pars)):
            return -np.inf
        alphal = self.samplealphal(pars)
        if isinstance(alphal, str):                  # "nolens": the model predicts no signal
            sig = self.survey.alphasigma.value
            return -0.5 * np.sum(self.data**2 / sig**2 - np.log(2*np.pi*sig**2))
        if isinstance(alphal, float):
            return -np.inf
        
        #FIXME
        if self.massprofile.type == 'ps':
            # if ps, need to do snr check and re-sample if any have too high snr. this stops walkers from getting too stuck.
             alphal = self.snr_check(alphal, pars)
             if isinstance(alphal, str):
                return -np.inf
        if np.any(np.isnan(alphal)): #FIXME Should not be needed!
            return -np.inf
        try:
            diff = alphal.value - self.data
        except:
            return -np.inf
        chisq = -0.5 * np.sum((diff)**2 / self.survey.alphasigma.value**2 -np.log(2 * np.pi * self.survey.alphasigma.value**2))
        # self.chisq.append([pars,chisq]) #specific to 1 par case
        return chisq

    def lnlike_noise(self,pars):
        mu, logvar = pars
        loglike = np.sum(np.log(scipy.stats.norm(loc=mu,
            scale=np.sqrt(np.exp(logvar))).pdf(self.data)))
        return loglike

    @staticmethod
    def find_nlens(Ml_, survey):
        volume = survey.fov_rad**2 * survey.maxdlens**3 / 3.
        mass = (RHO_DM*volume).to(u.Msun)
        nlens_k = mass.value/Ml_
        return np.ceil(nlens_k)

    def load_data(self):
        print('Creating data vector')
        #if False: #Switch to if statement below for CDM as null hypothesis for WDM
        if self.massfunction!=None:
            if (self.massfunction.Name == 'WDM Stream' or self.massfunction.Name == 'WDM Lensing'):
                self.data = AccelData(self.survey, nstars=self.nstars,
                    ndims=self.ndims,wdm=True).data.to_numpy()
            else:
                self.data = AccelData(self.survey, nstars=self.nstars,
                                      ndims=self.ndims).data.to_numpy()
        else:
            self.data = AccelData(self.survey, nstars=self.nstars,
                                  ndims=self.ndims).data.to_numpy()

    def load_prior(self):
        print('Loading prior')
        rv = gsh.initialize_dist(target=self.survey.target,
                rmax=self.survey.maxdlens.to(u.kpc).value)
        self.rdist = rv
        self.bs = np.logspace(
                np.log10(1e-8), 
                np.log10(np.sqrt(2) * self.survey.fov_rad),
                self.nstars)
        print(f'Prior loaded (bs_upper_factor={self.bs_upper_factor}, bs=[{self.bs.min():.2e}, {self.bs.max():.2e}])')

    def make_diagnostic_plots(self):
        #flatchain = self.sampler.get_chain(flat=True, discard=self.ntune)
        flatchain, __ = self.prune_chains()
        if len(flatchain) == 0:
            return 'Null flatchain'
        print('flatchain:', np.shape(flatchain))
        if self.massfunction:
            plot.plot_emcee(flatchain,
                    self.nstars, self.nsamples,self.ndims, self.massprofile,
                    self.survey.name, self.usefraction,massfunction=self.massfunction)
            extra_string = f'chainsplot_{self.survey.name}_{self.massprofile.type}_{self.massfunction.Name}'
        else:
            plot.plot_emcee(flatchain,
                            self.nstars, self.nsamples, self.ndims, self.massprofile,
                            self.survey.name, self.usefraction)
            extra_string = f'chainsplot_{self.survey.name}_{self.massprofile.type}'

        if self.usefraction == True:
            extra_string += '_frac'
        outpath = make_file_path(RESULTSDIR, [np.log10(self.nstars),
            np.log10(self.nsamples), self.ndims],
            extra_string= extra_string,
            ext='.png')
        plot.plot_chains(self.sampler.get_chain(), outpath) #Shape: (samples, chains, params)
        if self.massfunction:
            extra_string = f'logprob_{self.survey.name}_{self.massprofile.type}_{self.massfunction.Name}'
        else:
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
        # print('In make new mass 3')
        mptype = self.massprofile.type
        kwargs = self.massprofile.kwargs
        i = 1
        if self.massfunction:
            if mptype == 'ps':
                i = 0
            mftype = self.massfunction.Name
            if mftype == 'PowerLaw':
                logalpha = pars[i+0]
                logM0 = pars[i+1]
                newmf = mf.PowerLaw(m_l=self.massfunction.m_l,logM_0=logM0, logalpha=logalpha,sur=self.survey)
            elif mftype == 'Tinker':
                logkb = pars[i+0]*1/u.Mpc
                nb = pars[i+1]
                newmf = mf.Tinker(m_l=self.massfunction.m_l,logkb=logkb,nb=nb,sur=self.survey)
            elif mftype == 'TinkerMishra':
                logkb = pars[i+0]*1/u.Mpc
                nb = pars[i+1]
                newmf = mf.Tinker_Mishra(m_l=self.massfunction.m_l,logkb=logkb,nb=nb,sur=self.survey)#, k_b=k_b, n_b=n_b, k_s=k_s)
            elif mftype == 'CDM':
                b = pars[i+0]
                newmf = mf.CDM_Test(m_l=self.massfunction.m_l, b = b,sur=self.survey)
            elif mftype == 'WDM Stream':
                logmwdm = pars[i+0]
                gamma = pars[i+1]
                beta = pars[i+2]
                newmf = mf.WDM_stream(m_l=self.massfunction.m_l,logmwdm=logmwdm,gamma=gamma, beta=beta,sur=self.survey)#, loga_cdm=loga_cdm,b_cdm=b_cdm,logc_cdm=logc_cdm)
            elif mftype == 'PBH':
                logmass = pars[i+0]
                sigma = pars[i+1]
                newmf = mf.PBH_Gaussian(logmass = logmass,sigma = sigma, sur=self.survey)

            # newmp = []
            # n_l = np.asarray(newmf.n_l).astype(int)
            newmp = []
            mlens = getattr(newmf, 'm_lens', newmf.m_l)   # lens mass of each n_l entry (bin centres)
            n_l = np.asarray(newmf.n_l).astype(int)
            n_lens = n_l.sum()
            if n_lens == 0:
                return "nolens"
            nonzero = np.nonzero(n_l)[0]   

            if n_lens>1:
                nonzero = np.nonzero(newmf.n_l)
                sampling_pool = np.repeat(nonzero, newmf.n_l[nonzero])
                ran_samp = np.random.choice(sampling_pool, self.nstars, replace=True)

            if mptype == 'ps':
                if n_lens == 1:
                    index = np.nonzero(newmf.n_l.astype(int))
                    # kwargs['Ml'] = int(newmf.m_l[index[0]]) * u.Msun ##Check if this should change when PBH isnt changed yet
                    kwargs['Ml'] = float(mlens[index[0]][0]) * u.Msun
                    newmp = mp.PointSource(**kwargs)
                else:
                    for index in ran_samp:
                        kwargs['Ml'] = mlens[index] * u.Msun
                        newmp.extend([mp.PointSource(**kwargs)])

            elif mptype == 'gaussian':
                kwargs['R0'] = 10**pars[0]*u.pc
                if n_lens ==1:
                    index = np.nonzero(newmf.n_l)
                    kwargs['Ml'] = float(mlens[index[0]][0]) * u.Msun
                    newmp = mp.Gaussian(**kwargs)
                else:
                    for index in ran_samp:
                        kwargs['Ml'] = mlens[index] * u.Msun
                        newmp.extend([mp.Gaussian(**kwargs)])
            elif mptype == 'nfw':
                kwargs['c200'] = 10**pars[0]
                if n_lens == 1: ##Case with only one lens, add the mass profile corresponding to it as an object not list
                    index = np.nonzero(newmf.n_l.astype(int))
                    kwargs['Ml'] = float(mlens[index[0]][0]) * u.Msun
                    newmp = mp.NFW(**kwargs)
                else:
                    for index in ran_samp:
                        kwargs['Ml'] = mlens[index] * u.Msun
                        newmp.extend([mp.NFW(**kwargs)])
            else:
                raise NotImplementedError("""Need to add this mass profile/mass function to
                sampler.""")
            return newmp, newmf #Array of mp
        else:
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
