'''
MCMC sampler
'''
import time

import emcee
import dill
import pickle
import numpy as np
#import corner
import pandas as pd
#TEST
import pymc as pm
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
from time import *
RHO_DM = gsh.density(8.*u.kpc)


class Sampler():

    def __init__(self, nstars=None, nsamples=1000, nchains=8, ntune=1000,
            ndims=2, minlogMl=np.log10(1e0), maxlogMl=np.log10(1e8), MassProfile=mp.PointSource(**{'Ml' :
                1.e7*u.Msun}),MassFunction=None, SNRcutoff=10., survey=None, overwrite=True,
            usefraction=False):
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
        if npar_mf == 0: #Only Mass-Profile tested
            #print('Initializing mass profile')
            p0 = np.random.rand(nwalkers, npar_mp)*2+self.minlogMl #-- Kris needed to not get stuck sampling
            if self.massprofile.type == 'gaussian':
                p0[:, 1] = np.random.rand(nwalkers)*(self.logradmax-self.logradmin) + self.logradmin
            elif self.massprofile.type == 'nfw':
                p0[:, 1] = np.random.rand(nwalkers)*(self.logconcmax-self.logconcmin) + self.logconcmin
            if self.usefraction:
                p0[:, -1] = np.random.rand(nwalkers)
        else:
            #print('Initializing mass profile and mass function')
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

        ## set different likelihood if noise mcmc.
        if self.massprofile.type == 'noise':
            npar = 2
            p0 = np.random.rand(nwalkers, npar)*1.e-3
            sampler = emcee.EnsembleSampler(nwalkers, npar, self.lnlike_noise)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, npar, self.lnlike)

        # Burn-in
        # state = sampler.sample(p0, 100)
        # sampler.reset()

        ##Testing auto-correlation time for samples
        max_n = self.ntune+self.nsamples
        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(max_n)
        old_tau = np.inf
        # Now we'll sample for up to max_n steps
        print("Sampling..")
        ctr = 0
        # start = time()
        #sampler.run_mcmc(p0, max_n, progress=True)
        # for sample in sampler.sample(p0, iterations=max_n, progress=True):
        #     # Only check convergence every 100 steps
        #     if sampler.iteration % 100:
        #         continue
        #
        #     # Compute the autocorrelation time so far
        #     # Using tol=0 means that we'll always get an estimate even
        #     # if it isn't trustworthy
        #     tau = sampler.get_autocorr_time(tol=0)
        #     autocorr[index] = np.mean(tau)
        #     index += 1
        #
        #     # Check convergence
        #     converged = np.all(tau * 50 < sampler.iteration)
        #     converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        #     if converged and ctr==0:
        #         print('Converged at..', sampler.iteration)
        #         ctr+=1
        #     old_tau = tau
        # end = time()
        # serial_time = end-start
        # print("Serial took {0:.1f} seconds".format(serial_time))
        # print("Sampling..")
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, npar, self.lnlike, pool=pool)
            start = time()
            sampler.run_mcmc(p0, max_n, progress=True)
            end = time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))
            #print("{0:.1f} times faster than serial".format(serial_time / multi_time))
        ## run sampler
        #print("Sampling..")
        #sampler.run_mcmc(p0, self.ntune+self.nsamples, progress=True)
        # burnin = int(2 * np.max(old_tau))
        # thin = int(0.5 * np.min(old_tau))
        # print("burn-in: {0}".format(burnin))
        # print("thin: {0}".format(thin))

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
            p_mf[:,i] = np.random.rand(nwalkers) + self.massfunction.param_range[self.massfunction.param_names[i]][0]
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
        ## Samples p(alpha_l | M_l)
        #print('In samplealpha')
        if self.usefraction:
            f = pars[-1]
        else:
            f = 1.
        if not self.massfunction:
            newmassprofile = self.make_new_mass(pars)
            log10Ml = pars[0]

            Ml = 10**log10Ml
            nlens = int(np.ceil(f*Sampler.find_nlens(Ml, self.survey)))
            priorpdf = pdf(self.bs, a1=self.survey.fov_rad, a2=self.survey.fov_rad,
                           n=nlens)
        else:
            newmassprofile, newmassfunction = self.make_new_mass(pars)
            nlens = np.ceil(f*newmassfunction.n_l)
            if sum(nlens) ==0:
                print('no lens in sampler 2')
                nlens[0] = 1
            priorpdf = pdf(self.bs, a1=self.survey.fov_rad, a2=self.survey.fov_rad,
                n=sum(nlens))
        #print(priorpdf, self.bs, sum(nlens))
        if np.any(np.isnan(priorpdf)):
            return -np.inf

        priorpdfspline = UnivariateSpline(np.log10(self.bs[priorpdf>0]),
                np.log10(priorpdf[priorpdf>0]), ext='zeros', s=0)
        dists = self.rdist.rvs(self.nstars) * u.kpc
        x = np.log10(self.bs)
        y = 10**priorpdfspline(np.log10(self.bs))
        sci_s = scipy.interpolate.interp1d(x, y, fill_value='extrapolate')
        sci = sci_s(x)
        temp = np.random.choice(x, self.nstars, p=sci / sum(sci)) #* dists
        self.beff_avg.append(np.average(temp))
        beff = 10**(np.ones(temp.shape)*np.average(temp)) * dists
        # beff = 10 ** (temp) * dists
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
            ## default to b perp. to v. gives larger signal for NFW halo
            bvec[:,0] = beff
            vvec[:, 1] = vl
        ## add units back in because astropy is dumb...or really probably it's
        ## me.
        bvec *= u.kpc
        vvec *= u.km / u.s
        ## get alphal given sampled other params.

        alphal = lm.alphal(newmassprofile, bvec, vvec)
        #print('Outside lensing model')
        ## if only sampling in 1D, get magnitude of vec.
        if self.ndims == 1:
            alphal = np.linalg.norm(alphal, axis=1)
        return alphal

    def snr_check(self, alphal0, pars, maxiter=100): #FIXME handle array as alphal0 -- only for ps
        if ~np.any(np.isfinite(alphal0)): return "error"

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
                return "error"
        alphal = alphal[:self.nstars, :]
        return alphal

    def lnlike(self,pars):
        if ~np.isfinite(self.logprior(pars)):
            return -np.inf
        alphal = self.samplealphal(pars)
        #FIXME
        if self.massprofile.type == 'ps':
            # if ps, need to do snr check and re-sample if any have too high snr. this stops walkers from getting too stuck.
             alphal = self.snr_check(alphal, pars)
             if alphal == "error":
                 return -np.inf
        if np.any(np.isnan(alphal)): #FIXME Should not be needed!
            return -np.inf
        try:
            #print('Trying')
            diff = alphal.value - self.data
        except:
            return -np.inf
        chisq = -0.5 * np.sum((diff)**2 / self.survey.alphasigma.value**2 -np.log(2 * np.pi * self.survey.alphasigma.value**2))
        self.chisq.append([pars,chisq]) #specific to 1 par case
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
        if (self.massfunction.Name == 'WDM Stream' or self.massfunction.Name == 'WDM Lensing'):
            self.data = AccelData(self.survey, nstars=self.nstars,
                ndims=self.ndims,wdm=True).data.to_numpy()
        else:
            self.data = AccelData(self.survey, nstars=self.nstars,
                                  ndims=self.ndims).data.to_numpy()

    def load_prior(self):
        print('Loading prior')
        rv = gsh.initialize_dist(target=self.survey.target,
                rmax=self.survey.maxdlens.to(u.kpc).value)
        self.rdist = rv
        self.bs = np.logspace(-8, np.log10(np.sqrt(2)*self.survey.fov_rad),
                self.nstars)
        #print(self.bs)
        print('Prior loaded')

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

    def make_new_mass(self,pars): #FIXME For mf does this need to be mf + mp? or just mf?
        #print('In make new mass')
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
                newmf = mf.PowerLaw(m_l=self.massfunction.m_l,logM_0=logM0, logalpha=logalpha)
            elif mftype == 'Tinker':
                a = pars[i+0]
                b = pars[i+1]
                c = pars[i+2]
                #k_b = pars[i+4]
                #n_b = pars[i+5]
                #k_s = pars[i+6]
                newmf = mf.Tinker(m_l=self.massfunction.m_l,a= a, b= b, c= c)#, k_b=k_b, n_b=n_b, k_s=k_s)
            elif mftype == 'CDM':
                #loga = pars[i+0]
                b = pars[i+0]
                logc = pars[i+1]
                #print('before CDM makenewmass')
                newmf = mf.CDM_Test(m_l=self.massfunction.m_l, b = b,logc = logc)
                #print('after CDM makenewmass')
            elif mftype == 'WDM Stream':
                logmwdm = pars[i+0]
                gamma = pars[i+1]
                beta = pars[i+2]
                # loga_cdm = pars[i+3]
                # b_cdm = pars[i+4]
                # logc_cdm =pars[i+5]
                newmf = mf.WDM_stream(m_l=self.massfunction.m_l,logmwdm=logmwdm,gamma=gamma, beta=beta)#, loga_cdm=loga_cdm,b_cdm=b_cdm,logc_cdm=logc_cdm)
            elif mftype == 'WDM Lensing':
                mwdm = pars[i+0]
                beta = pars[i+1]
                newmf = mf.WDM_lensing(m_l=self.massfunction.m_l,mwdm=mwdm, beta=beta)
            elif mftype == 'Press Schechter':
                del_crit = pars[i+0]
                #b = pars[i+1]
                #logc = pars[i+2]
                newmf = mf.PressSchechter_test(m_l=self.massfunction.m_l,del_crit = del_crit)#,b = b,logc = logc)
            elif mftype == 'PBH':
                #print('inside pbh make new mass')
                logf_pbh = pars[i+0]
                #b = pars[i+1]
                #logc = pars[i+2]
                newmf = mf.PBH(m_l=self.massfunction.m_l,logf_pbh = logf_pbh)
            #else:
             #   raise NotImplementedError("""Need to add this mass function to
              #  sampler.""")
            newmp = []
            n_lens = sum(newmf.n_l.astype(int))
            if mptype == 'ps':
                if n_lens == 1:
                    index = np.nonzero(newmf.n_l.astype(int))
                    kwargs['Ml'] = int(newmf.m_l[index[0]]) * u.Msun
                    newmp = mp.PointSource(**kwargs)
                else:
                    for newmf_ml, num_lenses in zip(newmf.m_l, newmf.n_l.astype(int)):
                        kwargs['Ml'] = newmf_ml*u.Msun
                        newmp.extend([mp.PointSource(**kwargs)for _ in range(num_lenses)])
                    if len(newmp) == 0:
                        print('no lens makenewmass')
                        kwargs['Ml'] = int(newmf.m_l[0]) * u.Msun
                        newmp = mp.PointSource(**kwargs)

            elif mptype == 'gaussian':
                kwargs['R0'] = 10**pars[0]*u.pc
                if n_lens ==1:
                    index = np.nonzero(newmf.n_l)
                    kwargs['Ml'] = newmf.m_l[index[0]] * u.Msun
                    newmp = mp.Gaussian(**kwargs)
                else:
                    for newmf_ml, num_lenses in zip(newmf.m_l, newmf.n_l):
                        kwargs['Ml'] = newmf_ml * u.Msun
                        newmp.extend([mp.Gaussian(**kwargs) for _ in range(num_lenses)])
                    if len(newmp) == 0:
                        kwargs['Ml'] = newmf.m_l[0] * u.Msun
                        newmp = mp.Gaussian(**kwargs)
            elif mptype == 'nfw':
                kwargs['c200'] = 10**pars[0]
                #print('in nfw makenewmass')
                if n_lens == 1: ##Case with only one lens, add the mass profile corresponding to it as an object not list
                    index = np.nonzero(newmf.n_l.astype(int))
                    kwargs['Ml'] = int(newmf.m_l[index[0]]) * u.Msun
                    newmp = mp.NFW(**kwargs)
                else:
                    #print(newmf.n_l)
                    for ind123, (newmf_ml, num_lenses) in enumerate(zip(newmf.m_l, newmf.n_l.astype(int))):
                        #print(ind123)
                        kwargs['Ml'] = newmf_ml * u.Msun
                        newmp.extend([mp.NFW(**kwargs) for _ in range(num_lenses)])
                    if len(newmp) == 0: ##Case where there are no lens, assume 1 exists in the lowest mass bin
                        kwargs['Ml'] = int(newmf.m_l[0]) * u.Msun
                        newmp = mp.NFW(**kwargs)

            else:
                raise NotImplementedError("""Need to add this mass profile/mass function to
                sampler.""")
            # print(type(newmp[0]), np.size(newmp))
            #print('done w makenewmass')
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
