'''
MCMC sampler
'''

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
            if self.massprofile.type == 'nfw':
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
            if name == 'nfw':
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
        for sample in sampler.sample(p0, iterations=max_n, progress=True):
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence
            converged = np.all(tau * 50 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged and ctr==0:
                print('Converged at..', sampler.iteration)
                ctr+=1
            old_tau = tau

        ## run sampler
        #print("Sampling..")
        #sampler.run_mcmc(p0, self.ntune+self.nsamples, progress=True)
        # burnin = int(2 * np.max(old_tau))
        # thin = int(0.5 * np.min(old_tau))
        # print("burn-in: {0}".format(burnin))
        # print("thin: {0}".format(thin))

        samples = sampler.get_chain(discard=self.ntune, flat=True)
        print(f"95\% upper limit on c200: {np.percentile(samples[:, 0], 95)}")
        print(f"95\% upper limit on loga: {np.percentile(samples[:,1], 95)}")
        print(f"95\% upper limit on b: {np.percentile(samples[:, 2], 95)}")
        print(f"95\% upper limit on logc: {np.percentile(samples[:, 3], 95)}")
        # print(f"95\% upper limit on loga_cdm: {np.percentile(samples[:, 4], 95)}")
        # print(f"95\% upper limit on b_cdm: {np.percentile(samples[:, 5], 95)}")
        # print(f"95\% upper limit on logc_cdm: {np.percentile(samples[:, 6], 95)}")
        #print(f"95\% upper limit on k_s: {np.percentile(samples[:, 6], 95)}")

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
            if self.massprofile.type == 'nfw':
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
                par = pars[i+1]
                if par < min or par > max:
                    #print("outside param range for:", self.massfunction, self.massfunction.param_names[i])
                    return -np.inf
        return 0.

    def samplealphal(self, pars):
        ## Samples p(alpha_l | M_l)
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
            newmassprofile, newmassfunction = self.make_new_mass(pars) #FIXME Array of new mass profiles + new mass_function
            nlens = np.ceil(f*newmassfunction.n_l[2:])
            priorpdf = pdf(self.bs, a1=self.survey.fov_rad, a2=self.survey.fov_rad,
                n=sum(nlens))

        if np.any(np.isnan(priorpdf)):
            return -np.inf

        priorpdfspline = UnivariateSpline(np.log10(self.bs[priorpdf>0]),
                np.log10(priorpdf[priorpdf>0]), ext='zeros', s=0)
        #prior = pm.
        #prior = pm.distributions.continuous.Interpolated.dist(self.bs,10**priorpdfspline(np.log10(self.bs)))
        dists = self.rdist.rvs(self.nstars) * u.kpc
        x = self.bs
        y = 10**priorpdfspline(np.log10(self.bs))
        sci_s = scipy.interpolate.interp1d(x, y, fill_value='extrapolate')
        sci = sci_s(x)
        beff_p = np.random.choice(x, self.nstars, p=sci / sum(sci)) * dists
        beff = 0.001#*dists
        #beff = 0.001 #pm.draw(prior, draws=self.nstars)*dists #prior.random(size=self.nstars) * dists #FIXME change for pymc from pymc3 instead of prior.random() do pm.draw(prior)
        #print('beff',beff)
        #vl_d = np.array(pm.TruncatedNormal.dist(mu=0., sigma=220, lower=0,
                       # upper=550.))#.random(size=self.nstars))
        vl = scipy.stats.truncnorm.rvs(a=0, b=550., loc=0.,scale =220, size=self.nstars)
        bvec = np.zeros((self.nstars, 2))
        vvec = np.zeros((self.nstars, 2))
        if self.ndims==2:
            btheta = np.random.rand(self.nstars)* 2. * np.pi
            vtheta = np.random.rand(self.nstars)* 2. * np.pi
            bvec[:, 0] = beff_p * np.cos(btheta)
            bvec[:, 1] = beff_p * np.sin(btheta)
            vvec[:, 0] = vl * np.cos(vtheta)
            vvec[:, 1] = vl * np.sin(vtheta)
        else:
            ## default to b perp. to v. gives larger signal for NFW halo
            bvec[:,0] = beff_p
            vvec[:, 1] = vl
        ## add units back in because astropy is dumb...or really probably it's
        ## me.
        bvec *= u.kpc
        vvec *= u.km / u.s
        ## get alphal given sampled other params.
        alphal = lm.alphal(newmassprofile, bvec, vvec) #FIXME Test if it works for new_mass function and newmassprofile array-- for loop instead?
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
        alphal = self.samplealphal(pars) #FIXME for mf, gets array of alphal
        # if self.massprofile.type == 'ps':
        #     # if ps, need to do snr check and re-sample if any have too high snr. this stops walkers from getting too stuck.
        #      alphal = self.snr_check(alphal, pars)
        #      if alphal == "error":
        #          return -np.inf
        if np.any(np.isnan(alphal)): #FIXME Should not be needed!
            # print('alphal nan')
            return -np.inf
        try:
            diff = alphal.value - self.data
        except:
            return -np.inf
        chisq = -0.5 * np.sum((diff)**2 / self.survey.alphasigma.value**2 -np.log(2 * np.pi * self.survey.alphasigma.value**2))
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
        #flatchain = self.sampler.get_chain(flat=True, discard=self.ntune)
        flatchain, __ = self.prune_chains()
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
                newmf = mf.PowerLaw(logM_0=logM0, logalpha=logalpha)
            elif mftype == 'Tinker':
                A = pars[i+0]
                a = pars[i+1]
                b = pars[i+2]
                c = pars[i+3]
                k_b = pars[i+4]
                n_b = pars[i+5]
                k_s = pars[i+6]
                newmf = mf.Tinker(A= A, a= a, b= b, c= c, k_b=k_b, n_b=n_b, k_s=k_s)
            elif mftype == 'CDM':
                loga = pars[i+0]
                b = pars[i+1]
                logc = pars[i+2]
                newmf = mf.CDM(loga = loga, b = b,logc = logc)
            elif mftype == 'WDM Stream':
                m_wdm = pars[i+0]
                gamma = pars[i+1]
                beta = pars[i+2]
                # loga_cdm = pars[i+3]
                # b_cdm = pars[i+4]
                # logc_cdm =pars[i+5]
                newmf = mf.WDM_stream(m_wdm=m_wdm,gamma=gamma, beta=beta)#, loga_cdm=loga_cdm,b_cdm=b_cdm,logc_cdm=logc_cdm)
            elif mftype == 'Press Schechter':
                del_crit = pars[i+0]
                #b = pars[i+1]
                #logc = pars[i+2]
                newmf = mf.PressSchechter_test(del_crit = del_crit)#,b = b,logc = logc)
            #else:
             #   raise NotImplementedError("""Need to add this mass function to
              #  sampler.""")

            if mptype == 'ps':
                newmp = []
                for newmf_ml in newmf.m_l[2:]:
                    kwargs['Ml'] = newmf_ml*u.Msun
                    newmp.append(mp.PointSource(**kwargs))
            elif mptype == 'gaussian':
                # kwargs['R0'] = 10**pars[0]*u.pc
                newmp = []
                for newmf_ml in newmf.m_l[2:]:
                    kwargs['Ml'] = newmf_ml * u.Msun
                    newmp.append(mp.Gaussian(**kwargs))
                #newmp = mp.Gaussian(**kwargs)
            elif mptype == 'nfw':
                kwargs['c200'] = 10**pars[0]
                newmp = []
                for newmf_ml in newmf.m_l[2:]:
                    kwargs['Ml'] = newmf_ml * u.Msun
                    newmp.append(mp.PointSource(**kwargs))
                newmp = mp.NFW(**kwargs)
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
