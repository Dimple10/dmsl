import emcee
import dill
import pickle
import numpy as np
import corner
import pandas as pd
import pymc3 as pm
from scipy.spatial import distance
from scipy.stats import trim1
import exoplanet as xo
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric, Galactic
from collections import Counter

from dmsl.convenience import *
from dmsl.constants import *
from dmsl.paths import *
from dmsl.accel_data import AccelData
from dmsl.star_field import StarField
from dmsl.survey import *
import dmsl.plotting as plot
import dmsl.galaxy_subhalo_dist as gsh
from dmsl.prior import *

import dmsl.lensing_model as lm
import dmsl.background_model as bm
import dmsl.mass_profile as mp


class Sampler():

    def __init__(self, nstars=None, nsamples=1000, nchains=8, ntune=1000,
            ndims=2, minlogMl=np.log10(1e0), maxlogMl=np.log10(1e8), nbnlens=7,
            maxlognlens=3,nbsamples=7000, MassProfile=mp.PointSource(**{'Ml' :
                1.e7*u.Msun}), bcutoff = 0.1*u.pc, survey=None, overwrite=True,
            usefraction=False):
        self.nstars=nstars
        self.nbsamples=nbsamples
        self.nbnlens = nbnlens
        self.maxlognlens = maxlognlens
        self.nsamples=nsamples
        self.nchains=nchains
        self.ntune=ntune
        self.ndims=ndims
        self.minlogMl = minlogMl
        self.maxlogMl = maxlogMl
        #self.sigalphab = bm.sig_alphab()
        self.sigalphab = 0.*u.uas/u.yr**2
        self.massprofile = MassProfile
        self.logradmax = 4 #pc
        self.logradmin = -4
        self.bcutoff = bcutoff
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

        self.alphal = []
        self.counter = 0
        self.lastm = 0.
        self.load_starpos()
        self.load_data()
        self.load_prior()
        self.run_inference()
        if self.overwrite:
            self.make_diagnostic_plots()

    def run_inference(self):
        npar, nwalkers = 1, self.nchains
        if self.massprofile.type == 'gaussian':
            npar += self.massprofile.nparams - 1
        if self.usefraction:
            npar += 1
        p0 = np.random.rand(nwalkers, npar)*(self.maxlogMl-self.minlogMl)+self.minlogMl
        if self.massprofile.type == 'gaussian':
            p0[:, 1] = np.random.rand(nwalkers)**(self.logradmax-self.logradmin) + self.logradmin
        if self.usefraction:
            p0[:, -1] = np.random.rand(nwalkers)

        sampler = emcee.EnsembleSampler(nwalkers, npar, self.lnlike,
                moves=[(emcee.moves.DEMove(), 0.5),
                    (emcee.moves.DESnookerMove(), 0.5),])
        #sampler = emcee.EnsembleSampler(nwalkers, npar, self.lnlike,
        #        moves = emcee.moves.GaussianMove(cov=1))
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
            flatchain = self.prune_chains()
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
            if (frac < 0) or (frac>10.):
                return -np.inf
        if self.massprofile.type == 'gaussian':
            modelpars = self.massprofile.nparams  - 1
            for i in range(0, modelpars):
                logradius = pars[i+1]
                if logradius < self.logradmin or logradius > self.logradmax:
                    return -np.inf
        return 0.

    def lnlike(self,pars):
        if ~np.isfinite(self.logprior(pars)):
            return -np.inf
        newmassprofile = self.make_new_mass(pars)
        if self.usefraction:
            f = pars[-1]
        else:
            f = 1.
        log10Ml = pars[0]

        Ml = 10**log10Ml
        nlens = int(np.ceil(f*Sampler.find_nlens(Ml, self.survey)))

        prior = pm.distributions.continuous.Interpolated.dist(self.bs,
                pdf(self.bs, a1=self.survey.fov_rad,
                    a2=self.survey.fov_rad, n=nlens))
        dists = self.rdist.rvs(self.nstars)*u.kpc
        beff = prior.random(self.nstars)*dists
        if isinstance(self.bcutoff, dict):
            bmin = self.find_bmin(Ml*u.Msun, SNR=self.bcutoff['SNR']).value
        else:
            pmin = self.bcutoff
            bmin = np.quantile(beff.value, pmin)
        beff = beff[beff.value>bmin]
        count = 0
        while len(beff) < self.nstars:
            newbs = prior.random(self.nstars)*dists
            newbs = newbs[newbs.value>bmin]
            beff = np.append(beff,newbs)
            count +=1
            if count > 100:
                return -np.inf

        beff = beff[:self.nstars]


        vlens = np.array(pm.TruncatedNormal.dist(mu=0., sigma=220, lower=0,
                        upper=550.).random(size=nlens))
        if vlens.ndim < 1:
            vl = np.ones((self.nstars))*vlens
        else:
            vl = np.array([ vlens[j] for j in np.random.randint(0,nlens,
                size=self.nstars)])
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

        bvec *= u.kpc
        vvec *= u.km / u.s


        alphal = lm.alphal(newmassprofile, bvec, vvec)

        if self.ndims == 1:
            alphal = np.linalg.norm(alphal, axis=1)

        diff = alphal.value - self.data
        chisq = np.sum(diff**2/(self.survey.alphasigma.value**2+self.sigalphab.value**2))

        return -0.5*chisq

    @staticmethod
    def find_nlens(Ml_, survey):
        volume = survey.fov_rad**2 * survey.maxdlens**3 / 3.
        mass = RHO_DM*volume
        nlens_k = mass.value/Ml_
        return np.ceil(nlens_k)
    @staticmethod
    def place_lenses(nlens, survey):
        ## FIXME: should be random sphere not cube
        #physcoords = np.random.rand(totlens, 3)*survey.maxdlens
        #x= pm.Triangular.dist(lower=0, upper=FOV,c=FOV/2.).random(size=nlens)
        #y= pm.Triangular.dist(lower=0, upper=FOV,c=FOV/2.).random(size=nlens)
        ## FIXME: this isn't quite right since some lenses will be out of LOS
        ## pyramid.
        z = np.random.rand(nlens) * survey.maxdlens
        x = (np.random.rand(nlens) * survey.fov_rad - survey.fov_rad / 2.) * z
        y = (np.random.rand(nlens) * survey.fov_rad - survey.fov_rad / 2.) * z
        ## get on-sky position
        ## u = distance to MW center
        c = SkyCoord(u=z, v=y, w=x,
                frame='galactic', representation_type='cartesian')
        c.representation_type = 'spherical'
        ## filter out sources not anywhere near the FOV.
        #mask = survey.fov_center.separation(c) < survey.fov_deg*np.sqrt(2)/2.
        #lenses = c[mask]
        lenses = c
        return lenses

    def load_starpos(self):
        ## loads data or makes if file not found.
        print('Making star field')
        self.starpos = StarField(self.survey, nstars=self.nstars).starpos

    def load_data(self):
        print('Creating data vector')
        self.data = AccelData(self.survey, nstars=self.nstars,
                ndims=self.ndims).data.to_numpy()

    def load_prior(self):
        print('Loading prior')
        rv = gsh.initialize_dist(target=self.survey.target,
                rmax=self.survey.maxdlens.to(u.kpc).value)
        self.rdist = rv
        self.bs = np.logspace(-8, np.log10(np.sqrt(2)*self.survey.fov_rad), 100)
        print('Prior loaded')

    def get_b_min(self, mp, bvec, vvec = np.array([0., 220.])*u.km/u.s):
        alpha = np.linalg.norm(lm.alphal(mp, bvec, vvec))
        if alpha > 5.*self.survey.alphasigma:
            return np.linalg.norm(bvec)
        else:
            bvec /= 10.
            return self.get_b_min(mp, bvec)

    def find_bmin(self, m, SNR = 1):
        v = 220*u.km/u.s
        accmag = 4 * const.G * v**2 / (const.c**2)
        b = ((accmag*m*self.nstars**(0.5)/(SNR*self.survey.alphasigma))**(1./3.)).to(u.kpc, equivalencies = u.dimensionless_angles())
        return b

    def prune_chains(self, maxlengthfrac=0.05):
        chains = []
        for i in range(self.nchains):
            chain = self.sampler.get_chain()[self.ntune:, i,:]
            counts = Counter(chain[:,0])
            most = counts.most_common(1)
            if most[0][1] < int(self.nsamples*maxlengthfrac):
                chains.append(chain)
        flatchain = [item for sublist in chains for item in sublist]
        print(f"Chains have been pruned. {len(flatchain)/self.nsamples} chains remain")
        self.flatchain = flatchain
        return flatchain

    def make_diagnostic_plots(self):
        ##FIXME: should also thin out samples by half the autocorr time.
        try:
            autocorr = int(np.floor(self.sampler.get_autocorr_time()[0]/2))
        except:
            autocorr = 1
        flatchain = self.prune_chains()
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

    def make_new_mass(self,pars):
        mptype = self.massprofile.type
        kwargs = self.massprofile.kwargs

        if mptype == 'ps':
            kwargs['Ml'] = 10**pars[0]*u.Msun
            newmp = mp.PointSource(**kwargs)
        elif mptype == 'constdens':
            kwargs['Ml'] = 10**pars[0]*u.Msun
            newmp = mp.ConstDens(**kwargs)
        elif mptype == 'exp':
            kwargs['Ml'] = 10**pars[0]*u.Msun
            kwargs['rd'] = 10**pars[1]*u.pc
            newmp = mp.Exp(**kwargs)
        elif mptype == 'gaussian':
            kwargs['Ml'] = 10**pars[0]*u.Msun
            kwargs['R0'] = 10**pars[1]*u.pc
            newmp = mp.Gaussian(**kwargs)
        elif mptype == 'nfw':
            kwargs['Ml'] = 10**pars[0]*u.Msun
            newmp = mp.NFW(**kwargs)
        elif mptype =='tnfw':
            kwargs['Ml'] = 10**pars[0]*u.Msun
            kwargs['r0'] = 10**pars[1]*u.pc
            kwargs['rt'] = 10**pars[2]*u.pc
            newmp = mp.TruncatedNFW(**kwargs)
        else:
            raise NotImplementedError("""Need to add this mass profile to
            sampler.""")
        return newmp

