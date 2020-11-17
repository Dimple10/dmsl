import emcee
import pickle
import numpy as np
import corner
import pandas as pd
import pymc3 as pm
from scipy.spatial import distance
import exoplanet as xo
import astropy.units as u
from astropy.coordinates import SkyCoord, Galactocentric, Galactic

from dmsl.convenience import *
from dmsl.constants import *
from dmsl.paths import *
from dmsl.accel_data import AccelData
from dmsl.star_field import StarField
from dmsl.survey import *
import dmsl.plotting as plot
import dmsl.galaxy_subhalo_dist as gsh

import dmsl.lensing_model as lm
import dmsl.background_model as bm
import dmsl.mass_profile as mp


class Sampler():

    def __init__(self, nstars=None, nbsamples=100, nsamples=1000, nchains=8,
            ntune=1000, ndims=2, nblog10Ml=3, minlogMl=np.log10(1e0),
            maxlogMl=np.log10(1e8), minlogb =-15., maxlogb = 1.,
            MassProfile=mp.PointSource(**{'Ml' : 1.e7*u.Msun}),
            survey=None, overwrite=True, usefraction=False):
        self.nstars=nstars
        self.nbsamples=nbsamples
        self.nblog10Ml=nblog10Ml
        self.nsamples=nsamples
        self.nchains=nchains
        self.ntune=ntune
        self.ndims=ndims
        self.minlogMl = minlogMl
        self.maxlogMl = maxlogMl
        #self.sigalphab = bm.sig_alphab()
        self.sigalphab = 0.*u.uas/u.yr**2
        self.massprofile = MassProfile
        self.logradmax = 3.5 #pc
        self.logradmin = -3
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
        self.load_starpos()
        self.load_data()
        self.load_lensrdist()
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
       # sampler = emcee.EnsembleSampler(nwalkers, npar, self.lnlike,
       #         moves = emcee.moves.GaussianMove(cov=1.0))
        sampler.run_mcmc(p0, self.ntune+self.nsamples, progress=True)

        samples = sampler.get_chain(discard=self.ntune, flat=True)
        print(f"90\% upper limit on Ml: {np.percentile(samples[:,0], 90)}")
        ## save samples to class
        self.sampler = sampler

        ## save results
        if self.overwrite:
            pklpath = make_file_path(RESULTSDIR, [np.log10(self.nstars),
                np.log10(self.nsamples), self.ndims],
                extra_string=f'samples_{self.survey.name}_{self.massprofile.type}',
                ext='.pkl')
            with open(pklpath, 'wb') as buff:
                pickle.dump(samples, buff)
            print('Wrote {}'.format(pklpath))


    def logprior(self,pars):
        log10Ml = pars[0]
        if (log10Ml < self.minlogMl) or (log10Ml>self.maxlogMl):
            return -np.inf
        if self.usefraction:
            frac = pars[-1]
            if (frac < 0) or (frac>100):
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
        #nlens = int(np.ceil(f))
        FOV = self.survey.fov_rad
        dstar = self.survey.dstars
        #lenspos = Sampler.place_lenses(nlens, self.survey)
        ## FIXME: this isn't quite right since some lenses will be out of LOS
        ## pyramid.
        z = self.rdist.rvs(nlens)*u.kpc
        #print(np.sum(z > 0.5*u.kpc)/nlens)
        x = (np.random.rand(nlens) * self.survey.fov_rad)
        y = (np.random.rand(nlens) * self.survey.fov_rad)
        dists = distance.cdist(np.vstack([x, y]).T, self.starpos)
        ind = np.argmin(z[:, np.newaxis]*dists, axis=0)
        beff = np.min(z[:,np.newaxis]*dists, axis=0)
        vl = np.array(pm.TruncatedNormal.dist(mu=0., sigma=220, lower=0,
                        upper=550.).random(size=nlens))
        if vl.ndim < 1:
            vl = np.array([vl])
        bvec = np.zeros((self.nstars, 2))
        vvec = np.zeros((self.nstars, 2))
        if self.ndims==2:
            btheta = np.random.rand(self.nstars)* 2. * np.pi
            vtheta = np.random.rand(self.nstars)* 2. * np.pi
            bvec[:, 0] = beff * np.cos(btheta)
            bvec[:, 1] = beff * np.sin(btheta)
            vvec[:, 0] = vl[ind] * np.cos(vtheta)
            vvec[:, 1] = vl[ind] * np.sin(vtheta)
        else:
            ## default to b perp. to v. gives larger signal for NFW halo
            bvec[:,0] = beff
            vvec[:,1] = vl

        bvec *= u.kpc
        vvec *= u.km / u.s

        alphal = lm.alphal(newmassprofile, bvec, vvec)

        if self.ndims == 1:
            alphal = np.linalg.norm(alphal, axis=1)

        diff = alphal.value - self.data
        chisq = np.sum(diff**2/(self.survey.alphasigma.value**2+self.sigalphab.value**2))

        debug = True
        if debug == True:
            #b = np.linalg.norm(bvec, axis=1)
            #v = np.linalg.norm(vvec, axis=1)
            #alphalmag = np.linalg.norm(alphal, axis=1)
            #print(nlens)
            #prange(b)
            #print(np.average(b))
            #prange(v)
            #print(np.average(v))
            #prange(alphalmag)
            #print(np.average(alphalmag))
            #print(prange(newmassprofile.M(b)))
            #print(log10Ml)
            if (-0.5*chisq < -1071.2) and (-0.5*chisq > -1071.3) and (counter <
                    1):
                self.alphal.append(alphal)
                self.okay_params = [log10Ml, vvec, bvec]
                self.counter +=1 
            if -0.5*chisq > -999.:
                self.problem_params = [log10Ml, vvec, bvec]
                self.alphal.append(alphal)
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

    def load_lensrdist(self):
        rv = gsh.initialize_dist()
        self.rdist = rv

    def make_diagnostic_plots(self):
        ##FIXME: should also thin out samples by half the autocorr time.
        plot.plot_emcee(self.sampler.get_chain(flat=True, discard=self.ntune),
                self.nstars, self.nsamples,self.ndims, self.massprofile,
                self.survey.name, self.usefraction)
        outpath = make_file_path(RESULTSDIR, [np.log10(self.nstars),
            np.log10(self.nsamples), self.ndims],
            extra_string=f'chainsplot_{self.survey.name}_{self.massprofile.type}',
            ext='.png')
        plot.plot_chains(self.sampler.get_chain(), outpath)
        outpath = make_file_path(RESULTSDIR, [np.log10(self.nstars),
            np.log10(self.nsamples), self.ndims],
            extra_string=f'logprob_{self.survey.name}_{self.massprofile.type}',
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

