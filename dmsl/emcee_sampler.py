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
from dmsl.prior_sampler import PriorSampler, find_nlens_np
from dmsl.accel_data import AccelData
from dmsl.star_field import StarField
from dmsl.survey import *
import dmsl.plotting as plot

import dmsl.lensing_model as lm
import dmsl.background_model as bm
import dmsl.mass_profile as mp


class Sampler():

    def __init__(self, nstars=1000, nbsamples=100, nsamples=1000, nchains=8,
            ntune=1000, ndims=1, nblog10Ml=3, minlogMl=np.log10(1e0),
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

        self.load_starpos()
        self.load_data()
        self.run_inference()
        if self.overwrite:
            self.make_diagnostic_plots()

    def run_inference(self):
        npar, nwalkers = self.ndims, self.nchains
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
                moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),])
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
        nlens = int(np.ceil(f*find_nlens_np(Ml)))
        #nlens = int(np.ceil(f))
        FOV = self.survey.fov_rad
        x= pm.Triangular.dist(lower=0, upper=FOV,c=FOV/2.).random(size=nlens)
        y= pm.Triangular.dist(lower=0, upper=FOV,c=FOV/2.).random(size=nlens)
        lenspos = np.vstack([x,y]).T
        dists = distance.cdist(lenspos, self.starpos)
        beff = np.min(dists, axis=0)
        vl = pm.TruncatedNormal.dist(mu=0., sigma=220, lower=0,
                        upper=550.).random(1)
        if self.ndims==2:
            bstheta = pm.Uniform(name='bstheta', lower=0, upper=np.pi/2.)
            vltheta = pm.Uniform(name='vltheta', lower=0, upper=np.pi/2.)
        else:
            bstheta = None
            vltheta = None
        bvec = np.zeros((len(beff), 2))
        bvec[:,0] = beff
        bvec *= u.kpc
        vvec = np.zeros((len(beff), 2))
        vvec[:,1] = vl
        vvec *= u.km/u.s

        alphal = lm.alphal(newmassprofile, bvec, vvec)

        alphal = alphal[:, 0]

        diff = alphal.value - self.data
        chisq = np.sum(diff**2/(self.survey.alphasigma.value**2+self.sigalphab.value**2))
        return -0.5*chisq

    def load_starpos(self):
        ## loads data or makes if file not found.
        print('Making star field')
        self.starpos = StarField(self.survey, nstars=self.nstars).starpos

    def load_data(self):
        print('Creating data vector')
        self.data = AccelData(self.survey, nstars=self.nstars,
                ndims=self.ndims).data.to_numpy()

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

