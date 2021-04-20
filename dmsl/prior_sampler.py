'''
Sampler class for impact parameter prior
'''

import os
import numpy as np
import dill
import pickle
import pandas as pd
from scipy.spatial import distance
from scipy.stats import gaussian_kde

from dmsl  import __path__
from dmsl.survey import *
from dmsl.constants import *
from dmsl.convenience import *
from dmsl.paths import *
from dmsl.star_field import StarField
import dmsl.galaxy_subhalo_dist as gsh

class PriorSampler():

    def __init__(self, nstars=None, nbsamples=100,
            nbnlens=3, maxlognlens=3,
            survey=None, overwrite=True):
        self.nstars=nstars
        self.nbsamples = nbsamples
        self.nbnlens = nbnlens
        self.maxlognlens = maxlognlens
        self.overwrite = overwrite
        if survey is None:
            self.survey = Roman()
        else:
            self.survey = survey

        if nstars is None:
            print("""Defaulting to number of stars from survey...this might be
            too large for your computer memory...""")
            self.nstars = survey.nstars
            print(f"Nstars set to {self.nstars}")
        self.load_starpos()
        self.load_lensrdist()
        self.run_sampler()
        #self.make_diagnostic_plots()

    def get_dists(self,nlens):
        barray = np.zeros((self.nbsamples, self.nstars))
        for i in range(self.nbsamples):
#            z = self.rdist.rvs(nlens)*u.kpc
            z = 1.*u.kpc*np.ones(nlens)
            x = (np.random.rand(nlens) * self.survey.fov_rad)
            y = (np.random.rand(nlens) * self.survey.fov_rad)
            dists = distance.cdist(np.vstack([x, y]).T, self.starpos)
    #        beff = np.sum(1./(z[:, np.newaxis]*dists), axis=0)**(-1)
            beff = np.min(z[:, np.newaxis]*dists, axis=0)
            barray[i, :] = beff.to(u.kpc)

        self.test_convergence(nlens, barray)
        return barray

    def make_2dkde(self, stackedbs, stackednlens):
        samples = np.vstack([stackedbs, stackednlens])
        self.kde = gaussian_kde(np.log10(samples), bw_method=0.5)
        return self.kde

    def test_convergence(self, nlens, samples):
        lensamp = len(samples)
        len90 = int(0.9*lensamp)
        first90 = samples[:len90]
        getstats = lambda x: [np.average(x), np.std(x), np.percentile(x, 0.1)]
        ave90, std90, lower1090 = getstats(first90)
        aveful, stdful, lower10ful = getstats(samples)
        getfdiff = lambda x,y: np.abs(x-y)/y
        avefdiff = getfdiff(ave90, aveful)
        stdfdiff = getfdiff(std90, stdful)
        lower10diff = getfdiff(lower1090, lower10ful)
        print(nlens, avefdiff, stdfdiff, lower10diff)

    def run_sampler(self):
        nlensarray = np.logspace(0, self.maxlognlens, self.nbnlens, dtype=int)
        barray = np.zeros((len(nlensarray), self.nbsamples, self.nstars))
        for i,n in enumerate(nlensarray):
            barray[i,:, :] = self.get_dists(n)

        stackedb =  barray.flatten()
        self.barray = np.logspace(np.log10(np.min(stackedb)),
                np.log10(np.max(stackedb)), 20)
        stackedn = np.repeat(nlensarray, self.nstars*self.nbsamples)
        plogb = self.make_2dkde(stackedb, stackedn)

        ## now save to pickleable file.
        pklpath= make_file_path(RESULTSDIR, [np.log10(self.nstars),
            self.nbnlens, np.log10(self.nbsamples)],
            extra_string=f'plogb_{self.survey.name}_justmin',ext='.pkl')
        with open(pklpath, 'wb') as buff:
            dill.dump(plogb, buff)
        print('Wrote {}'.format(pklpath))
        print('B Min {}; B Max {}'.format(min(np.log10(stackedb)),
            max(np.log10(stackedb))))

    def load_starpos(self):
        ## loads data or makes if file not found.
        print('Making star field')
        self.starpos = StarField(self.survey, nstars=self.nstars).starpos

    def load_lensrdist(self):
        rv = gsh.initialize_dist(target=self.survey.target,
                rmax=self.survey.maxdlens.to(u.kpc).value)
        self.rdist = rv

