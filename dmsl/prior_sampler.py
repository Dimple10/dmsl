'''
Sampler class for impact parameter prior
'''

import os
import numpy as np
import pymc3 as pm
import pickle
import pandas as pd
from scipy.spatial import distance
from scipy.stats import gaussian_kde

from dmsl  import __path__
from dmsl.constants import *
from dmsl.convenience import *
from dmsl.paths import *
from dmsl.star_field import StarField

class PriorSampler():

    def __init__(self, nbsamples=100, log10Ml=3.,
            nmbins=5., nstars= 1000,overwrite=False):

        self.nstars = nstars
        self.nbsamples = nbsamples
        self.log10Ml = log10Ml
        starfile = STARPOSDIR+'nstars_'+str(int(np.log10(nstars)))+'.dat'
        try:
            self.star_pos = pd.read_csv(starfile).to_numpy()
        except:
            print('Need to make a new starfield...brb.')
            StarField(nstars=nstars)
            self.star_pos = pd.read_csv(starfile).to_numpy()

    def run_sampler(self):

        nlensk = self.find_nlens(10**self.log10Ml)
        logb_dists = []
        j = 0.
        while j <= self.nbsamples:
            Nlens = pm.Poisson.dist(mu=nlensk).random(size=1)
            lenspos = self.get_lens_pos(Nlens)
            dists = distance.cdist(lenspos, self.star_pos)
            ## use log instead of log10 because pymc3
            logbb = np.log(dists.reshape((np.shape(dists)[0]*np.shape(dists)[1])))
            logb_dists.append(logbb.tolist())
            j+=1
        logb = np.array(flatten(logb_dists))
        logbarray, plogb = self.make_kde(logb)
        self.plogb = plogb
        ## now save to pickleable file.
        pklpath= make_file_path(RESULTSDIR, [np.log10(self.nstars), self.log10Ml,
            np.log10(self.nbsamples)], extra_string='plogb',ext='.pkl')
        with open(pklpath, 'wb') as buff:
            pickle.dump(plogb, buff)
        print('Wrote {}'.format(pklpath))
        print('B Min {}; B Max {}'.format(np.exp(min(logbarray)),
            np.exp(max(logbarray))))

    def find_nlens(self, Ml_):

        volume = (LENS_FOV)**2*DLENS/3.
        mass = RHO_DM*volume
        nlens_k = mass.value/Ml_
        return np.ceil(nlens_k)

    def get_lens_pos(self,nlens_):

        ## randomly place in x, y triangle -- places close to middle of FOV should have more sources
        x= pm.Triangular.dist(lower=0, upper=FOV,c=FOV/2.).random(size=nlens_)
        y= pm.Triangular.dist(lower=0, upper=FOV,c=FOV/2.).random(size=nlens_)
        coords = np.vstack([x,y]).T
        return coords

    def make_kde(self,dist_):
        logbmin, logbmax = np.min(dist_), np.max(dist_)
        bs = np.linspace(logbmin, logbmax, self.nbsamples)
        logb = gaussian_kde(dist_)
        return bs, logb


def find_nlens_pm(self, Ml_):

    volume = (LENS_FOV)**2*DLENS/3.
    mass = RHO_DM*volume
    nlens_k = mass.value/Ml_
    return pm.math.ceil(nlens_k)
