
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy.spatial import distance
import exoplanet as xo
import matplotlib.pyplot as plt

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

ndims = 1
nstars = 1e3

def load_starpos():
    ## loads data or makes if file not found.
    fileinds = [np.log10(nstars)]
    filepath = make_file_path(STARPOSDIR, fileinds, ext='.dat',
            extra_string='nstars')
    if os.path.exists(filepath):
        starpos = pd.read_csv(filepath).to_numpy()
    else:
        StarField(nstars=nstars)
        starpos = pd.read_csv(filepath).to_numpy(dtype=np.float64)
    return starpos

def get_alphal(Ml):
    nlens = int(find_nlens_np(Ml))
    x= pm.Triangular.dist(lower=0, upper=FOV,c=FOV/2.).random(size=nlens)
    y= pm.Triangular.dist(lower=0, upper=FOV,c=FOV/2.).random(size=nlens)
    lenspos = np.vstack([x,y]).T
    dists = distance.cdist(lenspos, starpos)
    beff = np.sum(dists, axis=0)
    vl = pm.TruncatedNormal.dist(mu=0., sigma=220, lower=0,
                    upper=550.).random(1)
    if ndims==2:
        bstheta = pm.Uniform(name='bstheta', lower=0, upper=np.pi/2.)
        vltheta = pm.Uniform(name='vltheta', lower=0, upper=np.pi/2.)
    else:
        bstheta = None
        vltheta = None

    alphal = lm.alphal_np(Ml, beff, vl,
        btheta_=bstheta, vltheta_=vltheta)
    return np.average(alphal), np.var(alphal)

starpos = load_starpos()
Ml = np.logspace(2,8, 100)
alphals = np.array([get_alphal(M)[0] for M in Ml])
#varalpha = np.array([get_alphal(M)[1] for M in Ml])
plot.paper_plot()
FIGPATH = make_file_path(RESULTSDIR, [np.log10(nstars)], extra_string='alphalvM',
        ext='.png')
f = plt.figure(figsize=(6,4))
plt.loglog(Ml, alphals)
#plt.fill_between(Ml, alphals-varalpha, alphals+varalpha, alpha=0.8)
plt.ylim([1.e-9, 3.e0])
plt.axhline(WFIRST_SIGMA.value, color='black')
plt.text(np.min(Ml), WFIRST_SIGMA.value+0.1*WFIRST_SIGMA.value, r'$\rm{Roman~Sensitivity}$')
plt.xlabel(r'$M_l~[M_{\odot}]$')
plt.ylabel(r'$\alpha_l~[\mu\rm{as}/\rm{yr}^2]$')
plot.savefig(f, FIGPATH, writepdf=False)

