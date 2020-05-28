'''
Makes some diagnostic plots to see why there are divergences in pymc3.
'''

import numpy as np
import pymc3 as pm
import pickle
import matplotlib.pyplot as plt

from dmsl.constants import *
from dmsl.paths import *
from dmsl.convenience import *
from dmsl.accel_data import AccelData

nstars = 1e3
nsamples = 2e3
ndims = 1

## load trace
pklpath = make_file_path(RESULTSDIR, [np.log10(nstars), np.log10(nsamples), ndims],
    extra_string='trace', ext='.pkl')
with open(pklpath, 'rb') as buff:
    trace = pickle.load(buff)

## make some plots 
def pairplot_divergence(trace, ax=None, divergence=True, color='C3', divergence_color='C2'):
    theta = trace.get_values(varname='logbs', combine=True)
    logtau = trace.get_values(varname='logMl', combine=True)
    if not ax:
        f, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(theta, logtau, 'o', color=color, alpha=.5)
    if divergence:
        divergent = trace['diverging']
        ax.plot(theta[divergent], logtau[divergent], 'o', color=divergence_color)
    ax.set_xlabel('theta[0]')
    ax.set_ylabel('log(tau)')
#     ax.set_yscale('log')
    # ax.set_xscale('log')
    f.savefig('test.png')

pairplot_divergence(trace);
