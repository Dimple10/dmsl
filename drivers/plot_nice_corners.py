
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
from dmsl.plotting import *

nstars = 1e5
nsamples = 2e3
ndims = 1
fileinds =  [np.log10(nstars), np.log10(nsamples), ndims]
## load trace
pklpath = make_file_path(RESULTSDIR, fileinds,
    extra_string='trace', ext='.pkl')
with open(pklpath, 'rb') as buff:
    trace = pickle.load(buff)

## plot
path = make_file_path(RESULTSDIR,fileinds, extra_string='nice_lens_corner',
        ext='.png')
bkgpath = make_file_path(RESULTSDIR,fileinds, extra_string='nice_bkg_corner',
        ext='.png')
if ndims == 1:
    plot_nice_lens_corner_1D(trace, path)
    plot_nice_bkg_corner_1D(trace,bkgpath)
else:
    plot_nice_lens_corner_2D(trace,path)
