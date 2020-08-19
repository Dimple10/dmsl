'''
Makes nice sensitivity plot to compare with Mishra-Sharma results
'''

import numpy as np
import pymc3 as pm
import pickle
import matplotlib.pyplot as plt

from dmsl.constants import *
from dmsl.paths import *
from dmsl.convenience import *
from dmsl.plotting import *

nstars = 1e3
nsamples = 5e3
ndims = 1
massprofiletype = ['gaussian', 'exp', 'nfw', 'tnfw']
fileinds = [np.log10(nstars), np.log10(nsamples), ndims]
for t in massprofiletype:
    ## load flatchain
    pklpath = make_file_path(RESULTSDIR, fileinds,
            extra_string=f'samples_{t}', ext='.pkl')
    with open(pklpath, 'rb') as buff:
        samples = pickle.load(buff)

    ## plot
    path = make_file_path(RESULTSDIR,fileinds,
            extra_string=f'sensitivity_{t}', ext='.png')
    plot_sensitivity(samples, path)
