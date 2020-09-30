'''
Makes fake data vector
'''

import os
import numpy as np
import pandas as pd

from dmsl.paths import *
from dmsl.constants import *
from dmsl.convenience import *
from dmsl.plotting import *

seed = 0

class AccelData():

    def __init__(self, survey,nstars=1000, ndims=1):
        rng = np.random.default_rng(seed=seed)
        if ndims == 2:
            data = pd.DataFrame(rng.standard_normal(size=(nstars,ndims))*survey.alphasigma.value,
                    columns=['a_x', 'a_y'])
        else:
            data = pd.DataFrame(rng.standard_normal(size=(nstars,ndims))*survey.alphasigma.value,
                    columns=['a'])
        fileinds = [np.log10(nstars), ndims, survey.alphasigma.value]
        filepath = make_file_path(STARDATADIR, fileinds,
                extra_string=f'{survey.name}',ext='.dat')
        data.to_csv(filepath, index=False)
        print("Wrote to {}".format(filepath))
        self.plot_data(data,survey, nstars, ndims)
        self.data = data

    def plot_data(self, data, survey, nstars, ndims):
        fileinds = [np.log10(nstars), ndims, survey.alphasigma.value]
        filepath = make_file_path(STARDATADIR, fileinds,
                extra_string=f'{survey.name}_hist',ext='.png')
        make_histogram(data.to_numpy(),int(len(data)/100) + 1, r'\alpha', filepath)

