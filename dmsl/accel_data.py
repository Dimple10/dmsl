'''
Makes fake data vector
'''

import os
import numpy as np
import pandas as pd

from dmsl.paths import *
from dmsl.constants import *
from dmsl.convenience import *

class AccelData():

    def __init__(self, survey,nstars=1000, ndims=1):
        if ndims == 2:
            data = pd.DataFrame(np.random.randn(nstars,ndims)*survey.alphasigma.value,
                    columns=['a_x', 'a_y'])
        else:
            data = pd.DataFrame(np.random.randn(nstars,ndims)*survey.alphasigma.value,
                    columns=['a'])
        fileinds = [np.log10(nstars), ndims, survey.alphasigma.value]
        filepath = make_file_path(STARDATADIR, fileinds,
                extra_string=f'{survey.name}',ext='.dat')
        data.to_csv(filepath, index=False)
        print("Wrote to {}".format(filepath))
