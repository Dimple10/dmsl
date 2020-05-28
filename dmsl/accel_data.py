'''
Makes fake data vector
'''

import os
import numpy as np
import pandas as pd

from dmsl.paths import *
from dmsl.constants import *

class AccelData():

    def __init__(self, nstars=1000, ndims=1):
        if ndims == 2:
            data = pd.DataFrame(np.random.randn(nstars,ndims)*WFIRST_SIGMA.value,
                    columns=['a_x', 'a_y'])
        else:
            data = pd.DataFrame(np.random.randn(nstars,ndims)*WFIRST_SIGMA.value,
                    columns=['a'])
        outfile = STARDATADIR+str(int(np.log10(nstars)))+'_'+str(ndims)+'.dat'
        data.to_csv(outfile, index=False)
