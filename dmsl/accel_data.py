'''
Makes fake data vector
'''

import os
import numpy as np

from dmsl.paths import *
from dmsl.constants import *

class AccelData():

    def __init__(self, nstars=1000, ndims=1):
        self.data = np.random.randn(nstars,ndims)*WFIRST_SIGMA.value

        outfile = STARDATADIR+str(int(np.log10(nstars)))+'_'+str(ndims)+'.dat'
        np.savetxt(outfile, self.data)
