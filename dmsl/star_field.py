'''
Make files with star positions
'''

import os
import numpy as np
import pickle

from dmsl import __path__
from dmsl.constants import *
from dmsl.paths import *


class StarField():

    def __init__(self,nstars=1000):
        self.nstars = nstars

        self.make_field()
        self.save_field()

    def make_field(self):
        starpos = np.random.rand(self.nstars,2)*FOV
        self.starpos = starpos

    def save_field(self):
        outfile = STARDATADIR+'nstars_'+str(self.nstars)+'.dat'
        np.savetxt(outfile, self.starpos)

