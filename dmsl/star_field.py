'''
Make files with star positions
'''

import os
import numpy as np
import pickle
import pandas as pd

from dmsl import __path__
from dmsl.constants import *
from dmsl.paths import *
from dmsl.convenience import *


class StarField():

    def __init__(self,survey, nstars=1000):
        self.nstars = nstars
        self.survey = survey

        print('Making field...')
        self.make_field()
        print('Now saving field...')
        self.save_field()
        print('All done making star field!')

    def make_field(self):
        starpos = np.random.rand(self.nstars,2)*self.survey.fov_rad
        self.starpos = starpos

    def save_field(self):
        out = pd.DataFrame(self.starpos, columns=['x', 'y'])
        fileinds = [np.log10(self.nstars)]
        filepath = make_file_path(STARPOSDIR, fileinds, ext='.dat',
                extra_string=f'nstars_{self.survey.name}')
        out.to_csv(filepath, index=False)
        print("Wrote to {}".format(filepath))

