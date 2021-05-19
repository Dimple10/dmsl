'''
Make files with star positions
'''

import os
import numpy as np
import pickle
import pandas as pd
import astropy.units as u

from dmsl import __path__
from dmsl.paths import *
from dmsl.convenience import *
from dmsl.plotting import make_scatter

seed = 123456789
class StarField():

    def __init__(self,survey, nstars=1000):
        self.nstars = nstars
        self.survey = survey

        self.make_field()
        self.save_field()
        self.plot_field()

    def make_field(self):
        rng = np.random.default_rng(seed=seed)
        starpos = rng.random(size=(self.nstars,2))*self.survey.fov_rad
        self.starpos = starpos

    def save_field(self):
        out = pd.DataFrame(self.starpos, columns=['x', 'y'])
        fileinds = [np.log10(self.nstars)]
        filepath = make_file_path(STARPOSDIR, fileinds, ext='.dat',
                extra_string=f'nstars_{self.survey.name}')
        out.to_csv(filepath, index=False)
        print("Wrote to {}".format(filepath))


    def plot_field(self):
        fileinds = [np.log10(self.nstars)]
        filepath = make_file_path(STARPOSDIR, fileinds, ext='.png',
                extra_string=f'scatter_nstars_{self.survey.name}')
        labels = [r'x~[\rm{deg}]', r'y~[\rm{deg}]']
        make_scatter(self.starpos[:,0]*u.rad.to(u.deg),
                self.starpos[:,1]*u.rad.to(u.deg), labels,  filepath)

