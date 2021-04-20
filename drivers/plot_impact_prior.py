'''
makes plot of impact parameter prior
'''


import numpy as np
import matplotlib.pyplot as plt
import pickle
import astropy.units as u

import dmsl.galaxy_subhalo_dist as gsh

from dmsl.convenience import *
from dmsl.paths import *
from dmsl.plotting import *
from dmsl.survey import Roman
from dmsl.prior import *

## Set params
survey = Roman()
nlensarray = [1,10,100]
bs = np.logspace(-7, np.log10(survey.fov_rad), 1000)
FIGPATH = make_file_path(FINALDIR, {}, extra_string='prior',
        ext='.png')

## Load prior.
rv = gsh.initialize_dist(target=survey.target,
        rmax=survey.maxdlens.to(u.kpc).value)
rdist = rv
prior_init = impact_pdf(shapes='a1, a2, n')


## Make fig.
paper_plot()
f = plt.figure()
for n in nlensarray:
    pb = prior_init.pdf(bs, a1=survey.fov_rad,
        a2=survey.fov_rad, n=n)
    plt.plot(bs,  pb/np.max(pb), lw=2, label=f'$N_{{\\rm{{lens}}}}={n}$')

plt.legend()
plt.xlabel(r'$b~[\rm{kpc}]$')
plt.ylabel(r'$p(b)$')
plt.xscale('log')
savefig(f, FIGPATH)
