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
bs = np.logspace(-6, np.log10(np.sqrt(2)*survey.fov_rad), 100)
FIGPATH = make_file_path(FINALDIR, {}, extra_string='fig_prior',
        ext='.png')

## Load prior.
rv = gsh.initialize_dist(target=survey.target,
        rmax=survey.maxdlens.to(u.kpc).value)
rdist = rv


## Make fig.
paper_plot()
f = plt.figure()
for n in nlensarray:
    pb = pdf(bs, a1=survey.fov_rad,
        a2=survey.fov_rad, n=n)
    plt.plot(bs*u.rad.to(u.arcsec),  pb/np.max(pb), lw=2, label=f'$N_{{\\rm{{lens}}}}={n}$')

plt.axvline(survey.fov_rad*u.rad.to(u.arcsec), lw=2, linestyle='dashed',
        color='black')
plt.xlim([np.min(bs)*u.rad.to(u.arcsec), np.max(bs)*u.rad.to(u.arcsec)])
plt.legend()
plt.xlabel(r'$b~[\rm{arcsec}]$')
plt.ylabel(r'$p(b)$')
plt.xscale('log')
savefig(f, FIGPATH)

## DELETE FOR PUBLIC
FIGPATH = make_file_path(PAPERDIR, {}, extra_string='fig_prior',
        ext='.png')
savefig(f, FIGPATH)
