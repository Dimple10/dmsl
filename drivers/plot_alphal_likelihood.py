'''
plot_fig_corners.py

Makes corner plots for paper. Set lens types and surveys to cycle through at
the top.
'''

## load packages
import numpy as np
import matplotlib.pyplot as plt
import pickle
import astropy.units as u
import dill

from dmsl.convenience import *
from dmsl.paths import *
from dmsl.plotting import *
from dmsl.survey import Roman
from dmsl.sampler import Sampler
from scipy.ndimage import gaussian_filter
from dmsl.mass_profile import *


## load style and colors.
cs = paper_plot()

## define lens types and parameters to cycle through
lenstypes = ['ps', 'gaussian', 'nfw']
params = {'gaussian':[3, 0.5], 'ps':[3], 'nfw':[3, np.log10(13)]}
labels = {'ps':f'$\\rm{{Point~Source}}:$ $10^{params["ps"][0]}~{{\\mathrm{{M}}}}_{{\\odot}}$',
        'gaussian': f'$\\rm{{Gaussian}}:$ $10^{{{params["gaussian"][0]}}}~{{\\mathrm{{M}}}}_{{\\odot}}, 10^{{{params["gaussian"][1]}}}~{{\\mathrm{{pc}}}}$',
        'nfw' : f'$\\rm{{NFW:}}$ $10^{{{params["nfw"][0]}}}~{{\\mathrm{{M}}}}_{{\\odot}}, c_{{200}} = {10**params["nfw"][1]:.0f}$'}

gausprops = {'Ml':1.e8*u.Msun, 'R0':1.e-3*u.pc}
mgauss = Gaussian(**gausprops)
ps = PointSource(**{'Ml':1.e7*u.Msun})

f, ax = plt.subplots()
for l in lenstypes:
    ## initialize Sampler -- automatically runs chains to just set to 1 step.
    if l == 'ps':
        mp = ps
    elif l == 'gaussian':
        mp = mgauss
    s = Sampler(nstars=int(1e3), ntune=1, nsamples=1, nchains=8,
    minlogMl = 0, survey=Roman(), MassProfile=mp, usefraction=False,
    ndims=2, SNRcutoff = 1.e-5, overwrite=False)
    s.nstars = int(1.e4)
    alphas =s.samplealphal(params[f'{l}'])
    alphanorm = np.linalg.norm(alphas, axis=1)
    ax.hist(np.log10(alphanorm.value), bins=50, histtype='step', lw=2,
        label=labels[f'{l}'], density=True)
ax.axvline(np.log10(Roman().alphasigma.value), lw=2, linestyle='dashed', color='k')
f.legend(loc='upper left', fontsize=7, bbox_to_anchor=(0.56,0.88),   framealpha=1.0)
ax.set_xlim([-18, 3])
ax.set_xlabel(r'$\log_{10}\|\vec{\alpha_l}\|~[\mu\rm{as/yr}^2]$')
ax.set_ylabel(r'$p(\vec{\alpha}_l | M_l, \vec{X}_l)$')
FIGPATH = f'{FINALDIR}fig_alphal_likelihood'
savefig(f,FIGPATH,
        writepdf=False)
PAPERPATH = f'{PAPERDIR}fig_alphal_likelihood'
savefig(f, PAPERPATH, writepdf=True)
