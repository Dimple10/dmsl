'''
Make WFIRST-Like corner plot to compare to MS 2020.
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
from scipy.ndimage import gaussian_filter

## load style and colors
cs = paper_plot()

## set lens types and survey.
lenstypes = ['gaussian']
surveys = ['WFIRSTLike']
labels = {'gaussian': [r'$\log_{10} M_l~[\mathrm{M}_{\odot}]$',
            r'$\log_{10} R_0~[\rm{pc}]$',
            r'$\rm{Fraction~of~DM}$']}

def get_chains(lenstypes, surveys):
    chains = {}
    for survey in surveys:
        for l in lenstypes:
            tstring = f'{survey}_{l}'
            f = f'{RESULTSDIR}final/pruned_samples_{survey}_{l}_3_4_2.pkl'
            with open(f,'rb') as buff:
                chains[tstring] = dill.load(buff)
    return chains

def make_corner(chaindict, lenstype, survey, labeldict):
    chains = chaindict[f'{survey}_{lenstype}']

    FIGPATH = make_file_path(FINALDIR, {}, extra_string=f'fig_{survey}',
            ext='.png')
    f = plt.figure(figsize=(4,4))
    corner.hist2d(chains[:,0], chains[:,1], labels=labeldict[f'{lenstype}'], smooth=1.8,
                  levels=[0.68, 0.95],fig=f, color='steelblue',
                  max_n_ticks=10, **{'plot_datapoints':False});
    ax = f.axes
    if (np.shape(chains)[1] > 1) and (lenstype=='gaussian'):
        rs = np.linspace(np.min(chains[:,0]), np.max(chains[:,0]), 100)
        msline = 0.45*rs-2.
        ax[0].plot(rs, msline,
                c='black', linewidth=3, linestyle='dashed',
                label='Mishra-Sharma et al. (2020)')
        ax[0].legend(loc='upper left', fontsize=8)

    ax[0].set_xlabel(labeldict['gaussian'][0])
    ax[0].set_ylabel(labeldict['gaussian'][1])
    ax[0].set_ylim([-2,4])
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))

    f.tight_layout()
    savefig(f, FIGPATH)
    PAPERPATH = make_file_path(PAPERDIR, {}, extra_string=f'fig_{survey}',
            ext='.png')
    savefig(f,PAPERPATH,
            writepdf=False)

## MAIN
chains = get_chains(lenstypes, surveys)
for s in surveys:
    for l in lenstypes:
        make_corner(chains, l,s, labels)
