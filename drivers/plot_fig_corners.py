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
from scipy.ndimage import gaussian_filter

## load style and colors.
cs = paper_plot()

## set lens types, surveys, and labels for lens types.
lenstypes = ['gaussian', 'nfw', 'ps']
surveys = ['Roman']
labels = {'ps':[r'$\log_{10} M_l~[\mathrm{M}_{\odot}]$', r'$\rm{Fraction~of~DM}$'],
        'gaussian': [r'$\log_{10} M_l~[\mathrm{M}_{\odot}]$',
            r'$\log_{10} R_0~[\rm{pc}]$',
            r'$\rm{Fraction~of~DM}$'],
        'nfw' : [r'$\log_{10} M_l~[\mathrm{M}_{\odot}]$',
            r'$\log_{10} c_{200}$',
            r'$\rm{Fraction~of~DM}$']}

def get_chains(lenstypes, surveys):
    '''
    downloads chains from final directory
    '''
    chains = {}
    for survey in surveys:
        for l in lenstypes:
            tstring = f'{survey}_{l}'
            f = f'{FINALDIR}pruned_samples_{survey}_{l}_3_4_2.pkl'
            with open(f,'rb') as buff:
                chains[tstring] = dill.load(buff)
            tstring = f'{survey}_{l}_frac'
            f = f'{FINALDIR}pruned_samples_{survey}_{l}_frac_3_4_2.pkl'
            with open(f,'rb') as buff:
                chains[tstring] = dill.load(buff)
    return chains

def make_corner(chaindict, lenstype, survey, labeldict):
    '''
    makes corner plot for a chain
    '''
    chains = chaindict[f'{survey}_{lenstype}']
    fracchains = chaindict[f'{survey}_{lenstype}_frac']
    smoothparam = 2.2
    if lenstype == 'ps':
        smoothparam = 3.0 ## need more smoothing because needed to cut more walkers due to them getting stuck.

    FIGPATH = make_file_path(FINALDIR, {}, extra_string=f'fig_{lenstype}',
            ext='.png')
    f = plt.figure(figsize=(8,8))
    corner.corner(fracchains, labels=labeldict[f'{lenstype}'],
        smooth=smoothparam, levels=[0.68, 0.95],fig=f, smooth1d=smoothparam,
        color='steelblue', max_n_ticks=10, **{'plot_datapoints':False});
    ax = f.axes
    for i in range(0, np.shape(chains)[1]):
        axi = i*np.shape(fracchains)[1] + i
        if lenstype == 'ps':
            ax[axi].axvline(np.percentile(fracchains[:,i], 97.5), linestyle='dashed', lw=2,
                    color=cs[0])
            ax[axi].axvline(np.percentile(fracchains[:,i], 2.5), linestyle='dashed', lw=2,
                    color=cs[0])
            ax[axi].axvline(np.percentile(chains[:,i], 2.5), linestyle='dashed', linewidth=2,
                    color=cs[1])
            ax[axi].axvline(np.percentile(chains[:,i], 97.5), linestyle='dashed', linewidth=2,
                color=cs[1])
        n, b = np.histogram(
                chains[:,i], bins=15,
                    weights=np.ones(len(chains))*len(fracchains)/len(chains))
        n = gaussian_filter(n, 2.0)
        x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
        y0 = np.array(list(zip(n, n))).flatten()
        ax[axi].plot(x0, y0, c=cs[1], label='Fraction of DM = 1')
        if axi == 0:
            ax[axi].legend(loc='lower left', framealpha=1.0, fontsize=9)
        ax[axi].set_ylim([np.min(y0)*.1, np.max(y0)*1.1])
        ax[axi].set_xlim([0,np.max(x0)])
        ax[np.shape(fracchains)[1]].set_xlim([0,np.max(x0)])
    ax[np.shape(fracchains)[1]*np.shape(chains)[1]].set_ylim([0,1])
    if (np.shape(chains)[1] > 1):
        corner.hist2d(chains[:,0], chains[:,1], labels=labeldict[f'{lenstype}'], smooth=smoothparam,
                  levels=[0.68, 0.95], ax=ax[np.shape(chains)[1]+1],
                  color='orange', plot_density=False,
                  max_n_ticks=10,plot_datapoints=False,
                  **{'zorder':1}, label='Fraction of DM = 1');
        corner.hist2d(fracchains[:,0], fracchains[:,1], labels=labeldict[f'{lenstype}'], smooth=smoothparam,
                  levels=[0.68, 0.95], ax=ax[np.shape(chains)[1]+1],
                  color='steelblue',
                  max_n_ticks=10,plot_datapoints=False,
                  **{'zorder':1});
    if (np.shape(chains)[1] > 1) and (lenstype=='gaussian'):
        ax[np.shape(fracchains)[1]].set_ylim([-4,4])
        ax[np.shape(fracchains)[1]*np.shape(chains)[1]+1].set_xlim([-4,4])
    elif (np.shape(chains)[1] > 1) and (lenstype=='nfw'):
        ax[np.shape(fracchains)[1]].set_ylim([0,4])
        ax[np.shape(fracchains)[1]*np.shape(chains)[1]+1].set_xlim([0,4])

    ax[0].set_ylabel(r'$p(\log_{10} M_l)$');
    ax[-1].set_xlim([0,1])
    ax[-3].set_xlim([0,np.max(fracchains[:,0])])
    ax[-2].set_ylim([0,1])
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))

    f.tight_layout()
    savefig(f, FIGPATH)
    PAPERPATH = make_file_path(PAPERDIR, {}, extra_string=f'fig_{lenstype}',
            ext='.png')
    savefig(f,PAPERPATH,
            writepdf=False)
## MAIN
chains = get_chains(lenstypes, surveys)
for s in surveys:
    for l in lenstypes:
        make_corner(chains, l,s, labels)
