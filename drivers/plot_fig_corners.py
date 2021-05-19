
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

cs = paper_plot()

lenstypes = ['ps', 'gaussian', 'nfw']
surveys = ['Roman']
labels = {'ps':[r'$\log_{10} M_l~[\mathrm{M}_{\odot}]$', r'$\rm{Fraction~of~DM}$'],
        'gaussian': [r'$\log_{10} M_l~[\mathrm{M}_{\odot}]$',
            r'$\log_{10} R_0~[\rm{pc}]$',
            r'$\rm{Fraction~of~DM}$'],
        'nfw' : [r'$\log_{10} M_l~[\mathrm{M}_{\odot}]$',
            r'$\log_{10} c_{200}$',
            r'$\rm{Fraction~of~DM}$']}

def get_chains(lenstypes, surveys):
    chains = {}
    for survey in surveys:
        for l in lenstypes:
            tstring = f'{survey}_{l}'
            f = f'{RESULTSDIR}final/pruned_samples_{survey}_{l}_3_4_2.pkl'
            with open(f,'rb') as buff:
                chains[tstring] = dill.load(buff)
            tstring = f'{survey}_{l}_frac'
            f = f'{RESULTSDIR}final/pruned_samples_{survey}_{l}_frac_3_4_2.pkl'
            with open(f,'rb') as buff:
                chains[tstring] = dill.load(buff)
    return chains

def make_corner(chaindict, lenstype, survey, labeldict):
    chains = chaindict[f'{survey}_{lenstype}']
    fracchains = chaindict[f'{survey}_{lenstype}_frac']

    FIGPATH = make_file_path(FINALDIR, {}, extra_string=f'fig_{lenstype}',
            ext='.png')
    f = plt.figure(figsize=(8,8))
    corner.corner(fracchains, labels=labeldict[f'{lenstype}'], smooth=1.5,
                  levels=[0.68, 0.95],fig=f, smooth1d=1.0, color='steelblue',
                  max_n_ticks=10, **{'plot_datapoints':False});
    ax = f.axes
    for i in range(0, np.shape(chains)[1]):
        axi = i*np.shape(fracchains)[1] + i
        ax[axi].axvline(np.percentile(fracchains[:,0], 95), linestyle='dashed', lw=2,
                color=cs[0])
        ax[axi].axvline(np.percentile(fracchains[:,0], 5), linestyle='dashed', lw=2,
                color=cs[0])
        ax[axi].axvline(np.percentile(chains, 5), linestyle='dashed', linewidth=2,
                color=cs[1])
        ax[axi].axvline(np.percentile(chains, 95), linestyle='dashed', linewidth=2,
            color=cs[1])
        n, b = np.histogram(
                chains[:,i], bins=15,
                    weights=np.ones(len(chains))*len(fracchains)/len(chains))
        n = gaussian_filter(n, 0.8)
        x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
        y0 = np.array(list(zip(n, n))).flatten()
        ax[axi].plot(x0, y0, c=cs[1], label='Fraction of DM = 1')
        if axi == 0:
            ax[axi].legend(loc='lower left', framealpha=1.0, fontsize=9)
        ax[axi].set_ylim([np.min(y0)*.1, np.max(y0)*1.1])
        ax[axi].set_xlim([0,np.max(x0)])
        ax[np.shape(fracchains)[1]].set_xlim([0,np.max(x0)])
    ax[np.shape(fracchains)[1]*np.shape(chains)[1]].set_ylim([0,10])
    if (np.shape(chains)[1] > 1):
        corner.hist2d(chains[:,0], chains[:,1], labels=labeldict[f'{lenstype}'], smooth=1.5,
                  levels=[0.68, 0.95], ax=ax[np.shape(chains)[1]+1],
                  color='orange', plot_density=False,
                  max_n_ticks=10,plot_datapoints=False,
                  **{'zorder':1}, label='Fraction of DM = 1');
        corner.hist2d(fracchains[:,0], fracchains[:,1], labels=labeldict[f'{lenstype}'], smooth=1.5,
                  levels=[0.68, 0.95], ax=ax[np.shape(chains)[1]+1],
                  color='steelblue',
                  max_n_ticks=10,plot_datapoints=False,
                  **{'zorder':1});
    if (np.shape(chains)[1] > 1) and (lenstype=='gaussian'):
        ax[np.shape(fracchains)[1]].set_ylim([-4,4])
        ax[np.shape(fracchains)[1]*np.shape(chains)[1]+1].set_xlim([-4,4])
        #rs = np.linspace(np.min(chains[:,0]), np.max(chains[:,0]), 100)
        #msline = 0.45*rs-2.
        #ax[np.shape(fracchains)[1]].plot(rs, msline,
        #        c='black', linewidth=3, linestyle='dashed',
        #        label='Mishra-Sharma et al. (2020)')
        #ax[np.shape(fracchains)[1]].legend(loc='upper left', fontsize=8)
    elif (np.shape(chains)[1] > 1) and (lenstype=='nfw'):
        ax[np.shape(fracchains)[1]].set_ylim([0,8])
        ax[np.shape(fracchains)[1]*np.shape(chains)[1]+1].set_xlim([0,8])

    ax[0].set_ylabel(r'$p(\log_{10} M_l)$');
    ax[-1].set_xlim([0,10])
    ax[-3].set_xlim([0,np.max(fracchains[:,0])])
    ax[-2].set_ylim([0,10])
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))

    f.tight_layout()
    savefig(f, FIGPATH)
    ## DELETE FOR PUBLIC
    PAPERPATH = make_file_path(PAPERDIR, {}, extra_string=f'fig_{lenstype}',
            ext='.png')
    savefig(f,PAPERPATH,
            writepdf=False)

chains = get_chains(lenstypes, surveys)
for s in surveys:
    for l in lenstypes:
        make_corner(chains, l,s, labels)

## MAKE GAUSSIAN FIG
#with open('../results/final/pruned_samples_Roman_gaussian_3_4_2.pkl', 'rb') as buff:
#    chains = dill.load(buff)
#FIGPATH = make_file_path(FINALDIR, {}, extra_string='fig_gaussian',
#        ext='.png')
#f = plt.figure(figsize=(8,8))
#corner.corner(chains, labels=[r'$\log_{10} M_l~[\mathrm{M}_{\odot}]$', r'$\log_{10} R_0~[\rm{pc}]$'],
#              smooth=1, levels=[0.68,0.9], fig=f, max_n_ticks=10, smooth1d=0.8,
#              divergences=True, color='steelblue');
#plt.yticks(np.arange(int(min(chains[:,1])), int(max(chains[:,1]))+1, 1));
#ax = f.axes
#ax[0].axvline(np.percentile(chains[:,0], 90), linestyle='dashed', lw=2,
#        color=cs[0])
#ax[3].axvline(np.percentile(chains[:,1], 10), linestyle='dashed', lw=2,
#        color=cs[0])
#rs = np.linspace(np.min(chains[:,0]), np.max(chains[:,0]), 100)
#msline = 0.45*rs-2.
#ax[2].plot(rs, msline, c=cs[1], linewidth=4, linestyle='dashed',
#label='Mishra-Sharma et al. (2020)')
#ax[2].legend(loc='lower right', fontsize=8)
#ax[0].set_xlim([0, np.max(x0)])
#ax[2].set_xlim([0, np.max(x0)])
#ax[3].set_xlim([-4,4])
#ax[2].set_ylim([-4,4])
#ax[0].set_ylabel(r'$p(\log_{10} M_l)$');
#f.tight_layout()
#savefig(f, FIGPATH)
### DELETE FOR PUBLIC
#PAPERPATH = make_file_path(PAPERDIR, {}, extra_string='fig_gaussian',
#        ext='.png')
#savefig(f,PAPERPATH,
#        writepdf=False)
