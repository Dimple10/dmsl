
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
   # corner.corner(chains, labels=labeldict[f'{lenstype}'], smooth=2.0,
   #               levels=[0.68, 0.95],fig=f, smooth1d=2.0, color='steelblue',
   #               max_n_ticks=10, **{'plot_datapoints':False});
    corner.hist2d(chains[:,0], chains[:,1], labels=labeldict[f'{lenstype}'], smooth=1.8,
                  levels=[0.68, 0.95],fig=f, color='steelblue',
                  max_n_ticks=10, **{'plot_datapoints':False});
    ax = f.axes
    if (np.shape(chains)[1] > 1) and (lenstype=='gaussian'):
        rs = np.linspace(np.min(chains[:,0]), np.max(chains[:,0]), 100)
        msline = 0.45*rs-2.
        #ax[np.shape(chains)[1]].plot(rs, msline,
        #        c='black', linewidth=3, linestyle='dashed',
        #        label='Mishra-Sharma et al. (2020)')
        #ax[np.shape(chains)[1]].legend(loc='upper left', fontsize=8)
        ax[0].plot(rs, msline,
                c='black', linewidth=3, linestyle='dashed',
                label='Mishra-Sharma et al. (2020)')
        ax[0].legend(loc='upper left', fontsize=8)

    ax[0].set_xlabel(labeldict['gaussian'][0])
    ax[0].set_ylabel(labeldict['gaussian'][1])
    #ax[0].set_ylabel(r'$p(\log_{10} M_l)$');
    #ax[-1].set_xlim([-2,4])
    ax[0].set_ylim([-2,4])
    #ax[-2].set_xlim([np.min(chains[:,0]), np.max(chains[:,0])])
    #ax[0].set_xlim([np.min(chains[:,0]), np.max(chains[:,0])])
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))

    f.tight_layout()
    savefig(f, FIGPATH)
    ## DELETE FOR PUBLIC
    PAPERPATH = make_file_path(PAPERDIR, {}, extra_string=f'fig_{survey}',
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
