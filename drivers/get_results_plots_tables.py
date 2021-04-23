
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

def get_filenames():
    pass

def get_nums_for_table(chain, percent=90, twosided=True, side=None):
    ave = np.average(chains)
    if twosided:
        confup = np.percentile(chains, 100-(100-percent)/2.)
        confdown = np.percentile(chains, (100-percent)/2.)
    return ave, confup, confdown

def add_to_table():
    pass

def write_table():
    pass

## MAKE PS FIG
with open('../results/final/pruned_samples_Roman_ps_frac_3_4_2.pkl', 'rb') as buff:
    fracchains = dill.load(buff)
with open('../results/final/pruned_samples_Roman_ps_3_4_2.pkl', 'rb') as buff:
    chains = dill.load(buff)
FIGPATH = make_file_path(FINALDIR, {}, extra_string='fig_ps',
        ext='.png')
f = plt.figure(figsize=(8,8))
corner.corner(fracchains, labels=[r'$\log_{10} M_l~[\mathrm{M}_{\odot}]$', r'$\rm{Fraction~of~DM}$'], smooth=1.5,
              levels=[0.68, 0.95],fig=f, smooth1d=1.0, color='steelblue',
              max_n_ticks=10);
ax = f.axes
ax[0].axvline(np.percentile(fracchains[:,0], 95), linestyle='dashed', lw=2,
        color=cs[0])
ax[0].axvline(np.percentile(fracchains[:,0], 5), linestyle='dashed', lw=2,
        color=cs[0])
ax[3].axvline(np.percentile(fracchains[:,1], 90), linestyle='dashed', lw=2,
        color=cs[0])
n, b = np.histogram(
        chains[:,0], bins=15,
            weights=np.ones(len(chains))*len(fracchains)/len(chains))
n = gaussian_filter(n, 0.8)
x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
y0 = np.array(list(zip(n, n))).flatten()
ax[0].plot(x0, y0, c=cs[1], label='Fraction of DM = 1')
ax[0].legend(loc='lower left', framealpha=1.0, fontsize=9)
#corner.corner(chains, bins=15,fig=f, smooth1d=0.8)

ax[0].axvline(np.percentile(chains, 5), linestyle='dashed', linewidth=2,
        color=cs[1])
ax[0].axvline(np.percentile(chains, 95), linestyle='dashed', linewidth=2,
        color=cs[1])

plt.xlabel(r'$\log_{10} M_l~[\mathrm{M}_{\odot}]$');
ax[0].set_ylabel(r'$p(\log_{10} M_l)$');

ax[0].set_xlim([0,np.max(x0)])
ax[2].set_xlim([0,np.max(x0)])
ax[3].set_xlim([0,10])
ax[2].set_ylim([0,10])
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))

f.tight_layout()
savefig(f, FIGPATH)
## DELETE FOR PUBLIC
PAPERPATH = make_file_path(PAPERDIR, {}, extra_string='fig_ps',
        ext='.png')
savefig(f,PAPERPATH,
        writepdf=False)


## MAKE GAUSSIAN FIG
with open('../results/final/pruned_samples_Roman_gaussian_3_4_2.pkl', 'rb') as buff:
    chains = dill.load(buff)
FIGPATH = make_file_path(FINALDIR, {}, extra_string='fig_gaussian',
        ext='.png')
f = plt.figure(figsize=(8,8))
corner.corner(chains, labels=[r'$\log_{10} M_l~[\mathrm{M}_{\odot}]$', r'$\log_{10} R_0~[\rm{pc}]$'],
              smooth=1, levels=[0.68,0.9], fig=f, max_n_ticks=10, smooth1d=0.8,
              divergences=True, color='steelblue');
plt.yticks(np.arange(int(min(chains[:,1])), int(max(chains[:,1]))+1, 1));
ax = f.axes
ax[0].axvline(np.percentile(chains[:,0], 90), linestyle='dashed', lw=2,
        color=cs[0])
ax[3].axvline(np.percentile(chains[:,1], 10), linestyle='dashed', lw=2,
        color=cs[0])
rs = np.linspace(np.min(chains[:,0]), np.max(chains[:,0]), 100)
msline = 0.45*rs-2.
ax[2].plot(rs, msline, c=cs[1], linewidth=4, linestyle='dashed',
label='Mishra-Sharma et al. (2020)')
ax[2].legend(loc='lower right', fontsize=8)
ax[0].set_xlim([0, np.max(x0)])
ax[2].set_xlim([0, np.max(x0)])
ax[3].set_xlim([-4,4])
ax[2].set_ylim([-4,4])
ax[0].set_ylabel(r'$p(\log_{10} M_l)$');
f.tight_layout()
savefig(f, FIGPATH)
## DELETE FOR PUBLIC
PAPERPATH = make_file_path(PAPERDIR, {}, extra_string='fig_gaussian',
        ext='.png')
savefig(f,PAPERPATH,
        writepdf=False)
