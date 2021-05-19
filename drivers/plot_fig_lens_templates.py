'''
Plot vector field of acceleration signal
'''

## load packages
import sys
import numpy as np
import pymc3 as pm
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
        AutoMinorLocator, FuncFormatter)

from dmsl.paths import *
from dmsl.convenience import *
from dmsl.plotting import *
from dmsl.lensing_model import *
from dmsl.mass_profile import *


## set params
n = 16
vltheta = np.pi / 2.
vlthetadeg = vltheta * 180 / np.pi
vl = (1.e-3 * const.c).to(u.km / u.s).value
vvec = np.array([vl * np.cos(vltheta), vl * np.sin(vltheta)])
rarray = np.logspace(-2, 2, 2000) * u.kpc
nfwprops = {'Ml' : 1.e7 * u.Msun, 'c200' : 13.}
mnfw = NFW(**nfwprops)
mps = PointSource(**nfwprops)
size = 6. * mnfw.rs.value
fileinds = [n, vlthetadeg]
path = make_file_path(VECLENSDIR, fileinds, ext='.dat')


## print some useful info
print("Size of window: ", (size*u.rad).to(u.arcsec))
rschwarz = (2*const.G*nfwprops['Ml']/const.c**2).to(u.kpc).value
print('Size of PS lens: ', (rschwarz*u.rad))
print('Size of NFW lens: ', (mnfw.rs.value))

def make_grid(n, size):
    X = np.linspace(-size/2., size/2., n)
    Y = np.linspace(-size/2., size/2., n)
    U, V = np.meshgrid(X, Y)
    return U,V, U.reshape(n*n), V.reshape(n*n)

def make_vecfield(n, size, massprofile):
    U, V, X, Y = make_grid(n,size)

    vec_field = np.zeros((n**2, 2))
    for i, (x,y) in enumerate(zip(X,Y)):
        b = np.array([x,y])
        alpha= alphal(massprofile, b, vvec)
        vec_field[i, :] = alpha

    lensX = np.array(vec_field[:,0])
    lensY = np.array(vec_field[:,1])
    data = np.array([lensX, lensY]).T
    table = pd.DataFrame(data, columns=['lensX', 'lensY'])
    table.to_csv(path, index=False)
    print('{}: made {}'.format(datetime.now().isoformat(), path))
    return lensX, lensY

def load_vecfield(n, size):
    data = pd.read_csv(path)
    print('Succesfully loaded file.')
    lensX = data['lensX'].to_numpy()
    lensY = data['lensY'].to_numpy()
    return lensX, lensY

def plot_vecfield(x,y, lensx, lensy, rs_arcsec, ax, magnorm, cm):
    paper_plot()
    Q = ax.quiver(x/rs_arcsec,y/rs_arcsec,lensx/magnorm, lensy/magnorm,
            color=cm(norm(magnorm)), pivot='mid', units='width', scale=3./1.,
            scale_units='xy', width=0.006)
    circle1 = plt.Circle((0, 0), 1., color='k', fill=False,
            linewidth=0.5)
    ax.add_artist(circle1)
    circle2 = plt.Circle((0, 0), 2., color='k', fill=False,
            linewidth=0.5)
    ax.add_artist(circle2)
    circle3 = plt.Circle((0, 0), 3., color='k', fill=False,
            linewidth=0.5)
    ax.add_artist(circle3)
    ax.scatter(0.0,0.0, marker='x',color='black', s=36)
    return ax, sm

## make X,Y grid
_, _, X, Y = make_grid(n,size)
## get vector field
print('Brb need to create vector field')
lensX, lensY = make_vecfield(n, size, mnfw)
lensXps, lensYps = make_vecfield(n, size, mps)
## make plot and save
paper_plot()
plt.rcParams.update({
            'xtick.labelsize':8,
            'ytick.labelsize':8})
magnorm = np.sqrt(lensX**2+lensY**2)
magnorm2 = np.sqrt(lensXps**2+lensYps**2)
norm = matplotlib.colors.LogNorm(vmin=magnorm[np.nonzero(magnorm)].min(), vmax=magnorm.max())
cm = matplotlib.cm.Blues
sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])
outpath = make_file_path(FINALDIR,[], extra_string='fig_lens_template',
        ext='.png')

f = plt.figure(figsize=(11,8))
ax = AxesGrid(f, 121,  # similar to subplot(141)
                    nrows_ncols=(1, 2),
                    axes_pad=0.0,
                    label_mode="L",
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="single"
                    )
plot_vecfield(X,Y, lensX, lensY, mnfw.rs.value, ax[0], magnorm, cm)
plot_vecfield(X,Y, lensXps, lensYps, mnfw.rs.value, ax[1], magnorm2, cm)
ax[0].annotate('', xy=(-3, -2), xytext=(-3, -3),
        arrowprops=dict(facecolor='black', shrink=0.05, width=4, headwidth=8))
ax[0].text(-2.9, -2.9,r'$v_l$', fontsize=15)

cb1 = f.colorbar(sm, ax=ax[1], cax=ax.cbar_axes[0],
        label=r'$\|\alpha\|~[\mu\rm{as/yr}^2]$')
[a.xaxis.set_major_locator(MultipleLocator(1)) for a in ax]
[a.yaxis.set_major_locator(MultipleLocator(1)) for a in ax]
[a.xaxis.set_major_locator(MultipleLocator(1)) for a in ax]
[a.yaxis.set_major_formatter(FormatStrFormatter(r"\textbf{%i}")) for a in ax]
[a.xaxis.set_major_formatter(FormatStrFormatter(r"\textbf{%i}")) for a in ax]
[a.xaxis.set_minor_locator(MultipleLocator(0.5)) for a in ax]
[a.yaxis.set_minor_locator(MultipleLocator(0.5)) for a in ax]
[a.xaxis.tick_top() for a in ax]
[a.tick_params(bottom=True, top=True, which='both') for a in ax]
ax[0].set_xlabel(r'$b_x/r_s$')
ax[1].set_xlabel(r'$b_x/r_s$')
ax[0].set_ylabel(r'$b_y/r_s$')
savefig(f, outpath, writepdf=False)
outpath = make_file_path(PAPERDIR,[], extra_string='fig_lens_template',
        ext='.png')
savefig(f, outpath, writepdf=False)
