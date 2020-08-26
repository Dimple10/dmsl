'''
Makes plots
'''


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pymc3 as pm
import seaborn as sns
import corner
import numpy as np
import pandas as pd
import scipy

from datetime import datetime
from dmsl.convenience import flatten, prange, make_file_path
from dmsl.paths import *
from pymc3.backends.tracetab import trace_to_dataframe

def paper_plot():
    sns.set_context("paper")
    sns.set_style('ticks')
    sns.set_palette('colorblind')
    plt.rc('font', family='serif', serif='cm10')
    figparams = {
            'text.latex.preamble': [r'\usepackage{amsmath}', r'\boldmath',
            r'\bf'],
            'text.usetex':True,
            'axes.labelsize':20.,
            'xtick.labelsize':16,
            'ytick.labelsize':16,
            'figure.figsize':[6., 4.],
            'font.family':'DejaVu Sans',
            'legend.fontsize':12}
    plt.rcParams.update(figparams)
    cs = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return cs

def plot_trace(trace, outpath):
    fig = plt.figure(figsize=(7, 7))
    pm.traceplot(trace, divergences="bottom", compact=True)
    #plt.tight_layout()
    #savefig(fig, outpath, writepdf=0, dpi=100)
    plt.savefig(outpath, dpi=100)
    print('{}: made {}'.format(datetime.now().isoformat(), outpath))

def plot_emcee(flatchain,nstars, nsamples,ndims, massprofiletype, kwargs):
    for key in kwargs:
        if key ==  'rs':
            continue
        outpath = make_file_path(RESULTSDIR, [nstars, nsamples,
            ndims],extra_string=f'post_{massprofiletype}_{key}',
            ext='.png')
        plt.close('all')
        paper_plot()
        i = 0
        if key == 'Ml': i  = 0
        else: i = 1
        up95 = np.percentile(flatchain[:,i],68)
        fig = plt.figure()
        plt.hist(flatchain[:, i], 50, color="k", histtype="step", density=True);
        plt.axvline(up95)
        plt.xlabel(f'$\\log_{10} {key}$');
        plt.ylabel(f'$p({key})$');
        savefig(fig, outpath, writepdf=0, dpi=100)
    plt.close('all')
    fig = corner.corner(flatchain)
    outpath = make_file_path(RESULTSDIR, [nstars, nsamples,
        ndims],extra_string=f'corner_{massprofiletype}',
        ext='.png')
    savefig(fig, outpath, writepdf=0, dpi=100)

def plot_chains(samples, outpath):
    plt.close('all')
    paper_plot()
    fig = plt.figure()
    plt.plot(samples[:,:,0])
    plt.ylabel(r'$\log_{10} M_l$');
    plt.xlabel(r'N');
    savefig(fig, outpath, writepdf=0, dpi=100)

def plot_logprob(samples, outpath):
    plt.close('all')
    paper_plot()
    fig = plt.figure()
    plt.plot(np.abs(samples[100:]))
    plt.yscale('log')
    plt.ylabel(r'$\mid \log \mathcal{L} \mid$');
    plt.xlabel(r'N');
    savefig(fig, outpath, writepdf=0, dpi=100)

def plot_sensitivity(flatchain, outpath):
#     # Make a 2d normed histogram
#     H,xedges,yedges=np.histogram2d(flatchain[:,0], flatchain[:,1],bins=40,normed=True)
# 
#     norm=H.sum() # Find the norm of the sum
#     # Set contour levels
#     contour1=0.99
#     contour2=0.95
#     contour3=0.68
# 
#     # Set target levels as percentage of norm
#     target1 = norm*contour1
#     target2 = norm*contour2
#     target3 = norm*contour3
# 
#     # Take histogram bin membership as proportional to Likelihood
#     # This is true when data comes from a Markovian process
#     def objective(limit, target):
#         w = np.where(H>limit)
#         count = H[w]
#         return count.sum() - target
# 
#     # Find levels by summing histogram to objective
#     level1= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target1,))
#     level2= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target2,))
#     level3= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target3,))
# 
#     # For nice contour shading with seaborn, define top level
#     level4=H.max()
#     levels=[level1,level2,level3,level4]
    plt.close('all')
    cs = paper_plot()
    fig, ax = plt.subplots()
    xs = np.linspace(0, 8, 100)
    plt.plot(xs, 0.35*xs-2, c='black', label='Mishra-Sharma 95\% limit',
            linewidth=2)
    sns.kdeplot(flatchain[:,0], flatchain[:,1], ax=ax, shade=True,
            n_levels=3, cmap='Blues_r')
    # corner.hist2d(flatchain[:,0], flatchain[:,1], ax=ax, plot_datapoints=False,
    #         plot_density=False, fill_contours=True, cmap=cm.Blues_r,
    #             levels=[0.997])
    fig.legend()
    ax.set_xlim([0, 8])
    ax.set_ylim([-3., 3.])
    ax.set_xlabel(r'$\log_{10} M_l$');
    ax.set_ylabel(r'$\log_{10} R~[\rm{pc}]$');
    savefig(fig, outpath, writepdf=False, dpi=100)


def plot_corner(trace, outpath):
    # corner plot of posterior samples
    plt.close('all')
    corvars = [x for x in trace.varnames if x[-1] != '_']
    trace_df = trace_to_dataframe(trace, varnames=corvars)
    fig = corner.corner(trace_df,quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        title_fmt='.2g')
    savefig(fig, outpath, writepdf=0, dpi=100)

def plot_nice_lens_corner_1D(trace, outpath):
    # corner plot of posterior samples
    plt.close('all')
    paper_plot()
    #corvars = ['logMl', 'logbs', 'logabsalphal']
    corvars = ['logMl', 'logabsalphal']
    trace_df = trace_to_dataframe(trace, varnames=corvars)
    trace_df['logMl'] = np.log10(np.exp(trace_df['logMl']))
    #trace_df['logbs'] = np.log10(np.exp(trace_df['logbs']))
    trace_df['logabsalphal'] = np.log10(np.exp(trace_df['logabsalphal']))
    #labs = [r'$\log_{10} M_l$', r'$\log_{10} b$', r'$\log_{10}\alpha_l$']
    labs = [r'$\log_{10} M_l$', r'$\log_{10}\alpha_l$']
    fig = corner.corner(trace_df, labels=labs, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        title_fmt='.2g')
    savefig(fig, outpath, writepdf=0, dpi=100)

def plot_nice_bkg_corner_1D(trace, outpath):
    # corner plot of posterior samples
    plt.close('all')
    paper_plot()
    corvars = ['rmw', 'logabsalphab']
    trace_df = trace_to_dataframe(trace, varnames=corvars)
    trace_df['logabsalphab'] = np.log10(np.exp(trace_df['logabsalphab']))
    labs = [r'$r_{\rm{MW}}$',r'$\log_{10}\alpha_b$']
    fig = corner.corner(trace_df, labels=labs, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        title_fmt='.2g')
    savefig(fig, outpath, writepdf=0, dpi=100)
def plot_nice_lens_corner_2D(trace, outpath):
    # corner plot of posterior samples
    plt.close('all')
    paper_plot()
    corvars = ['logMl', 'logbs', 'logabsalphal', 'bstheta']
    trace_df = trace_to_dataframe(trace, varnames=corvars)
    trace_df['logMl'] = np.log10(np.exp(trace_df['logMl']))
    trace_df['logbs'] = np.log10(np.exp(trace_df['logbs']))
    alphalr = np.sqrt(np.exp(trace_df['logabsalphal__1'])**2+ np.exp(trace_df['logabsalphal__0'])**2)
    trace_df['logalphar'] = np.log10(alphalr)
    cornerdf = trace_df[['logMl', 'logbs', 'logalphar', 'bstheta']]
    labs = [r'$\log_{10} M_l$', r'$\log_{10} b$', r'$\log_{10}\alpha_l$',
    r'$b_{\theta}$']
    fig = corner.corner(cornerdf, labels=labs, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12},
                        title_fmt='.2g')
    savefig(fig, outpath, writepdf=0, dpi=100)

def savefig(fig, figpath, writepdf=True, dpi=450):
    ## stolen from luke
    fig.savefig(figpath, dpi=dpi, bbox_inches='tight')
    print('{}: made {}'.format(datetime.now().isoformat(), figpath))

    if writepdf:
        pdffigpath = figpath.replace('.png','.pdf')
        fig.savefig(pdffigpath, bbox_inches='tight', rasterized=True, dpi=dpi)
        print('{}: made {}'.format(datetime.now().isoformat(), pdffigpath))

    plt.close('all')

def plot_lens_vector_field(x,y,lensx, lensy, outpath):
    ## use a weird symlog because lensx, lensy < 1.
    symlog = lambda x:(np.sign(x)*np.log10(np.abs(x)))**(-1)
    prange(lensx)
    prange(lensy)
    M = np.log10(np.sqrt(lensy**2+lensx**2))
    norm = matplotlib.colors.Normalize()
    norm.autoscale(M)
    cm = matplotlib.cm.Blues
    sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    plt.close('all')
    paper_plot()
    fig1, ax1 = plt.subplots()
    Q = ax1.quiver(x/3600.,y/3600.,symlog(lensx), symlog(lensy), color=cm(norm(M)),
            pivot='mid', units='inches', scale=5./1.)
    plt.annotate('', xy=(-0.5, -0.4), xytext=(-0.5, -0.5),
            arrowprops=dict(facecolor='black', shrink=0.05))
    plt.text(-0.48, -0.46,r'$v_l$', fontsize=20)
    plt.scatter(0.0,0.0, marker='x',color='black', s=36)
    plt.colorbar(sm, label=r'$\log_{10}\|\alpha\|~[\mu \rm{as/yr}^2]$')
    plt.xlabel(r'$\rm{X}~[\rm{arcmin}]$')
    plt.ylabel(r'$\rm{Y}~[\rm{arcmin}]$')
    savefig(fig1, outpath, writepdf=False)
