'''
Makes plots
'''


import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import corner
import numpy as np
import pandas as pd

from datetime import datetime
from dmsl.convenience import flatten
from pymc3.backends.tracetab import trace_to_dataframe

def paper_plot():
    sns.set_context("paper")
    sns.set_style('ticks')
    sns.set_palette('colorblind')
    figparams = {
            'text.latex.preamble': [r'\usepackage{amsmath}'],
            'text.usetex':True,
            'axes.labelsize':20.,
            'xtick.labelsize':16,
            'ytick.labelsize':16,
            'figure.figsize':[14., 12.],
            'font.family':'DejaVu Sans',
            'legend.fontsize':12}
    plt.rcParams.update(figparams)
    cs = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_trace(trace, outpath):
    fig = plt.figure(figsize=(7, 7))
    pm.traceplot(trace, divergences="bottom", compact=True)
    #plt.tight_layout()
    #savefig(fig, outpath, writepdf=0, dpi=100)
    plt.savefig(outpath, dpi=100)
    print('{}: made {}'.format(datetime.now().isoformat(), outpath))

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
    corvars = ['logMl', 'logbs', 'logabsalphal']
    trace_df = trace_to_dataframe(trace, varnames=corvars)
    trace_df['logMl'] = np.log10(np.exp(trace_df['logMl']))
    trace_df['logbs'] = np.log10(np.exp(trace_df['logbs']))
    trace_df['logabsalphal'] = np.log10(np.exp(trace_df['logabsalphal']))
    labs = [r'$\log_{10} M_l$', r'$\log_{10} b$', r'$\log_{10}\alpha_l$']
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
