'''
Makes plots
'''


import matplotlib.pyplot as plt
import pymc3 as pm
import corner

from datetime import datetime
from dmsl.convenience import flatten
from pymc3.backends.tracetab import trace_to_dataframe


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
    fig = corner.corner(trace_df, quantiles=[0.16, 0.5, 0.84],
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
