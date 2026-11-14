'''
defines the pdf for the impact parameter
'''

import numpy as np
from scipy.stats import rv_continuous
from scipy.integrate import cumtrapz
from scipy.integrate import cumulative_trapezoid

from dmsl.convenience import *

def gs(s, a1, a2):
    gs = np.piecewise(s,
                      [s<a1**2, (a1**2<=s) & (s<a2**2), (a2**2<=s) & (s<(a1**2 + a2**2)), (s>(a1**2+a2**2))],
                   [lambda s: -2.*np.sqrt(s)/(a1**2*a2) - 2.*np.sqrt(s)/(a1*a2**2) + np.pi/(a1*a2) + s/(a1**2*a2**2),
                    lambda s: -2.*np.sqrt(s)/(a1**2*a2) -1./a2**2 + 2./(a1*a2)*np.arcsin(a1/np.sqrt(s)) + 2./(a1**2*a2)*np.sqrt(s-a1**2),
                   lambda s:-1./a2**2 + 2./(a1*a2)*np.arcsin(a1/np.sqrt(s)) + 2./(a1**2*a2)*np.sqrt(s-a1**2) - 1./a1**2 + 2./(a1*a2)*np.arcsin(a2/np.sqrt(s))+ 2./(a1*a2**2)*np.sqrt(s - a2**2) - np.pi/(a1*a2) -  s/(a1**2*a2**2),
                   lambda s: 0])
    return gs
_G_CACHE = {}

def _G_grid(a, beta_lo=1.e-13, n=6000):
    '''One-lens CDF G(beta^2) on a log grid of beta for a square field of side a [rad].'''
    if a not in _G_CACHE:
        beta = np.logspace(np.log10(beta_lo), np.log10(np.sqrt(2.) * a), n)
        s = beta**2
        G = cumulative_trapezoid(gs(s, a, a), s, initial=0.) + np.pi * s[0] / a**2
        _G_CACHE[a] = (np.log(beta), np.clip(G / G[-1], 1.e-300, 1. - 1.e-16))
    return _G_CACHE[a]

def sample_beta(n, a, N):
    '''beta [rad] of the nearest of N lenses for n independent stars, field of side a [rad].
    Inverse CDF of Q(beta) = 1 - (1 - G(beta^2))^N in log space, so the small-beta tail is resolved.'''
    logbeta, G = _G_grid(a)
    logQ = np.log(np.clip(-np.expm1(N * np.log1p(-G)), 1.e-300, 1.))
    logQ, idx = np.unique(logQ, return_index=True)
    return np.exp(np.interp(np.log(np.random.rand(n)), logQ, logbeta[idx]))

def pdf(b,a1,a2,n):
    G = cumtrapz(gs(b**2, a1,a2),x=b**2, initial=0)
    #print('G=',G)
    #print('1-G',(1-G))
    Q = 1-(1-G)**n
    #print('Q=',Q)
    tot = np.gradient(Q, b**2)
    #print('tot=',tot)
    return 2*b*tot
