'''
gets pdf for subhalo distances from sun
'''

import numpy as np
import astropy.units as u
import astropy.constants as const
import scipy.stats

## use same parameters as Mondino+ 2020
RHO_S = 0.003*u.Msun/u.pc**3
R_S = 18.*u.kpc
R_SUN = 8.*u.kpc


def density(r):
    denom = r / R_S * (1. + r / R_S)**2
    rhor = 4. * RHO_S / denom
    return (rhor).to(u.Msun/u.kpc**3)

def initialize_dist(target='GC', rmax=4.0, rmin=1.e-3, nbins=4000):
    '''
    p(d_L) proportional to rho(r(d_L)) * d_L^2 on [rmin, rmax] kpc: the DM density along the
    line of sight times the volume element of the survey cone. r = R_SUN - d_L towards the
    Galactic centre ('GC'), R_SUN + d_L otherwise. Returns a frozen distribution with .rvs(n) in kpc.
    '''
    edges = np.linspace(rmin, rmax, nbins + 1)
    d = 0.5 * (edges[1:] + edges[:-1])
    rsun = R_SUN.to(u.kpc).value
    r = rsun - d if target == 'GC' else rsun + d
    w = density(r * u.kpc).value * d**2
    return scipy.stats.rv_histogram((w, edges))()

# def initialize_dist(target='GC', rmax=1.0):
#     if target == 'GC':
#         rarray = np.linspace(0.001,4.0, 1000)*u.kpc
#         mw_r = R_SUN - rarray
#         rhor = density(mw_r)
#         rsamples = 1. - (rhor-np.min(rhor))/(np.max(rhor)-np.min(rhor))
#         '''
#         rho(r) monotonically decreases as r increases. Largest probability is
#         at lowest r == farthest from sun towards MW center. By rescaling this
#         to a 0-1 distribution, we still get the correct relationship between
#         r's. So, can just sample from this distribution, but need 1 - because r
#         array is backwards since need to subtract from RSUN
#         '''
#     elif target == 'out':
#         rarray = np.linspace(0.,rmax, 1000)*u.kpc
#         mw_r = R_SUN + rarray
#         rhor = density(mw_r).value
#         rsamples = (rhor - np.min(rhor))/(np.max(rhor)-np.min(rhor))*rmax

#     hist = np.histogram(rsamples, bins=10)
#     hist_dist = scipy.stats.rv_histogram(hist)
#     return hist_dist()


