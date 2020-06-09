'''
Functions for calculating the background model
Version 0: v simple model -- just uses distance from core and fixed core mass.
'''

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
import pymc3 as pm

from dmsl.constants import *
from dmsl.convenience import prange

def alphab(r_, rtheta_=None):
    alpha_units = (const.G*MASS_MWCORE/(1.*u.kpc**3)).to(u.uas/u.yr**2,
            equivalencies=u.dimensionless_angles())
    if rtheta_ == None:
        alphafromr = alpha_units.value*1./(r_**3)
    else:
        rx = r_*pm.math.cos(rtheta_)
        ry = r_*pm.math.sin(rtheta_)
        rxy = pm.math.stack(rx, ry)
        alphafromr = alpha_units.value*1./(rxy**3)

    return alphafromr

def get_theta_to_GC(lcen=1.*u.deg, bcen=-1*u.deg):
    x = np.array([lcen.value-FOV_DEG.value, lcen.value+FOV_DEG.value])
    y = np.array([bcen.value-FOV_DEG.value, bcen.value+FOV_DEG.value])
    coords = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    wfirst = SkyCoord(l=coords[:,0]*u.deg, b=coords[:,1]*u.deg, frame='galactic')
    gc = SkyCoord(l=0*u.deg, b=0*u.deg, frame='galactic')
    theta = (wfirst.position_angle(gc).to(u.deg) - 270*u.deg).to(u.rad)
    prange(theta)
    return theta.value


