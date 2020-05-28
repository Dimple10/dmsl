'''
Functions to calculate lensing signature
'''

import numpy as np
import astropy.units as u
import astropy.constants as const
import exoplanet as xo
import pymc3 as pm


def alphal(Ml_, b_, vl_, btheta_=None, vltheta_=None):

    if vltheta_ == None:
        acc_units = (4*const.G*u.Msun*(1.*u.km/u.s)**2/const.c**2*1./(1.*u.kpc)**3).to(u.uas/u.yr**2,
                equivalencies=u.dimensionless_angles())
        alphal = acc_units.value*Ml_*vl_**2/b_**3
        return alphal
    else:
        bmag, bunit = make_mag_units_vecs(b_, btheta_)
        vlmag, vlunit = make_mag_units_vecs(vl_, vltheta_)
        acc_units = (4*const.G*u.Msun*(1.*u.km/u.s)**2/const.c**2*1./(1.*u.kpc)**3).to(u.uas/u.yr**2,
                equivalencies=u.dimensionless_angles())
        accmag = acc_units.value*Ml_*vlmag**2/bmag**3
        direction = bunit - vlunit
        if xo.eval_in_model(direction).all() == 0.:
            ## FIXME ?
            direction = bunit
        alphal = direction*accmag
        return alphal

def alphal_theta(Ml_, b_, vl_, btheta_=None, vltheta_=None):
    bmag, bunit = make_mag_units_vecs(b_, btheta_)
    vlmag, vlunit = make_mag_units_vecs(vl_, vltheta_)
    direction = bunit - vlunit
    theta = btheta_- vltheta_
    return theta

def alphal_direction(Ml_, b_, vl_, btheta_=None, vltheta_=None):
    bmag, bunit = make_mag_units_vecs(b_, btheta_)
    vlmag, vlunit = make_mag_units_vecs(vl_, vltheta_)
    direction = bunit - vlunit
    return direction

def alphal_r(Ml_, b_, vl_, btheta_=None, vltheta_=None):
    bmag, bunit = make_mag_units_vecs(b_, btheta_)
    vlmag, vlunit = make_mag_units_vecs(vl_, vltheta_)
    acc_units = (4*const.G*u.Msun*(1.*u.km/u.s)**2/const.c**2*1./(1.*u.kpc)**3).to(u.uas/u.yr**2,
            equivalencies=u.dimensionless_angles())
    accmag = acc_units.value*Ml_*vlmag**2/bmag**3
    return accmag

def make_mag_units_vecs(r, theta):
    x = r*pm.math.cos(theta)
    y = r*pm.math.sin(theta)
    xyvec = pm.math.stack(x,y)
    mag = xyvec.norm(2)
    unitvec = xyvec/mag
    return mag, unitvec
