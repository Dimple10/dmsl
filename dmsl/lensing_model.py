'''
Functions to calculate lensing signature
'''

import numpy as np
import astropy.units as u
import astropy.constants as const
import exoplanet as xo
import pymc3 as pm

from dmsl.convenience import pmprint, pmpshape, pshape

def alphal(Ml_, b_, vl_, btheta_=None, vltheta_=None):

    if vltheta_ == None:
        acc_units = (4*const.G*u.Msun*(1.*u.km/u.s)**2/const.c**2*1./(1.*u.kpc)**3).to(u.uas/u.yr**2,
                equivalencies=u.dimensionless_angles())
        vec_part = alphal_vec_exp(b_, vl_, 0., 0.).norm(2)
        alphal = acc_units.value*Ml_*vec_part
        return alphal
    else:
        acc_units = (4*const.G*u.Msun*(1.*u.km/u.s)**2/const.c**2*1./(1.*u.kpc)**3).to(u.uas/u.yr**2,
                equivalencies=u.dimensionless_angles())
        accmag = acc_units.value*Ml_
        vec_part = alphal_vec_exp(b_, vl_, btheta_, vltheta_)
        alphal = accmag*vec_part
        return alphal

def alphal_vec_exp(b, vl, btheta, vltheta,  vldot = None):
    bmag, bunit = make_mag_units_vecs(b, btheta)
    vvec = make_xyvec(vl, vltheta)[:,None]
    term1 = -8.*(pm.math.dot(vvec.T, bunit)**2/bmag**3)*bunit
    if vldot != None:
        term2 = 2.*(pm.math.dot(bunit, vldot)/bmag**2)*bunit
        term5 = -1.*vldot/bmag**2
    else:
        term2 = 0.
        term5 = 0.
    term3 = 2.*vl**2/bmag**3*bunit
    term4 = 4.*pm.math.dot(vvec.T, bunit)*vvec/bmag**3
    return term1 + term2 + term3 + term4 + term5

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
    mag = xyvec.norm(2,axis=0)
    unitvec = xyvec/mag
    return mag, unitvec

def make_xyvec(r, theta):
    x = r*pm.math.cos(theta)
    y = r*pm.math.sin(theta)
    xyvec = pm.math.stack(x,y)
    return xyvec



def alphal_np(Ml_, b_, vl_, btheta_=None, vltheta_=None):

    if vltheta_ == None:
        acc_units = (4*const.G*u.Msun*(1.*u.km/u.s)**2/const.c**2*1./(1.*u.kpc)**3).to(u.uas/u.yr**2,
                equivalencies=u.dimensionless_angles())
        vec_part = np.linalg.norm(alphal_vec_exp_np(b_, vl_, 0., 0.), axis=0)
        alphal = acc_units.value*Ml_*vec_part
        return alphal
    else:
        acc_units = (4*const.G*u.Msun*(1.*u.km/u.s)**2/const.c**2*1./(1.*u.kpc)**3).to(u.uas/u.yr**2,
                equivalencies=u.dimensionless_angles())
        accmag = acc_units.value*Ml_
        vec_part = alphal_vec_exp_np(b_, vl_, btheta_, vltheta_)
        alphal = accmag*vec_part
        return alphal

def alphal_vec_exp_np(b, vl, btheta, vltheta,  vldot = None):
    bmag, bunit = make_mag_units_vecs_np(b, btheta)
    vvec = make_xyvec_np(vl, vltheta)
    term1 = -8.*(np.dot(bunit.T, vvec).reshape(len(bmag))**2/bmag**3).T*bunit
    return term1
    if vldot != None:
        term2 = 2.*(np.dot(bunit, vldot)/bmag**2)*bunit
        term5 = -1.*vldot/bmag**2
    else:
        term2 = 0.
        term5 = 0.
    term3 = 2.*vl**2/bmag**3*bunit
    term4 = 4.*np.dot(bunit.T, vvec).reshape(len(bmag))*vvec/bmag**3
    alphal_vec = term1 + term2 + term3 + term4 + term5
    return alphal_vec

def alphal_theta_np(Ml_, b_, vl_, btheta_=None, vltheta_=None):
    bmag, bunit = make_mag_units_vecs_np(b_, btheta_)
    vlmag, vlunit = make_mag_units_vecs_np(vl_, vltheta_)
    direction = bunit - vlunit
    theta = btheta_- vltheta_
    return theta

def make_mag_units_vecs_np(r, theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    xyvec = np.vstack([x,y])
    mag = np.linalg.norm(xyvec,axis=0)
    unitvec = xyvec/mag
    return mag, unitvec

def make_xyvec_np(r, theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    xyvec = np.vstack([x,y])
    return xyvec
