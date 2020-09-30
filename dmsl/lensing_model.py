'''
Functions to calculate lensing signature
'''

import numpy as np
import astropy.units as u
import astropy.constants as const
import exoplanet as xo
import pymc3 as pm

from scipy.interpolate import UnivariateSpline
from dmsl.convenience import pmprint, pmpshape, pshape



def alphal_np(Ml_, b_, vl_, dlens, btheta_=None, vltheta_=None, Mlprofile = None):

    if vltheta_ == None:
        acc_units = (4*const.G*u.Msun*(vl_*u.km/u.s)**2/const.c**2*1./(dlens)**3).to(u.uas/u.yr**2,
                equivalencies=u.dimensionless_angles())
        if Mlprofile == None:
            vec_part = np.linalg.norm(alphal_vec_exp_np(b_, vl_, 0., 0., dlens), axis=0)
            alphal = acc_units.value*Ml_*vec_part
        else:
            vec_part = np.linalg.norm(alphal_vec_exp_np(b_, vl_, 0., 0., dlens,
                Mlprofile=Mlprofile), axis=0)
            alphal = acc_units.value*vec_part

        return alphal
    else:
        acc_units = (4*const.G*u.Msun*(vl_*u.km/u.s)**2/const.c**2*1./(dlens)**3).to(u.uas/u.yr**2,
                equivalencies=u.dimensionless_angles())
        if Mlprofile == None:
            accmag = acc_units.value*Ml_
            vec_part = alphal_vec_exp_np(b_, vl_, btheta_, vltheta_, dlens)
        else:
            vec_part = alphal_vec_exp_np(b_, vl_, btheta_, vltheta_, dlens,
                Mlprofile=Mlprofile)
            accmag = acc_units.value
        alphal = accmag*vec_part
        return alphal

def alphal_vec_exp_np(b, vl, btheta, vltheta, dlens,  vldot = None, Mlprofile = None):
    bmag, bunit = make_mag_units_vecs_np(b, btheta)
    vmag, vunit = make_mag_units_vecs_np(vl, vltheta)
    ## A(b) term
    Aterm1 = -8.*(np.dot(bunit.T, vunit).reshape(len(bmag))**2).T*bunit
    Aterm2 = 0.
    Aterm3 = 2.*bunit
    Aterm4 = 4.*np.dot(bunit.T, vunit).reshape(len(bmag))*vunit
    Aterm5 = 0.
    if vldot != None:
        Aterm2 += 2.*(np.dot(bunit, vldot))*bunit*bmag
        Aterm5 += -1.*vldot*bmag
    Aterm = 1./bmag**3*(Aterm1 + Aterm2 + Aterm3 + Aterm4 + Aterm5)
    if Mlprofile != None:
        Mlb = UnivariateSpline(Mlprofile.bs.value, Mlprofile.profile.value,
                s=0., ext='zeros')
        Mlp = UnivariateSpline(Mlprofile.bs.value, Mlprofile.mprime.value,
                s=0., ext='zeros')
        Mlpp = UnivariateSpline(Mlprofile.bs.value, Mlprofile.mpprime.value,
                s=0., ext='zeros')
        Aterm *= Mlb(bmag)
#        Aterm *= Mlprofile.M(b*dlens).value
        #print(bmag, Mlb(bmag)/1.e7, Mlp(bmag)/1.e7, Mlpp(bmag)/1.e7)
        ## B(b) term
        Bterm1 = 5.*(np.dot(bunit.T, vunit).reshape(len(bmag))**2).T*bunit
        Bterm2 = 2.*np.dot(bunit.T, vunit).reshape(len(bmag))*vunit
        Bterm3 = 0.
        Bterm4 = -1.*bunit
        if vldot != None:
            Bterm3 += -1.*(np.dot(bunit, vldot))*bunit*bmag
        Bterm = 1./bmag**2*(Bterm1 + Bterm2 + Bterm3 + Bterm4)
        Bterm *= Mlp(bmag)
        #Bterm *= Mlprofile.mprime_func(b*dlens).value

        ## C(b) term
        Cterm = 1./bmag*(np.dot(bunit.T, vunit).reshape(len(bmag))**2).T*bunit
        Cterm *= Mlpp(bmag)
        #Cterm *= Mlprofile.mpprime_func(b*dlens).value
    else:
        Bterm = 0.
        Cterm = 0.
    #print(Aterm, Bterm, Cterm)
    alphal_vec = Aterm + Bterm + Cterm
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
################################# old ##########################
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
    # FIXME: delete or update with Ml profile.
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

