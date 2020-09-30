'''
Functions to calculate lensing signature
'''

import numpy as np
import astropy.units as u
import astropy.constants as const


def alphal(Ml, bvec, vvec):
    '''
    Calculates the apparent acceleration due to lensing.
    Ml must be a mass profile instance.
    bvec must be in physical units -- if not a Quantity, will be assumed to be
    in kpc.
    vvvec must be in length / unit time units -- if not a Quantity, will be
    assumed to be in km/s.
    '''
    if isinstance(bvec, u.Quantity) == False:
        bvec *= u.kpc
    if isinstance(vvec, u.Quantity) == False:
        vvec *= u.km / u.s

    b = np.linalg.norm(bvec)
    v = np.linalg.norm(vvec)
    accmag = 4 * const.G * v**2 * Ml.M(b) / (const.c**2 * b**3)
    vec_part = alphal_vec(Ml, bvec, vvec)
    alphal = accmag * vec_part
    return (alphal).to(u.uas / u.yr**2, equivalencies = u.dimensionless_angles())

def alphal_vec(Ml, bvec, vvec, vdotvec = None):
    '''
    Calculates the unitless part of the apparent acceleration due to lensing
    (i.e. the linalg part)
    Ml must be a mass profile instance.
    bvec must be in physical units -- if not a Quantity, will be assumed to be
    in kpc.
    vvvec must be in length / unit time units -- if not a Quantity, will be
    assumed to be in km/s.
    '''
    if isinstance(bvec, u.Quantity) == False:
        bvec *= u.kpc
    if isinstance(vvec, u.Quantity) == False:
        vvec *= u.km / u.s
    b = np.linalg.norm(bvec)
    bunit = bvec / b
    v = np.linalg.norm(vvec)
    vunit = vvec / v
    bdotv = np.dot(bunit,vunit)
    ## A(b) term
    Aterm1 = -8. * (bdotv)**2 * bunit
    Aterm2 = 0.
    Aterm3 = 2. * bunit
    Aterm4 = 4.* bdotv * vunit
    Aterm5 = 0.
    if vdotvec != None:
        ## FIXME: not totally right. need to figure out magnitudes
        vdot = np.linalg.norm(vdotvec)
        vdotunit = vdotvec / vdot
        Aterm2 += 2. * (np.dot(bunit, vdotunit)) * bunit
        Aterm5 += -1. * vdotunit
    Aterm = Aterm1 + Aterm2 + Aterm3 + Aterm4 + Aterm5
    ## B(b) term
    Bterm1 = 5. * (bdotv)**2 * bunit
    Bterm2 = -2.* (bdotv) * vunit
    Bterm3 = 0.
    Bterm4 = -1. * bunit
    if vdotvec != None:
        ## FIXME
        Bterm3 += -1. * (np.dot(bunit, vdotunit)) * bunit
    Bterm = Bterm1 + Bterm2 + Bterm3 + Bterm4
    ## C(b) term
    Cterm = (bdotv)**2 * bunit
    ## Put it all together, make each term unitless
    Term1 = Aterm
    Term2 = Ml.Mprime(b) / Ml.M(b) * b * Bterm
    Term3 = Ml.Mpprime(b) / Ml.M(b) * b**2 * Cterm
    alphal_vec = Term1 + Term2 + Term3
    return alphal_vec

## def alphal_theta_np(Ml_, b_, vl_, btheta_=None, vltheta_=None):
##     bmag, bunit = make_mag_units_vecs_np(b_, btheta_)
##     vlmag, vlunit = make_mag_units_vecs_np(vl_, vltheta_)
##     direction = bunit - vlunit
##     theta = btheta_- vltheta_
##     return theta
## 
## def make_mag_units_vecs_np(r, theta):
##     x = r*np.cos(theta)
##     y = r*np.sin(theta)
##     xyvec = np.vstack([x,y])
##     mag = np.linalg.norm(xyvec,axis=0)
##     unitvec = xyvec/mag
##     return mag, unitvec
## 
## def make_xyvec_np(r, theta):
##     x = r*np.cos(theta)
##     y = r*np.sin(theta)
##     xyvec = np.vstack([x,y])
##     return xyvec
################################# old ##########################
## def alphal(Ml_, b_, vl_, btheta_=None, vltheta_=None):
## 
##     if vltheta_ == None:
##         acc_units = (4*const.G*u.Msun*(1.*u.km/u.s)**2/const.c**2*1./(1.*u.kpc)**3).to(u.uas/u.yr**2,
##                 equivalencies=u.dimensionless_angles())
##         vec_part = alphal_vec_exp(b_, vl_, 0., 0.).norm(2)
##         alphal = acc_units.value*Ml_*vec_part
##         return alphal
##     else:
##         acc_units = (4*const.G*u.Msun*(1.*u.km/u.s)**2/const.c**2*1./(1.*u.kpc)**3).to(u.uas/u.yr**2,
##                 equivalencies=u.dimensionless_angles())
##         accmag = acc_units.value*Ml_
##         vec_part = alphal_vec_exp(b_, vl_, btheta_, vltheta_)
##         alphal = accmag*vec_part
##         return alphal
## 
## def alphal_vec_exp(b, vl, btheta, vltheta,  vldot = None):
##     # FIXME: delete or update with Ml profile.
##     bmag, bunit = make_mag_units_vecs(b, btheta)
##     vvec = make_xyvec(vl, vltheta)[:,None]
##     term1 = -8.*(pm.math.dot(vvec.T, bunit)**2/bmag**3)*bunit
##     if vldot != None:
##         term2 = 2.*(pm.math.dot(bunit, vldot)/bmag**2)*bunit
##         term5 = -1.*vldot/bmag**2
##     else:
##         term2 = 0.
##         term5 = 0.
##     term3 = 2.*vl**2/bmag**3*bunit
##     term4 = 4.*pm.math.dot(vvec.T, bunit)*vvec/bmag**3
##     return term1 + term2 + term3 + term4 + term5
## 
## def alphal_theta(Ml_, b_, vl_, btheta_=None, vltheta_=None):
##     bmag, bunit = make_mag_units_vecs(b_, btheta_)
##     vlmag, vlunit = make_mag_units_vecs(vl_, vltheta_)
##     direction = bunit - vlunit
##     theta = btheta_- vltheta_
##     return theta
## 
## def alphal_direction(Ml_, b_, vl_, btheta_=None, vltheta_=None):
##     bmag, bunit = make_mag_units_vecs(b_, btheta_)
##     vlmag, vlunit = make_mag_units_vecs(vl_, vltheta_)
##     direction = bunit - vlunit
##     return direction
## 
## def alphal_r(Ml_, b_, vl_, btheta_=None, vltheta_=None):
##     bmag, bunit = make_mag_units_vecs(b_, btheta_)
##     vlmag, vlunit = make_mag_units_vecs(vl_, vltheta_)
##     acc_units = (4*const.G*u.Msun*(1.*u.km/u.s)**2/const.c**2*1./(1.*u.kpc)**3).to(u.uas/u.yr**2,
##             equivalencies=u.dimensionless_angles())
##     accmag = acc_units.value*Ml_*vlmag**2/bmag**3
##     return accmag
## 
## def make_mag_units_vecs(r, theta):
##     x = r*pm.math.cos(theta)
##     y = r*pm.math.sin(theta)
##     xyvec = pm.math.stack(x,y)
##     mag = xyvec.norm(2,axis=0)
##     unitvec = xyvec/mag
##     return mag, unitvec
## 
## def make_xyvec(r, theta):
##     x = r*pm.math.cos(theta)
##     y = r*pm.math.sin(theta)
##     xyvec = pm.math.stack(x,y)
##     return xyvec
## 
