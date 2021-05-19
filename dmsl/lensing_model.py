'''
Functions to calculate lensing signature
'''

import numpy as np
import astropy.units as u
import astropy.constants as const
from dmsl.convenience import pshape


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

    if bvec.ndim > 1:
        b = np.linalg.norm(bvec, axis=1)
        v = np.linalg.norm(vvec, axis=1)
    else:
        b = np.linalg.norm(bvec)
        v = np.linalg.norm(vvec)
    accmag = 4 * const.G * v**2 / (const.c**2 * b**3)
    vec_part = alphal_vec(Ml, bvec, vvec)
    if accmag.ndim == 1:
        alphal = accmag[:,np.newaxis] * vec_part
    else:
        alphal = accmag*vec_part
    return (alphal).to(u.uas / u.yr**2, equivalencies = u.dimensionless_angles())

def alphal_vec(Ml, bvec, vvec, vdotvec = None):
    '''
    Calculates the vector part of the apparent acceleration due to lensing
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
    if bvec.ndim > 1:
        b = np.linalg.norm(bvec, axis=1)
        v = np.linalg.norm(vvec, axis=1)
    else:
        b = np.array([np.linalg.norm(bvec).value])*bvec.unit
        v = np.array([np.linalg.norm(vvec).value])*vvec.unit

    bunit = bvec / b[:, np.newaxis]
    vunit = vvec / v[:, np.newaxis]
    if bunit.ndim == 1:
        bdotv = np.dot(bunit, vunit)
    else:
        bdotv = np.einsum('ij, ij-> i', bunit, vunit).reshape(-1,1)
    ## A(b) term
    Aterm1 = -8. * (bdotv)**2 * bunit
    Aterm2 = 0.
    Aterm3 = 2. * bunit
    Aterm4 = 4.* bdotv * vunit
    Aterm5 = 0.
    if vdotvec != None:
        ## FIXME: not totally right. need to figure out magnitudes
        vdot = np.linalg.norm(vdotvec, axis=1)
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
    Cterm = -1. * (bdotv)**2 * bunit
    ## Put it all together, make each term unitless
    Term1 = Ml.M(b)[:, np.newaxis] * Aterm
    Term2 = (Ml.Mprime(b) * b)[:, np.newaxis] * Bterm
    Term3 = (Ml.Mpprime(b) * b**2)[:, np.newaxis] * Cterm
    alphal_vec = Term1 + Term2 + Term3
    return alphal_vec
