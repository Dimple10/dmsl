'''
Functions to calculate lensing signature
'''

import numpy as np
import astropy.units as u
import astropy.constants as const
import exoplanet as xo


def alphal(Ml_, b_, vl_):

    if xo.eval_in_model(vl_).shape == ():
        acc_units = (4*const.G*u.Msun*(1.*u.km/u.s)**2/const.c**2*1./(1.*u.kpc)**3).to(u.uas/u.yr**2,
                equivalencies=u.dimensionless_angles())
        alphal = acc_units.value*Ml_*vl_**2/b_**3
        print(xo.eval_in_model(alphal))
    else:
        bmag = b_.norm(2)
        vlmag = vl_.norm(2)
        acc_units = (4*const.G*u.Msun*(1.*u.km/u.s)**2/const.c**2*1./(1.*u.kpc)**3).to(u.uas/u.yr**2,
                equivalencies=u.dimensionless_angles())
        accmag = acc_units.value*Ml_*vlmag**2/bmag**3
        bunit = b_/bmag
        vlunit = -1*vl_/vlmag
        direction = bunit + vlunit
        alphal = direction*accmag
    return alphal
