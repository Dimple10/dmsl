'''
Functions for calculating the background model
Version 0: v simple model -- just uses distance from core and fixed core mass.
'''

import numpy as np
import astropy.units as u
import astropy.constants as const

from dmsl.constants import *

def alphab(r_):
    alpha_units = (const.G*MASS_MWCORE/(1.*u.kpc**3)).to(u.uas/u.yr**2,
            equivalencies=u.dimensionless_angles())
    alphafromr = alpha_units.value*1./(r_**3)
    return alphafromr
