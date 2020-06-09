'''
not actually a test -- just calculating crossing time of lenses
'''

import numpy as np
import astropy.units as u

from dmsl.constants import *


def crossing_time(dx, v):
    t = dx/v
    return t.to(u.yr)

onskymotion = (300*u.km/u.s*1./(1.*u.kpc)*180/np.pi*u.deg).to(u.arcsec/u.yr)

print('We want to know how long it would take a lens to cross:', LENS_FOV)
print('It will take: ', crossing_time(LENS_FOV, 300*u.km/u.s))
print('We assumed a velocity of 300 km/s at 1 kpc from us.')
print('This is an onsky motion of {}'.format(onskymotion))


nstars = 1e8
avespace = np.sqrt(FOV**2/nstars)
timetonext = crossing_time(avespace/2.*u.kpc, 300*u.km/u.s)

print('Now let us consider a related question:')
print('If a lens is right on top of a star, how long until it is halfway to the next star?')
print('Typical spacing of stars:', (avespace*u.rad).to(u.arcsec))
print('Then time to halfway is:', timetonext)
