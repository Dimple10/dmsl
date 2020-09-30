'''
test full acceleration form
'''
import numpy as np
import astropy.units as u
import astropy.constants as const
from dmsl.lensing_model import *
from dmsl.mass_profile import *

Ml = PointSource(**{'Ml':1.e7*u.Msun})
bvec = np.array([0.0, 1.0])*u.kpc
vvec = np.array([0.0, 1.e-3*const.c.to(u.km/u.s).value])*u.km/u.s

al = alphal(Ml, bvec, vvec)
print(f'Using a mass type: {Ml.nicename}')
print(f'For bvector: {bvec}')
print(f'and vvector: {vvec}')
print(f'I calculate an acceleration of: {al}')


Ml = NFW(**{'Ml':1.e7*u.Msun})
bvec = np.array([0.0, 1.0])*u.kpc
vvec = np.array([0.0, 1.e-3*const.c.to(u.km/u.s).value])*u.km/u.s

al = alphal(Ml, bvec, vvec)
print(f'Using a mass type: {Ml.nicename}')
print(f'For bvector: {bvec}')
print(f'and vvector: {vvec}')
print(f'I calculate an acceleration of: {al}')
print(f'Rs is: {Ml.rs}')
rsvec = np.array([0,1])*Ml.rs*.99
print(f'For b=Rs, acceleration is: {alphal(Ml, rsvec, vvec)}')
