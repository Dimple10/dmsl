'''
Defines mass profiles
'''

import os
import numpy as np
import astropy.units as u
import astropy.constants as const
import pandas as pd

from scipy.integrate import cumtrapz
from scipy.interpolate import UnivariateSpline

from astropy.cosmology import WMAP9 as cosmo
RHO_CRIT = cosmo.critical_density(0.)

class MassProfile:

    def __init__(self, profiletype, **kwargs):
        self.type = profiletype
        self.kwargs = kwargs

    def M(self):
        raise NotImplementedError("""Woops! You need to specify an actual halo
        type before you can get the mass profile!""")
        pass

    def Mprime(self, bs, profile):
        mprime = np.gradient(profile, bs, edge_order=2)
        self.mprime = mprime
        return mprime

    def Mpprime(self, bs, mprime):
        mpprime = np.gradient(mprime, bs, edge_order=2)
        self.mpprime = mpprime
        return mpprime

class PointSource(MassProfile):

    def __init__(self, **kwargs):
        super().__init__('ps')
        self.kwargs = kwargs
        self.params = ['Ml']
        self.nicename = 'Point~Source'
        self.nparams = len(self.kwargs)

    def M(self, b):
        try:
            return self.kwargs['Ml']* np.ones((len(b)))
        except:
            return self.kwargs['Ml']

    def Mprime(self, b):
        '''
        use at your own peril
        '''
        try:
            return np.zeros((len(b))) * self.Ml.unit / b.unit
        except:
            return 0. * self.kwargs['Ml'] / b.unit

    def Mpprime(self, b):
        '''
        use at your own peril
        '''
        return self.Mprime(b) * 1. / b.unit


class Gaussian(MassProfile):
    def __init__(self, **kwargs):
        super().__init__('gaussian')
        self.kwargs = kwargs
        self.nicename = 'Gaussian'
        self.nparams = len(self.kwargs)
        self.M(1.*u.kpc)

    def M(self, b):
        m0 = self.kwargs['Ml']
        r0 = self.kwargs['R0']
        denom = 2.*np.sqrt(2)*np.pi**(1.5)*r0**3
        m = m0 * (1 - np.exp(-b ** 2 / (2 * r0 ** 2)))
        return m.to(u.Msun)

    def Mprime(self, b):
        m0 = self.kwargs['Ml']
        r0 = self.kwargs['R0']
        mp = (m0 * b / r0 ** 2) * np.exp(-b ** 2 / (2 * r0 ** 2))
        return mp.to(u.Msun/u.kpc)

    def Mpprime(self, b):
        m0 = self.kwargs['Ml']
        r0 = self.kwargs['R0']
        mpp = m0 * (-((b ** 2 * np.exp(-(b ** 2 / (2 * r0 ** 2)))) / r0 ** 4) + np.exp(
            -(b ** 2 / (2 * r0 ** 2))) / r0 ** 2)
        return mpp.to(u.Msun/u.kpc**2)

class NFW(MassProfile):

    def __init__(self, **kwargs):
        super().__init__('nfw')
        self.kwargs = kwargs
        self.nicename = 'NFW'
        self.nparams = len(self.kwargs)
        self.M(1.*u.kpc) ## use this to get rs and rhos

    def M(self, b):
        M200 = self.kwargs['Ml']
        try:
            c200 = self.kwargs['c200']
        except:
            print("Setting concentration to MW value")
            c200 = 13. ## MW value
            self.kwargs['c200'] = c200
        delta_c = (200 / 3.) * c200**3 / (np.log(1 + c200) - c200 / (1 + c200))
        rhos = RHO_CRIT * delta_c
        rs = (M200 / ((4 / 3.) * np.pi * c200**3 * 200 * RHO_CRIT))**(1 / 3.)
        self.rhos = rhos.to( u.Msun / u.kpc**3 )
        self.rs = rs.to( u.kpc )
        self.delta_c = delta_c
        x = ( b / rs ).to('').value
        Mr = 4. * np.pi * rhos * rs**3 * ( np.log( x / 2. ) + self.F(x) )
        ## some instability in np arctanh (?) causes weird mass values below certain x.
        Mr[x<1.e-3] = 0.*u.Msun
        return Mr.to(u.Msun)

    def Mprime(self, b):
        x = ( b / self.rs ).to('').value
        mlprime = 4. * np.pi * self.rhos * self.rs**2 * ( (1. / x) + self.dFdx(x) )
        return mlprime.to( u.Msun / u.kpc )

    def Mpprime(self, b):
        x = ( b / self.rs ).to('').value
        mlpprime = 4 * np.pi * self.rs * self.rhos * ( (-1. / x**2) + self.d2Fdx2(x) )
        return mlpprime.to( u.Msun / u.kpc**2 )

    def F(self, x):
        '''
        Helper function for NFW profile.
        shamelessly stolen from @smsharma/astrometry-lensing-corrections
        '''
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                x == 1.0,
                1.0,
                np.where(x <= 1.0, np.arctanh(np.sqrt(1.0 - x ** 2)) / (np.sqrt(1.0 - x ** 2)),
                         np.arctan(np.sqrt(x ** 2 - 1.0)) / (np.sqrt(x ** 2 - 1.0))),
            )
    def dFdx(self, x):
        '''
        First derivative of helper function for NFW profile.
        with inspiration from @smsharma/astrometry-lensing-corrections
        '''
        return (1 - x ** 2 * self.F(x)) / (x * (x ** 2 - 1))

    def d2Fdx2(self, x):
        '''
        Second derivative of helper function for NFW profile.
        with inspiration from @smsharma/astrometry-lensing-corrections
        '''
        return (1 - 3 * x ** 2 + (x ** 2 + x ** 4) * self.F(x) + (x ** 3 -
            x**5) * self.dFdx(x)) / ( x ** 2 * (-1 + x ** 2) ** 2)

class Noise(MassProfile):
    '''
    dummy class to allow for likelihood calculation of just white noise
    '''
    def __init__(self, **kwargs):
        super().__init__('noise')
        self.kwargs = kwargs
        self.nicename = 'Noise'
        self.nparams = 2


class From_File(MassProfile):
    '''
    use at your own peril
    '''

    def __init__(self,filename,name,**kwargs):
        super().__init__('file')
        self.filename = filename
        self.nicename = name
        self.kwargs = kwargs
        if os.path.exists(filename):
            print('Loading mass data')
            massdat = pd.read_csv(filename)
        else:
            raise FileNotFoundError("No file with this name!")

        self.bs, self.profile = self.get_profile(massdat, **kwargs)
        self.nparams = len(self.kwargs)
        self.get_mprime(self.bs, self.profile)
        self.get_mpprime(self.bs, self.mprime)

    def get_profile(self, massdat, **kwargs):
        r = massdat['r'].to_numpy()*u.kpc
        m = kwargs['Ml']
        mr = massdat['m'].to_numpy()*u.Msun
        spline = UnivariateSpline(r, mr*m, s=0.)
        bs = kwargs['rs']
        return bs, spline(bs)*u.Msun
