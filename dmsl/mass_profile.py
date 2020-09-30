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

from dmsl.constants import RHO_CRIT

class MassProfile:

    def __init__(self, profiletype, **kwargs):
        self.type = profiletype
        self.kwargs = kwargs

    def get_profile(self):
        raise NotImplementedError("""Woops! You need to specify an actual halo
        type before you can get the mass profile!""")
        pass

    def get_mprime(self, bs, profile):
        mprime = np.gradient(profile, bs, edge_order=2)
        self.mprime = mprime
        return mprime

    def get_mpprime(self, bs, mprime):
        mpprime = np.gradient(mprime, bs, edge_order=2)
        self.mpprime = mpprime
        return mpprime

class ConstDens(MassProfile):

    def __init__(self, **kwargs):
        super().__init__('constdens')
        self.kwargs = kwargs
        self.params = ['Ml']
        self.nicename = 'Const.~Dens.'
        self.bs, self.profile = self.get_profile(**kwargs)
        self.nparams = len(self.kwargs)
        self.get_mprime(self.bs, self.profile)
        self.get_mpprime(self.bs, self.mprime)

    def get_profile(self, **kwargs):
        rho0 = kwargs['Ml']/u.pc**3
        try:
            rs = kwargs['rs']
        except:
            rs = np.linspace(0, 1, 100)*u.kpc
        m = 4./3.*np.pi*rho0*rs**3
        return rs, m.to(u.Msun)

class Exp(MassProfile):

    def __init__(self, **kwargs):
        super().__init__('exp')
        self.kwargs = kwargs
        self.nicename = 'Exponential'
        self.bs, self.profile = self.get_profile(**kwargs)
        self.nparams = len(self.kwargs)
        self.get_mprime(self.bs, self.profile)
        self.get_mpprime(self.bs, self.mprime)

    def get_profile(self, **kwargs):
        m0 = kwargs['Ml']
        rd = kwargs['rd']
        try:
            rs = kwargs['rs']
        except:
            rs = np.linspace(0, 1, 100)*u.kpc
        m = m0*np.exp(-1.*rs/rd)
        return rs, m.to(u.Msun)

class Gaussian(MassProfile):

    def __init__(self, **kwargs):
        super().__init__('gaussian')
        self.kwargs = kwargs
        self.nicename = 'Gaussian'
        self.bs, self.profile = self.get_profile(**kwargs)
        self.nparams = len(self.kwargs)
        self.get_mprime(self.bs, self.profile)
        self.get_mpprime(self.bs, self.mprime)

    def get_profile(self, **kwargs):
        m0 = kwargs['Ml']
        r0 = kwargs['R0']
        try:
            rs = kwargs['rs']
        except:
            rs = np.linspace(0, 1, 100)*u.kpc
        denom = 2.*np.sqrt(2)*np.pi**(1.5)*r0**3
        rho = (m0/denom*np.exp(-0.5*rs**2/r0**2)).to(u.Msun/u.kpc**3)
        ## FIXME: can do this analytically.
        m = cumtrapz(rs, rho, initial=0.)
        return rs, m*u.Msun

class NFW(MassProfile):

    def __init__(self, **kwargs):
        super().__init__('nfw')
        self.kwargs = kwargs
        self.nicename = 'NFW'
        self.bs, self.profile = self.get_profile(**kwargs)
        self.nparams = len(self.kwargs)
        self.get_mprime(self.bs, self.rhos, self.rs)
        self.get_mpprime(self.bs, self.rhos, self.rs)

    def get_profile(self, **kwargs):
        M200= kwargs['Ml']
        #rs = kwargs['rs']
        try:
            r = kwargs['rarray']
        except:
            raise ValueError("Need to specify r array for profile")

        try:
            c200 = kwargs['c200']
        except:
            print("Setting concentration to MW value")
            c200 = 13. ## MW value
            kwargs['c200'] = c200
        delta_c = (200 / 3.) * c200 ** 3 / (np.log(1 + c200) - c200 / (1 + c200))
        rhos = RHO_CRIT * delta_c
        rs = (M200 / ((4 / 3.) * np.pi * c200 ** 3 * 200 * RHO_CRIT)) ** (1 / 3.)  # NFW scale radius
        self.rhos = rhos
        self.rs = rs
        x = (r / rs).to('').value
        mr = 4 * np.pi * rhos * rs ** 3 * (np.log(x / 2.) + self.F(x))
        return r, mr.to(u.Msun)

    def M(self, r):
        x = r.value / self.rs.value
        M = 4 * np.pi * self.rhos * self.rs ** 3 * (np.log(x / 2.) + self.F(x))
        return M.to(u.Msun)
    ##### next few functions stolen from mishra sharma
    def F(self, x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                x == 1.0,
                1.0,
                np.where(x <= 1.0, np.arctanh(np.sqrt(1.0 - x ** 2)) / (np.sqrt(1.0 - x ** 2)),
                         np.arctan(np.sqrt(x ** 2 - 1.0)) / (np.sqrt(x ** 2 - 1.0))),
            )
    def dFdx(self, x):
        """ Helper function for NFW deflection, from astro-ph/0102341 eq. (49)
        """
        return (1 - x ** 2 * self.F(x)) / (x * (x ** 2 - 1))

    def d2Fdx2(self, x):
        """ Helper function for NFW deflection, derivative of dFdx
        """
        return (1 - 3 * x ** 2 + (x ** 2 + x ** 4) * self.F(x) + (x ** 3 - x**5)
                * self.dFdx(x)) / ( x ** 2 * (-1 + x ** 2) ** 2)

    def get_mprime(self,r, rhos, rs):
        num = 16.*np.pi*rhos*rs*r
        denom = (1. + r/rs)**2
        x = r.value / rs.value
        dMdb = 4 * np.pi * rhos * rs ** 2 * ((1 / x) + self.dFdx(x))
        self.mprime = num/denom
        return self.mprime

    def mprime_func(self, r):
        x = r.value / self.rs.value
        dMdb = 4 * np.pi * self.rhos * self.rs ** 2 * ((1 / x) + self.dFdx(x))
        print(self.rhos, self.rs, self.dFdx(x))
        print(dMdb)
        return dMdb.to(u.Msun/u.kpc)

    def get_mpprime(self, r, rhos, rs):
        num = 16*np.pi*rhos*rs**3*(rs-r)
        denom = (r+rs)**3
        x = r.value / rs.value
        d2Mdb2 = 4 * np.pi * rs * rhos * (-1 / x ** 2 + self.d2Fdx2(x))
        self.mpprime = num/denom
        return self.mpprime

    def mpprime_func(self, r):
        x = r.value / self.rs.value
        d2Mdb2 = 4 * np.pi * self.rs * self.rhos * (-1 / x ** 2 + self.d2Fdx2(x))
        return d2Mdb2.to(u.Msun/u.kpc**2)

class TruncatedNFW(MassProfile):

    def __init__(self, **kwargs):
        super().__init__('tnfw')
        self.kwargs = kwargs
        self.nicename = 'Truncated NFW'
        self.bs, self.profile = self.get_profile(**kwargs)
        self.nparams = len(self.kwargs)
        self.get_mprime(self.bs, self.profile)
        self.get_mpprime(self.bs, self.mprime)

    def get_profile(self, **kwargs):
        m0 = kwargs['Ml']
        r0 = kwargs['r0']
        rt = kwargs['rt']
        try:
            rs = kwargs['rs']
        except:
            rs = np.linspace(0, 1, 100)*u.kpc
        ## mishra-sharma eqn 17 my r0 = their rs
        term1 = m0/(4*np.pi*rs*(rs+r0)**2)
        term2 = rt**2/(rs**2+rt**2)
        rhor = (term1*term2).to(u.Msun/u.kpc**3)
        mr = cumtrapz(rs, rhor, initial=0.)
        return rs, mr*u.Msun

class Compact(MassProfile):

    def __init__(self):
        super().__init__('compact')
        self.kwargs = kwargs
        self.nicename = 'compact'
        self.bs, self.profile = self.get_profile()
        self.get_mprime(self.bs, self.profile)
        self.get_mpprime(self.bs, self.mprime)

    def get_profile(self, *kwargs):
        m0 = kwargs['Ml']
        r0 = kwargs['r0']
        n = kwargs['n']
        try:
            rs = kwargs['rs']
        except:
            rs = np.linspace(0, 1, 100)*u.kpc
        mr = m0*(rs/r0)**n
        return rs, mr.to(u.Msun)


class From_File(MassProfile):

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
