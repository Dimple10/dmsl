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
        self.get_mprime(self.bs, self.profile)
        self.get_mpprime(self.bs, self.mprime)

    def get_profile(self, **kwargs):
        m0 = kwargs['Ml']
        r0 = kwargs['r0']
        rho0 = (m0/(4*np.pi*r0**3)).to(u.Msun/u.kpc**3)
        try:
            rs = kwargs['rs']
        except:
            rs = np.linspace(0, 1, 100)*u.kpc
        term1 = np.log(1.+rs/r0)
        term2 = -1*rs/r0/(1+rs/r0)
        mr = m0*(term1 + term2)
        return rs, mr.to(u.Msun)

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
