'''
Defines mass profiles
'''

import os
import numpy as np
import astropy.units as u
import astropy.constants as const


class MassProfile:

    def __init__(self, profiletype):
        self.type = profiletype

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
        self.nicename = 'Const.~Dens.'
        self.bs, self.profile = self.get_profile(**kwargs)
        self.get_mprime(self.bs, self.profile)
        self.get_mpprime(self.bs, self.mprime)

    def get_profile(self, **kwargs):
        rho0 = kwargs['rho0']
        try:
            rs = kwargs['rs']
        except:
            rs = np.linspace(0, 1, 100)*u.kpc
        m = 4./3.*np.pi*rho0*rs**3
        return rs, m.to(u.Msun)

class Exp(MassProfile):

    def __init__(self, **kwargs):
        super().__init__('exp')
        self.nicename = 'Exponential'
        self.bs, self.profile = self.get_profile(**kwargs)
        self.get_mprime(self.bs, self.profile)
        self.get_mpprime(self.bs, self.mprime)

    def get_profile(self, **kwargs):
        m0 = kwargs['M0']
        rd = kwargs['rd']
        try:
            rs = kwargs['rs']
        except:
            rs = np.linspace(0, 1, 100)*u.kpc
        m = m0*np.exp(-1.*rs/rd)
        return rs, m.to(u.Msun)

class Compact(MassProfile):

    def __init__(self):
        super().__init__('compact')
        self.bs, self.profile = self.get_profile()
        self.get_mprime(self.bs, self.profile)
        self.get_mpprime(self.bs, self.mprime)

    def get_profile(self, *kwargs):
        pass


