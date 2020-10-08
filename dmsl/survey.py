import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from dataclasses import dataclass

@dataclass
class Survey:
    name: str
    fov_deg: float
    fov_center: SkyCoord
    maxdlens: float
    dstars: float
    alphasigma: float ##error on acceleration measurement
    nstars: int

    def __post_init__(self):
        self.fov_rad: float = self.fov_deg.to(u.rad).value
        self.stars_fov: float  = (self.dstars*self.fov_rad).to(u.kpc)
        self.lens_fov: float = (self.maxdlens*self.fov_rad).to(u.kpc)

    def update(self):
        self.__post_init__()
    ##FIXME: add a method to check units.


@dataclass
class Roman(Survey):
    name: str = 'Roman'
    fov_deg: float = np.sqrt(0.28)*u.deg
    ## FIXME: put in actual l,b for EML.
    fov_center: SkyCoord = SkyCoord(l=0.*u.deg, b=0.*u.deg, frame='galactic')
    maxdlens: float = 1.*u.kpc
    dstars: float = 8.*u.kpc
    alphasigma: float = 0.1*u.uas/u.yr**2
    nstars: int = int(1.e7)

    def __post_init__(self):
        super().__post_init__()

@dataclass
class GaiaMC(Survey):
    name: str = 'GaiaMC'
    fov_deg: float = 10.*u.deg
    ## FIXME
    fov_center: SkyCoord = SkyCoord(l=0.*u.deg, b=0.*u.deg, frame='galactic')
    maxdlens: float = 5.*u.kpc ##FIXME
    dstars: float = 48.*u.kpc ##LMC
    alphasigma: float = 100.*u.uas/u.yr**2
    nstars: int = int(1.e5) ##FIXME

    def __post_init__(self):
        super().__post_init__()
