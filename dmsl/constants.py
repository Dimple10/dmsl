'''
Constants used in project
'''
import astropy.units as u
import numpy as np
### ---
FB_BINS = 10
DSTARS = 8.*u.kpc
#DSTARS = 48.*u.kpc##LMC
DLENS  = 1.*u.kpc
#DLENS  = 5.*u.kpc ## for LMC
FOV_DEG = np.sqrt(0.28)*u.deg
#FOV_DEG = 10.*u.deg ## gaia
FOV = np.sqrt(0.28)*np.pi/180 ##in radians
#FOV = 10.*np.pi/180 ##in radians #gaia
STAR_FOV = (FOV*DLENS).to(u.kpc) ## i.e. 1 side of WFIRST FOV to physical distance @ galactic center.
LENS_FOV = (FOV*DLENS).to(u.kpc)
#RHO_DM = 1.e12*u.Msun/(4./3.*np.pi*(100.*u.kpc)**3) ## BT Table 1.2, p. 16 using DM half mass and radius
RHO_DM = (0.008*u.Msun/u.pc**3).to(u.Msun/u.kpc**3) ## Bovy+Tremaine 2012 
MASS_MWCORE = 4.5e9*u.Msun ## from BT Table 1.2 I think...NEED TO FIX TO VARY WITH RADIUS.
RD_MW = 2.*u.kpc ## BT p.12
WFIRST_SIGMA = 0.1*u.uas/u.yr**2 ## from mishra-sharma 2020; average/star over 10 years.
#WFIRST_SIGMA = 100*u.uas/u.yr**2 ## trying for Gaia DR 2.. \sigma_\mu = 0.2mas/yr over 22 months ~ 2 years
