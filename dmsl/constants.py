'''
Constants used in project
'''
import astropy.units as u
import numpy as np
### ---
FB_BINS = 10
DSTARS = 8.*u.kpc
DLENS  = 1.*u.kpc
FOV = np.sqrt(0.28)*np.pi/180 ##in radians
STAR_FOV = (FOV*DLENS).to(u.kpc) ## i.e. 1 side of WFIRST FOV to physical distance @ galactic center.
LENS_FOV = (FOV*DLENS).to(u.kpc)
RHO_DM = 1.e12*u.Msun/(4./3.*np.pi*(100.*u.kpc)**3) ## BT Table 1.2, p. 16 using DM half mass and radius
MASS_MWCORE = 4.5e9*u.Msun ## from BT Table 1.2 I think...NEED TO FIX TO VARY WITH RADIUS.
RD_MW = 2.*u.kpc ## BT p.12
WFIRST_SIGMA = 0.1*u.uas/u.yr**2 ## from mishra-sharma 2020; average/star over 10 years.
