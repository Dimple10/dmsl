
'''
Plot vector field of acceleration signal
'''
import sys
import numpy as np
import pymc3 as pm
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from dmsl.paths import *
from dmsl.convenience import *
from dmsl.plotting import *
from dmsl.lensing_model import *
from dmsl.mass_profile import *

## set params
n = 16
arcsecsize = 3600.
#size = (10*12.*u.arcsec).to(u.rad).value
#size = (12.*u.arcsec).to(u.rad).value
size = 6. ## rad
ml = 1.e3
vltheta = np.pi/2.
vlthetadeg = vltheta*180/np.pi
vl = (1.e-3*const.c).to(u.km/u.s).value
rarray = np.logspace(-2, 2, 2000)*u.kpc
nfwprops = {'Ml':1.e7*u.Msun,'rarray': rarray}
mnfw = NFW(**nfwprops)
fileinds = [n, np.log10(arcsecsize), vlthetadeg]
path = make_file_path(VECLENSDIR, fileinds, ext='.dat')

## useful conversion
radtoarcsec = (1.*u.rad).to(u.arcsec).value

## print some useful info
print("Size of window: ", (size*u.rad).to(u.arcsec))
rschwarz = (2*const.G*ml*u.Msun/const.c**2).to(u.kpc).value
print('Size of PS lens: ', (rschwarz*u.rad).to(u.arcsec))
print('Size of NFW lens: ', (mnfw.rs.value*radtoarcsec))
print("Check that magnitude is right:")
mcheck = NFW(**{'Ml':1.e7*u.Msun, 'rarray':rarray})
alphacheckpoint = alphal_np(1.e7, 1., (1.e-3*const.c).to(u.km/u.s).value, 1*u.kpc,
        Mlprofile=None)
alphacheck = alphal_np(1.e7, 1., (1.e-3*const.c).to(u.km/u.s).value, 1*u.kpc,
        Mlprofile=mcheck)
print("For vT2018 eqn 2.8, but PS, alpha is:", alphacheckpoint)
print("For vT2018 eqn 2.8, alpha is:", alphacheck)

def make_grid(n, size):
    X = np.linspace(-size/2., size/2., n)
    Y = np.linspace(-size/2., size/2., n)
    U, V = np.meshgrid(X, Y)
    return(U,V)

def make_vecfield(n, size):
    U,V = make_grid(n,size)
    r = np.sqrt(U**2+V**2).reshape(n*n)
    theta = np.arctan2(V,U).reshape(n*n)
    printU = U.reshape(n*n)
    printV = V.reshape(n*n)

    vec_field = np.zeros((n**2, 2))
    for i, (b,t) in enumerate(zip(r,theta)):
        alpha= alphal_np(1.e7, b, vl, 1.*u.kpc, btheta_=t, vltheta_=vltheta,
                Mlprofile=mnfw)[:,0]
        vec_field[i, :] = alpha

    lensX = np.array(vec_field[:,0])
    lensY = np.array(vec_field[:,1])
    data = np.array([lensX, lensY]).T
    table = pd.DataFrame(data, columns=['lensX', 'lensY'])
    table.to_csv(path, index=False)
    print('{}: made {}'.format(datetime.now().isoformat(), path))
    return lensX, lensY

def load_vecfield(n, size):
    data = pd.read_csv(path)
    print('Succesfully loaded file.')
    lensX = data['lensX'].to_numpy()
    lensY = data['lensY'].to_numpy()
    return lensX, lensY

## make X,Y grid
U,V = make_grid(n,size)
## get vector field
#try:
#    lensX, lensY = load_vecfield(n,size)
#except:
print('Brb need to create vector field')
lensX, lensY = make_vecfield(n, size)
## reshape to grid 
lensX.reshape(n,n)
lensY.reshape(n,n)
## make plot and save
outpath = make_file_path(RESULTSDIR,fileinds, extra_string='lens_vector_field',
        ext='.png')
plot_lens_vector_field(U*radtoarcsec,V*radtoarcsec, lensX, lensY,
        mnfw.rs.value*radtoarcsec, outpath)
