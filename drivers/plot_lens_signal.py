
'''
Plot vector field of acceleration signal
'''

import numpy as np
import pymc3 as pm
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from dmsl.constants import *
from dmsl.paths import *
from dmsl.convenience import *
from dmsl.plotting import *
from dmsl.lensing_model import *

## set params
n = 20
arcsecsize = 3600.
size = (arcsecsize*u.arcsec).to(u.rad).value
ml = 1.e8
vltheta = np.pi/2.
vlthetadeg = vltheta*180/np.pi
vl = 100.

fileinds = [n, np.log10(arcsecsize), vlthetadeg]
path = make_file_path(VECLENSDIR, fileinds, ext='.dat')

## useful conversion
radtoarcsec = (1.*u.rad).to(u.arcsec).value

## print some useful info
print("Size of window: ", (size*u.rad).to(u.arcsec))
rschwarz = (2*const.G*ml*u.Msun/const.c**2).to(u.kpc).value
print('Size of lens: ', (rschwarz*u.rad).to(u.arcsec))

def make_grid(n, size):
    X = np.linspace(-size/2., size/2., n)
    Y = np.linspace(-size/2., size/2., n)
    U, V = np.meshgrid(X, Y)
    return(U,V)

def make_vecfield(n, size):
    U,V = make_grid(n,size)
    r = np.sqrt(U**2+V**2).reshape(n*n)
    theta = np.tan(V/U).reshape(n*n) - np.pi

    array = []
    with pm.Model() as model:
        for b, t in zip(r,theta):
            array.append(alphal(ml, b,vl, t, vltheta))
        vec_field = flatten(xo.eval_in_model(array))

    lensX = np.array(vec_field[::2])
    lensY = np.array(vec_field[1::2])
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
try:
    lensX, lensY = load_vecfield(n,size)
except:
    print('Brb need to create vector field')
    lensX, lensY = make_vecfield(n, size)
## reshape to grid 
lensX.reshape(n,n)
lensY.reshape(n,n)
## make plot and save
outpath = make_file_path(RESULTSDIR,fileinds, extra_string='lens_vector_field',
        ext='.png')
plot_lens_vector_field(U*radtoarcsec,V*radtoarcsec, lensX, lensY, outpath)
