
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
vltheta = np.pi / 2.
vlthetadeg = vltheta * 180 / np.pi
vl = (1.e-3 * const.c).to(u.km / u.s).value
vvec = np.array([vl * np.cos(vltheta), vl * np.sin(vltheta)])
rarray = np.logspace(-2, 2, 2000) * u.kpc
nfwprops = {'Ml' : 1.e7 * u.Msun, 'c200' : 13.}
mnfw = NFW(**nfwprops)
mps = PointSource(**nfwprops)
size = 6. * mnfw.rs.value
fileinds = [n, vlthetadeg]
path = make_file_path(VECLENSDIR, fileinds, ext='.dat')


## print some useful info
print("Size of window: ", (size*u.rad).to(u.arcsec))
rschwarz = (2*const.G*nfwprops['Ml']/const.c**2).to(u.kpc).value
print('Size of PS lens: ', (rschwarz*u.rad))
print('Size of NFW lens: ', (mnfw.rs.value))

def make_grid(n, size):
    X = np.linspace(-size/2., size/2., n)
    Y = np.linspace(-size/2., size/2., n)
    U, V = np.meshgrid(X, Y)
    return U,V, U.reshape(n*n), V.reshape(n*n)

def make_vecfield(n, size):
    U, V, X, Y = make_grid(n,size)

    vec_field = np.zeros((n**2, 2))
    for i, (x,y) in enumerate(zip(X,Y)):
        b = np.array([x,y])
        alpha= alphal(mnfw, b, vvec)
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
_, _, X, Y = make_grid(n,size)
## get vector field
#try:
#    lensX, lensY = load_vecfield(n,size)
#except:
print('Brb need to create vector field')
lensX, lensY = make_vecfield(n, size)
## make plot and save
outpath = make_file_path(RESULTSDIR,fileinds, extra_string='lens_vector_field',
        ext='.png')
plot_lens_vector_field(X,Y, lensX, lensY,
        mnfw.rs.value, outpath)
