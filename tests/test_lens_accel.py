'''
test full acceleration form
'''

import pymc3 as pm
import numpy as np
import exoplanet as xo

from dmsl.lensing_model import *


with pm.Model() as model:
    ml = 1.e8
    b = 0.1
    vl = 100.
    btheta = 0.
    vltheta = np.pi/2.
    print("first test vec expression function")
    a = alphal_vec_exp(b, vl, btheta, vltheta)
    print(xo.eval_in_model(a))
    print("Now test magnitude only alphal")
    b = alphal(ml, b,vl)
    print(xo.eval_in_model(b))
    print("Finally, test the 2D version")
    c = alphal(ml, b,vl, btheta, vltheta)
    print(xo.eval_in_model(c))

