'''
Plots mcmc results for different survey values to make sure basic scalings
apply.
'''
import numpy as np
import astropy.units as u

import dmsl.survey as s
import dmsl.emcee_sampler as es
from dmsl.mass_profile import *
import dmsl.plotting as myplot
from dmsl.paths import *


## set survey
survey = s.Roman()

### define mass profiles
gaussprops = {'Ml':1.e6*u.Msun,'R0':0.01*u.kpc}
mgauss = Gaussian(**gaussprops)
nfwprops = {'Ml': 1.e7 * u.Msun, 'c200': 1.e4}
mnfw = NFW(**nfwprops)
mps = PointSource(**{'Ml': 1.e5 * u.Msun})

massprofile = mps

### sigma_alpha scaling.
print("Plotting acceleration sensitivity scaling")
sigmas = [1.e-4*u.uas/u.yr**2, 0.1*u.uas/u.yr**2, 10.*u.uas/u.yr**2]
mlsamples = {}
for i, sig in enumerate(sigmas):
    survey.alphasigma = sig
    survey.update()
    rsampler = es.Sampler(nstars=int(1e3), ntune=int(1e3), nsamples=int(1e3), ndims=2,
        MassProfile=massprofile, survey=survey, overwrite=False)
    mlsamples[f'$\\sigma_\\alpha = {sig.value}$'] = rsampler.sampler.get_chain(flat=True, discard=rsampler.ntune)[:,0]

filename = RESULTSDIR+'sigma_alpha_scaling.png'
myplot.plot_hists(mlsamples, filename)

