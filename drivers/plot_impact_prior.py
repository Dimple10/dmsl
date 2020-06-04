'''
makes plot of impact parameter prior
'''


import numpy as np
import matplotlib.pyplot as plt
import pickle


from dmsl.convenience import *
from dmsl.paths import *
from dmsl.plotting import *

nstars = 1e3
nblog10Ml = 3.
nbsamples = 100
FIGPATH = make_file_path(RESULTSDIR, [np.log10(nstars)], extra_string='prior',
        ext='.png')

fileinds = [np.log10(nstars), nblog10Ml, np.log10(nbsamples)]
pklpath = make_file_path(RESULTSDIR, fileinds, extra_string='plogb',
        ext='.pkl')
if os.path.exists(pklpath):
    plogb  = pickle.load(open(pklpath, 'rb'))

else:
    ps = PriorSampler(nstars=nstars, nbsamples=nbsamples,
            log10Ml=nblog10Ml)
    ps.run_sampler()
    plogb = ps.plogb

barray = np.linspace(-12, 0, 1000)
actualb = np.exp(barray)
paper_plot()
f = plt.figure()
plt.plot(actualb,  plogb(barray))
plt.xlabel(r'$b~[\rm{kpc}]$')
plt.ylabel(r'$p(b)$')
plt.xscale('log')
savefig(f, FIGPATH)
