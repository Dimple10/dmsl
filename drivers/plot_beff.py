
'''
makes plot of effective impact parameter
'''


import numpy as np
import matplotlib.pyplot as plt
import pickle


from dmsl.convenience import *
from dmsl.paths import *
from dmsl.plotting import *
from dmsl.prior_sampler import find_nlens_np
nstars = 1e3
nblog10Ml = 3.
nbsamples = 100
FIGPATH = make_file_path(RESULTSDIR, [np.log10(nstars)], extra_string='beffvsM',
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
bmax = actualb[np.argmax(plogb(barray))]


marray = np.logspace(-3., 8., 20)
nlens = np.array([find_nlens_np(m) for m in marray])
print(nlens)
beff = 1./bmax**3*nlens
paper_plot()
f = plt.figure()
plt.plot(marray, beff)
plt.plot(marray, beff*marray, label='with Mass factor')
plt.xlabel(r'$M_l~[M_{\odot}]$')
plt.ylabel(r'$N_{\rm{lens}}/b^3$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
savefig(f, FIGPATH)
