import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
import pandas as pd

from dmsl.mass_profile import *
from dmsl.plotting import paper_plot, savefig
from dmsl.paths import *
import dmsl.lensing_model as lm
from dmsl.prior_sampler import find_nlens_np

def plot_massprofile(m_):
    paper_plot()
    f = plt.figure()
    plt.plot(m_.bs, m_.profile, c='black', label=r'$M(b)$')
    plt.plot(m_.bs, m_.mprime, c='black', linestyle='dashed', label=r"$M'(b)$")
    plt.plot(m_.bs, m_.mpprime, c='black', linestyle='dotted', label=r"$M''(b)$")
    plt.legend()
    if (np.any(m_.mprime < 0.)) | (np.any(m_.mpprime < 0.)):
        plt.yscale('linear')
        plt.xscale('linear')
    else:
        plt.yscale('log')
        plt.xscale('log')
    plt.xlabel(f'$b~[\\rm {m_.bs.unit}]$')
    figpath = RESULTSDIR+'test_profile_'+m_.type+'.png'
    savefig(f, figpath, writepdf=False)

bs = np.logspace(-3, 0, 1001)[1:]
props = {'Ml':1.e6*u.Msun, 'rs':bs*u.kpc}
m = ConstDens(**props)
exprops = {'Ml':1.e8*u.Msun, 'rs':bs*u.kpc, 'rd':0.01*u.kpc}
mexp = Exp(**exprops)

gaussprops = {'Ml':1.e6*u.Msun, 'rs':bs*u.kpc, 'R0':0.01*u.kpc}
gauss = Gaussian(**gaussprops)


## create a mass profile for file to test.
r = np.logspace(-2,-1,1000)*u.kpc
mr = (r/(0.01*u.kpc))**(-2.)
res = np.array([r, mr]).T
table = pd.DataFrame(res, columns=['r', 'm'])
filepath = 'test_mr.dat'
table.to_csv(filepath, index=False)

kwargs = {'Ml': 1.e6*u.Msun, 'rs':bs*u.kpc}
mfile = From_File(filepath, 'Test',**kwargs)

plot_massprofile(mexp)
plot_massprofile(gauss)
plot_massprofile(mfile)

alphapoint = lm.alphal_np(1.e8, bs, 300.)
alphas = lm.alphal_np(None, bs, 300., btheta_=None, vltheta_=None, Mlprofile = m)
alphasexp = lm.alphal_np(None, bs, 300., btheta_=None, vltheta_=None,
        Mlprofile = mexp)
alphasgauss = lm.alphal_np(None, bs, 300., btheta_=None, vltheta_=None,
        Mlprofile = gauss)

alphasfile = lm.alphal_np(None, bs, 300., btheta_=None, vltheta_=None,
        Mlprofile = mfile)
paper_plot()
f = plt.figure()
plt.plot(bs, alphapoint, c='black', label=r'$\rm{Point~Lens}$')
plt.plot(bs, alphas, label=f'$\\rm {m.nicename}$')
plt.plot(bs, alphasexp, label=f'$\\rm {mexp.nicename}$')
plt.plot(bs, alphasgauss, label=f'$\\rm {gauss.nicename}$')
plt.plot(bs, alphasfile, label=f'$\\rm {mfile.nicename}$')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.ylim([1.e-12, 1.])
plt.xlabel(f'$b~[\\rm {m.bs.unit}]$')
plt.ylabel(r'$\alpha_l~[\mu\rm{as}/\rm{yr}^2]$')

figpath = RESULTSDIR+'test_alphasprofile.png'
savefig(f, figpath, writepdf=False)


## check peak in posterior i'm seeing
# props = {'rho0': 3.e2*u.Msun/u.pc**3, 'rs':bs*u.kpc}
# m = ConstDens(**props)
# alphas = lm.alphal_np(None, bs, 300., btheta_=None, vltheta_=None, Mlprofile = m)
# print(np.average(alphas))
# print(find_nlens_np(1.e2))
