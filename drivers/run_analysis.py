from dmsl.sampler import *
import datetime
from dmsl.mass_profile import *
from dmsl import survey as s


ps = PointSource(**{'Ml':1.e5*u.Msun})
gaussprops = {'Ml':1.e6*u.Msun,'R0':0.01*u.kpc}
mgauss = Gaussian(**gaussprops)
nfwprops = {'Ml':1.e6*u.Msun, 'c200': 13}
nfw = NFW(**nfwprops)
nn = 1e3
roman = s.Roman(alphasigma = 0.1*u.uas/u.yr**2*(nn/1.e8)**(1./2.))
wfirstlike = s.WFIRSTLike(alphasigma = 0.1*u.uas/u.yr**2*(nn/(1.e11*0.05*4*np.pi)))
print(roman.alphasigma)

# time1 = datetime.datetime.now()
# print("Running P.S. case, no fraction")
# s = Sampler(nstars=int(nn), ntune=int(1e3), nsamples=int(1e4), survey=roman,
#         MassProfile=ps, usefraction=False, nchains=128, ndims=2, minlogMl = 0, SNRcutoff=1.0)
# print('This took',str(datetime.datetime.now()-time1))
#
# time1 = datetime.datetime.now()
# print("Running P.S. case, yes fraction")
# s = Sampler(nstars=int(nn), ntune=int(1e3), nsamples=int(1e4), survey=roman,
#        MassProfile=ps, usefraction=True, nchains=128, ndims=2, minlogMl = 0, SNRcutoff=1.0)
# print('This took',str(datetime.datetime.now()-time1))


# time1 = datetime.datetime.now()
# print("Running gaussian case, no fraction")
# s = Sampler(nstars=int(nn), ntune=int(1e3), nsamples=int(1e4), survey=roman,
#        MassProfile=mgauss, usefraction=False, nchains=64, ndims=2)
# print('This took',str(datetime.datetime.now()-time1))
#
# time1 = datetime.datetime.now()
# print("Running gaussian case, yes fraction")
# s = Sampler(nstars=int(nn), ntune=int(1e3), nsamples=int(1e4), survey=roman,
#        MassProfile=mgauss, usefraction=True, nchains=64, ndims=2)
# print('This took',str(datetime.datetime.now()-time1))

# time1 = datetime.datetime.now()
# print("Running NFW case, no fraction")
# s = Sampler(nstars=int(nn), ntune=int(1e3), nsamples=int(1e4), survey=roman,
#        MassProfile=nfw, usefraction=False, nchains=64, ndims=2)
# print('This took',str(datetime.datetime.now()-time1))
#
# time1 = datetime.datetime.now()
# print("Running NFW case, yes fraction")
# s = Sampler(nstars=int(nn), ntune=int(1e3), nsamples=int(1e4), survey=roman,
#        MassProfile=nfw, usefraction=True, nchains=64, ndims=2)
# print('This took',str(datetime.datetime.now()-time1))

# time1 = datetime.datetime.now()
# print("Running Mishra-Sharma case, no fraction")
# s = Sampler(nstars=int(nn), ntune=int(1e3), nsamples=int(1e4),
#         survey=wfirstlike, minlogMl = 0.,
#         MassProfile=mgauss, usefraction=False, nchains=64, ndims=2)
# print('This took',str(datetime.datetime.now()-time1))
