'''
Test that sampler works
'''


#from dmsl.sampler import *
from dmsl.sampler import *
import datetime
from dmsl.mass_profile import *
from dmsl.survey import *



nfwprops = {'Ml':1.e6*u.Msun}
mnfw = NFW(**nfwprops)

ps = PointSource(**{'Ml':1.e7*u.Msun})

gausprops = {'Ml':1.e8*u.Msun, 'R0':1.e-3*u.pc}
mgauss = Gaussian(**gausprops)

nn = 1e3

Gaia = GaiaMC(alphasigma=100.*u.uas/u.yr**2*np.sqrt(1./1e4))
roman = Roman(alphasigma=0.1*u.uas/u.yr**2*np.sqrt(1./1.e8))
## extra factor = correction for number of stars

time1 = datetime.datetime.now()
print('First let`s do a short run with only magnitude of the acceleration.')
s = Sampler(nstars=int(nn), ntune=int(1e3), nsamples=int(5e3), nchains=128,
        minlogMl = 7, maxlogMl = 9,
        maxlognlens=4.1, survey = Gaia, MassProfile=mgauss, usefraction=True, ndims=2,
        bcutoff={'SNR': 100.})
#s = Sampler(nstars=int(nn), ntune=int(1e3), nsamples=int(5e3), nchains=32,
#        minlogMl = 3,
#        maxlognlens=2.5, MassProfile=ps, usefraction=True, ndims=2,
#        bcutoff={'SNR': 10.})
#print('This took',str(datetime.datetime.now()-time1))


# time1 = datetime.datetime.now()
# print('Now let`s do a real 2D run.')
# print('The current time is {}'.format(time1))
# s = Sampler(nstars=int(nn),ndims=2, ncores=4)
# print('This took',str(datetime.datetime.now()-time1))
