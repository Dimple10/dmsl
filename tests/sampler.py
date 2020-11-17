'''
Test that sampler works
'''


#from dmsl.sampler import *
from dmsl.emcee_sampler import *
import datetime
from dmsl.mass_profile import *



nfwprops = {'Ml':1.e6*u.Msun}
mnfw = NFW(**nfwprops)

ps = PointSource(**{'Ml':1.e7*u.Msun})

gausprops = {'Ml':1.e6*u.Msun, 'R0':1.e3*u.pc}
mgauss = Gaussian(**gausprops)

nn = 1e3

time1 = datetime.datetime.now()
#print('First let`s do a short run with only magnitude of the acceleration.')
s = Sampler(nstars=int(nn), ntune=int(1e3), nsamples=int(1e3),
        MassProfile=mgauss, usefraction=False, ndims=1)
print('This took',str(datetime.datetime.now()-time1))


# time1 = datetime.datetime.now()
# print('Now let`s do a real 2D run.')
# print('The current time is {}'.format(time1))
# s = Sampler(nstars=int(nn),ndims=2, ncores=4)
# print('This took',str(datetime.datetime.now()-time1))
