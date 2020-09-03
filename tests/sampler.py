'''
Test that sampler works
'''


#from dmsl.sampler import *
from dmsl.emcee_sampler import *
import datetime
from dmsl.mass_profile import *

bs = np.logspace(-3, 0, 1001)[1:]

props = {'Ml':1.e6*u.Msun, 'rs':bs*u.kpc}
m = ConstDens(**props)

exprops = {'Ml':1.e8*u.Msun, 'rs':bs*u.kpc, 'rd':0.01*u.kpc}
mexp = Exp(**exprops)

gaussprops = {'Ml':1.e6*u.Msun, 'rs':bs*u.kpc, 'R0':0.01*u.kpc}
mgauss = Gaussian(**gaussprops)

nfwprops = {'Ml':1.e6*u.Msun, 'rs':bs*u.kpc, 'r0':0.01*u.kpc}
mnfw = NFW(**nfwprops)

tnfwprops = {'Ml':1.e6*u.Msun, 'rs':bs*u.kpc, 'r0':0.01*u.kpc, 'rt':1*u.kpc}
mtnfw = TruncatedNFW(**tnfwprops)


nn = 1e3

time1 = datetime.datetime.now()
#print('First let`s do a short run with only magnitude of the acceleration.')
s = Sampler(nstars=int(nn), ntune=int(1e2), nsamples=int(1e3),
        MassProfile=mgauss)
print('This took',str(datetime.datetime.now()-time1))


# time1 = datetime.datetime.now()
# print('Now let`s do a real 2D run.')
# print('The current time is {}'.format(time1))
# s = Sampler(nstars=int(nn),ndims=2, ncores=4)
# print('This took',str(datetime.datetime.now()-time1))
