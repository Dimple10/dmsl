'''
Test prior
'''


import numpy as np
import datetime

from dmsl.prior_sampler import *


ps = PriorSampler(nbsamples=100, log10Ml=3.,
            nstars=int(1e3), overwrite=False)
time1 = datetime.datetime.now()
print('First let`s do a short run with only 1e3 stars.')
ps.run_sampler()
print('This took',str(datetime.datetime.now()-time1))


#ps = PriorSampler(nbsamples=100, log10Ml=3.,
#            nstars=int(1e5), overwrite=False)
time1 = datetime.datetime.now()
print('Now let`s try 1e5 stars.')
print('Starting at {}'.format(time1))
#ps.run_sampler()
print('This took',str(datetime.datetime.now()-time1))


#ps = PriorSampler(nbsamples=1, log10Ml=3.,
#            nstars=int(1e8), overwrite=False)
time1 = datetime.datetime.now()
print('Finally let`s do the 1e7  stars.')
print('Starting at {}'.format(time1))
#ps.run_sampler()
print('This took',str(datetime.datetime.now()-time1))
