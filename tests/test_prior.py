'''
Test prior
'''


import numpy as np
import datetime


from dmsl.prior_sampler import *
from dmsl.survey import *

Gaia = GaiaMC()

print('First let`s do a short run with only 1e3 stars.')
time1 = datetime.datetime.now()
ps = PriorSampler(nstars=1000, nbsamples=2000,
            nbnlens=7, maxlognlens=3,
            overwrite=False)
print('This took',str(datetime.datetime.now()-time1))


