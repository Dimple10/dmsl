'''
Test that sampler works
'''


from dmsl.sampler import *
import datetime


time1 = datetime.datetime.now()
print('First let`s do a short run with only magnitude of the acceleration.')
s = Sampler(nstars=int(1e5))
print('This took',str(datetime.datetime.now()-time1))


time1 = datetime.datetime.now()
print('Now let`s do a real 2D run.')
print('The current time is {}'.format(time1))
s = Sampler(nstars=int(1e5),ndims=2)
print('This took',str(datetime.datetime.now()-time1))
