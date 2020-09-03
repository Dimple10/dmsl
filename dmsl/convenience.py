'''
Random useful functions

Includes:
flatten
pmprint
prange
pshape
psum
make_file_path


'''
import numpy as np
import pymc3 as pm
import exoplanet as xo
#from varname import nameof, varname

# flatten lists
flatten = lambda l: [item for sublist in l for item in sublist]

# print in pymc3 model
pmprint = lambda x: print(xo.eval_in_model(x))
pmpshape = lambda x: print(np.shape(xo.eval_in_model(x)))

# useful print statements
prange = lambda _: print('min:', np.min(_), 'max:', np.max(_))
pshape = lambda _: print(np.shape(_))
psum = lambda _: print(np.sum(_))

## make file path names
def make_file_path(directory, array_kwargs, extra_string=None, ext='.dat'):
        s = '_'
        string_kwargs = [str(int(i)) for i in array_kwargs]
        string_kwargs = np.array(string_kwargs, dtype='U35')
        if (extra_string !=None) and (len(extra_string)>35):
            raise TypeError('Extra string must have less than 35 characters')
        if extra_string !=None:
            string_kwargs = np.insert(string_kwargs, 0, extra_string)
        kwpath = s.join(string_kwargs)
        return directory+kwpath+ext

