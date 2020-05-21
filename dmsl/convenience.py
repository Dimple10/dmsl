'''
Random useful functions
'''
import numpy as np

# flatten lists
flatten = lambda l: [item for sublist in l for item in sublist]

## make file path names
def make_file_path(directory, array_kwargs, extra_string=None, ext='.dat'):
        s = '_'
        string_kwargs = [str(int(i)) for i in array_kwargs]
        string_kwargs = np.array(string_kwargs, dtype='U25')
        if len(extra_string)>25:
            raise TypeError('Extra string must have less than 25 characters')
        if extra_string !=None:
            string_kwargs = np.insert(string_kwargs, 0, extra_string)
        kwpath = s.join(string_kwargs)
        return directory+kwpath+ext
