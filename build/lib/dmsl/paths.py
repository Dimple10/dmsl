'''
Defines paths for all the things.
'''

import os
from dmsl import __path__

STARPOSDIR = os.path.join(os.path.dirname(__path__._path[0]), 'data/star_pos/')
STARDATADIR = os.path.join(os.path.dirname(__path__._path[0]),
        'data/star_accel/')
VECLENSDIR = os.path.join(os.path.dirname(__path__._path[0]),
        'data/vec_lens/')
RESULTSDIR = os.path.join(os.path.dirname(__path__._path[0]), 'results/')
FINALDIR = os.path.join(os.path.dirname(__path__._path[0]), 'results/final/')

PAPERDIR = '/Users/kpardo/Dropbox/Apps/Overleaf/dm_substructure_lensing/'

