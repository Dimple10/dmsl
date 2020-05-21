'''
Defines paths for all the things.
'''

import os
from dmsl import __path__

STARPOSDIR = os.path.join(os.path.dirname(__path__._path[0]), 'data/star_pos/')
STARDATADIR = os.path.join(os.path.dirname(__path__._path[0]),
        'data/star_accel/')
RESULTSDIR = os.path.join(os.path.dirname(__path__._path[0]), 'results/')

