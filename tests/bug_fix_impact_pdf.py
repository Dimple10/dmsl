from dmsl.prior import *

import numpy as np
from scipy.integrate import cumtrapz
from dmsl.survey import Roman

survey = Roman()
bs = np.logspace(-7, np.log10(np.sqrt(2)*survey.fov_rad), 100)
print(gs(bs**2,survey.fov_rad, survey.fov_rad))
print(cumtrapz(gs(bs**2, survey.fov_rad, survey.fov_rad), x=bs**2, initial=0))

prior_init = impact_pdf(shapes='a1, a2, n')

pb = pdf(bs, a1=survey.fov_rad,
    a2=survey.fov_rad, n=1)
print(pb)

'''
What did I learn from this bug?
1. Don't use scipy rv_continuous. It doesn't like piecewise functions where you
also integrate.
2. Make sure bs has the correct values -- too much subdivision or too few
messes everything up. (i.e. CDF > 1 or weird PDF values).
'''

