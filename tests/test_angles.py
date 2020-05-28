import numpy as np


from dmsl.background_model import *
from dmsl.constants import FOV

print("Theta at center:", get_theta_to_GC())
print("Largest theta:", get_theta_to_GC(l=1.*u.deg-FOV/2.*180/np.pi*u.deg,
    b=-1*u.deg-FOV/2.*180/np.pi*u.deg))
print("Smallest theta:", get_theta_to_GC(l=1.*u.deg+FOV/2.*180/np.pi*u.deg,
    b=-1*u.deg+FOV/2.*180/np.pi*u.deg))
