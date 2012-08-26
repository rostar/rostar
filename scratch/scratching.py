
'''
Created on 5.8.2012

@author: Q
'''
import numpy as np
from scipy.optimize import fsolve

def opt(x,a):
    x2 = x.reshape(4,1)**2/a
    return x - np.sum(x2, axis = 1)

print fsolve(opt, np.array([1.,2,3,4]), args = (np.linspace(2,4,10).reshape(1,10)))