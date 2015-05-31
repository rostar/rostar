from scipy.special import gamma
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from T05.fiber_tests.filament_statistics import FilamentTestsEvaluation
from scipy.stats import norm

Vf10 = np.array([12.6, 15.0 , 10.9 , 17.1, 12.6])

print Vf10.mean()
print np.std(Vf10) / Vf10.mean()

# Em = 25e3
# Ef = 182e3
# Vf1= 0.01
# Vf2= 0.015 
# Ec1 = Ef*Vf1 + Em*(1.-Vf1)
# Ec2 = Ef*Vf2 + Em*(1.-Vf2)
#  
# sigmacmin1 = np.array([2.94, 2.58, 2.53, 3.01, 3.31])
# sigmacmin2 = np.array([2.78, 3.43, 3.03, 3.47, 3.59])
#  
# sigmammin1 = sigmacmin1 / Ec1 * Em
# sigmammin2 = sigmacmin2 / Ec1 * Em
# print 'mean minimum Lc strength 1: ', sigmammin1.mean()
# shape = 150.
# print 'scale of the minimum: ', sigmammin1.mean()/gamma(1. + 1./shape)
# print 'scale of the local: ', sigmammin1.mean()/gamma(1. + 1./shape) * (1./(250.+1.)) ** (- 1./shape)
#  
# print 'mean minimum Lc strength 2: ', sigmammin2.mean()
# shape = 150.
# print 'scale of the minimum 2: ', sigmammin2.mean()/gamma(1. + 1./shape)
# print 'scale of the local 2: ', sigmammin2.mean()/gamma(1. + 1./shape) * (1./(250.+1.)) ** (- 1./shape)
#  
#  
# print 'mean minimum Lc strength 2: ', sigmammin2.mean()
#   
# fte = FilamentTestsEvaluation(data = sigmammin1, length = 250)
# shape1, scale1 = fte.maximum_likelihood()
# fte.data = sigmammin2
# shape2, scale2 = fte.maximum_likelihood()
#   
# fL = lambda x: (1./(250.+1.)) ** (- 1./x)
# sm1 = scale1 * fL(shape1)
# sm2 = scale2 * fL(shape2)
# print 'local scale 1 = ', sm1, sm1 * gamma(1 + 1./shape1) / fL(shape1)
# print 'local scale 2 = ', sm2, sm2 * gamma(1 + 1./shape2) / fL(shape2) 