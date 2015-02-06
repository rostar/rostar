'''
Created on Feb 4, 2015

@author: rostislavrypl
'''

import numpy as np
from scipy.stats import weibull_min
from matplotlib import pyplot as plt
import skimage
import pymc

m = 3.
s = 4.
t = np.linspace(0.00001, 10, 200)
h = lambda t, m, s: m/s * (t/s)**(m-1)
h0 = h(t, m, s)
AF = 2.
h_AF = h(t, m, s/AF)
h_PH = 2. * h(t, m, s)
#plt.plot(t, h0, label='baseline')
#plt.plot(t, h_AF, label='AF')
#plt.plot(t, h_PH, label='PH')

plt.plot(t, h_PH/h0)
plt.plot(t, h_AF/h0)
plt.legend(loc='best')
plt.show()