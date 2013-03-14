'''
Created on Mar 14, 2013

@author: rostar
'''

from etsproxy.traits.api import HasTraits, Float, Int, Property
from numpy import linspace, argmax, sqrt
from numpy.random import rand
from math import exp
from scipy.stats import weibull_min, norm
from matplotlib import pyplot as plt
import numpy as np

L0 = 100.
m = 5.0
s = 0.02
Ef = 72e3

def filaments_ld():
    eps_fu = weibull_min(m, scale=s).ppf(np.linspace(0.001,0.999,50))
    for eps in eps_fu:
        plt.plot([0, eps, eps], [0, eps*Ef, 0], color='grey')
    mu_eps = weibull_min(m, scale=s).stats('m')
    plt.plot([0, mu_eps, mu_eps],[0, mu_eps*Ef, 0],color='black', lw=3, ls='dashed')

def bundle_ld():
    e = np.linspace(0.0, 1.5*s, 1000)
    plt.plot(e, e*Ef*(1-weibull_min(m, scale=s).cdf(e)), color='black', lw=3)
    plt.xlim(0,1.6*s)

from scipy.special import gamma
def filaments_SE():
    l_arr = np.linspace(1.0, 1000., 1000)
    sl = Ef * s * (L0/l_arr)**(1./m)
    mu_arr = sl * gamma(1+1./m)
    plt.plot(l_arr, mu_arr, color='black', lw=2, ls='dashed')

def bundle_SE():
    l_arr = np.linspace(1.0, 1000., 1000)
    sl = Ef * s * (L0/l_arr/m)**(1./m)
    mu_arr = sl * np.exp(-1./m)
    plt.plot(l_arr, mu_arr, color='black', lw=2)
    plt.ylim(0)

#filaments_ld()
#bundle_ld()
filaments_SE()
bundle_SE()
plt.show()
