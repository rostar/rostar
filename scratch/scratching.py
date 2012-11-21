import numpy as np
from scipy.optimize import fsolve
from scipy.stats import weibull_min
from matplotlib import pyplot as plt
from scipy.optimize import broyden2
from scipy.special import gammainc
from scipy.interpolate import griddata
from mayavi import mlab as m
from math import pi
import time as t




m = 5.0
s = 0.02
Ef = 72e3
L0 = 50.
T = 0.02/50.

def P(e):
    return 1. - np.exp(-(e/s)**m)

def Pf(e, L):
    return 1. - np.exp(-L/L0*(e/s)**m)

def Pf2(e, a):
    f = a * (e/s)**m / (m+1)
    return 1. - np.exp(-2. / L0 * f)

def Pf3(e, tau, l):
    f = (((tau * l - 1)*((e*(1-tau*l)))/s)**m + (e/s)**m) / (m+1) / tau
    return 1. - np.exp(-2. / L0 * f)

def muL(L, e):
    nom = L * e ** 2 * gammainc(2./m, (e/s)**m)/m/s**2 + L**2/2.
    de = L * e **2 * gammainc(1./m, (e/s)**m)/m/s + L
    return nom/de
#    
##print Pf(0.02, 50.)
##print Pf2(0.02, 50.)
##print Pf3(0.02, 1. / 50., 5.)
#
#e_arr = np.linspace(0, 0.02, 1000)
#L_arr = np.linspace(0, 50., 1000)
#print np.sum(P(0.02 - e_arr) * L_arr) / np.sum(P(0.02 - e_arr))
#print muL(50., 0.02)
##plt.show()