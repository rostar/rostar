import numpy as np
from scipy.optimize import fsolve
from scipy.stats import weibull_min
from matplotlib import pyplot as plt
from scipy.optimize import broyden2

m = 5.0
s = 0.02
Ef = 72e3
L0 = 50.

def Pf(e, L):
    return 1. - np.exp(-L/L0*(e/s)**m)

def Pf2(e, a):
    f = a * (e/s)**m / (m+1)
    print 'f=', f
    return 1. - np.exp(-2. / L0 * f)

def Pf3(e, tau, l):
    f = (((tau * l - 1)*((e*(1-tau*l)))/s)**m + (e/s)**m) / (m+1) / tau
    print 'f=', f
    return 1. - np.exp(-2. / L0 * f)

print Pf(0.02, 50.)
print Pf2(0.02, 50.)
print Pf3(0.02, 1./50., 5.)

#L_arr = np.linspace(0, 200, 100)
#plt.plot(L_arr, Pf(L_arr, 0.02))
#plt.show()