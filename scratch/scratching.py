import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import weibull_min
from scipy.optimize import brentq
from scipy.integrate import cumtrapz

def H(x):
    return x >= 0.0

Ef = 200e3
r = 0.003
tau = 0.1
m = 5.
s = 0.03
L0 = 100.

def CDFw(e, L):
    return 1 - np.exp(-L/L0 * (e/s)**m)

# failure probability between 0 and a at e
def CDFa(e):
    T = 2. * tau / r / Ef
    a = e / T
    return 1. - np.exp(-a * 2 * (e / s) ** m / (m + 1) / L0)

# failure probability density between 0 and a at e
def PDFa(e):
    T = 2. * tau / r / Ef
    a = e / T
    return np.exp(-a * 2 * (e / s) ** m / L0 / (m + 1)) * 2 * (e / s) ** m / (T * L0)

# failure probability between 0 and L at e
def CDF_L(e, L):
    T = 2. * tau / r / Ef
    a = e / T
    Pf = 1. - np.exp(-a * (e / s) ** m * (1 - (1 - L / a) ** (m + 1)) / (m + 1) / L0)
    return Pf / CDFa(e)

def h(e, de, x, dx):
    a = e / 2. / tau * r * Ef
    h_xe = (1 - CDFa(e)) * H(a - x) / s ** m * 2 * (e * (1. - x/a)) ** (m-1) * m # * e ** (m - 1) * 2 * ((1. - x/a)/s) ** (m) * m
    return h_xe / L0

from scipy.optimize import fsolve
from scipy.special import gammainc, gamma

def evans(T):
    suma = (L0/r*(s* Ef)**m*tau*(m+1))**(1./(m+1.))
    alpha = (T/suma)**(m+1)
    return L0/2. * ((s* Ef)/suma)**m * gamma((m+2)/(m+1)) * gammainc((m+2)/(m+1), alpha) / CDFa(T/Ef)

e_arr = np.linspace(0.001, 0.05, 100)
plt.plot(e_arr, evans(e_arr * Ef))

from stats.spirrid import make_ogrid as orthogonalize
from mayavi import mlab
nx = 200
ne = 200
e_arr = np.linspace(0.0001, 0.04, nx)
x_arr = np.linspace(0.0, 80., ne)
dx = x_arr[1] - x_arr[0]
de = e_arr[1] - e_arr[0]
h = h(e_arr.reshape(ne, 1), de, x_arr.reshape(1, nx), dx)
pdf_x = np.trapz(h, e_arr.flatten(), axis=0)
print 'sum = ', np.trapz(pdf_x.flatten(), x_arr.flatten())
print 'mu_L =', np.trapz(pdf_x.flatten() * x_arr, x_arr.flatten())
#ctrl_arr = orthogonalize([np.arange(len(e_arr.flatten())), np.arange(len(x_arr.flatten()))])
#mlab.surf(e_arr[0], e_arr[1], h / np.max(h) * nx)
#mlab.show()
#plt.plot(x_arr.flatten(), pdf_x.flatten())
#plt.show()

def simulation(e_max):
    T = 2. * tau / r / Ef
    a = e_max / T
    nz = 100
    z_arr = np.linspace(a/2./nz, a - a/2./nz, nz)
    eps = e_max - T * z_arr
    dz = a/nz
    fu_arr = weibull_min(m, scale=s * (dz/L0)**(-1./m)).ppf(np.minimum(np.random.rand(nz),
                                                                       np.random.rand(nz)))
    max_diff = np.max(eps - fu_arr)
    if max_diff > 0.0:
        return z_arr[np.argmax(eps - fu_arr)], e_max - max_diff 
    else:
        return None, None

def MC(n, e_arr):
    Ls = []
    es = []
    for i in range(n):
        l, e = simulation(e_arr[-1])
        if l == None:
            pass
        else:
            Ls.append(l)
            es.append(e)
    es = np.array(es)
    Ls = np.array(Ls)
    muL = []
    for ei in e_arr:
        mask = es < ei
        muL.append(np.mean(Ls[mask]))
    return muL

from scipy.optimize import fsolve

def MC_CDFa(n, e_arr):
    nx = 1000.
    emax = e_arr[-1]
    T = 2. * tau / r / Ef
    amax = emax / T
    dx = amax/nx
    z_arr = np.linspace(amax/2./nx, amax - amax/2./nx, nx)
    eu = []
    def residuum(ef0, fu):
        return np.min(fu - ef0 + T * z_arr)
    for i in range(n):
        fu_arr = weibull_min(m, scale=s * (dx/L0)**(-1./m)).ppf(np.minimum(np.random.rand(nx),
                                                                       np.random.rand(nx)))
        eu.append(fsolve(residuum, 0.001, args=(fu_arr)))

    cdf = []
    for ei in e_arr:
        cdf.append(np.sum(np.array(eu) < ei))
    return np.array(cdf)/float(n)

l_arr = np.linspace(0.0, 90., 50)
e_arr = np.linspace(0.001, 0.05, 100)
e_arr2 = np.linspace(0.015, 0.07, 20)
mc = MC(10000, e_arr)
print mc[-1]
plt.plot(e_arr, mc, 'ro')
plt.xlabel('eps crack')
plt.ylabel('mu_L')
plt.legend(loc='best')
plt.show()
