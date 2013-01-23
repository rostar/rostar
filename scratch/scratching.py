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
s = 0.02
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
    s_mod = s * (100. / 1.196143) ** (1./m) #0.04847
    h_xe = (1 - CDFa(e)) * H(a - x) * e ** (m - 1) * 2 * (1. - x/a) ** (m) * m / s_mod ** m
    return h_xe * x * dx * de

from scipy.optimize import fsolve

def find_s():
    def residuum(c):
        e_arr = np.linspace(0.0001, 0.04, 2000).reshape(2000, 1)
        x_arr = np.linspace(0.0, 80., 2000).reshape(1, 2000)
        mu_q = h(e_arr, 0.04/2000., x_arr, 80. / 2000., c)
        return np.sum(mu_q) - 1.0
    print fsolve(residuum, 0.02)

#find_s()
from stats.spirrid import make_ogrid as orthogonalize
from mayavi import mlab
e_arr = np.linspace(0.0001, 0.04, 200).reshape(200, 1)
x_arr = np.linspace(0.0, 80., 200).reshape(1, 200)
mu_q = h(e_arr, 0.04/200., x_arr, 80. / 200.)
print np.sum(mu_q)
ctrl_arr = orthogonalize([np.arange(len(e_arr.flatten())), np.arange(len(x_arr.flatten()))])
mlab.surf(e_arr[0], e_arr[1], mu_q * 50000.)
mlab.show()
#mu_q = h(e_arr, 0.04/200., x_arr, 60. / 300.)
#pdf_x = np.trapz(mu_q, e_arr.flatten(), axis = 0)
#plt.plot(x_arr.flatten(), pdf_x.flatten())
#print np.trapz(pdf_x.flatten(), x_arr.flatten())
#plt.show()

def simulation(e):
    T = 2. * tau / r / Ef
    a = e / T
    nz = 100
    z_arr = np.linspace(a/2./nz, a - a/2./nz, nz)
    eps = e - T * z_arr
    dz = a/nz
    fu_arr = weibull_min(m, scale=s * (dz/L0)**(-1./m)).ppf(np.minimum(np.random.rand(nz),
                                                                       np.random.rand(nz)))
    max_diff = np.max(eps - fu_arr)
    if max_diff > 0.0:
        return z_arr[np.argmax(eps - fu_arr)]
    else:
        return None

def MC(n, e_arr):
    L = []
    for e in e_arr:
        ls = []
        for i in range(n):
            l = simulation(e)
            if l == None:
                pass
            else:
                ls.append(l)
        L.append(np.mean(np.array(ls)))
    return L

from scipy.optimize import fsolve

def MC_CDFa(n, e_arr):
    nx = 500.
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
    

def PMF(e, l_arr):
    T = 2. * tau / r / Ef
    a = e / T
    dx = l_arr[1]
    PMF = np.exp(dx/L0*(e * (1-l_arr/a)/s)**m -a * (e / s) ** m / (m + 1) / L0)
    return PMF / np.sum(PMF)

def MUL_at_e(e):
    T = 2. * tau / r / Ef
    a = e / T
    l_arr = np.linspace(0.0, a, 300)
    return np.trapz(PMF(e, l_arr) * l_arr, l_arr)

l_arr = np.linspace(0.0, 90., 50)
e_arr = np.linspace(0.001, 0.05, 100)
e_arr2 = np.linspace(0.015, 0.07, 20)
plt.plot(e_arr, CDFa(e_arr))
plt.xlabel('eps crack')
plt.ylabel('mu_L')
plt.legend(loc='best')
plt.show()
