import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import weibull_min
from scipy.special import gammainc, gamma
from scipy.integrate import cumtrapz

def H(x):
    return x >= 0.0

Ef = 200e3
r = 0.003
tau = 5.01
m = 10.9
s = 0.015
L0 = 100.

def CDFw(e, L):
    return 1 - np.exp(-L/L0 * (e/s)**m)

# failure probability between 0 and a at e
def CDFa(e):
    T = 2. * tau / r
    a = e / T
    return 1 - np.exp(-a * 2 * Ef * (e / s) ** m / (m + 1) / L0)

# failure probability density between 0 and a at e
def PDFa(e):
    T = 2. * tau / r
    a = e * Ef / T
    return 2 * (e / s) ** m / (T/Ef * L0) * np.exp(-a * 2 * (e / s) ** m / (m + 1) / L0)

# percent point function
def PPFa(p):
    T = 2. * tau / r
    return (-0.5 * np.log(1.-p) * T / Ef * L0 * (m+1) * s**m) ** (1./(m+1))

# failure probability between 0 and L at e
def CDF_L(e, L):
    T = 2. * tau / r / Ef
    a = e / T
    Pf = 1. - np.exp(-2 * a * (e / s) ** m * (1 - (1 - L / a) ** (m + 1)) / (m + 1) / L0)
    return Pf / CDFa(e)

def mean_xi():
    T = 2. * tau / r
    n = (m+1)
    c = 2. * Ef / T / L0 / n / s**m
    mu_xi = c**(-1./n)/n * gamma(1./n)
    return mu_xi 

def h(e, x):
    a = e / 2. / tau * r * Ef
    h_xe = (1 - CDFa(e)) * H(a - x) * 2 * m * (e * (1. - x/a)) ** (m-1) / L0 / s ** m 
    return h_xe

def g_z(e, x):
    T = 2. * tau / r / Ef
    a = e / 2. / tau * r * Ef
    h_xe = (1 - CDFa(e)) * H(a - x) * 2 * m * (e * (1. - x/a)) ** (m-1) / L0 / s ** m
    pdf = m/a * (1-x/a)**(m-1) * H(a - x)
    return pdf #h_xe / PDFa(e)

def cdfL(e, x):
    a = e / 2. / tau * r * Ef
    return 1 - (1 - x/a)**m

def muLate(e):
    T = 2. * tau / r / Ef
    a = e / T
    return a/(m+1)

def evans(T):
    suma = (L0/r*(s* Ef)**m*tau*(m+1))**(1./(m+1.))
    alpha = (T/suma)**(m+1)
    return L0/2. * ((s* Ef)/suma)**m * gamma((m+2)/(m+1)) * gammainc((m+2)/(m+1), alpha) / CDFa(T/Ef)

def muH(e):
    T = 2. * tau / r
    n = (m+1)
    c = 2. * Ef / T / L0 / n / s**m
    I = c**(-1./n)/n * gamma(1./n) * gammainc(1./n, c*e**n) - e*(1-CDFa(e))
    return I * Ef / T / n / CDFa(e)

def simulation(e_max):
    T = 2. * tau / r / Ef
    a = e_max / T
    nz = 500
    z_arr = np.linspace(a/2./nz, a - a/2./nz, nz)
    eps = e_max - T * z_arr
    dz = a/nz
    fu_arr = weibull_min(m, scale=s * (dz/L0)**(-1./m)).ppf(np.minimum(np.random.rand(nz), np.random.rand(nz)))
    max_diff = np.max(eps - fu_arr)
    if max_diff > 0.0:
        L = z_arr[np.argmax(eps - fu_arr)]
        eu = e_max - max_diff
        return L, eu, eu / T / L
    else:
        return None, None, None

def MC(n, e_arr):
    Ls = []
    es = []
    fact = []
    for i in range(n):
        l, e, f = simulation(e_arr[-1])
        if l == None:
            pass
        else:
            Ls.append(l)
            es.append(e)
            fact.append(f)
    es = np.array(es)
    Ls = np.array(Ls)
    Fa = np.array(fact)
    muL = []
    muLL = []
    for ei in e_arr:
        mask = es < ei
        muL.append(np.mean(Ls[mask]))
        muLL.append(np.mean(Fa[mask]))
    return muL, muLL

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

e_arr = np.linspace(0.02, 0.05, 100)
z_arr = np.linspace(0.01, 10, 200)
plt.plot(z_arr, g_z(z_arr, 0.03))
#plt.plot(e_arr, evans(e_arr * Ef))
#plt.plot(e_arr, muH(e_arr))
#plt.plot(e_arr, MC(20000, e_arr)[0])
plt.show()
