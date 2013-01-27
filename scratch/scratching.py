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
m = 7.
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
    return np.exp(-a * 2 * (e / s) ** m / (m + 1) / L0) * 2 * (e / s) ** m / (T * L0)

# percent point function
def PPFa(p):
    T = 2. * tau / r / Ef
    return (-0.5 * np.log(1.-p) * T * L0 * (m+1) * s**m) ** (1./(m+1))

# failure probability between 0 and L at e
def CDF_L(e, L):
    T = 2. * tau / r / Ef
    a = e / T
    Pf = 1. - np.exp(-a * (e / s) ** m * (1 - (1 - L / a) ** (m + 1)) / (m + 1) / L0)
    return Pf / CDFa(e)

def h(e, x):
    a = e / 2. / tau * r * Ef
    h_xe = (1 - CDFa(e)) * H(a - x) * 2 * m * (e * (1. - x/a)) ** (m-1) / L0 / s ** m 
    return h_xe

def pdfL(e, x):
    T = 2. * tau / r / Ef
    a = e / 2. / tau * r * Ef
    h_xe = (1 - CDFa(e)) * H(a - x) * 2 * m * (e * (1. - x/a)) ** (m-1) / L0 / s ** m
    pdf = m*T * (1-x/a)**(m-1) / e * H(a - x)
    print 'mean,', T * a**2/e/(m+1), a, a/(T * a**2/e/(m+1))
    return pdf #h_xe / PDFa(e)

def muLate(e):
    T = 2. * tau / r / Ef
    a = e / T
    return a/(m+1)

from scipy.optimize import fsolve
from scipy.special import gammainc, gamma

def evans(T):
    suma = (L0/r*(s* Ef)**m*tau*(m+1))**(1./(m+1.))
    alpha = (T/suma)**(m+1)
    return L0/2. * ((s* Ef)/suma)**m * gamma((m+2)/(m+1)) * gammainc((m+2)/(m+1), alpha) / CDFa(T/Ef)

def evans2(e):
    T = 2. * tau / r / Ef
    n = (m+1)
    c = 2. / L0 / T / n / s**m
    C = 2. / L0 / T**2 / n / s**m
    print C * ( + 0.*(c*0.**n)**(-1./n) * gamma(1./n)) / (c * n**2)
    return - C * (n * e * np.exp(-c*e**n) + e*(c*e**n)**(-1./n) * (gamma(1./n) - gamma(1./n) * gammainc(1./n, c*e**n))) / (c * n**2)

e_arr = np.linspace(0.005, 0.05, 100)
plt.plot(e_arr, evans(e_arr * Ef))
#print evans2(e_arr)
#plt.plot(e_arr, evans2(e_arr), lw=3, ls='dashed')
#plt.show()

from stats.spirrid import make_ogrid as orthogonalize
from mayavi import mlab
nx = 200
ne = 200
e_arr = np.linspace(0.0001, 0.04, ne)
x_arr = np.linspace(0.0, 80., nx)
#plt.plot([0.035], np.trapz(pdfL(0.035, x_arr) * x_arr, x_arr), 'go')
ee_arr = np.linspace(0.0, 0.05, ne)
#plt.plot([0.05], np.trapz(muLate(ee_arr) * PDFa(ee_arr), ee_arr), 'bo')
#print 'mean num', np.trapz(pdfL(0.035, x_arr) * x_arr, x_arr)
#h = h(e_arr.reshape(ne, 1), x_arr.reshape(1, nx))
#pdf_x = np.trapz(h, e_arr.flatten(), axis=0)
#print 'sum = ', np.trapz(pdf_x.flatten(), x_arr.flatten())
#print 'mu_L =', np.trapz(pdf_x.flatten() * x_arr, x_arr.flatten())
#ctrl_arr = orthogonalize([np.arange(len(e_arr.flatten())), np.arange(len(x_arr.flatten()))])
#mlab.surf(e_arr[0], e_arr[1], h / np.max(h) * nx)
#mlab.show()
#plt.plot(x_arr.flatten(), pdf_x.flatten())
#plt.show()

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

l_arr = np.linspace(0.0, 90., 50)
e_arr = np.linspace(0.001, 0.05, 100)
e_arr2 = np.linspace(0.015, 0.07, 20)
mc, fact = MC(10000, e_arr)
plt.plot(e_arr, mc, 'ro')
plt.plot(e_arr, fact, 'bo')
plt.xlabel('eps crack')
plt.ylabel('mu_L')
plt.legend(loc='best')
plt.show()
