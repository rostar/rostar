import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import weibull_min
from scipy.optimize import brentq

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
    return 1. - np.exp(-a * (e / s) ** m / (m + 1) / L0)

# failure probability density between 0 and a at e
def PDFa(e):
    T = 2. * tau / r / Ef
    a = e / T
    return np.exp(-a * (e / s) ** m / L0 / (m + 1)) * (e / s) ** m / (T * L0)

# failure probability between 0 and L at e
def CDF_L(e, L):
    T = 2. * tau / r / Ef
    a = e / T
    Pf = 1. - np.exp(-a * (e / s) ** m * (1 - (1 - L / a) ** (m + 1)) / (m + 1) / L0)
    return Pf / CDFa(e)

# failure probability for dx at x given failure at e
def Pf_dx_L(e, L, dx):
    T = 2. * tau / r / Ef
    a = e / T
    
    CDF = np.exp((a * (e / s) ** m * (1 - L / a) ** (m + 1)) / (m + 1) / L0)
    return CDF

#l_arr = np.linspace(0.0, 80., 1000)
#plt.plot(l_arr, CDFL2(0.02, l_arr))
#plt.show()

# failure probability density at L, e
def PDFLgiven_e(e, L):
    T = 2. * tau / r / Ef
    a = e / T
    pdf = m * dx * (1 - L/a) / L0 / s / a * (e * (1.-L/a)/s)**(m-1)
    return pdf * H(a - L)

# failure probability density at L, e
def PDFL(e, L, dx):
    T = 2. * tau / r / Ef
    a = e / T
    pdf = (1. - CDFa(e)) * m * dx * (1 - L/a) / L0 / s / a * (e * (1.-L/a)/s)**(m-1)
    return pdf * H(a - L)

#e_arr = np.linspace(0.0001, 0.05, 500).reshape(500,1)
#l_arr = np.linspace(0., 60., 200).reshape(1, 200)
#
##print np.trapz(PDFL(e_arr, l_arr), e_arr, axis = 0)
#PdF = PDFL(e_arr, l_arr, 60./200.)
#p = np.trapz(PdF, e_arr, axis = 0).flatten()
#P = np.trapz(p, l_arr.flatten())
#print P
#plt.plot(l_arr.flatten(), np.trapz(PDFL(e_arr, l_arr), e_arr, axis = 0).flatten())
#plt.show()

def muL(e):
    T = 2. * tau / r / Ef
    a = e / T
    L_arr = np.linspace(0., a, 100)
    pdfL = PDFL(e, L_arr) / CDFa(e)
    mul = np.trapz(pdfL * L_arr, L_arr)
    return mul

def muLate(e):
    muL_lst = []
    e_arr = np.linspace(0.0001, e, 100)
    for ei in e_arr:
        muL_lst.append(muL(ei))
    muL_arr = np.array(muL_lst)
    PDFa_arr = PDFa(e_arr) / (CDFa(e))
    muLate_value = np.trapz(muL_arr * PDFa_arr, e_arr)
    return muLate_value

def muLate_arr(e_arr):
    muLate_lst = []
    for eii in e_arr:
        muLate_lst.append(muLate(eii))
    return np.array(muLate_lst)

def scalar_mu_L(e):
    c = 1. / CDFa(e)
    T = 2. * tau / r / Ef
    a = e / T
    z_arr = np.linspace(0, a, 300)
    integ = np.trapz(CDFL2(e, z_arr), z_arr)
    z_arr * PDFa(e)
    return a - c * integ

def mu_L(e_arr):
    muL = []
    for e in e_arr:
        muL.append(scalar_mu_L(e))
    return np.array(muL)

def mu_L_at_e(e):
    T = 2. * tau / r / Ef
    a = e / T
    z_arr = np.linspace(0.0, a, 50)
    z_arr_1 = np.hstack((0.0, z_arr[:-1]))
    z_arr_2 = np.hstack((z_arr[1:], z_arr[-1]))
    integ = np.trapz(z_arr * (1 - CDFL(e, z_arr_1)) * CDFw(e - T*z_arr, z_arr[1]-z_arr[0]) * (1. - CDFa(e - T*z_arr_2)), z_arr)
    c = 1. / np.trapz((1 - CDFL(e, z_arr_1)) * CDFw(e - T * z_arr, z_arr[1]-z_arr[0]) * (1. - CDFa(e - T*z_arr_2)), z_arr)
    return c * integ

def scalar_mu_L_rr(e):
    e_arr = np.linspace(0.0001, e, 100)
    muL = []
    for ei in e_arr:
        muL.append(mu_L_at_e(ei))
    muL_arr = np.array(muL)
    PDF = PDFa(e_arr)
    c = 1. / CDFa(e)
    integ = np.trapz(PDF * muL_arr, e_arr)
    return c * integ

def mu_L_rr(e_arr):
    muL = []
    for e in e_arr:
        muL.append(scalar_mu_L_rr(e))
        #muL.append(mu_L_at_e(e))
    return np.array(muL)

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

from math import pi
from scipy.special import gamma, gammainc

def evans():
    s = 0.02 * 200e3
    suma = (L0 * s**m * tau * (m+1)/r)**(1./(m+1))
    h = L0/2. * (s/suma)**m * gamma((m+2)/(m+1))
    return h

def evans_arr(e):
    s = 0.02 * 200e3
    suma = ((L0 * s**m * tau * (m+1))/r)**(1./(m+1))
    alpha = (e * 200e3/suma)**(m+1)
    h = L0/2. * (s/suma) ** m * gammainc((m+2)/(m+1), alpha)
    return h

l_arr = np.linspace(0.0, 60., 100)
e_arr = np.linspace(0.001, 0.07, 100)
e_arr2 = np.linspace(0.015, 0.07, 20)
#plt.plot(e_arr, evans_arr(e_arr), label='Evans')
#plt.plot(e_arr, muLate_arr(e_arr), label='new')
plt.plot(e_arr2, MC(50000, e_arr2), 'ro', label='MC')
#plt.plot(e_arr, CDFa(e_arr) / np.max(CDFa(e_arr)) * 10, label='10xCDF')
#plt.plot(e_arr, PDFa(e_arr), label='PDFa')
#plt.plot(l_arr, PDFL(0.02, l_arr), label='PDFL')
#plt.plot(e_arr, mu_L(e_arr), label='mu_L Curtin')
#plt.plot(e_arr, mu_L_rr(e_arr), label='mu_L RR')
plt.xlabel('eps crack')
plt.ylabel('mu_L')
#plt.plot(e_arr, e_arr / 2. / tau * r * Ef, label='a')
plt.legend(loc='best')
plt.show()
