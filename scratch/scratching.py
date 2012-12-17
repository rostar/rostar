import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz

Ef = 200e3
r = 0.003
tau = 0.1
m = 5.
s = 0.02
L0 = 100.

# failure probability between 0 and a
def CDFa(e):
    T = 2. * tau / r / Ef
    a = e / T
    return 1. - np.exp(-a * (e / s) ** m / (m + 1) / L0)

# failure probability density between 0 and a
def PDFa(e):
    T = 2. * tau / r / Ef
    a = e / T
    return np.exp(-a * (e / s) ** m / (m + 1) / L0) * ((e / s) ** m * m / (T * L0 * (m + 1)))

# failure probability between 0 and L
def CDFL(e, L):
    T = 2. * tau / r / Ef
    a = e / T
    return 1. - np.exp(-a * (e / s) ** m * (1-(1-L/a)**(m+1)) / (m + 1) / L0)

def scalar_mu_L(e):
    c = 1. / CDFa(e)
    T = 2. * tau / r / Ef
    a = e / T
    z_arr = np.linspace(0, a, 300)
    integ = np.trapz(CDFL(e, z_arr), z_arr)
    z_arr * PDFa(e)
    return a - c * integ

def mu_L(e_arr):
    muL = []
    for e in e_arr:
        muL.append(scalar_mu_L(e))
    return np.array(muL)

def scalar_mu_L_rr(e):
    e_arr = np.linspace(0.0001, e, 300)
    muL_arr = mu_L(e_arr)
    PDF = PDFa(e_arr)
    plt.figure()
    plt.plot(e_arr, PDF)
    plt.show()
    c = 1. / CDFa(e)
    print np.trapz(PDF, e_arr) / c
    integ = np.trapz(PDF * muL_arr, e_arr)
    return c * integ

def mu_L_rr(e_arr):
    muL = []
    for e in e_arr:
        muL.append(scalar_mu_L_rr(e))
    return np.array(muL)

e_arr = np.linspace(0.0001, 0.1, 20)
plt.plot(e_arr, CDFa(e_arr)/np.max(CDFa(e_arr)) * 10, label='CDF')
#plt.plot(e_arr, PDFa(e_arr), label='PDFa')
#plt.plot(e_arr, CDFL(e_arr, 2.))
plt.plot(e_arr, mu_L(e_arr), label='mu_L Curtin')
plt.plot(e_arr, mu_L_rr(e_arr), label='mu_L RR')
plt.xlabel('eps crack')
plt.ylabel('mu_L')
#plt.plot(e_arr, e_arr / 2. / tau * r * Ef, label='a')
plt.legend(loc='best')
plt.show()
