import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz
from scipy.stats import weibull_min
from scipy.optimize import brentq

Ef = 200e3
r = 0.003
tau = 0.1
m = 5.
s = 0.02
L0 = 100.

def CDFw(e, L):
    return 1 - np.exp(-L/L0 * (e/s)**m)

# failure probability between 0 and a
def CDFa(e):
    T = 2. * tau / r / Ef
    a = e / T
    return 1. - np.exp(-a * (e / s) ** m / (m + 1) / L0)

# failure probability density between 0 and a
def PDFa(e):
    T = 2. * tau / r / Ef
    a = e / T
    return np.exp(-a * (e / s) ** m / L0 / (m + 1)) * (e / s) ** m / (T * L0)

# failure probability between 0 and L
def CDFL(e, L):
    T = 2. * tau / r / Ef
    a = e / T
    return 1. - np.exp(-a * (e / s) ** m * (1 - (1 - L / a) ** (m + 1)) / (m + 1) / L0)

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

def mu_L_at_e(e):
    T = 2. * tau / r / Ef
    a = e / T
    z_arr = np.linspace(0.0, a, 10)
    z_arr_1 = np.hstack((0.0, z_arr[:-1]))
    z_arr_2 = np.hstack((z_arr[1:], z_arr[-1]))
    integ = np.trapz(z_arr * (1 - CDFL(e, z_arr_1)) * CDFw(e - T*z_arr, z_arr[1]-z_arr[0]) * (1. - CDFa(e - T*z_arr_2)), z_arr)
    c = 1. / np.trapz((1 - CDFL(e, z_arr_1)) * CDFw(e - T * z_arr, z_arr[1]-z_arr[0]) * (1. - CDFa(e - T*z_arr_2)), z_arr)
    print e, c * integ
    plt.plot(z_arr, (1 - CDFL(e, z_arr_1)) * CDFw(e - T*z_arr, z_arr[1]-z_arr[0]) * (1. - CDFa(e - T*z_arr_2)) * c, label='1')
    z_arr = np.linspace(0.0, a, 50)
    z_arr_1 = np.hstack((0.0, z_arr[:-1]))
    z_arr_2 = np.hstack((z_arr[1:], z_arr[-1]))
    integ = np.trapz(z_arr * (1 - CDFL(e, z_arr_1)) * CDFw(e - T*z_arr, z_arr[1]-z_arr[0]) * (1. - CDFa(e - T*z_arr_2)), z_arr)
    c = 1. / np.trapz((1 - CDFL(e, z_arr_1)) * CDFw(e - T * z_arr, z_arr[1]-z_arr[0]) * (1. - CDFa(e - T*z_arr_2)), z_arr)
    print e, c * integ
    plt.plot(z_arr, (1 - CDFL(e, z_arr_1)) * CDFw(e - T*z_arr, z_arr[1]-z_arr[0]) * (1. - CDFa(e - T*z_arr_2)) * c, label='2')    
    z_arr = np.linspace(0.0, a, 120)
    z_arr_1 = np.hstack((0.0, z_arr[:-1]))
    z_arr_2 = np.hstack((z_arr[1:], z_arr[-1]))
    integ = np.trapz(z_arr * (1 - CDFL(e, z_arr_1)) * CDFw(e - T*z_arr, z_arr[1]-z_arr[0]) * (1. - CDFa(e - T*z_arr_2)), z_arr)
    c = 1. / np.trapz((1 - CDFL(e, z_arr_1)) * CDFw(e - T * z_arr, z_arr[1]-z_arr[0]) * (1. - CDFa(e - T*z_arr_2)), z_arr)
    print e, c * integ
    plt.plot(z_arr, (1 - CDFL(e, z_arr_1)) * CDFw(e - T*z_arr, z_arr[1]-z_arr[0]) * (1. - CDFa(e - T*z_arr_2)) * c, label='3')  
    plt.legend()
    plt.show()
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
    eps = e - T*z_arr
    dz = a/nz
    fu_arr1 = weibull_min(m, scale=s * (dz/L0)**(-1./m)).ppf(np.random.rand(nz))
    fu_arr2 = weibull_min(m, scale=s * (dz/L0)**(-1./m)).ppf(np.random.rand(nz))
    fu_arr = np.minimum(fu_arr1, fu_arr2)
    def residuum(epsf0):
        return np.min(fu_arr1 - (epsf0 - T*z_arr))
    try:
        epsu = brentq(residuum, 0.0000001, e)
        idx = np.argmin(fu_arr - (epsu - T*z_arr))
        return z_arr[idx] 
    except:
        pass
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

def MC2(n, e_arr):
    T = 2. * tau / r / Ef
    amax = e_arr[-1] / T
    nz = 500
    z_arr = np.linspace(amax/2./nz, amax - amax/2./nz, nz)
    dz = amax / nz
    strengths = []
    for i in range(n):
        rn = np.minimum(np.random.rand(nz), np.random.rand(nz))
        strengths.append(weibull_min(m, scale=s * (dz/L0)**(-1./m)).ppf(rn))

    def residuum(epsf0, fu_arr):
        return np.min(fu_arr - (epsf0 - T*z_arr))

    mu_l_e = []
    for e in e_arr:
        l = []
        for f in strengths:
            try:
                eu = brentq(residuum, 0.00001, e, args=(f))
                l.append(z_arr[np.argmin(f - (eu - T*z_arr))])
            except:
                pass
        if len(l) == 0:
            mu_l_e.append(0.0)
        else:
            mean_l = np.mean(np.array(l))
            mu_l_e.append(mean_l)
    return mu_l_e


e_arr = np.linspace(0.001, 0.07, 100)
e_arr2 = np.linspace(0.015, 0.07, 20)
#plt.plot(e_arr2, MC(2000, e_arr2), 'ro', label='MC')
#plt.plot(e_arr, MC2(500, e_arr), 'bo')
plt.plot(e_arr, CDFa(e_arr) / np.max(CDFa(e_arr)) * 10, label='10xCDF')
#plt.plot(e_arr, PDFa(e_arr), label='PDFa')
#plt.plot(e_arr, CDFL(e_arr, 2.))
plt.plot(e_arr, mu_L(e_arr), label='mu_L Curtin')
plt.plot(e_arr, mu_L_rr(e_arr), label='mu_L RR')
plt.xlabel('eps crack')
plt.ylabel('mu_L')
#plt.plot(e_arr, e_arr / 2. / tau * r * Ef, label='a')
plt.legend(loc='best')
plt.show()
