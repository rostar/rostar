'''
Created on 2.2.2013

@author: Q
'''
from indep_CB_model import CBResidual
from stats.spirrid.spirrid import SPIRRID
from stats.spirrid.rv import RV
from matplotlib import pyplot as plt
import numpy as np

E_f = 200e3
V_f = 0.01
r = 0.003
tau = .5
#m = 5.
s = 0.015
L_0 = 100.
Pf = RV('uniform', loc=0.0, scale=1.0)

def strength_vs_tau(tau_arr, m_arr, w_arr):
    for mi in m_arr:
        strength_arr1 = np.zeros_like(tau_arr)
        strength_arr2 = np.zeros_like(tau_arr)
        wmax_arr = np.zeros_like(tau_arr)
        for i, tau in enumerate(tau_arr):
            spirrid = SPIRRID(q=CBResidual(include_pullout=True),
            sampling_type='PGrid',
            evars=dict(w=w_arr),
            tvars=dict(tau=tau, E_f=E_f, V_f=V_f, r=r,
                       m=mi, L_0=L_0, s=s, Pf=Pf),
            n_int=5000)
            if isinstance(r, RV):
                r_arr = np.linspace(r.ppf(0.001), r.ppf(0.999), 300)
                Er = np.trapz(r_arr ** 2 * r.pdf(r_arr), r_arr)
            else:
                Er = r ** 2
            sigma_c1 = spirrid.mu_q_arr / Er
            strength1 = np.max(sigma_c1)
            if strength1 == sigma_c1[-1]:
                print 'w range to small'
            strength_arr1[i] = np.max(sigma_c1)
            wmax_arr[i] = w_arr[np.argmax(sigma_c1)]

#            spirrid = SPIRRID(q=CBResidual(include_pullout=False),
#            sampling_type='PGrid',
#            evars=dict(w=w_arr),
#            tvars=dict(tau=tau, E_f=E_f, V_f=V_f, r=r,
#                       m=mi, L_0=L_0, s=s, Pf=Pf),
#            n_int=1000)
#            if isinstance(r, RV):
#                r_arr = np.linspace(r.ppf(0.001), r.ppf(0.999), 300)
#                Er = np.trapz(r_arr ** 2 * r.pdf(r_arr), r_arr)
#            else:
#                Er = r ** 2
#            sigma_c2 = spirrid.mu_q_arr / Er
#            strength_arr2[i] = sigma_c2[np.argmax(sigma_c1)]
        T = 2. * tau_arr / r
        a = np.sqrt(wmax_arr * T / E_f) * E_f / T
        CDF = 1. - np.exp(-a * 2 * (np.sqrt(wmax_arr * T / E_f) / s) ** mi / (mi + 1) / L_0)
        plt.plot(tau_arr, CDF, lw=2, color='blue')
        #plt.plot(tau_arr, (strength_arr1-strength_arr2)/strength_arr1, lw=2, color='black')
        #x = np.log10(tau_arr)
        #y = np.log10(strength_arr)
        #plt.plot(np.diff(x) / np.diff(y), lw=2, color='black')
        #plt.plot(tau_arr, strength_arr2, lw=1, color='blue')
        plt.legend(loc='best')
    plt.show()


from scipy.special import gamma
def random_xi(w_arr, m_arr):
#    T = 2. * tau / r
#    m_ref = 5.0
#    n_ref = (m_ref+1)
#    c_ref = 2. * E_f / T / L_0 / n_ref / s**m_ref
#    mu_xi_ref = c_ref**(-1./n_ref)/n_ref * gamma(1./n_ref)
#    median_xi_ref = (-0.5 * np.log(1.-.5) * T / E_f * L_0 * (n_ref) * s**m_ref) ** (1./(n_ref))
#    xi_ref = mu_xi_ref
    for mi in m_arr:
#        ni = mi+1
#        si = (xi_ref * ni / gamma(1./ni))**(ni/mi) * (2 * E_f/ni/L_0/T)**(1./mi)
#        CDF = 1. - np.exp(-xi_ref * E_f / T * 2 * (xi_ref / si) ** mi / ni / L_0)
#        cb = CBResidual(include_pullout=False)
#        determ = cb(w_arr, tau, E_f, V_f, r, mi, L_0, si, CDF) / r**2
#        plt.plot(w_arr, determ, label='determ m = ' + str(mi))
       
        spirrid = SPIRRID(q=CBResidual(include_pullout=False),
                    sampling_type='PGrid',
                    evars=dict(w=w_arr),
                    tvars=dict(tau=tau, E_f=E_f, V_f=V_f, r=r,
                               m=mi, L_0=L_0, s=s, Pf=Pf),
                    n_int=5000)
        if isinstance(r, RV):
            r_arr = np.linspace(r.ppf(0.001), r.ppf(0.999), 300)
            Er = np.trapz(r_arr ** 2 * r.pdf(r_arr), r_arr)
        else:
            Er = r ** 2
        sigma_c = spirrid.mu_q_arr / Er
        plt.plot(w_arr, sigma_c, lw=2, label='m = ' + str(mi))
    plt.title('constant mean')
    plt.xlabel('crack opening w[mm]')
    plt.ylabel('composite stress [MPa]')
    plt.legend(loc='best')
    plt.show()


tau_arr = np.linspace(0.05, .5, 30)
m_arr = np.array([5.0, 10.0])
w_arr = np.linspace(0, 1.5, 500)
strength_vs_tau(tau_arr, m_arr, w_arr)
#random_xi(w_arr, m_arr)
#from scipy.stats import weibull_min
#l_arr = np.linspace(5.0,150.,100)
#med1 = []
#med2 = []
#med3 = []
#mu1 = []
#mu2 = []
#mu3 = []
#for l in l_arr:
#    med1.append(weibull_min(0.5, scale = (100./l)**(1./0.5)*0.02).ppf(0.5))
#    med2.append(weibull_min(2., scale = (100./l)**(1./2.)*0.02).ppf(0.5))
#    med3.append(weibull_min(10., scale = (100./l)**(1./10.)*0.02).ppf(0.5))
#    mu1.append(weibull_min(0.5, scale = (100./l)**(1./0.5)*0.02).stats('m'))
#    mu2.append(weibull_min(2., scale = (100./l)**(1./2.)*0.02).stats('m'))
#    mu3.append(weibull_min(10., scale = (100./l)**(1./10.)*0.02).stats('m'))
#plt.plot(l_arr, med1)
#plt.plot(l_arr, med2)
#plt.plot(l_arr, med3)
##plt.plot(l_arr, mu1, lw=2)
##plt.plot(l_arr, mu2, lw=2)
##plt.plot(l_arr, mu3, lw=2)
#print 100. * np.log(2)
#plt.ylim(0,0.2)
#plt.show()