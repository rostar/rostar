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
m = 5.
sV0 = 3.e-3
Pf = RV('uniform', loc=0.0, scale=1.0)

def H(x):
    return x >= 0.0

def Fig1_general_diagram():
    w_arr = np.linspace(0,1.0,1000)
    cb = CBResidual(include_pullout=True)
    total = SPIRRID(q=cb,
                sampling_type='PGrid',
                evars=dict(w=w_arr),
                tvars=dict(tau=tau, E_f=E_f, V_f=V_f, r=r,
                           m=m, sV0=sV0, Pf=Pf),
                n_int=1000)
    if isinstance(r, RV):
        r_arr = np.linspace(r.ppf(0.001), r.ppf(0.999), 300)
        Er = np.trapz(r_arr ** 2 * r.pdf(r_arr), r_arr)
    else:
        Er = r ** 2
    total = total.mu_q_arr / Er    
    plt.plot(w_arr, total, lw=2, color='black')
    
    cb = CBResidual(include_pullout=False)
    broken = SPIRRID(q=cb,
                sampling_type='PGrid',
                evars=dict(w=w_arr),
                tvars=dict(tau=tau, E_f=E_f, V_f=V_f, r=r,
                           m=m, sV0=sV0, Pf=Pf),
                n_int=1000)
    if isinstance(r, RV):
        r_arr = np.linspace(r.ppf(0.001), r.ppf(0.999), 300)
        Er = np.trapz(r_arr ** 2 * r.pdf(r_arr), r_arr)
    else:
        Er = r ** 2
    broken = broken.mu_q_arr / Er    
    plt.plot(w_arr, broken, lw=2, ls='dashed', color='black')
    plt.plot(w_arr, total-broken, lw=2, ls='dashed', color='black')   
    plt.ylim(0, 30)
    plt.xlim(0, .9)
    plt.xlabel('w [mm]')
    plt.ylabel('sigma_c [MPa]')
    plt.show()

from math import pi
from scipy.optimize import fsolve
def Fig2_rand_xi():
    def get_scale(mu_xi, m):
        def optimize(s):
            p = np.linspace(0., .9999, 1000)
            T = 2. * tau / r
            ef0_break = (-0.5 * np.log(1.-p) * T / E_f * (m+1) * s**m) ** (1./(m+1))
            return np.trapz(1-p, ef0_break) - mu_xi
        return fsolve(optimize, mu_xi)
    
    T = 2. * tau / r
    p = np.linspace(0., .999999, 1000)
    w_arr = np.linspace(0,1.0,1000)
    cb = CBResidual(include_pullout=True)
    if isinstance(r, RV):
        r_arr = np.linspace(r.ppf(0.001), r.ppf(0.999), 300)
        Er = np.trapz(r_arr ** 2 * r.pdf(r_arr), r_arr)
    else:
        Er = r ** 2
    for mi in [4., 12., 100.]:
        s = get_scale(0.02, mi)
        sV0 = float(s * (pi*r**2)**(1./mi))
#        ef0_break = (-0.5 * np.log(1.-p) * T / E_f * (mi+1) * s**mi) ** (1./(mi+1))
#        mu_e = np.trapz(1-p, ef0_break)
#        plt.plot(ef0_break, p, lw=2, color='black')
#        plt.ylim(0,1.1)
        total = SPIRRID(q=cb,
                sampling_type='PGrid',
                evars=dict(w=w_arr),
                tvars=dict(tau=tau, E_f=E_f, V_f=V_f, r=r,
                           m=mi, sV0=sV0, Pf=Pf),
                n_int=1000)
        result = total.mu_q_arr / Er    
        plt.plot(w_arr, result, lw=2, color='black')
    plt.xlabel('w [mm]')
    plt.ylabel('sigma_c [MPa]')
    plt.xlim(0,0.85)
    plt.ylim(0,45.)
    plt.show()

from scipy.special import gammainc, gamma
def Fig3_rand_xi_T():
    n = m+1
    tau = np.linspace(0.2, 2.0, 10)
    T = 2. * tau / r
    s = sV0 / (pi * r ** 2)**(1./m)
    c = 2*E_f/n/s**m/T
    gam = gamma(1./n) * gammainc(1./n, 1./m)
    intact = E_f * V_f * (m*c)**(-1./n) * np.exp(-1./m)
    broken = E_f * V_f * (1./(n**2 * c **(1./n)) * gam - 1./(n*(m*c)**(1./n)) * np.exp(-1./m))
    strength = intact + broken
    plt.plot(T, strength, lw=2, color='black')
    plt.show()

def Fig4_discrete_r():
    for ri in [0.005, 0.01, 0.015]: 
        m = 4.
        n = m+1
        T = 2. * tau / ri
        w_arr = np.linspace(0,.65,1000)
        xi_med = ((np.log(2.)*tau*n*sV0**m)/(ri**3 * E_f * pi))**(1./n)
        epsf0 = np.sqrt(T * w_arr / E_f)
        sigmac = E_f * V_f * epsf0 * H(xi_med - epsf0)
        plt.plot(w_arr, sigmac, lw=2, color='black')
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    for mi in [4.0, 5.0, 6.0]:
        r_arr = np.linspace(0.001, 0.020, 500)
        wstar = E_f/2./tau*r_arr *((np.log(2.)*tau*(mi+1)*sV0**mi)/(r_arr**3 * E_f * pi))**(2./(mi+1))
        xi_med = ((np.log(2.)*tau*(mi+1)*sV0**mi)/(r_arr**3 * E_f * pi))**(1./(mi+1))
        strength = E_f * V_f * xi_med
        ax1.loglog(r_arr, strength, lw=2, color='black')
        ax2.loglog(r_arr, wstar, lw=2, color='black', ls='dashed')
    plt.xlim(0,0.02)
    plt.ylim(0)
    plt.show()

def Fig5_rand_r():
    w_arr = np.linspace(0, .6, 1000)
    cb = CBResidual(include_pullout=True)
    for i, ri in enumerate([RV('norm', loc=0.01, scale=.0001),
                            RV('norm', loc=0.01, scale=.002),
                            RV('norm', loc=0.01, scale=.003)]):
        total = SPIRRID(q=cb,
                sampling_type='PGrid',
                evars=dict(w=w_arr),
                tvars=dict(tau=tau, E_f=E_f, V_f=V_f, r=ri,
                           m=4.0, sV0=sV0, Pf=0.5),
                n_int=500)
        if isinstance(ri, RV):
            r_arr = np.linspace(ri.ppf(0.001), ri.ppf(0.999), 300)
            Er = np.trapz(r_arr ** 2 * ri.pdf(r_arr), r_arr)
        else:
            Er = ri ** 2
        result = total.mu_q_arr / Er
        plt.plot(w_arr, result, lw=2, color='black')
    plt.xlabel('w [mm]')
    plt.ylabel('sigma_c [MPa]')
    plt.show()

def Fig6_discrete_tau():
    for taui in [0.2, 0.5, 2.0]: 
        m = 4.
        r = 0.01
        n = m+1
        T = 2. * taui / r
        w_arr = np.linspace(0,.65,1000)
        xi_med = ((np.log(2.)*taui*n*sV0**m)/(r**3 * E_f * pi))**(1./n)
        epsf0 = np.sqrt(T * w_arr / E_f)
        sigmac = E_f * V_f * epsf0 * H(xi_med - epsf0)
        plt.plot(w_arr, sigmac, lw=2, color='black')
    plt.ylim(0,40)
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    for mi in [2.0, 5.0, 10.0]:
        tau_arr = np.linspace(0.3, 2., 500)
        wstar = E_f/2./tau_arr*r *((np.log(2.)*tau_arr*(mi+1)*sV0**mi)/(r**3 * E_f * pi))**(2./(mi+1))
        xi_med = ((np.log(2.)*tau_arr*(mi+1)*sV0**mi)/(r**3 * E_f * pi))**(1./(mi+1))
        strength = E_f * V_f * xi_med
        ax1.loglog(tau_arr, strength, lw=2, color='black')
        ax2.loglog(tau_arr, wstar, lw=2, color='black', ls='dashed')
    plt.xlim(0.3,2.0)
    plt.ylim(0)
    plt.show()

def Fig7_rand_tau():
    r = 0.01
    w_arr = np.linspace(0, .6, 1000)
    cb = CBResidual(include_pullout=True)
    for i, taui in enumerate([RV('norm', loc=.5, scale=.005),
                            RV('norm', loc=0.5, scale=.1),
                            RV('norm', loc=0.5, scale=.15)]):
        total = SPIRRID(q=cb,
                sampling_type='PGrid',
                evars=dict(w=w_arr),
                tvars=dict(tau=taui, E_f=E_f, V_f=V_f, r=r,
                           m=4.0, sV0=sV0, Pf=0.5),
                n_int=500)
        if isinstance(r, RV):
            r_arr = np.linspace(r.ppf(0.001), r.ppf(0.999), 300)
            Er = np.trapz(r_arr ** 2 * r.pdf(r_arr), r_arr)
        else:
            Er = r ** 2
        result = total.mu_q_arr / Er
        plt.plot(w_arr, result, lw=2, color='black')
    plt.xlabel('w [mm]')
    plt.ylabel('sigma_c [MPa]')
    plt.show()

#Fig1_general_diagram()
#Fig2_rand_xi()
#Fig3_rand_xi_T()
#Fig4_discrete_r()
#Fig5_rand_r()
#Fig6_discrete_tau()
Fig7_rand_tau()

