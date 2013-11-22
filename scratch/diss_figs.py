'''
Created on 12 Oct 2013

@author: Q
'''
'''
Created on Oct 7, 2013

@author: rostar
'''

import numpy as np
from quaducom.micro.resp_func.CB_clamped_rand_xi import CBClampedRandXi
from quaducom.micro.resp_func.CB_clamped import CBClamped
from quaducom.micro.resp_func.cb_short_fiber import CBShortFiber
from spirrid.spirrid import SPIRRID
from spirrid.rv import RV
from matplotlib import pyplot as plt
from scipy.stats import weibull_min
from stats.pdistrib.weibull_fibers_composite_distr import fibers_MC, fibers_CB_rigid, fibers_dry
from scipy.optimize import minimize_scalar
import pickle
from stats.spirrid import make_ogrid as orthogonalize
from mayavi import mlab
from math import pi
from scipy.stats import norm

cb = CBClampedRandXi()
spirrid = SPIRRID(q=cb, sampling_type='PGrid',
                  theta_vars=dict(tau=0.1,
                                  E_f=200e3,
                                  V_f=0.01,
                                  r=0.00345,
                                  m=5.0,
                                  sV0=0.0026,
                                  lm=1000.),
                  n_int=100)

def fiber():
    cb = CBClamped()
    w = np.linspace(0.0, 1., 300)
    sigmaCB = []
    sigmaMC = []
    for wi in w:
        sigmaCB.append(cb(wi, .1, 240e3, 0.01, 0.0035, 7.0, 0.0046, 5000., 0.5))
        sigmaMC.append(cb(wi, .1, 240e3, 0.01, 0.0035, 7.0, 0.0046, 20., 0.5))
    plt.plot(w, np.array(sigmaMC)/0.00345**2, label='MC')
    plt.plot(w, np.array(sigmaCB)/0.00345**2, label='CB')
    plt.legend()
    plt.show()

def lcs_effect():
    def sigmac(w, lm, tau, m_xi):
        if isinstance(w, np.ndarray):
            pass
        else:
            w = np.array([w])
        spirrid.eps_vars['w'] = w
        spirrid.theta_vars['lm'] = lm
        spirrid.theta_vars['tau'] = tau
        spirrid.theta_vars['m'] = m_xi
        r = spirrid.theta_vars['r']
        if isinstance(r, RV):
            r_arr = np.linspace(r.ppf(0.001), r.ppf(0.999), 300)
            Er = np.trapz(r_arr ** 2 * r.pdf(r_arr), r_arr)
        else:
            Er = r ** 2
        sigma_c = spirrid.mu_q_arr / Er
        return sigma_c

    def maxsigma(lm, tau, m_xi):
        def minfunc(w):
            res = sigmac(w, lm, tau, m_xi)
            return - res * (w < 200.) + 1e-5 * w**2
        w_max = minimize_scalar(minfunc, bracket=(0.0001, 0.0002))
        return w_max.x, sigmac(w_max.x, lm, tau, m_xi)

    def m_effect():
        m_xi_arr = np.linspace(1.,20.,15)
        m_tau_arr = np.linspace(0.3, 5.0, 10)
        try:
            res_file = open('res_m.pkl', 'rb' )
            res_arr = pickle.load(res_file)
            res_file.close()
#            for res_part in res_arr:
#                plt.plot(m_xi_arr, res_part)
#            plt.show()
            eps_vars = orthogonalize([np.arange(15), np.arange(10)])
            resCB = np.ones_like(res_arr)
            resMC = res_arr
            mlab.surf(eps_vars[0], eps_vars[1], resCB * 10)
            mlab.surf(eps_vars[0], eps_vars[1], resMC * 10)
            mlab.show()
        except:
            res_arr = []
            for m_tau in m_tau_arr:
                strength_CB = []
                strength_MC = []
                for m_xi in m_xi_arr:
                    wmCB, smCB = maxsigma(1000., RV('weibull_min', shape=m_tau, scale=0.1, loc=0.0), m_xi)
                    wmMC, smMC = maxsigma(1., RV('weibull_min', shape=m_tau, scale=0.1, loc=0.0), m_xi)
        #            w_CB = np.linspace(0.0,100./m**2,200)
        #            w_MC = np.linspace(0.0,1.5/m**2,200)
        #            plt.plot(wmCB, smCB, 'ro')
        #            plt.plot(wmMC, smMC, 'bo')
        #            sigma_c_CB = sigmac(w_CB, 1000., RV('weibull_min', shape=3.0, scale=0.1, loc=0.0), m)
        #            sigma_c_MC = sigmac(w_MC, 1.0, RV('weibull_min', shape=3.0, scale=0.1, loc=0.0), m)
        #            plt.plot(w_CB, sigma_c_CB, label='CB')
        #            plt.plot(w_MC, sigma_c_MC, label='MC')
        #            plt.show()
                    strength_CB.append(smCB)
                    strength_MC.append(smMC)
                CB_arr = np.ones_like(np.array([strength_CB]))
                MC_arr = np.array([strength_MC]) / np.array([strength_CB])
                res_arr.append(MC_arr.flatten())
                plt.plot(m_xi_arr, CB_arr.flatten(), lw=2., color='black')
                plt.plot(m_xi_arr, MC_arr.flatten(), label=str(m_tau)+' MC')
            res_arr = np.array(res_arr)
            file_res = open('res_m.pkl', 'wb')
            pickle.dump(res_arr, file_res, -1)
            file_res.close()
            plt.legend(loc='best')
            plt.ylim(0)
            plt.show()

    def T_effect():
        strength_CB = []
        strength_MC = []
        m_arr = np.linspace(.5, 5., 30)
        for m in m_arr:
            #strengths.append(sigmac_max(l)[0])
            #print 'strentgth = ', strengths[-1]
            w_CB = np.linspace(0.0,.5,100)
            w_MC = np.linspace(0.0,.03,100)
            sigma_c_CB = - sigmac(w_CB, 1000., RV('weibull_min', shape=m, scale=0.1, loc=0.0), 5.0)
            sigma_c_MC = - sigmac(w_MC, 1.0, RV('weibull_min', shape=m, scale=0.1, loc=0.0), 5.0)
            strength_CB.append(np.max(sigma_c_CB))
            strength_MC.append(np.max(sigma_c_MC))
    #         plt.plot(w_CB, sigma_c_CB, label='CB')
    #         plt.plot(w_CB, sigma_c_MC, label='MC')
    #         plt.show()
        COV = [np.sqrt(weibull_min(m, scale=0.1).var())/weibull_min(m, scale=0.1).mean() for m in m_arr]
        CB_arr = np.ones_like(np.array([strength_CB]))
        MC_arr = np.array([strength_MC]) / np.array([strength_CB])
        plt.plot(COV, CB_arr.flatten())
        plt.plot(COV, MC_arr.flatten())
        plt.ylim(0)
        plt.show()
    
    #T_effect()
    m_effect()

def Gxi():
    r, tau, Ef, m, sV0 = 0.00345, 0.1, 200e3, 3.0, 0.0026
    e = np.linspace(0.001, 0.04, 20)
    a = e * Ef / (2. * tau / r)
    L = 10.0
    rat = 2.*a/L
        
    wfd = fibers_dry(m=m, sV0=sV0)
    CDFdry = wfd.cdf(e, r, 2*a)
    
    wfcbr = fibers_CB_rigid(m=m, sV0=sV0)
    CDFCB = wfcbr.cdf(e, 2*tau/Ef/r, r)
    
    wfmc = fibers_MC(m=m, sV0=sV0, Ll=L, Lr=L)
    CDFexct = wfmc.cdf_exact(e, 2*tau/Ef/r, r)
    CDFrypl = wfmc.cdf(e, 2*tau/Ef/r, r, a, a)
    
    #linear scale
#    plt.plot(e, CDFrypl, label='Rypl')
#    plt.plot(e, CDFexct, label='exact')
#    plt.plot(e, CDFdry, label='Phoenix 1992')
#    plt.plot(e, CDFCB, label='CB')
    
    # Weibull plot
#    print rat
    plt.plot(np.log(e), np.log(-np.log(1.0 - CDFdry)), label='Phoenix 1992')
    plt.plot(np.log(e), np.log(-np.log(1.0 - CDFCB)), label='CB')
    plt.plot(np.log(e), np.log(-np.log(1.0 - CDFrypl)), label='Rypl')
    plt.plot(np.log(e), np.log(-np.log(1.0 - CDFexct)), label='exact')

    #plt.ylim(0)
    plt.legend(loc='best')
    plt.show()

cb = CBShortFiber()
lf = 14.0
sf_spirrid = SPIRRID(q=cb, sampling_type='PGrid',
                      theta_vars=dict(tau=1.8,
                                  E_f=200e3,
                                  r=0.088,
                                  le=RV('uniform', scale=lf/2., loc=0.0),
                                  phi=RV('sin2x', scale=1.0),
                                  f=.87,
                                  xi=20e10),
                  n_int=100)

def short_det_vs_rand():
    w = np.linspace(0.0,0.02,200)
    sf_spirrid.eps_vars['w'] = w
    sf_spirrid.theta_vars['le'] = lf/2.
    sf_spirrid.theta_vars['phi'] = 0.3
    Ef = sf_spirrid.theta_vars['E_f']
    plt.plot(w, sf_spirrid.mu_q_arr)
    plt.legend(loc='best')
    plt.ylim(0)
    plt.show()

def short_fibers_f():
    w = np.linspace(0.0,0.02,200)
    Ef = sf_spirrid.theta_vars['E_f']
    Vf = 0.01
    sf_spirrid.eps_vars['w'] = w
    sf_spirrid.theta_vars['f'] = 1.0
    plt.plot(w, sf_spirrid.mu_q_arr * Ef * Vf / 2., label='1.0')
    sf_spirrid.theta_vars['f'] = 0.5
    plt.plot(w, sf_spirrid.mu_q_arr * Ef * Vf / 2., label='0.5')
    sf_spirrid.theta_vars['phi'] = 0.0
    plt.plot(w, sf_spirrid.mu_q_arr * Ef * Vf, label='aligned')
    plt.legend(loc='best')
    plt.show()
    
def short_fibers_lf():
    w = np.linspace(0.0,0.02,200)
    Ef = sf_spirrid.theta_vars['E_f']
    Vf = 0.01
    sf_spirrid.eps_vars['w'] = w
    sf_spirrid.theta_vars['le'] = RV('uniform', scale=5.0, loc=0.0)
    plt.plot(w, sf_spirrid.mu_q_arr * Ef * Vf / 2., label='10.0')
    sf_spirrid.theta_vars['le'] = RV('uniform', scale=7.0, loc=0.0)
    plt.plot(w, sf_spirrid.mu_q_arr * Ef * Vf / 2., label='14.0')
    sf_spirrid.theta_vars['le'] = RV('uniform', scale=9.0, loc=0.0)
    plt.plot(w, sf_spirrid.mu_q_arr * Ef * Vf / 2., label='18.0')
    plt.legend(loc='best')
    plt.show()
    
def short_fibers_strength_f():
    sf_spirrid.eps_vars['w'] = np.array([1000.])
    f_arr = np.linspace(0.0, 1.5, 100)
    func = (np.exp(f_arr * pi / 2.) + 1) / (f_arr**2 + 4)
    plt.plot(f_arr, np.ones_like(f_arr), label='aligned')
    plt.plot(f_arr, func, label='analyt')
    plt.ylim(0,2)
    plt.show()

def short_fibers_strength_var():
    sf_spirrid.codegen.implicit_var_eval=True
    sf_spirrid.eps_vars['w'] = np.array([1000.])
    var = sf_spirrid.var_q_arr
    mu = sf_spirrid.mu_q_arr
    cov_e = np.sqrt(var)/mu

    Vf = 0.01
    r = sf_spirrid.theta_vars['r']
    Af = pi * r ** 2
    Lc = 100.
    Ac = 1600.
    lf = 14.0

    Ac_arr = np.linspace(1600.0, 6400., 200)
    COV_Ac = np.sqrt(2. * Af / Vf / Ac_arr) * np.sqrt(cov_e**2 + (1. - lf/2./Lc))
    #plt.plot(Ac_arr, COV_Ac, label='COV')

    Vf_arr = np.linspace(0.01, 0.04, 200)
    COV_Vf = np.sqrt(2. * Af / Vf_arr / Ac) * np.sqrt(cov_e**2 + (1. - lf/2./Lc))
    #plt.plot(Vf_arr, COV_Vf, label='COVVf',ls='dashed',lw=3)

    COV_lf = []
    lf_arr = np.linspace(1., 30., 100)
    for lfi in lf_arr:
        sf_spirrid.theta_vars['le'] = RV('uniform', scale=lf/2., loc=0.0)
        var = sf_spirrid.var_q_arr
        mu = sf_spirrid.mu_q_arr
        cov_e = np.sqrt(var)/mu
        COV = np.sqrt(2. * Af / Vf / Ac) * np.sqrt(cov_e**2 + (1. - lfi/2./Lc))
        COV_lf.append(COV)
    #plt.plot(lf_arr, COV_lf, label='COV_lf')

    COV_Lc = []
    Lc_arr = np.linspace(50., 1000., 100)
    for Lci in Lc_arr:
        var = sf_spirrid.var_q_arr
        mu = sf_spirrid.mu_q_arr
        cov_e = np.sqrt(var)/mu
        COV = np.sqrt(2. * Af / Vf / Ac) * np.sqrt(cov_e**2 + (1. - lf/2./Lci))
        COV_Lc.append(COV)
    plt.plot(Lc_arr, COV_Lc, label='COV_Lc')

    plt.ylim(0,0.08)
    plt.legend()
    plt.show()

def hybrid():
    cb = CBClampedRandXi()
    w = np.linspace(0.0, 1., 300)
    spirrid = SPIRRID(q=cb, sampling_type='PGrid',
                      eps_vars=dict(w=w),
                  theta_vars=dict(tau=0.1,
                                  E_f=200e3,
                                  V_f=0.01,
                                  r=0.00345,
                                  m=5.0,
                                  sV0=0.0026,
                                  lm=1000.),
                  n_int=100)
    sigma_c1 = spirrid.mu_q_arr / spirrid.theta_vars['r'] ** 2
    plt.plot(w, sigma_c1,label='1')
    spirrid.theta_vars['E_f'] = 70e3
    spirrid.theta_vars['r'] = 0.013
    sigma_c2 = spirrid.mu_q_arr / spirrid.theta_vars['r'] ** 2
    plt.plot(w, sigma_c2,label='2')
    plt.plot(w, sigma_c2 + sigma_c1)
    plt.legend()
    plt.show()

def short_fibers_CHOB():
    cb = CBShortFiber()
    Ef = 70e3
    Vf = 0.015
    r=0.15
    Lc = 100.
    Ac = 1600.
    lf = 14.
    spirrid = SPIRRID(q=cb, sampling_type='PGrid',
                      eps_vars=dict(w=np.array([100.0])),
                      theta_vars=dict(tau=1.67,
                                  E_f=Ef,
                                  r=r,
                                  le=RV('uniform', scale=lf/2., loc=0.0),
                                  phi=RV('sin2x', scale=1.0),
                                  f=.99,
                                  xi=20e10),
                  n_int=100
                  )
    
    spirrid.codegen.implicit_var_eval=True
    var_e = spirrid.var_q_arr
    mu_e = spirrid.mu_q_arr
    print mu_e
    Af = pi * r ** 2
    p = lf / 2. / Lc
    n = Ac * Lc * Vf / Af / lf
    mu_strength = Ef * Vf / 2. * mu_e
    var_strength  = (Ef * Vf / 2.) ** 2 / n / p * (var_e + (1. - p)*mu_e**2)
    n = np.arange(10)
    distr = norm(loc=mu_strength, scale=var_strength**(0.5))
    sig_arr = np.linspace(mu_strength/2., mu_strength*1.5, 100)
    plt.plot(sig_arr, distr.cdf(sig_arr), label='strength distr')
    plt.legend()
    plt.show()
   
#fiber()
#lcs_effect()
#Gxi()
#short_det_vs_rand()
#short_fibers_f()
#short_fibers_lf()
#short_fibers_strength_f()
#short_fibers_strength_var()
hybrid()
#short_fibers_CHOB()