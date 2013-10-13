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
from spirrid.spirrid import SPIRRID
from spirrid.rv import RV
from matplotlib import pyplot as plt
from scipy.stats import weibull_min


def fiber():
    cb = CBClamped()
    w = np.linspace(0.0, 1., 300)
    sigmaCB = []
    sigmaMC = []
    for wi in w:
        sigmaCB.append(cb(wi, .1, 240e3, 0.01, 0.0035, 7.0, 0.0046, 5000., 0.5))
        sigmaMC.append(cb(wi, .1, 240e3, 0.01, 0.0035, 7.0, 0.0046, 30., 0.5))
    plt.plot(w, np.array(sigmaMC)/0.00345**2, label='MC')
    plt.plot(w, np.array(sigmaCB)/0.00345**2, label='CB')
    plt.legend()
    plt.show()

def lcs_effect():
    
    def sigmac(w, lm, tau, m_xi):
        if isinstance(w, np.ndarray):
            pass
        else:
            print w
            w = np.array([w])
        cb = CBClampedRandXi()
        spirrid = SPIRRID(q=cb, sampling_type='PGrid',
                          eps_vars=dict(w=w),
                          theta_vars=dict(tau=tau,
                                          E_f=200e3,
                                          V_f=0.01,
                                          r=0.00345,
                                          m=m_xi,
                                          sV0=0.0026,
                                          lm=lm),
                          n_int=100)
        r = spirrid.theta_vars['r']
        if isinstance(r, RV):
            r_arr = np.linspace(r.ppf(0.001), r.ppf(0.999), 300)
            Er = np.trapz(r_arr ** 2 * r.pdf(r_arr), r_arr)
        else:
            Er = r ** 2
        sigma_c = spirrid.mu_q_arr / Er
        return - sigma_c
    
    def m_effect():
        strength_CB = []
        strength_MC = []
        m_arr = np.linspace(1.,20.,3)
        for m in m_arr:
            w_CB = np.linspace(0.0,100./m**2,500)
            w_MC = np.linspace(0.0,1.5/m**2,500)
            sigma_c_CB = - sigmac(w_CB, 1000., RV('weibull_min', shape=3.0, scale=0.1, loc=0.0), m)
            sigma_c_MC = - sigmac(w_MC, 1.0, RV('weibull_min', shape=3.0, scale=0.1, loc=0.0), m)
            strength_CB.append(np.max(sigma_c_CB))
            strength_MC.append(np.max(sigma_c_MC))
            plt.plot(w_CB, sigma_c_CB, label='CB')
            plt.plot(w_CB, sigma_c_MC, label='MC')
            plt.show()
        CB_arr = np.ones_like(np.array([strength_CB]))
        MC_arr = np.array([strength_MC]) / np.array([strength_CB])
        plt.plot(m_arr, CB_arr.flatten())
        plt.plot(m_arr, MC_arr.flatten())
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

fiber()
#lcs_effect()
