'''
Created on Feb 20, 2015

@author: rostislavrypl
'''

from multi_scale_CB_model import GFRCMicro, GFRCMeso, GFRCMacro
import matplotlib.pyplot as plt
import numpy as np
from quaducom.micro.resp_func.cb_short_fiber import CBShortFiber
from math import pi
from spirrid.rv import RV

def random_tau():
    w_arr = np.linspace(0.,.3,200)
    micro = GFRCMicro(resp_func=CBShortFiber)
    meso = GFRCMeso(micro_model=micro)
    macro = GFRCMacro(meso_model=meso,
                      W=40., H=40., L=200., Lf=18.,
                      r=8e-3, Vf=.00020, N_fil=100,
                      Ef=70e3, snub=0.5, xi=1e15,
                      tau = RV('uniform', loc=0.1, scale=0.7),
                      w_arr=w_arr)
    mean_resp, var_resp = macro.CB_response_asymptotic
    mean_N_fib = macro.N_fib_bridging.mean()
    
    plt.plot(w_arr, mean_resp * macro.N_fil * pi*(macro.r)**2 / macro.Vf / mean_N_fib, lw=2, label='tau $\sim$ $\mathcal{U}(0.1,0.8)$')
    
    for t in np.linspace(0.1,0.8,8):
        macro.tau = t
        mean_resp, var_resp = macro.CB_response_asymptotic
        mean_N_fib = macro.N_fib_bridging.mean()
        plt.plot(w_arr, mean_resp * macro.N_fil * pi*(macro.r)**2 / macro.Vf / mean_N_fib,
                 lw=1, color='black', label='tau = ' + str(t))
    
    #plt.plot(w_arr, mean_r + np.sqrt(var_r), lw=2)
    #plt.plot(w_arr, mean_r - np.sqrt(var_r), lw=2)
    plt.xlabel('crack opening w [mm]')
    plt.ylabel('force [N]')
    #plt.legend(loc='best')
    plt.show()

def multiple_phi():
    w_arr = np.linspace(0.,.3,200)
    micro = GFRCMicro(resp_func=CBShortFiber)
    meso = GFRCMeso(micro_model=micro)
    macro = GFRCMacro(meso_model=meso, phi=0.0,
                      W=40., H=40., L=200., Lf=18.,
                      r=8e-3, Vf=.00020, N_fil=100,
                      Ef=70e3, snub=0.5, xi=1e15,
                      tau = RV('uniform', loc=0.1, scale=0.7),
                      w_arr=w_arr, spall=0.5)
    
    for i,p in enumerate(np.linspace(15.0, 75.0, 5)):
        macro.phi = p/180. * pi
        mean_resp, var_resp = macro.CB_response_asymptotic
        mean_N_fib = macro.N_fib_bridging.mean()
        plt.plot(w_arr, mean_resp * macro.N_fil * pi*(macro.r)**2 / macro.Vf / mean_N_fib,
                 lw=1 + (i/2), label='phi = ' + str(p))
    
    #plt.plot(w_arr, mean_r + np.sqrt(var_r), lw=2)
    #plt.plot(w_arr, mean_r - np.sqrt(var_r), lw=2)
    plt.xlabel('crack opening w [mm]')
    plt.ylabel('force [N]')
    plt.legend(loc='best')
    plt.show()

random_tau()