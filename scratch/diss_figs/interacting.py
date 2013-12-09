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
from stats.pdistrib.weibull_fibers_composite_distr import fibers_MC, fibers_CB_rigid, fibers_dry, fibers_CB_elast
from math import pi
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import ContinuousFibers
from stats.pdistrib.weibull_fibers_composite_distr import WeibullFibers, fibers_MC
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx_view import CompositeCrackBridge, CompositeCrackBridgeView
import copy

reinf = ContinuousFibers(r=0.0035,
                      tau=0.1,
                      V_f=0.05,
                      E_f=200e3,
                      xi=fibers_MC(m=5.0, sV0=10.006),
                      label='carbon',
                      n_int=200)

model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf],
                             Ll=100.,
                             Lr=100.,
                             )

ccb_view = CompositeCrackBridgeView(model=model)

def BC_effect(w_arr):
    model.Ll = 1e10
    model.Lr = 1e10
    sigma_c_arr = ccb_view.sigma_c_arr(w_arr)
    plt.plot(w_arr, sigma_c_arr, lw=2, color='black', label='w-sigma')
    model.Ll = 10.
    model.Lr = 1e10
    sigma_c_arr = ccb_view.sigma_c_arr(w_arr)
    plt.plot(w_arr, sigma_c_arr, lw=2, color='black', label='w-sigma')
    model.Ll = 10.
    model.Lr = 30.
    sigma_c_arr = ccb_view.sigma_c_arr(w_arr)
    plt.plot(w_arr, sigma_c_arr, lw=2, color='black', label='w-sigma')

def FBM_limit(w_arr):
    sV0 = 0.006
    m = 5.0
    model.Ll = 50.
    model.Lr = 50.
    model.E_m = 25e3
    reinf.tau = 0.2
    reinf.xi = fibers_CB_elast(m=m, sV0=sV0)
    sigma_c_arr = ccb_view.sigma_c_arr(w_arr)
    plt.plot(w_arr / 100., sigma_c_arr / 0.05, lw=2, label='w-sigma1')
    reinf1 = copy.copy(reinf)
    reinf1.tau = 0.1
    model.reinforcement_lst = [reinf1]
    sigma_c_arr = ccb_view.sigma_c_arr(w_arr)
    plt.plot(w_arr / 100., sigma_c_arr / 0.05, lw=2, label='w-sigma2')
    reinf2 = copy.copy(reinf)
    reinf2.tau = 0.05
    model.reinforcement_lst = [reinf2]
    sigma_c_arr = ccb_view.sigma_c_arr(w_arr)
    plt.plot(w_arr / 100., sigma_c_arr / 0.05, lw=2, label='w-sigma2')
    reinf3 = copy.copy(reinf)
    reinf3.tau = 0.005
    model.reinforcement_lst = [reinf3]
    sigma_c_arr = ccb_view.sigma_c_arr(w_arr)
    plt.plot(w_arr / 100., sigma_c_arr / 0.05, lw=2, label='w-sigma2')

    e = w_arr / (100.)
    F = lambda x: 1 - np.exp(-100. * reinf.r ** 2 * pi * (x / sV0) ** m)
    sigma = e * 200e3 * (1. - F(e))
    plt.plot(e, sigma, lw=4, color='red', label='FBM')
    
def CBR_limit(w_arr):
    model.Ll = 1e10
    model.Lr = 1e10
    model.E_m = 5e3
    sigma_c_arr = ccb_view.sigma_c_arr(w_arr)
    plt.plot(w_arr, sigma_c_arr, lw=2, label='w-sigma1')
    model.E_m = 20e3
    sigma_c_arr = ccb_view.sigma_c_arr(w_arr)
    plt.plot(w_arr, sigma_c_arr, lw=2, label='w-sigma2')
    model.E_m = 100e3
    sigma_c_arr = ccb_view.sigma_c_arr(w_arr)
    plt.plot(w_arr, sigma_c_arr, lw=2, label='w-sigma2')
    model.E_m = 25e10
    sigma_c_arr = ccb_view.sigma_c_arr(w_arr)
    plt.plot(w_arr, sigma_c_arr, lw=3, color='black', label='rigid')
    
def CBe_limit(w_arr):
    model.Ll = 1e10
    model.Lr = 1e10
    model.E_m = 25e3
    reinf.xi = 100.
    sigma_c_arr = ccb_view.sigma_c_arr(w_arr)
    plt.plot(w_arr, sigma_c_arr / reinf.E_f / reinf.V_f, lw=2, label='numerical')
    def analytical(w_arr):
        T = 2 * reinf.tau / reinf.r
        return np.sqrt((T * model.E_c * w_arr) / (reinf.E_f * model.E_m * (1 - reinf.V_f)))
    plt.plot(w_arr, analytical(w_arr), lw=4, ls='dashed', label='analytical')

BC_effect(np.linspace(0.0, .6, 200))
# FBM_limit(np.linspace(0.0, 3.0, 200))
# CBR_limit(np.linspace(0.0, 4.0, 200))
# CBe_limit(np.linspace(0.0, 4.0, 200))

plt.legend(loc='best')
plt.show()
