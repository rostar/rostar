'''
Created on 15. 7. 2014

@author: admin
'''
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx_view import CompositeCrackBridge
from matplotlib import pyplot as plt
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement_old import Reinforcement, ContinuousFibers
from stats.pdistrib.weibull_fibers_composite_distr import fibers_MC, WeibullFibers
import numpy as np
from spirrid.rv import RV

def CB_response():
    reinf = ContinuousFibers(r=0.0035,
                          tau=RV('uniform', loc=1., scale=3.0),
                          V_f=0.1,
                          E_f=200e3,
                          xi=RV('weibull_min', loc=0., shape=1000., scale=.01),
                          label='carbon',
                          n_int=30)
    
    ccb = CompositeCrackBridge(E_m=25e13,
                                 reinforcement_lst=[reinf],
                                 Ll=2000.,
                                 Lr=2000.,
                                 w=.3)
    
    
    w_arr = np.linspace(0.0,.05,500)
    sigma_lst = []
    for w in w_arr:
        ccb.w = w
        damage = ccb.damage
        eps = ccb._epsf0_arr * (1. - damage)
        sigma_lst.append(eps)
    
    mean = np.array(sigma_lst)
    mean = np.mean(mean, 1)
    
    plt.plot(w_arr, sigma_lst)
    plt.plot(w_arr, mean, lw=3, ls='dashed', color='red')
    plt.show()

def profiles():
    reinf = ContinuousFibers(r=0.0035,
                      tau=RV('uniform', loc=1., scale=3.0),
                      V_f=0.1,
                      E_f=200e3,
                      xi=RV('weibull_min', loc=0., shape=1000., scale=.01),
                      label='carbon',
                      n_int=30)

    ccb = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf],
                                 Ll=1.,
                                 Lr=1.,
                                 w=.0056)

#     ccb = CompositeCrackBridge(E_m=25e13,
#                                  reinforcement_lst=[reinf],
#                                  Ll=2.,
#                                  Lr=2.,
#                                  w=.0088)

    ccb.damage
    for i, depsf in enumerate(ccb.sorted_depsf):
        epsf_x = np.maximum(ccb._epsf0_arr[i] - depsf * np.abs(ccb._x_arr), ccb._epsm_arr)
        if i == 0:
            plt.plot(ccb._x_arr, epsf_x, color='black', label='fibers')
        else:
            plt.plot(ccb._x_arr, epsf_x, color='black')
    plt.plot(ccb._x_arr, ccb._epsm_arr, lw=2, color='blue', label='matrix')
    plt.legend(loc='best')
    plt.ylabel('matrix and fiber strain [-]')
    plt.ylabel('long. position [mm]')
    plt.show()


#profiles()
CB_response()