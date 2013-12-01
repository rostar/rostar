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
from math import pi
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import ContinuousFibers
from stats.pdistrib.weibull_fibers_composite_distr import WeibullFibers, fibers_MC
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx_view import CompositeCrackBridge, CompositeCrackBridgeView


reinf = ContinuousFibers(r=0.0035,
                      tau=0.1,
                      V_f=0.05,
                      E_f=180e3,
                      xi=fibers_MC(m=4.0, sV0=0.003),
                      label='carbon',
                      n_int=10)

model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf],
                             Ll=100.,
                             Lr=100.,
                             )

ccb_view = CompositeCrackBridgeView(model=model)

def BC_effect(w_arr):
    reinf.xi = 100.
    model.Ll = 30.
    model.Lr = 50.
    sigma_c_arr = ccb_view.sigma_c_arr(w_arr)
    plt.plot(w_arr, sigma_c_arr, lw=2, color='black', label='w-sigma')

# TODO: check energy for combined reinf
# energy(np.linspace(.0, .15, 100))
#    sigma_c = np.linspace(1., 7., 7)
# profile(0.031)
w = np.linspace(0.0, 1.6, 200)
BC_effect(w)
# energy(w)
# bundle at 20 mm
# sigma_bundle = 70e3*w/20.*np.exp(-(w/20./0.03)**5.)
# plt.plot(w,sigma_bundle)
# plt.plot(ccb_view.sigma_c_max[1], ccb_view.sigma_c_max[0], 'bo')
# sigma_f(np.linspace(.0, .16, 50))
plt.legend(loc='best')
plt.show()
