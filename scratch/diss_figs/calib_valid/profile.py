'''
Created on Sep 20, 2012

The CompositeCrackBridge class has a method for evaluating fibers and matrix
strain in the vicinity of a crack bridge.
Fiber diameter and bond coefficient can be set as random variables.
Reinforcement types can be combined by creating a list of Reinforcement
instances and defining it as the reinforcement_lst Trait in the
CompositeCrackBridge class.
The evaluation is array based.

@author: rostar
'''

from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import ContinuousFibers
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx_view import CompositeCrackBridgeView
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx import CompositeCrackBridge

if __name__ == '__main__':
    from stats.pdistrib.weibull_fibers_composite_distr import fibers_MC
    import matplotlib.pyplot as plt    
    from spirrid.rv import RV
    from math import pi
    from scipy.special import gamma
    import numpy as np
    
    tau_shape = 0.0650
    tau_loc = 1410e-6
    xi_shape = 7.1
    xi_scale = 0.0069
    #xi_scale = 1578. / (182e3 * (pi * 3.5e-3 **2 * 500. * e)**(-1./xi_shape))
    mu_tau = 0.075
    tau_scale = (mu_tau - tau_loc)/tau_shape

    reinf_cont = ContinuousFibers(r=3.5e-3,
                          tau=RV('gamma', loc=tau_loc, scale=tau_scale, shape=tau_shape),
                          V_f=0.01,
                          E_f=181e3,
                          xi=fibers_MC(m=xi_shape, sV0=xi_scale),
                          label='carbon',
                          n_int=500)

    model = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf_cont],
                                 Ll=50.,
                                 Lr=200.,
                                 )

    ccb_view = CompositeCrackBridgeView(model=model)

    def profile(w):
        ccb_view.model.w = w
        plt.plot(ccb_view.x_arr, ccb_view.epsm_arr, lw=2, label='Ll=' + str(ccb_view.model.Ll))
        plt.plot(ccb_view.x_arr, ccb_view.epsf_arr, color='red', lw=2)
        plt.xlabel('position [mm]')
        plt.ylabel('strain')

    def sigma_c_w(w_arr):
        sigma_c_arr, u_arr, damage_arr = ccb_view.sigma_c_arr(w_arr, damage=False, u=True)
        plt.plot(w_arr, sigma_c_arr/reinf_cont.V_f, lw=2, color='black', label='w-sigma')
        #plt.plot(w_arr, damage_arr, lw=2, color='red', label='damage')
        #plt.plot(u_arr, sigma_c_arr, lw=2, label='u-sigma')
        #plt.plot(ccb_view.sigma_c_max[1], ccb_view.sigma_c_max[0], 'bo')
        plt.xlabel('w,u [mm]')
        plt.ylabel('$\sigma_c$ [MPa]')
        plt.legend(loc='best')

    w_arr = np.linspace(0.0,2.0, 100)
    #sigma_c_w(w_arr)
    profile(0.05)

    plt.legend(loc='best')
    plt.show()
      
#     for i, depsf in enumerate(ccb.cont_fibers.sorted_depsf):
#         epsf_x = np.maximum(ccb._epsf0_arr[0][i] - depsf * np.abs(ccb._x_arr), ccb._epsm_arr)
# #             if i == 0:
# #                 plt.plot(ccb._x_arr, epsf_x, color='blue', label='fibers')
# #             else:
#         plt.plot(ccb._x_arr, epsf_x, color='black', alpha=1-0.5*ccb.cont_fibers.damage[i])
#     
#     
#     epsf0_combined = np.hstack((ccb._epsf0_arr[0], ccb._epsf0_arr[1]))
#     plt.plot(np.zeros_like(epsf0_combined), epsf0_combined, 'ro', label='maximum')
#     plt.plot(ccb._x_arr, ccb._epsm_arr, lw=2, color='blue', label='matrix')
#     plt.plot(ccb._x_arr, ccb._epsf_arr, lw=2, color='red', label='mean fiber')
#     plt.legend(loc='best')
#     plt.ylabel('matrix and fiber strain [-]')
# #     plt.ylabel('long. position [mm]')
#     plt.show()
