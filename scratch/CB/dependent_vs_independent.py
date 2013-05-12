'''
Created on Oct 5, 2012

compares the dependent fibers model and SPIRRID implementation

conclusion - the mean fiber and matrix strains don't differ much for infinite fiber strength,
but there are huge differences in the stress state of individual 'fibers'
so that if breakage of fibers is included, the differences in damage are significant.

@author: rostar
'''

from dependent_fibers.depend_CB_model import CompositeCrackBridge
from dependent_fibers.reinforcement import Reinforcement
from dependent_fibers.depend_CB_postprocessor import CompositeCrackBridgePostprocessor
from spirrid.rv import RV
from matplotlib import pyplot as plt
import numpy as np
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress import \
    CBEMClampedFiberStress, CBEMClampedFiberStressSP
from spirrid.spirrid import SPIRRID


if __name__ == '__main__':

    r = 0.00345#RV('uniform', loc=0.002, scale=0.002)
    V_f = 0.01
    tau = RV('weibull_min', shape=1.5, scale=.2)
    E_f = 200e3
    E_m = 25e3
    l = 0.0#RV('uniform', scale=10., loc=2.)
    theta = 0.0
    xi = RV('weibull_min', scale=0.02, shape=5.0)
    phi = 1.
    Ll = 50.
    Lr = 40.
    n_int = 100

    spirrid_plot = False

    def profile(w):
        reinf1 = Reinforcement(r=r, tau=tau, V_f=V_f, E_f=E_f, xi=xi, n_int=n_int)
        ccb = CompositeCrackBridge(E_m=E_m, reinforcement_lst=[reinf1], Ll=Ll, Lr=Lr, w=w)
        ccb_view = CompositeCrackBridgePostprocessor(model=ccb)
        plt.plot(ccb_view.x_arr, ccb_view.epsm_arr, color='red')
        plt.plot(ccb_view.x_arr, ccb_view.mu_epsf_arr, color='red', label='ODE')

#         cb_prof = CBEMClampedFiberStressSP()
#         s = SPIRRID(q=cb_prof,
#                     sampling_type='LHS',
#                     eps_vars=dict(w=np.array([w]),
#                                x=np.linspace(-Ll, Lr, n_int**2)),
#                     theta_vars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
#                                E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
#                     n_int=n_int)
#     
#         plt.plot(np.linspace(-Ll,Lr,n_int**2), s.mu_q_arr.flatten()/E_f, color='blue',
#                  label='SPIRRID')
#         epsy_arr = s.mu_q_arr.flatten() / E_f
#         epsm_arr = (np.max(epsy_arr) - epsy_arr) * E_f * V_f / (1.-V_f) / E_m
#         plt.plot(np.linspace(-Ll, Lr, n_int**2), epsm_arr, color='blue')
#         plt.legend(loc='best')
#         print 'SPIRRID w = ', np.trapz(epsy_arr-epsm_arr, np.linspace(-Ll,Lr,n_int**2))
#         print 'DOE w = ', ccb_view.w_evaluated
        plt.show()

    def eps_w(w_arr):
        reinf1 = Reinforcement(r=r, tau=tau, V_f=V_f, E_f=E_f, xi=xi, n_int=n_int)
        ccb = CompositeCrackBridge(E_m=E_m, reinforcement_lst=[reinf1], Ll=Ll, Lr=Lr)
        ccb_view = CompositeCrackBridgePostprocessor(model=ccb)
        sigma = []
        for w in w_arr:
            ccb.w = w
            sigma.append(ccb_view.sigma_c)
        plt.plot(w_arr, sigma, color='red', label='dependent fibers model')
# 
#         cb_prof = CBEMClampedFiberStress()
#         s = SPIRRID(q=cb_prof,
#                     sampling_type='LHS',
#                     eps_vars=dict(w=w_arr),
#                     theta_vars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
#                                E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
#                     n_int=n_int)
# 
#         plt.plot(w_arr, s.mu_q_arr * V_f, label='SPIRRID')
#         plt.legend(loc='best')
        plt.show()

    profile(.02)
    #eps_w(np.linspace(0., .8, 200))