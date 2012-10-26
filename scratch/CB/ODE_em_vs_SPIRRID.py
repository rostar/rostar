'''
Created on Oct 5, 2012

compares ODE_em and SPIRRID implementation

@author: rostar
'''

from dependent_fibers.composite_crack_bridge import CompositeCrackBridge
from dependent_fibers.composite_crack_bridge import Reinforcement
from stats.spirrid.rv import RV
from matplotlib import pyplot as plt
import numpy as np
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress import \
    CBEMClampedFiberStress, CBEMClampedFiberStressSP
from stats.spirrid.spirrid import SPIRRID


if __name__ == '__main__':

    r = 0.00345#RV('uniform', loc=0.002, scale=0.002)
    V_f = 0.7
    tau = RV('weibull_min', shape=3., scale=.3)
    E_f = 200e3
    E_m = 25e3
    l = 0.0#RV('uniform', scale=10., loc=2.)
    theta = 0.0
    xi = 200.#RV('weibull_min', scale=0.02, shape=5)
    phi = 1.
    Ll = 5.
    Lr = 40.
    n_int = 20

    def profile(w):
        reinf1 = Reinforcement(r=r, tau=tau, V_f=V_f, E_f=E_f, xi=xi, n_int=n_int)
        ccb = CompositeCrackBridge(E_m=E_m, reinforcement_lst=[reinf1], Ll=Ll, Lr=Lr, w=w)
        plt.plot(ccb.x_arr, ccb.em_arr, color='red')
        plt.plot(ccb.x_arr, ccb.ey_arr, color='red', label='ODE')

        cb_prof = CBEMClampedFiberStressSP()
        s = SPIRRID(q=cb_prof,
                    sampling_type='PGrid',
                    evars=dict(w=np.array([w]),
                               x=np.linspace(-Ll, Lr, n_int**2)),
                    tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                               E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
                    n_int=n_int)
    
        plt.plot(np.linspace(-Ll,Lr,n_int**2), s.mu_q_arr.flatten()/E_f, color='blue',
                 label='SPIRRID')
        epsy_arr = s.mu_q_arr.flatten() / E_f
        epsm_arr = (np.max(epsy_arr) - epsy_arr) * E_f * V_f / (1.-V_f) / E_m
        plt.plot(np.linspace(-Ll, Lr, n_int**2), epsm_arr, color='blue')
        plt.legend(loc='best')
        print 'SPIRRID w = ', np.trapz(epsy_arr-epsm_arr, np.linspace(-Ll,Lr,n_int**2))
        print 'DOE w = ', ccb.w_evaluated
        plt.show()

    def eps_w(w_arr):
        reinf1 = Reinforcement(r=r, tau=tau, V_f=V_f, E_f=E_f, xi=xi, n_int=n_int)
        ccb = CompositeCrackBridge(E_m=E_m, reinforcement_lst=[reinf1], Ll=Ll, Lr=Lr)
        eps = []
        for w in w_arr:
            ccb.w = w
            eps.append(ccb.max_norm_stress)
        plt.plot(w_arr, eps, color='red', label='ODE')

        cb_emtrx = CBEMClampedFiberStress()
        s = SPIRRID(q=cb_emtrx,
                sampling_type='PGrid',
                evars=dict(w=w_arr),
                tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                           E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
         n_int=n_int)

        plt.plot(w_arr, s.mu_q_arr, color='blue', label='SPIRRID')
        plt.legend(loc='best')
        plt.show()

    profile(.2)
    eps_w(np.linspace(0, .5, 50))
