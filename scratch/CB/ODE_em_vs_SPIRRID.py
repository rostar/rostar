'''
Created on Oct 5, 2012

compares ODE_em and SPIRRID implementation

@author: rostar
'''

from composite_crack_bridge import CompositeCrackBridge
from composite_crack_bridge import Reinforcement
from stats.spirrid.rv import RV
from matplotlib import pyplot as plt
import numpy as np
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress import \
    CBEMClampedFiberStress, CBEMClampedFiberStressSP
from stats.spirrid.spirrid import SPIRRID


if __name__ == '__main__':

    reinf1 = Reinforcement(r=0.00345,#RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=.5, scale=1.),
                          V_f=0.5,
                          E_f=200e3,
                          n_int=100)

    ccb = CompositeCrackBridge(E_m=25e3,
                               w=0.4,
                               reinforcement_lst=[reinf1],
                               Ll=5.,
                               Lr=40.)

    def profile(w):
        ccb.w = w
        plt.plot(ccb.profile[0], ccb.profile[1], label='matrix1')
        plt.plot(ccb.profile[0], ccb.profile[2], label='yarn')
        plt.legend(loc='best')
        #plt.show()

    def eps_w(w_arr):
        eps = []
        for w in w_arr:
            #print 'w_ctrl=', w
            ccb.w = w
            eps.append(np.max(ccb.profile[2]))
        plt.plot(w_arr, eps, label='ld')
        #plt.legend()
        #plt.show()

    profile(4.)
    #eps_w(np.linspace(0., 4., 50))


    # filaments
    r = 0.00345
    V_f = 0.5
    tau = RV('uniform', loc=.5, scale=1.)
    E_f = 200e3
    E_m = 25e3
    l = 0.0#RV('uniform', scale=10., loc=2.)
    theta = 0.0
    xi = 200.#RV('weibull_min', scale=0.02, shape=5)
    phi = 1.
    Ll = 5.
    Lr = 40.

    w = np.linspace(0, 4., 50)

    cb_emtrx = CBEMClampedFiberStress()
    s = SPIRRID(q=cb_emtrx,
         sampling_type='PGrid',
         evars=dict(w=w),
         tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                    E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
         n_int=100)


    #plt.plot(w, s.mu_q_arr/200e3, lw=2, color='red',
    #         label='SPIRRID')
    #plt.show()

    cb_prof = CBEMClampedFiberStressSP()
    s = SPIRRID(q=cb_prof,
     sampling_type='PGrid',
     evars=dict(w=np.array([4.]),
                x=np.linspace(-Ll,Lr,200)),
     tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
     n_int=100)
    
    plt.plot(np.linspace(-Ll,Lr,200), s.mu_q_arr.flatten()/200e3, lw=2, color='red',
             label='SPIRRID')
    plt.show()   