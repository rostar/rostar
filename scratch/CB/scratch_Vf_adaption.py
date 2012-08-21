
from cb_adapt_Vf import CBEMClampedFiberStressVf
from cb_adapt_Vf_w_controled import CBEMClampedFiberStressVfw
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress import CBEMClampedFiberStress
from stats.spirrid.spirrid import SPIRRID
from stats.spirrid.rv import RV
from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':

    #----------------------------------------------------
    # controled by BC points and without residual stress
    # ---------------------------------------------------
    
    # filaments
    r = 0.00345
    V_f = 0.03
    tau = 0.5#RV('uniform', loc=0.05, scale=2.)
    E_f = 200e3
    E_m = 25e3
    l = 0.0
    theta = 0.0
    phi = 1.
    Ll = 100.
    Lr = 100.
    xi = RV('weibull_min', shape=5., scale=.02)

    w = np.linspace(0, .7, 400)

    def no_res_stress_CB():
        cb = CBEMClampedFiberStressVf()
        s = SPIRRID(q=cb,
             sampling_type='LHS',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=300)

        w_adapt = []
        q_adapt = []
        damage = 0.0
        vf = V_f
        for i in range(200):
            s.q = CBEMClampedFiberStressVf()
            s.tvars['V_f'] = vf
            mu = s.mu_q_arr
            damag = s.q.damage[1:]
            damage = np.array(damag)[np.array(damag) > damage][0]
            if damage == 1.0:
                print 'kaputt'
                break
            vf = V_f * (1.0 - damage)
            plt.plot(w[:np.argwhere(damag == damage)[0]],
                     mu[:np.argwhere(damag == damage)[0]] / V_f,
                     color='blue')

    #-----------------------------------------------
    # controled by crack opening no residual stress
    # ----------------------------------------------

    def no_res_stress_w():
        cb = CBEMClampedFiberStressVfw()
        s = SPIRRID(q=cb,
             sampling_type='LHS',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=300)

        w_adapt = []
        q_adapt = []
        damage = 0.0
        vf = V_f
        for i in range(200):
            s.q = CBEMClampedFiberStressVfw()
            s.tvars['V_f'] = vf
            mu = s.mu_q_arr
            damag = s.q.damage[1:]
            damage = np.array(damag)[np.array(damag) > damage][0]
            if damage == 1.0:
                print 'kaputt'
                break
            vf = V_f * (1.0 - damage)
            plt.plot(w[:np.argwhere(damag == damage)[0]],
                     mu[:np.argwhere(damag == damage)[0]] / V_f,
                     color='red')

    def no_adaption():
        cb = CBEMClampedFiberStress()
        s = SPIRRID(q=cb,
             sampling_type='LHS',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=1000)
        plt.plot(w, s.mu_q_arr, lw=2, color='black', label='no adaption')

no_res_stress_CB()
no_res_stress_w()
no_adaption()
plt.legend()
plt.show()