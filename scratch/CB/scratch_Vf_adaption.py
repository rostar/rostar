
from cb_adapt_Vf import CBEMClampedFiberStressVf
from cb_adapt_Vf_w_controled import CBEMClampedFiberStressVfw
from cb_damage_w import CBDamageW
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
    V_f = 0.01
    tau = RV('uniform', loc=0.2, scale=1.)
    E_f = 200e3
    E_m = 25e10
    l = 1.0
    theta = 0.0
    phi = 1.
    Ll = 100.
    Lr = 100.
    xi = RV('weibull_min', shape=5., scale=.02)

    ctrl_damage = np.linspace(0.0, 0.99, 100)
    w = np.linspace(0, 1., 300)
    n_int = 20

    def no_res_stress_CB():
        cb = CBEMClampedFiberStressVf()
        s = SPIRRID(q=cb,
             sampling_type='LHS',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=n_int)

        vf = V_f
        for D in ctrl_damage:
            s.q = CBEMClampedFiberStressVf()
            s.tvars['V_f'] = vf
            s.tvars['V_f'] *= 1.0000001
            mu = s.mu_q_arr
            damage_arr = s.q.damage[1:]
            damage = np.array(damage_arr)[np.array(damage_arr) > D][0]
            vf = V_f * (1.0 - damage)
            plt.plot(w[:np.argwhere(damage_arr == damage)[0]],
                     mu[:np.argwhere(damage_arr == damage)[0]] / V_f,
                     color='blue')

    #-----------------------------------------------
    # controled by crack opening no residual stress
    # ----------------------------------------------

    def no_res_stress_w():
        cb = CBEMClampedFiberStressVfw()
        s = SPIRRID(q=cb,
             sampling_type='TGrid',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=n_int)

        f = s.mu_q_arr
        print s.q.damage
        plt.plot(w, s.q.damage[1:], color = 'blue')
#        vf = V_f
#        for D in ctrl_damage:
#            s.q = CBEMClampedFiberStressVfw()
#            s.tvars['V_f'] = vf
#            print vf
#            s.tvars['V_f'] *= 1.0000001
#            mu = s.mu_q_arr
#            damage_arr = s.q.damage[1:]
#            damage = np.array(damage_arr)[np.array(damage_arr) > D][0]
#            vf = V_f * (1.0 - damage)
#            plt.plot(w[:np.argwhere(damage_arr == damage)[0]],
#                     mu[:np.argwhere(damage_arr == damage)[0]] / V_f,
#                     color='red')

    def damage():
        cb = CBDamageW()
        s = SPIRRID(q=cb,
             sampling_type='LHS',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=n_int)

        plt.plot(w, 1. - s.mu_q_arr/400., color = 'red')

    def no_adaption():
        cb = CBEMClampedFiberStress()
        s = SPIRRID(q=cb,
             sampling_type='LHS',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=200)
        plt.plot(w, s.mu_q_arr, lw=2, color='black', label='no adaption')

#no_res_stress_CB()
no_res_stress_w()
damage()
#no_adaption()
plt.legend(loc='best')
plt.show()
