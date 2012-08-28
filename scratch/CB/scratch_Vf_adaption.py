
from cb_adapt_Vf import CBEMClampedFiberStressVf
from cb_damage_w import CBDamageW, CBDamageCB
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress import \
CBEMClampedFiberStress, CBEMClampedFiberStressSP
from stats.spirrid.spirrid import SPIRRID
from stats.spirrid.rv import RV
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import newton, brentq


if __name__ == '__main__':

    #----------------------------------------------------
    # controled by BC points and without residual stress
    # ---------------------------------------------------

    # filaments
    r = 0.00345
    V_f = 0.05
    tau = RV('uniform', loc=0.2, scale=1.)
    E_f = 200e3
    E_m = 25e3
    l = 1.0
    theta = 0.0
    phi = 1.
    Ll = 70.
    Lr = 70.
    xi = RV('weibull_min', shape=5., scale=.02)
    x = np.linspace(-Ll, Lr, 300)

    ctrl_damage = np.linspace(0.0, 0.999, 50)
    w = np.linspace(0, 1.4, 300)
    n_int = 50

    spirrid_plot = False

    def no_res_stress_CB():
        cb = CBEMClampedFiberStressVf()
        s = SPIRRID(q=cb,
             sampling_type='PGrid',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=n_int)

        damage_func = CBDamageCB()
        s_damage = SPIRRID(q=damage_func,
                    sampling_type='PGrid',
                    tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi,
                               phi=phi, E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
                    n_int=n_int)

        def residuum(w, D):
            s_damage.evars['w'] = np.array([w])
            s_damage.tvars['V_f'] = (1. - D) * V_f
            return s_damage.mu_q_arr - (1. - D)

        for D in ctrl_damage:
            print D
            wD = brentq(residuum, 0.0, 5.0, args=(D,))
            s.q = CBEMClampedFiberStressVf()
            s.evars['w'] = np.array([wD])
            s.tvars['V_f'] = (1.0 - D) * V_f
            mu = s.mu_q_arr
            plt.plot(wD, mu, 'b^')
            if spirrid_plot == True:
                s.evars['w'] = w
                plt.plot(w, s.mu_q_arr, color='blue', lw=0.2)


    #-----------------------------------------------
    # controled by crack opening no residual stress
    # ----------------------------------------------

    def no_res_stress_w():
        cb = CBEMClampedFiberStress()
        s = SPIRRID(q=cb,
             sampling_type='PGrid',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=n_int)

        damage_func = CBDamageW()
        s_damage = SPIRRID(q=damage_func,
                    sampling_type='PGrid',
                    tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi,
                               phi=phi, E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
                    n_int=n_int)

#        for vf in np.linspace(0.01, 0.1, 20):
#            s_damage.tvars['V_f'] = vf
#            s_damage.evars['w'] = w
#            plt.plot(w, s_damage.mu_q_arr)
#        plt.show()

        def residuum(w, D):
            s_damage.evars['w'] = np.array([w])
            s_damage.tvars['V_f'] = (1. - D) * V_f
            return s_damage.mu_q_arr - (1. - D)

        for D in ctrl_damage:
            print D
            wD = brentq(residuum, 0.0, 5.0, args=(D,))
            s.q = CBEMClampedFiberStress()
            s.evars['w'] = np.array([wD])
            s.tvars['V_f'] = (1.0 - D) * V_f
            mu = s.mu_q_arr
            plt.plot(wD, mu, 'ro')
            if spirrid_plot == True:
                s.evars['w'] = w
                plt.plot(w, s.mu_q_arr, color='red', lw=0.2)


    def no_adaption():
        cb = CBEMClampedFiberStress()
        s = SPIRRID(q=cb,
             sampling_type='PGrid',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=100)
        #plt.figure()
        plt.plot(w, s.mu_q_arr, lw=2, color='red', label='w ctrl')

        cb = CBEMClampedFiberStressVf()
        s = SPIRRID(q=cb,
             sampling_type='PGrid',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=100)
        #plt.figure()
        plt.plot(w, s.mu_q_arr, lw=2, color='blue', label='BC ctrl')

no_res_stress_CB()
no_res_stress_w()
no_adaption()
plt.legend(loc='best')
plt.show()
