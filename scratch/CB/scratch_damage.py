from w_omega import WOmega, WOmegaDamage
from u_omega import UOmega, UOmegaDamage
from stats.spirrid.spirrid import SPIRRID
from stats.spirrid.rv import RV
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import brentq


if __name__ == '__main__':

    #----------------------------------------------------
    # controled by BC points and without residual stress
    # ---------------------------------------------------

    # filaments
    r = 0.00345
    V_f = 0.04
    tau = RV('uniform', loc=0.2, scale=1.)
    E_f = 200e3
    E_m = 25e3
    l = 1.0
    theta = 0.0
    phi = 1.
    Ll = 70.
    Lr = 70.
    xi = RV('weibull_min', shape=5., scale=.02)

    ctrl_damage = np.linspace(0.0, .999, 50)
    w = np.linspace(0, 1.2, 100)
    n_int = 50

    spirrid_plot = False

    def w_omega():
        cb = WOmega()
        s = SPIRRID(q=cb,
             sampling_type='PGrid',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=n_int)

        damage_func = WOmegaDamage()
        s_damage = SPIRRID(q=damage_func,
                    sampling_type='PGrid',
                    tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi,
                               phi=phi, E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
                    n_int=n_int)

        def residuum(w, omega):
            s_damage.evars['w'] = np.array([w])
            s_damage.tvars['omega'] = omega
            return s_damage.mu_q_arr - omega

        for omega in ctrl_damage:
            print 'w_omega; damage = ', omega
            wD = brentq(residuum, 0.0, 5.0, args=(omega,))
            s.q = WOmega()
            s.evars['w'] = np.array([wD])
            s.tvars['omega'] = omega
            mu = s.mu_q_arr
            plt.plot(wD, mu, 'ro')
            if spirrid_plot == True:
                s.evars['w'] = w
                plt.plot(w, s.mu_q_arr, color='red', lw=0.2)

    def u_w():
        cb = UOmega()
        s = SPIRRID(q=cb,
             sampling_type='PGrid',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=n_int)

        damage_func = UOmegaDamage()
        s_damage = SPIRRID(q=damage_func,
                    sampling_type='PGrid',
                    tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi,
                               phi=phi, E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
                    n_int=n_int)

        def residuum(omega, w):
            s_damage.evars['w'] = np.array([w])
            s_damage.tvars['omega'] = omega
            return s_damage.mu_q_arr - omega

        for wi in w:
            omega = brentq(residuum, 0.0, 1. - 1e-10, args=(wi,))
            print 'w_u; damage = ', omega
            s.evars['w'] = np.array([wi])
            s.tvars['omega'] = omega
            mu = s.mu_q_arr
            plt.plot(wi, mu, 'ko')

    def u_omega():
        cb = UOmega()
        s = SPIRRID(q=cb,
             sampling_type='PGrid',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=n_int)

        damage_func = UOmegaDamage()
        s_damage = SPIRRID(q=damage_func,
                    sampling_type='PGrid',
                    tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi,
                               phi=phi, E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
                    n_int=n_int)

        def residuum(w, omega):
            s_damage.evars['w'] = np.array([w])
            s_damage.tvars['omega'] = omega
            return s_damage.mu_q_arr - omega

        for omega in ctrl_damage:
            print 'u_omega; damage = ', omega
            wD = brentq(residuum, 0.0, 5.0, args=(omega,))
            s.q = UOmega()
            s.evars['w'] = np.array([wD])
            s.tvars['omega'] = omega
            mu = s.mu_q_arr
            plt.plot(wD, mu, 'b*')
            if spirrid_plot == True:
                s.evars['w'] = w
                plt.plot(w, s.mu_q_arr, color='blue', lw=0.2)

    def no_damage():
        cb = WOmega()
        s = SPIRRID(q=cb,
             sampling_type='PGrid',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr, omega=0.0),
             n_int=100)
        plt.plot(w, s.mu_q_arr, lw=2, color='red', label='w ctrl')

        cb = UOmega()
        s = SPIRRID(q=cb,
             sampling_type='PGrid',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr, omega=0.0),
             n_int=200)
        plt.plot(w, s.mu_q_arr, lw=2, color='blue', label='BC ctrl')

u_omega()
w_omega()
u_w()
no_damage()
plt.legend(loc='best')
plt.show()
