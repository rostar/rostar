from w_omega import WOmega, WOmegaDamage
from u_omega import UOmega, UOmegaDamage
from u_analytical import UOmegaAnalyt
from w_analyt_iter import WAnalytIter
from stats.spirrid.spirrid import SPIRRID
from stats.spirrid.rv import RV
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import brentq
from math import e

if __name__ == '__main__':

    #----------------------------------------------------
    # controled by BC points and without residual stress
    # ---------------------------------------------------

    # filaments
    r = 0.00345
    V_f = 0.04
    tau = 0.5#RV('uniform', loc=0.2, scale=1.)
    E_f = 200e3
    E_m = 25e3
    l = 0.0
    theta = 0.0
    phi = 1.
    Ll = 70.
    Lr = 70.
    xi = RV('weibull_min', shape=5., scale=.02)

    ctrl_damage = np.linspace(0.0, .99999, 500)
    w = np.linspace(0, .57, 100)
    n_int = 500

    spirrid_plot = False

    def w_omega_spirrid():
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
            wD = brentq(residuum, 0.0, 5.0, args=(omega,))
            s.q = WOmega()
            s.evars['w'] = np.array([wD])
            s.tvars['omega'] = omega
            mu = s.mu_q_arr
            plt.plot(wD, mu, 'ro')
            if spirrid_plot == True:
                s.evars['w'] = w
                plt.plot(w, s.mu_q_arr, color='red', lw=0.2)

    def u_u():
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
            s.evars['w'] = np.array([wi])
            s.tvars['omega'] = omega
            mu = s.mu_q_arr
            plt.plot(wi, mu, 'ko')

    def u_omega_spirrid():
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
        #plt.plot(w, s.mu_q_arr, lw=2, color='blue', label='BC ctrl')

    def u_analytical_iterative():
        a = UOmegaAnalyt()
        w, q = a(ctrl_damage, tau, l, E_f, E_m,
                 theta, xi, phi, Ll, Lr, V_f, r)
        plt.plot(w, q, 'r*')

    def w_analytical_iterative():
        a = WAnalytIter()
        w, q = a.eval_w_q(tau, l, E_f, E_m, theta, xi,
                          phi, Ll, Lr, V_f, r, ctrl_damage)
        plt.plot(w, q, 'r*')

    # statistical homogenization of damage function
    def w_analytical():

        def crackbridge(w, tau, E_f, E_m, V_f, r, omega):
            Kf = E_f * V_f * (1 - omega)
            Km = E_m * (1 - V_f)
            Kc = Kf + Km
            T = 2. * tau * V_f * (1. - omega) / r
            c = np.sqrt(Kc * T / Km / Kf)
            return c * np.sqrt(w) * (1 - omega)

        def w_omega(tau, E_f, E_m, V_f, r, omega, m, s):
            Kf = E_f * V_f * (1 - omega)
            Km = E_m * (1 - V_f)
            Kc = Kf + Km
            T = 2. * tau * V_f * (1. - omega) / r
            return (-np.log(1. - omega)) ** (2. / m) \
                    * s ** 2 * Km * Kf / Kc / T
        w_lst = []
        for omega in ctrl_damage:
            w_lst.append(w_omega(tau, E_f, E_m, V_f, r, omega, 5.0, 0.02))
        epsf = crackbridge(np.array(w_lst), tau, E_f, E_m, V_f, r, ctrl_damage)
        plt.plot(np.array(w_lst), epsf * E_f,
                 color='blue')

    def u_analytical():

        def crackbridge(u, tau, E_f, E_m, V_f, r, omega, L):
            Kf = E_f * V_f * (1 - omega)
            Km = E_m * (1 - V_f)
            Kc = Kf + Km
            T = 2. * tau / r / E_f
            c = Kf * T * L
            a = Kc / 2. / Km ** 2
            b = 4. * Km ** 2 * u * T
            eps = a * (np.sqrt(c ** 2. + b) - c)
            return eps * (1. - omega)

        def u_omega(tau, E_f, E_m, V_f, r, omega, m, s, L):
            Kf = E_f * V_f * (1. - omega)
            Km = E_m * (1 - V_f)
            Kc = Kf + Km
            T = 2. * tau / r / E_f
            u = (1. / Kc**2 / T)*(((-np.log(1. - omega))**(2./m) * s**2 * Km**2
                + (-np.log(1. - omega))**(1./m) * s * Kc * Kf * T * L))
            return u

        u_lst = []
        for omega in ctrl_damage:
            u_lst.append(u_omega(tau, E_f, E_m, V_f, r, omega, 5.0, 0.02, Ll+Lr))
        epsf = crackbridge(np.array(u_lst), tau, E_f, E_m, V_f, r, ctrl_damage, Ll+Lr)
#        plt.plot(np.array(u_lst), ctrl_damage,
#                 color='brown', label='u-damage')
        plt.plot(np.array(u_lst), epsf * E_f,
                 color='red')

#w_omega_spirrid()
#w_analytical_iterative()
#w_analytical()
#u_analytical()
#no_damage()
#u_u()
#u_omega_spirrid()
#u_analytical_iterative()
for vf in np.linspace(0.001, 0.1, 10):
    V_f = vf
    u_analytical()
    w_analytical()
plt.legend(loc='best')
plt.show()
 