'''
Created on Sep 30, 2011

The example implements micromechanical responses of a crack opening
in a composite material where a single discrete crack is bridged
by:
a) single homogeneous filament
b) yarn with random filament properties
The diagram shows a comparison between the formulation with discrete
filament strength and length dependent strength with residual filament
stress due to pullout of the broken filament from the matrix.
BC - fixed displacement at both ends (Ll and Lr) of the filament.
@author: rostar
'''

from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress_residual import \
    CBEMClampedFiberStressResidualSP
from iter_cb import CBEMClampedFiberStressResidualSPIter
from stats.spirrid.spirrid import SPIRRID
from stats.spirrid.rv import RV
from matplotlib import pyplot as plt
import numpy as np
from iteration_CB import CBIter
from etsproxy.traits.api import HasTraits, cached_property, Float, Property, Bool
from scipy.optimize import fsolve
from scipy.stats import weibull_min, uniform
import copy


class Profile(HasTraits):

    r = Float(0.00345)
    mtau = Float(5.)
    stau = Float(0.5)
    V_f = Float(0.04)
    E_m = Float(25e3)
    E_f = Float(200e3)
    w = Float(0.5)
    fwd_Euler = Bool(False)
    midpoint_method = Bool(True)

    Km = Property(depends_on='E_m, V_f')

    @cached_property
    def _get_Km(self):
        return self.E_m * (1. - self.V_f)

    Kf = Property(depends_on='E_f, V_f')

    @cached_property
    def _get_Kf(self):
        return self.E_f * self.V_f

    Kc = Property(depends_on='E_f, E_m, V_f')

    @cached_property
    def _get_Kc(self):
        return self.Km + self.Kf

    def tau_distr(self):
        return weibull_min(100.0, scale=1.0)

    def T_epsm(self, tau):
        return 2. * tau * self.V_f / self.r / self.E_m / (1. - self.V_f)

    def T_epsf(self, tau):
        return 2. * tau / self.r / self.E_f

    def deps_m(self, T):
        tau_max = T * self.r * (1 - self.V_f) * self.E_m / 2. / self.V_f
        tau_min = self.tau_distr().ppf(0.0001)
        range = self.tau_distr().ppf(0.9999) - self.tau_distr().ppf(0.0001)
        if tau_max > self.tau_distr().ppf(0.9999):
            tau_max = self.tau_distr().ppf(0.9999) + 0.01 * range
        tau_arr = np.linspace(tau_min - 0.01 * range, tau_max, 500)
        return self.T_epsm(np.trapz(tau_arr * self.tau_distr().pdf(tau_arr), tau_arr))

    def residuum(self, Tepsm, eps_m_x, x_x):
        tau = Tepsm * self.r * (1 - self.V_f) * self.E_m / 2. / self.V_f
        res = eps_m_x[-1] * x_x[-1] + self.T_epsf(tau) * x_x[-1] ** 2 / 2. - np.trapz(eps_m_x, x_x)
        return self.w / 2. - res

    def eps_m(self, x_arr):
        eps_m_x = [0.0]
        deps_m_x = [self.T_epsm(self.tau_distr().mean())]
        T_epsm_x = []
        x_x = [0.0]
        for xi in x_arr[1:]:
            if self.fwd_Euler == True:
                eps_m_x.append(eps_m_x[-1] + deps_m_x[-1] * (xi - x_x[-1]))
            elif self.midpoint_method == True:
                h = (xi - x_x[-1])
                eps_midp = (eps_m_x[-1] + h/2. * deps_m_x[-1])
                eps_lst = copy.deepcopy(eps_m_x)
                eps_lst.append(eps_midp)
                x_lst = copy.deepcopy(x_x)
                x_lst.append(x_x[-1]+h/2.)
                T_midp = fsolve(self.residuum, 0.0, args=(eps_lst, x_lst))
                deps_midp = self.deps_m(T_midp)
                eps_m_x.append(eps_m_x[-1] + h * deps_midp)
            x_x.append(xi)
            T_epsm_x.append(fsolve(self.residuum, 0.0, args=(eps_m_x, x_x)))
            deps_m_x.append(self.deps_m(T_epsm_x[-1]))
        return eps_m_x

if __name__ == '__main__':
    # filaments
    r = 0.00345
    V_f = 0.04
    tau = RV('weibull_min', shape=100.0, scale=1.0)
    E_f = 200e3
    E_m = 25e3
    l = 0.0
    theta = 0.0
    phi = 1.
    Ll = 40.
    Lr = 40.
    s0 = 100.0205
    m = 5.0
    Pf = 0.5#RV('uniform', loc=0., scale=1.0)

    w = 0.5
    x = np.linspace(-30, 30, 1000)

    cb = CBEMClampedFiberStressResidualSP()
    s = SPIRRID(q=cb,
         sampling_type = 'PGrid',
         evars = dict(x = x),
         tvars = dict(w = w, tau = tau, l = l, E_f = E_f, theta = theta, m = m, phi = phi,
                    E_m = E_m, r = r, V_f = V_f, Ll = Ll, Lr = Lr, s0 = s0, Pf = Pf),
         n_int = 100)

    n_int = s.n_int
    pi = np.linspace(0.5 / n_int, 1. - 0.5 / n_int, n_int)

    def iter():
        load = np.ones(len(x)) * np.max(s.mu_q_arr) * V_f
        sigma_m = (load - s.mu_q_arr * V_f) / (1. - V_f)
        # initial spirrid guess
        eps_f = s.mu_q_arr / E_f
        eps_m = sigma_m / E_m
        plt.plot(x, eps_f, lw=1, color='black',
                 ls = 'solid', label = 'initial SPIRRID guess')
        plt.plot(x, eps_m, lw=1, color='black',
                 ls='solid')

        w_means = []
        w_stdevs = []
        w_global = []

        cb_iter = CBEMClampedFiberStressResidualSPIter()
        w_err = []
        for i in pi:
            t = tau.ppf(i)
            eps_ff = cb_iter(w, x, t, l, E_f, E_m, theta, Pf,
                     phi, Ll, Lr, V_f, r, s0, m) / E_f
            eps_ff = np.maximum(eps_ff, eps_m)
            w_err.append(np.trapz(eps_ff - eps_m, x))
        w_err = np.array(w_err)
        w_means.append(w_err.mean())
        w_stdevs.append(w_err.std())
        w_global.append(np.trapz(eps_f - eps_m, x))

        iters = 1

        for j in range(iters):
            cbi = CBIter(eps_m=eps_m, x_arr=x, q_init=cb.q)
            si = SPIRRID(q=cbi,
                         sampling_type='PGrid',
                         evars = dict(x = x),
                         tvars = dict(w = w, tau = tau, l = l, E_f = E_f, theta=theta, m=m, phi=phi,
                        E_m = E_m, r = r, V_f = V_f, Ll = Ll, Lr = Lr, s0 = s0, Pf=Pf),
            n_int = 100)

            load = np.ones(len(x)) * np.max(si.mu_q_arr * E_f) * V_f
            sigma_m = (load - si.mu_q_arr * E_f * V_f) / (1. - V_f)
            eps_f = si.mu_q_arr
            eps_m = sigma_m / E_m
            w_err = []
            for i, n in enumerate(pi):
                t = tau.ppf(n)
                q = cbi.q[i]
                Tf = 2. * t / r
                e_x = (q - Tf * np.abs(x)) / E_f
                eps_ff = np.maximum(e_x, eps_m)
                w_err.append(np.trapz(eps_ff - eps_m, x))

            w_err = np.array(w_err)
            w_means.append(w_err.mean())
            w_stdevs.append(w_err.std())
            print 'iter =', j
            w_global.append(np.trapz(eps_f - eps_m, x))

        plt.plot(x, si.mu_q_arr, color = 'blue', label = 'after %i iterations' % iters)
        plt.plot(x, eps_m, color = 'blue')
        plt.legend()
        plt.title('fibers and matrix mean strain')
        plt.xlabel('x position [mm]')
        plt.ylabel('strain [-]')
        p = Profile()
        xx = np.linspace(0, 30, 500)
        plt.plot(xx, p.eps_m(xx), lw=2, ls='dashed', color='red')
#        plt.figure()
#        plt.plot(w_means, label = 'mean of filaments')
#        plt.plot(w_global, label = 'global')
#        plt.title('mean of w for individual filaments')
#        plt.xlabel('number of iterations')
#        plt.ylabel('mean w')
#        plt.legend()
#        plt.figure()
#        plt.plot(np.array(w_stdevs) / np.array(w_means))
#        plt.title('COV of w for individual filaments')
#        plt.xlabel('number of iterations')
#        plt.ylabel('stdev w')
        plt.legend()
        plt.show()

    iter()
