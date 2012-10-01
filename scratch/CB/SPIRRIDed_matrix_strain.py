'''
Created on Sep 20, 2012

@author: rostar
'''

import numpy as np
from stats.spirrid.rv import RV
from matplotlib import pyplot as plt
from etsproxy.traits.api import HasTraits, cached_property, \
    Float, Property, Instance, List, Int
from scipy.integrate import cumtrapz
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from scipy.optimize import brentq


class Reinforcement(HasTraits):

    #r = Float(0.00345)
    V_f = Float(0.4)
    E_m = Float(25e3)
    E_f = Float(200e3)
    #xi = Instance(rv_continuous)
    #tau = Instance(rv_continuous)
    n_int = Int

    depsf_arr = Property(depends_on='n_int')
    @cached_property
    def _get_depsf_arr(self):
        weights = 1.0
        no_rv = 0
        if isinstance(self.tau, RV):
            tau = self.tau.ppf(
                np.linspace(.5 / self.n_int, 1. - .5 / self.n_int, self.n_int))
            weights *= 1. / self.n_int
            no_rv += 1
        else:
            tau = self.tau
        if isinstance(self.r, RV):
            r = self.r.ppf(
                np.linspace(.5 / self.n_int, 1. - .5 / self.n_int, self.n_int))
            weights *= 1. / self.n_int
            no_rv += 1
        else:
            r = self.r

        if isinstance(tau, np.ndarray) and isinstance(r, np.ndarray):
            r = r.reshape(1, self.n_int)
            tau = tau.reshape(self.n_int, 1)
        return 2. * tau / r / self.E_f, weights


class CompositeCrackBridge(HasTraits):

    reinforcement_lst = List(Instance(Reinforcement))
    w = Float

    sorted_depsf = Property(depends_on='reinforcement_lst')
    @cached_property
    def _get_sorted_depsf(self):
        return np.sort(self.reinforcement_lst[0].depsf_arr[0])[::1]

    def depsm_mu_tau(self, mu_tau):
        T = self.T_mu_tau.get_values(mu_tau)
        deb_yarn = self.tau_distr().cdf(T)
        E_mtrx = self.V_f * (1 - deb_yarn) * self.E_f + (1. - self.V_f) * self.E_m
        return 2. * mu_tau * self.V_f / self.r / E_mtrx

    def depsf_mu_tau(self, mu_tau):
        return 2. * mu_tau / self.r / self.E_f

    mu_tau_T = Property
    @cached_property
    def _get_mu_tau_T(self):
        T = np.linspace(self.tau_distr().ppf(0.0001),
                                self.tau_distr().ppf(0.9999),
                                500)
        int_value = np.hstack((0.0, cumtrapz(self.tau_distr().cdf(T), T)))
        mu_tau = T * self.tau_distr().cdf(T) - int_value
        return MFnLineArray(xdata=T, ydata=mu_tau)

    T_mu_tau = Property
    @cached_property
    def _get_T_mu_tau(self):
        T = self.mu_tau_T.xdata
        mu_tau = self.mu_tau_T.ydata
        return MFnLineArray(xdata=mu_tau, ydata=T)

    def depsm_T(self, T):
        if T > self.tau_distr().ppf(0.9999):
            T = self.tau_distr().ppf(0.9999)
        mu_tau = self.mu_tau_T.get_values(T)
        return self.depsm_mu_tau(mu_tau)

    def get_T(self, eps_m_x, x_x):
        um = np.trapz(eps_m_x, x_x)
        c = eps_m_x[-1] * x_x[-1] - um
        depsf = 2. * (self.w / 2. - c) / x_x[-1] ** 2
        T = depsf * self.r * self.E_f / 2.
        return T

    eps_m = Property(depends_on='w, reinforcement+')
    @cached_property
    def _get_eps_m(self):
        u_m_x = [0.0]
        eps_m_x = [0.0]
        deps_m_x = [0.0]
        weight = self.reinforcement_lst[0].depsf_arr[1]
        deps_m_x[0] += np.sum(self.sorted_depsf) * weight
        x_x = [0.0]
        for depsf in self.sorted_depsf:
            if self.fwd_Euler == True:
                eps_m_x.append(eps_m_x[-1] + deps_m_x[-1] * (xi - x_x[-1]))
            elif self.midpoint_method == True:
                h = (xi - x_x[-1])
                eps_midp = (eps_m_x[-1] + h / 2. * deps_m_x[-1])
                eps_lst = eps_m_x + [eps_midp]
                x_lst = x_x + [x_x[-1] + h / 2.]
                T_midp = self.get_T(eps_lst, x_lst)
                deps_midp = self.depsm_T(T_midp)
                eps_m_x.append(eps_m_x[-1] + h * deps_midp)
            x_x.append(xi)
            T = self.get_T(eps_m_x, x_x)
            deps_m_x.append(self.depsm_T(T))
        return np.array(eps_m_x)

    def eps_f(self, w, x, tau, E_f, E_m, xi, Ll, Lr, V_f, r):
        pass

    um_line = Property(depends_on = 'w')
    @cached_property
    def _get_um_line(self):
        um = cumtrapz(self.eps_m, self.x_arr)
        um = np.hstack((np.array([0.]),um))
        return MFnLineArray(xdata=self.x_arr, ydata=um)

    epsm_line = Property(depends_on = 'w')
    @cached_property
    def _get_epsm_line(self):
        return MFnLineArray(xdata=self.x_arr, ydata=self.eps_m)

    def MC_residuum(self, x, T_f):
        c = self.epsm_line.get_values(x) * x - self.um_line.get_values(x)
        return c + T_f * x ** 2 / 2. - self.w/2.

    def MC(self, n_sim):
        epsy = np.zeros_like(self.x_arr)
        for sim in range(n_sim):
            tau = self.tau_distr().ppf(np.random.rand(1))
            x = brentq(self.MC_residuum, self.x_arr[0], self.x_arr[-1], args=(self.mu_tau_f(tau)))
            T_f = self.epsm_line.get_values(x) + self.mu_tau_f(tau) * x
            T_f = np.maximum(T_f - self.mu_tau_f(tau) * self.x_arr, self.eps_m)
            epsy += T_f
        epsy /= float(n_sim)
        plt.plot(self.x_arr, epsy)
        print 'w = ', np.trapz(epsy - self.eps_m, self.x_arr)
        print 'force = ', epsy * self.E_f * self.V_f + self.eps_m * self.E_m * (1-self.V_f)
        plt.ylim(0)

if __name__ == '__main__':

    reinf = Reinforcement(r=RV('uniform', loc=0.003, scale=0.001),
                          tau=RV('weibull_min', shape=3., scale=0.3),
                          V_f=0.2,
                          E_m=25e3,
                          E_f=200e3,
                          n_int=5)

    print reinf.depsf_arr

#    def plot_profile(w):
#        ms = MatrixStrain()
#        ms.fwd_Euler = False
#        ms.midpoint_method = True
#        ms.x_arr = np.linspace(0, 50., 300)
#        ms.w = w
#        epsm = ms.eps_m
#        sigc = epsm[-1] * (ms.V_f * ms.E_f + (1. - ms.V_f) * ms.E_m)
#        epsf = (sigc - epsm * ms.E_m * (1. - ms.V_f)) / ms.V_f / ms.E_f
#        plt.plot(ms.x_arr, ms.eps_m, label='matrix strain')
#        plt.plot(ms.x_arr, epsf, label='yarn strain')
#        plt.legend(loc='best')
#        print np.trapz(epsf-epsm, ms.x_arr)
#        ms.MC(100)
#        plt.show()
#
#    def plot_w_epsf(w_arr):
#        ms = MatrixStrain()
#        ms.fwd_Euler = False
#        ms.midpoint_method = True
#        ms.x_arr = np.linspace(0, 20., 300)
#        elst = []
#        wlst = []
#        for w in w_arr:
#            ms.w = w
#            epsm = ms.eps_m
#            sigc = epsm[-1] * (ms.V_f * ms.E_f + (1. - ms.V_f) * ms.E_m)
#            epsf = (sigc - epsm * ms.E_m * (1. - ms.V_f)) / ms.V_f / ms.E_f
#            elst.append(np.max(epsf))
#            w2 = 2 * np.trapz(epsf - epsm, ms.x_arr)
#            wlst.append(w2)
#        plt.plot(w_arr, elst, label='ctrl')
#        plt.plot(wlst, elst, label='eval')
#        plt.legend(loc='best')
#        plt.show()
#
#    plot_w_epsf(np.linspace(0., 0.7, 10))
#    #plot_profile(0.5)

