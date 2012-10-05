'''
Created on Sep 20, 2012

@author: rostar
'''

import numpy as np
from scipy.stats import weibull_min, uniform
from matplotlib import pyplot as plt
from etsproxy.traits.api import HasTraits, cached_property, Float, Property, Bool, Array
from scipy.integrate import cumtrapz
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from scipy.optimize import brentq


class MatrixStrain(HasTraits):

    r = Float(0.00345)
    V_f = Float(0.125)
    E_m = Float(25e3)
    E_f = Float(200e3)
    w = Float(0.4)
    x_arr = Array
    fwd_Euler = Bool(False)
    midpoint_method = Bool(False)

    def tau_distr(self):
        return uniform(loc=0.1, scale=3.)

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

    eps_m = Property(depends_on='w')
    @cached_property
    def _get_eps_m(self):
        eps_m_x = [0.0]
        deps_m_x = [self.depsm_mu_tau(self.tau_distr().mean())]
        x_x = [0.0]
        for xi in self.x_arr[1:]:
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

##################################################################
# damage
##################################################################

    def xi_distr(self):
        return weibull_min(self.mxi, scale=self.sxi)

    def eps_f(self):
        '''stiffness of every 'filament' multiplied by damage_func(T)'''
        pass

    um_line = Property(depends_on='w')
    @cached_property
    def _get_um_line(self):
        um = cumtrapz(self.eps_m, self.x_arr)
        um = np.hstack((np.array([0.]),um))
        return MFnLineArray(xdata=self.x_arr, ydata=um)

    epsm_line = Property(depends_on='w')
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
            x = brentq(self.MC_residuum, self.x_arr[0], self.x_arr[-1], args=(self.depsf_mu_tau(tau)))
            T_f = self.epsm_line.get_values(x) + self.depsf_mu_tau(tau) * x
            T_f = np.maximum(T_f - self.depsf_mu_tau(tau) * self.x_arr, self.eps_m)
            epsy += T_f
        epsy /= float(n_sim)
        plt.plot(self.x_arr, epsy)
        #print 'w = ', np.trapz(epsy - self.eps_m, self.x_arr)
        #print 'force = ', epsy * self.E_f * self.V_f + self.eps_m * self.E_m * (1-self.V_f)
        plt.ylim(0)

if __name__ == '__main__':

    def plot_profile(w):
        ms = MatrixStrain()
        ms.fwd_Euler = False
        ms.midpoint_method = True
        ms.x_arr = np.linspace(0, 25., 300)
        ms.w = w
        epsm = ms.eps_m
        sigc = epsm[-1] * (ms.V_f * ms.E_f + (1. - ms.V_f) * ms.E_m)
        epsf = (sigc - epsm * ms.E_m * (1. - ms.V_f)) / ms.V_f / ms.E_f
        plt.plot(ms.x_arr, ms.eps_m, label='matrix strain')
        plt.plot(ms.x_arr, epsf, label='yarn strain')
        plt.legend(loc='best')
        #print np.trapz(epsf-epsm, ms.x_arr)
        ms.MC(10)
        plt.show()

    def plot_w_epsf(w_arr):
        ms = MatrixStrain()
        ms.fwd_Euler = False
        ms.midpoint_method = True
        ms.x_arr = np.linspace(0, 20., 300)
        elst = []
        wlst = []
        for w in w_arr:
            ms.w = w
            epsm = ms.eps_m
            sigc = epsm[-1] * (ms.V_f * ms.E_f + (1. - ms.V_f) * ms.E_m)
            epsf = (sigc - epsm * ms.E_m * (1. - ms.V_f)) / ms.V_f / ms.E_f
            elst.append(np.max(epsf))
            w2 = 2 * np.trapz(epsf - epsm, ms.x_arr)
            wlst.append(w2)
        plt.plot(w_arr, elst, label='ctrl')
        plt.plot(wlst, elst, label='eval')
        plt.legend(loc='best')
        plt.show()

    #plot_w_epsf(np.linspace(0., 0.7, 10))
    plot_profile(0.4)

