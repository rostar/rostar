
'''
Created on 14.6.2012

@author: Q
'''

import numpy as np
from scipy.stats import weibull_min, uniform
from matplotlib import pyplot as plt
from etsproxy.traits.api import HasTraits, cached_property, Float, Property, Bool
from scipy.integrate import cumtrapz
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray


class MatrixStrain(HasTraits):

    r = Float(0.00345)
    mtau = Float(5.)
    stau = Float(0.5)
    V_f = Float(0.04)
    E_m = Float(25e3)
    E_f = Float(200e3)
    w = Float(0.4)
    fwd_Euler = Bool(False)
    midpoint_method = Bool(False)

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
        return weibull_min(self.mtau, scale=self.stau)
        #return weibull_min(self.mtau, scale=self.stau)

    def T_epsm(self, tau):
        return 2. * tau * self.V_f / self.r / self.E_m / (1. - self.V_f)

    def T_epsf(self, tau):
        return 2. * tau / self.r / self.E_f

    int_G = Property

    @cached_property
    def _get_int_G(self):
        tau_range = np.linspace(self.tau_distr().ppf(0.0001),
                                self.tau_distr().ppf(0.9999),
                                500)
        int_value = np.hstack((0.0, cumtrapz(self.tau_distr().cdf(tau_range), tau_range)))

        return MFnLineArray(xdata=tau_range, ydata=int_value)

    def deps_m(self, T):
        tau_max = T * self.r * (1 - self.V_f) * self.E_m / 2. / self.V_f
        if tau_max > self.tau_distr().ppf(0.9999):
            tau_max = self.tau_distr().ppf(0.9999)
        result = tau_max * self.tau_distr().cdf(tau_max) - self.int_G.get_values(tau_max)
        return self.T_epsm(result)

    def get_T(self, eps_m_x, x_x):
        um = np.trapz(eps_m_x, x_x)
        c = eps_m_x[-1] * x_x[-1] - um
        Tf = 2 * (self.w / 2. - c) / x_x[-1] ** 2
        tau = Tf * self.r * self.E_f / 2.
        return self.T_epsm(tau)

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
                eps_midp = (eps_m_x[-1] + h / 2. * deps_m_x[-1])
                eps_lst = eps_m_x + [eps_midp]
                x_lst = x_x + [x_x[-1] + h / 2.]
                T_midp = self.get_T(eps_lst, x_lst)
                deps_midp = self.deps_m(T_midp)
                eps_m_x.append(eps_m_x[-1] + h * deps_midp)
            x_x.append(xi)
            T_epsm_x.append(self.get_T(eps_m_x, x_x))
            deps_m_x.append(self.deps_m(T_epsm_x[-1]))
#        t_arr = np.array([2.0] + T_epsm_x).flatten()
#        x_arr = np.array(x_x)
#        epsm = np.array(eps_m_x)
#        tau_x = MFnLineArray(ydata=x_arr[::-1], xdata=(t_arr * self.r *
#                                                       (1 - self.V_f) *
#                                                       self.E_m / 2. / self.V_f)[::-1])
#        x_epsm = MFnLineArray(xdata=x_arr, ydata=epsm)
#        taufil_arr = np.linspace(0.225, 2.175, 40)
#        xfil_arr = tau_x.get_values(taufil_arr)[::-1]
#        x_arr = np.hstack((np.linspace(0.0, xfil_arr[0] - 0.1, 50), xfil_arr))
#        for i, t in enumerate(taufil_arr[::-1]):
#            t = self.T_epsf(t)
#            x = xfil_arr[:i + 1]
#            epsm = x_epsm.get_values(x_arr)
#            epsf = epsm[i] + x[-1] * t - x_arr * t
#            epsf = np.maximum(epsf, epsm)
#            print np.trapz(epsf - epsm, x_arr)
#            plt.plot(x_arr, epsf)
#            plt.plot(x_arr, epsm)
#        plt.show()
        return eps_m_x

if __name__ == '__main__':
    ms = MatrixStrain()
    ms.fwd_Euler = False
    ms.midpoint_method = True
    x = np.linspace(0, 30., 1000)
    epsm = np.array(ms.eps_m(x))
    sigc = epsm[-1] * (ms.V_f * ms.E_f + (1. - ms.V_f) * ms.E_m)
    epsf = (sigc - epsm * ms.E_m * (1. - ms.V_f)) / ms.V_f / ms.E_f
    print np.trapz(epsf - epsm, x)
    plt.plot(x, ms.eps_m(x), label='matrix strain')
    plt.plot(x, epsf, label='yarn strain')
    plt.legend()
    plt.show()

