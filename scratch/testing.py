
'''
Created on 14.6.2012

@author: Q
'''
import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.stats import weibull_min
from matplotlib import pyplot as plt
from stats.spirrid.spirrid import SPIRRID
from stats.spirrid.rv import RV
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress import \
    CBEMClampedFiberStress
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress_residual import \
    CBEMClampedFiberStressResidual
from math import e
from etsproxy.traits.api import HasTraits, cached_property, Float, Property
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray


class Profile(HasTraits):

    r = Float(0.00345)
    mtau = Float(5.)
    stau = Float(0.5)
    V_f = Float(0.04)
    E_m = Float(25e3)
    E_f = Float(200e3)
    w = Float(0.4)

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

    Tmfn = Property(depends_on='V_f, mtau, stau, r, E_m, E_f, w')

    @cached_property
    def _get_Tmfn(self):
        tau_arr = np.linspace(0.0, self.tau_distr().ppf(0.99), 500)
        a = lambda t: np.sqrt(2. * self.w * self.Km /
                      (2./self.r/self.E_f *
                       (t + np.trapz(self.tau_distr().cdf(tau_arr), tau_arr))))
        a_vect = np.vectorize(a)
        x_arr = a_vect(tau_arr)
        return MFnLineArray(xdata=x_arr[::-1], ydata=2./self.r/self.E_f*tau_arr[::-1])

    def T_x(self, x):
        return self.Tmfn.get_values(x)



if __name__ == '__main__':
    p = Profile()
    x = np.linspace(0, 6000, 100)
    t = p.T_x(x)
    plt.plot(x, t)
    plt.show()
    







