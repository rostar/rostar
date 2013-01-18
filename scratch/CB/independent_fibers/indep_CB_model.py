'''
Created on Jan 15, 2013

@author: rostar
'''
#---#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Jun 14, 2010 by: rch

from etsproxy.traits.api import \
    Float, Str, implements

import numpy as np

from stats.spirrid.i_rf import \
    IRF

from stats.spirrid.rf import \
    RF

from matplotlib import pyplot as plt
from math import pi


def H(x):
    return x >= 0.0


class CBResidual(RF):
    '''
    Crack bridged by a fiber with constant
    frictional interface to rigid; free fiber end;
    '''

    implements(IRF)

    title = Str('crack bridge with rigid matrix')

    tau = Float(2.5, auto_set=False, enter_set=True, input=True,
                distr=['uniform', 'norm'])

    l = Float(10.0, auto_set=False, enter_set=True, input=True,
              distr=['uniform'], desc='free length')

    r = Float(0.013, auto_set=False, enter_set=True, input=True,
              distr=['uniform', 'norm'], desc='fiber radius')

    E_f = Float(72e3, auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])

    E_m = Float(30e3, auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])

    l0 = Float(10., auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])

    m = Float(5., auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])

    s0 = Float(0.02, auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])

    V_f = Float(0.0175, auto_set=False, enter_set=True, input=True,
              distr=['uniform'])

    Pf = Float(0.0, auto_set=False, enter_set=True, input=True,
              distr=['uniform'])

    w = Float(auto_set=False, enter_set=True, input=True,
               distr=['uniform'], desc='crack width',
               ctrl_range=(0.0, 1.0, 10))

    x = Float(auto_set=False, enter_set=True, input=True,
               distr=['uniform'], desc='crack width',
               ctrl_range=(0.0, 1.0, 10))

    x_label = Str('crack opening [mm]')
    y_label = Str('composite stress [MPa]')

    C_code = Str('')

    def omega(self, f_inf, w, T, E_f, r, m, l0, s0, Pf):
        epsf0 = f_inf / (pi * r ** 2) / E_f
        a = epsf0 / T
        PfL = 1 - np.exp(-(a * (epsf0 / s0) ** m) / (l0 * (m + 1)))
        PDF = (1./l0) * (epsf0 * (1 - x / a)/s0) ** m * np.exp(-(a * (epsf0 / s0) ** m) / (l0 * (m + 1)))
        muL = np.trapz(x * PDF, x)
        return PfL, muL

    def __call__(self, w, tau, E_f, V_f, r, m, l0, s0, Pf):
        #defining variables
        T = 2. * tau / r
        f_inf = np.sqrt(T * w / E_f) * pi * r ** 2 * E_f * np.ones_like(Pf)
        omega, muL = self.omega(f_inf, w, T, E_f, r, m, l0, s0, Pf)
        f = f_inf * (1 - omega) + muL * 2 * pi * r * tau * omega
        return f

from stats.spirrid.spirrid import SPIRRID
from stats.spirrid.rv import RV


def CB_composite_stress(w, tau, E_f, V_f, r, m, l0, s0, Pf, n_int):
    cb = CBResidual()
    s = SPIRRID(q=cb,
                sampling_type='PGrid',
                evars=dict(w=w),
                tvars=dict(tau=tau, E_f=E_f, V_f=V_f, r=r,
                           m=m, l0=l0, s0=s0, Pf=Pf),
                n_int=n_int)

    if isinstance(r, RV):
        r_arr = np.linspace(r.ppf(0.001), r.ppf(0.999), 300)
        Er = np.trapz(r_arr ** 2 * r.pdf(r_arr), r_arr)
    else:
        Er = r ** 2
    sigma_c = s.mu_q_arr * V_f / (pi * Er)
    plt.plot(w, sigma_c, color='blue', label='SPIRRID')
    plt.xlabel('w [mm]')
    plt.ylabel('sigma_c [MPa]')
    plt.show()

if __name__ == '__main__':
    w = np.linspace(0, 3.0, 100)
    tau = RV('uniform', loc=0.01, scale=.10)
    E_f = 72e3
    V_f = 0.1
    r = RV('uniform', loc=0.002, scale=0.004)
    Pf = RV('uniform', loc=0.0, scale=1.0)
    m = 5.
    l0 = 100.
    s0 = 0.017
    n_int = 50
    #cb = CBResidual()
    #plt.plot(w, cb(w, 0.5, E_f, V_f, r, m, l0, s0, Pf) / (pi * r ** 2))
    #plt.show()
    CB_composite_stress(w, tau, E_f, V_f, r, m, l0, s0, Pf, n_int)
