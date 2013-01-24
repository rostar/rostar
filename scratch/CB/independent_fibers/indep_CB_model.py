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
from scipy.special import gammainc, gamma


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

    w = Float(auto_set=False, enter_set=True, input=True,
               distr=['uniform'], desc='crack width',
               ctrl_range=(0.0, 1.0, 10))

    x = Float(auto_set=False, enter_set=True, input=True,
               distr=['uniform'], desc='crack width',
               ctrl_range=(0.0, 1.0, 10))

    x_label = Str('crack opening [mm]')
    y_label = Str('composite stress [MPa]')

    C_code = Str('')

    def __call__(self, w, tau, E_f, V_f, r, m, l0, s0):
        #defining variables
        T = 2. * tau / r
        ef0_inf = np.sqrt(T * w / E_f)
        a_inf = ef0_inf * E_f / T
        Pf = 1. - np.exp(-a_inf * 2 * (ef0_inf / s0) ** m / (m + 1) / l0)

        psi = (l0/r*(s0* E_f)**m*tau*(m+1))**(1./(m+1.))
        alpha = (ef0_inf * E_f / psi)**(m+1)
        muL_failed = l0/2. * ((s0* E_f)/psi)**m * gamma((m+2)/(m+1)) * gammainc((m+2)/(m+1), alpha) / Pf
        ef0_failed = T * muL_failed / E_f

        ef0_tot = ef0_inf * (1-Pf) + ef0_failed * Pf
        return ef0_tot

from stats.spirrid.spirrid import SPIRRID
from stats.spirrid.rv import RV


def CB_composite_stress(w, tau, E_f, V_f, r, m, l0, s0, n_int):
    cb = CBResidual()
    s = SPIRRID(q=cb,
                sampling_type='PGrid',
                evars=dict(w=w),
                tvars=dict(tau=tau, E_f=E_f, V_f=V_f, r=r,
                           m=m, l0=l0, s0=s0),
                n_int=n_int)

    if isinstance(r, RV):
        r_arr = np.linspace(r.ppf(0.001), r.ppf(0.999), 300)
        Er = np.trapz(r_arr ** 2 * r.pdf(r_arr), r_arr)
    else:
        Er = r ** 2
    sigma_c = s.mu_q_arr * V_f / (pi * Er)
    plt.plot(w, sigma_c, color='blue', label='SPIRRID')
    plt.xlabel('w [mm]')
    plt.ylabel('epsfmax [-]')
    plt.show()

if __name__ == '__main__':
    w = np.linspace(0, 5.0, 500)
    tau = RV('uniform', loc=0.01, scale=.50)
    E_f = 72e3
    V_f = 0.1
    r = RV('uniform', loc=0.002, scale=0.004)
    m = 5.
    l0 = 100.
    s0 = 0.017
    n_int = 50
    #cb = CBResidual()
    #plt.plot(w, cb(w, 0.5, E_f, V_f, r, m, l0, s0, Pf) / (pi * r ** 2))
    #plt.show()
    CB_composite_stress(w, tau, E_f, V_f, r, m, l0, s0, n_int)

