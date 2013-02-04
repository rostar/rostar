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
    Float, Str, implements, Bool

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

    L_0 = Float(10., auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])

    m = Float(5., auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])

    s = Float(0.02, auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])

    V_f = Float(0.0175, auto_set=False, enter_set=True, input=True,
              distr=['uniform'])

    w = Float(auto_set=False, enter_set=True, input=True,
               distr=['uniform'], desc='crack width',
               ctrl_range=(0.0, 1.0, 10))

    x = Float(auto_set=False, enter_set=True, input=True,
               distr=['uniform'], desc='crack width',
               ctrl_range=(0.0, 1.0, 10))
    
    include_pullout = Bool(True)

    x_label = Str('crack opening [mm]')
    y_label = Str('composite stress [MPa]')

    C_code = Str('')

    def __call__(self, w, tau, E_f, V_f, r, m,  L_0, s, Pf):
        #strain and debonded length of intact fibers
        T = 2. * tau / r
        ef0_inf = np.sqrt(T * w / E_f)
        # strain at fiber breakage
        ef0_break = (-0.5 * np.log(1.-Pf) * T / E_f *  L_0 * (m+1) * s**m) ** (1./(m+1))
        # debonded length at fiber breakage
        a_break = ef0_break * E_f / T
        #mean pullout length of broken fibers
        mu_Lpo = a_break / (m + 1)
        # strain carried by broken fibers
        ef0_residual = T / E_f * mu_Lpo

        if self.include_pullout == True:
            ef0_tot = ef0_residual * H(ef0_inf - ef0_break) + ef0_inf * H(ef0_break - ef0_inf)
        else:
            ef0_tot = ef0_inf * H(ef0_break - ef0_inf)
        return ef0_tot * E_f * V_f * r**2

if __name__ == '__main__':
    from stats.spirrid.spirrid import SPIRRID
    from stats.spirrid.rv import RV

    
    def CB_composite_stress(w, tau, E_f, V_f, r, m, L_0, s, Pf, n_int):
        cb = CBResidual()
        m_arr = np.linspace(.2, 15., 30)
        r_arr = np.linspace(.001, .01, 5)
        for ri in r_arr:
            sig_max = []
            for mi in m_arr:
                spirrid = SPIRRID(q=cb,
                            sampling_type='PGrid',
                            evars=dict(w=w),
                            tvars=dict(tau=tau, E_f=E_f, V_f=V_f, r=ri,
                                       m=mi, L_0= L_0, s=s, Pf=Pf),
                            n_int=n_int)
                if isinstance(r, RV):
                    r_arr = np.linspace(r.ppf(0.001), r.ppf(0.999), 300)
                    Er = np.trapz(r_arr ** 2 * r.pdf(r_arr), r_arr)
                else:
                    Er = r ** 2
                sigma_c = spirrid.mu_q_arr / Er
                if np.max(sigma_c) == sigma_c[-1]:
                    print 'NO NO NO'    
                sig_max.append(np.max(sigma_c))
            
            plt.plot(m_arr, sig_max, lw=2, label='r =' + str(ri))
        #plt.ylim(0, 35)
        plt.xlabel('m')
        plt.ylabel('sigma_c [MPa]')
        plt.legend(loc='best')
        plt.show()
    w = np.linspace(0, 3.5, 300)
    tau = 0.5#RV('weibull_min', shape=3., scale=.03)
    E_f = 200e3
    V_f = 0.01
    r = 0.003#RV('uniform', loc=0.001, scale=0.004)
    m = 5.
    L_0 = 100.
    s = 0.01
    Pf = RV('uniform', loc=0., scale=1.0)
    n_int = 300
    #cb = CBResidual()
    #plt.plot(w, cb(w, 0.5, E_f, V_f, r, m,  L_0, s, Pf) / (pi * r ** 2))
    #plt.show()
    CB_composite_stress(w, tau, E_f, V_f, r, m, L_0, s, Pf, n_int)
