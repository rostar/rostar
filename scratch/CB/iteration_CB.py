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
    Float, Str, implements, Range, Array, Property, cached_property, List
import numpy as np
from stats.spirrid.i_rf import IRF
from stats.spirrid.rf import RF
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import copy

def H(x):
    return x >= 0.0


class CBIter(RF):
        '''
        Stress profile for a crack bridged by a fiber with constant
        frictional interface to an elastic matrix; clamped fiber end;
        residual stress carried by broken filaments.
        '''

        implements(IRF)
    
        title = Str('crack bridge - clamped fiber with constant friction')
    
        Pf = Range(0, 1, auto_set=False, enter_set=True, input=True,
                    distr=['uniform'])
    
        m = Float(5.0, auto_set=False, enter_set=True, input=True,
                    distr=['weibull_min', 'uniform'],
                    desc='filament Weibull shape parameter')
    
        s0 = Float(.02, auto_set=False, enter_set=True, input=True,
                    distr=['weibull_min', 'uniform'],
                    desc='filament Weibull scale parameter at l = 10 mm')
    
        tau = Float(2.5, auto_set=False, enter_set=True, input=True,
                    distr=['uniform', 'norm'])
    
        l = Float(10.0, auto_set=False, enter_set=True, input=True,
                  distr=['uniform'], desc='free length')
    
        r = Float(0.013, auto_set=False, input=True,
                  enter_set=True, desc='fiber radius in mm')
    
        E_f = Float(72e3, auto_set=False, enter_set=True, input=True,
                      distr=['uniform'])
    
        E_m = Float(30e3, auto_set=False, enter_set=True, input=True,
                      distr=['uniform'])
    
        V_f = Float(0.0175, auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])
    
        theta = Float(0.01, auto_set=False, enter_set=True, input=True,
                      distr=['uniform', 'norm'], desc='slack')
    
        phi = Float(1., auto_set=False, enter_set=True, input=True,
                      distr=['uniform', 'norm'], desc='bond quality')
    
        Ll = Float(1., auto_set=False, enter_set=True, input=True,
                  distr=['uniform'], desc='embedded length - left',
                   ctrl_range=(0.0, 1.0, 10))
    
        Lr = Float(.5, auto_set=False, enter_set=True, input=True,
                  distr=['uniform'], desc='embedded length - right',
                   ctrl_range=(0.0, 1.0, 10))
    
        w = Float(auto_set=False, enter_set=True, input=True,
                   distr=['uniform'], desc='crack width',
                   ctrl_range=(0.0, 1.0, 10))
    
        x = Float(auto_set=False, enter_set=True, input=True,
                   distr=['uniform'], desc='crack width',
                   ctrl_range=(0.0, 1.0, 10))

        eps_m = Array
        x_arr = Array
    
        x_label = Str('crack opening [mm]')
        y_label = Str('force [N]')
    
        C_code = Str('')

        x = Float(0.0, auto_set=False, enter_set=True, input=True,
                  distr=['uniform'], desc='distance from crack')

        x_label = Str('position [mm]')
        y_label = Str('force [N]')

        C_code = Str('')
        
        params = List

        def residuum(self, q, q_shape, w, x, tau, l, E_f, E_m, theta, Pf,
                     phi, Ll, Lr, V_f, r, s0, m):
            Tf = 2. * tau / r
            #stress in the free length
            l = l * (1 + theta)
            q = q.reshape(q_shape)
            q_l = q * H(l / 2 - abs(x))
            #stress in the part, where fiber transmits stress to the matrix
            q_e = (q - Tf * (abs(x) - l / 2.)) * H(abs(x) - l / 2.)
            #q_e = q_e * H(x + Ll) * H (Lr - x)
            #putting all parts together
            q_x = q_l + q_e
            eps_m = self.eps_m.reshape((1,1000))
            eps_f = np.maximum(q_x / E_f, eps_m)
            w_iter = np.trapz(eps_f - self.eps_m, x)
            return (w - w_iter).flatten()

        q_init = Array(Float)

        q = Property()

        @cached_property
        def _get_q(self):
            varlist = self.params
            varlist.append(self.q_init)
            reshaped = []
            shape_len = 1
            for var in varlist:
                if isinstance(var, np.ndarray):
                    shape = tuple(list(var.shape) + [1])
                    var = var.reshape(shape)
                    shape_len = len(var.shape)
                reshaped.append(var)
            q_init = np.array(reshaped.pop(), dtype='float64')
            x_shape = np.ones(shape_len)
            x_shape[-1] = len(self.x_arr)
            x_arr = self.x_arr.reshape(x_shape)
            reshaped.insert(1, x_arr)
            reshaped.insert(0, q_init.shape)
            q = fsolve(self.residuum, q_init.flatten(), args=(tuple(reshaped)))
            q = q.reshape(q_init.shape)
            return q

        def __call__(self, w, x, tau, l, E_f, E_m, theta, Pf,
                     phi, Ll, Lr, V_f, r, s0, m):
            self.params = [w, tau, l, E_f,
                    E_m, theta, Pf, phi, Ll, Lr, V_f, r, s0, m]
            Tf = 2. * tau / r
            if isinstance(tau,np.ndarray):
                Tf = Tf.reshape(len(tau),1)
            #stress in the free length
            l = l * (1 + theta)
            q = self.q
            q_l = q * H(l / 2. - abs(x))
            #stress in the part, where fiber transmits stress to the matrix
            q_e = (q - Tf * (abs(x) - l / 2.)) * H(abs(x) - l / 2.)
            #q_e = q_e * H(x + Ll) * H (Lr - x)

            #putting all parts together
            q_x = q_l + q_e
            eps_f = q_x / E_f
            eps_m = self.eps_m[np.argwhere(x==self.x_arr)]
            eps_f = np.maximum(eps_f,eps_m)
            return eps_f
