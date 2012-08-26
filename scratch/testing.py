'''
Created on 14.6.2012

@author: Q
'''
import numpy as np
from matplotlib import pyplot as plt
from stats.spirrid.spirrid import SPIRRID
from stats.spirrid.rv import RV
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress import \
    CBEMClampedFiberStress
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress_residual import \
    CBEMClampedFiberStressResidual

# filaments
r = 0.00345
V_f = 0.0103
tau = RV('uniform', loc = 0.02, scale = .1) # 0.5
E_f = 200e3
E_m = 25e3
l = RV( 'uniform', scale = 20., loc = 10. )
theta = 0.0#RV( 'uniform', scale = .01, loc = 0. )
xi = RV( 'weibull_min', scale = 0.02, shape = 5 ) # 0.017
phi = 1.
Ll = 20.
Lr = 30.
Pf = RV('uniform', loc = 0.0, scale = 1.0)
w = np.linspace(0,1.2,200)
m = 5.
s0 = 0.02

import time as t
tt = t.clock()
rf = CBEMClampedFiberStress()
spirrid = SPIRRID(q = rf,
                   sampling_type = 'TGrid',
                   evars = dict(w = w),
                   tvars = dict( tau = tau,
                                 l = l,
                                 Ll = Ll,
                                 Lr = Lr,
                                 E_f = E_f,
                                 theta = theta,
                                 xi = xi,
                                 phi = phi,
                                 E_m = E_m,
                                 r = r,
                                 V_f = V_f
                                 ),
                   n_int = 20)

#plt.plot(w, spirrid.mu_q_arr, lw = 2, label = 'no residuum')
print t.clock() - tt

tt = t.clock()
rf = CBEMClampedFiberStressResidual()
spirrid = SPIRRID(q = rf,
                   sampling_type = 'TGrid',
                   evars = dict(w = w),
                   tvars = dict( tau = tau,
                                 l = l,
                                 Ll = Ll,
                                 Lr = Lr,
                                 E_f = E_f,
                                 theta = theta,
                                 Pf = Pf,
                                 phi = phi,
                                 E_m = E_m,
                                 r = r,
                                 V_f = V_f,
                                 m = m,
                                 s0 = s0
                                 ),
                   n_int = 20)


plt.plot(w, spirrid.mu_q_arr * V_f, lw = 2, label = 'force residuum')
print t.clock()-tt
plt.legend(loc = 'best')
plt.show()