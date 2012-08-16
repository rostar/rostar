'''
Created on Mar 14, 2012

@author: rostar
'''

from etsproxy.traits.api import HasTraits
from stats.spirrid.spirrid import SPIRRID
from stats.spirrid.rv import RV
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress import \
    CBEMClampedFiberStress
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress_residual import \
    CBEMClampedFiberStressResidual
import numpy as np
from matplotlib import pyplot as plt
import time

r = 0.00345
V_f = 0.0103
tau = RV('uniform', loc=0.02, scale=.2)
E_f = 200e3
E_m = 25e3
l = RV('uniform', scale=20., loc=2.)
theta = 0.0
xi = RV('weibull_min', scale=0.02, shape=5)
phi = 1.
Ll = 40.
Lr = 20.
s0 = 0.02
m = 5.0
w = np.linspace(0, 2.5, 200)
Pf = RV('uniform', loc=0., scale=1.0)


tt = time.clock()
rf = CBEMClampedFiberStress()
spirrid1 = SPIRRID(q=rf,
               sampling_type='PGrid',
               evars=dict(w=w),
          tvars=dict(Ll=Ll,
                       Lr=Lr,
                       tau=tau,
                        l=l,
                        E_f=E_f,
                        theta=theta,
                        xi=xi,
                        phi=phi,
                        E_m=E_m,
                        r=r,
                        V_f=V_f,
                             ),
                   n_int=30)

plt.plot(w, spirrid1.mu_q_arr, lw=2, label='breakage in crack')
print time.clock() - tt

tt = time.clock()
rf = CBEMClampedFiberStressResidual()

spirrid2 = SPIRRID(q=rf,
               sampling_type='PGrid',
                evars=dict(w=w),
          tvars=dict(Ll=Ll,
                       Lr=Lr,
                       tau=tau,
                        l=l,
                        E_f=E_f,
                        theta=theta,
                        Pf=Pf,
                        phi=phi,
                        E_m=E_m,
                        r=r,
                        V_f=V_f,
                        m=m,
                        s0=s0
                             ),
                   n_int=30)
plt.plot(w, spirrid2.mu_q_arr, lw=2, label='residual')

print time.clock() - tt
plt.xlabel('crack opening w [mm]')
plt.ylabel('filament stress $\sigma_f$ [MPa]')
plt.legend()
plt.show()
