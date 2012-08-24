'''
Created on Mar 14, 2012

@author: rostar
'''

from etsproxy.traits.api import HasTraits
from stats.spirrid.spirrid import SPIRRID
from stats.spirrid.rv import RV
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress import \
    CBEMClampedFiberStress, CBEMClampedFiberStressSP
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress_residual import \
    CBEMClampedFiberStressResidualSP
import numpy as np
from matplotlib import pyplot as plt
import time

# filaments
r = 0.00345
V_f = 0.1
tau = RV('uniform', loc=.01, scale=.2)
E_f = 200e3
E_m = 25e3
l = 0.0
theta = 0.0
phi = 1.
Ll = 50.
Lr = 50.
xi = RV('weibull_min', shape=5., scale=0.01)
m = 5.0
s0 = 100.
Pf = 0.5

x = np.linspace(-Ll, Lr, 200)
w = 0.2

def get_qmax():
    cb = CBEMClampedFiberStress()
    sq = SPIRRID(q=cb,
         sampling_type='PGrid',
         evars=dict(w=np.array([w, 0.05])),
         tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                    E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
         n_int=200)
    return sq.mu_q_arr[0]
qmax = get_qmax()

cb = CBEMClampedFiberStressSP()
s = SPIRRID(q=cb,
     sampling_type='PGrid',
     evars=dict(x=x),
     tvars=dict(w=w, tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
     n_int=200)

load = qmax * V_f
eps_m = (load - s.mu_q_arr * V_f) / (1. - V_f) / E_m
eps_f = s.mu_q_arr / E_f
plt.plot(x, eps_f, color='red', label='fiber', lw = 2)
plt.plot(x, eps_m, color='blue', label='mtrx', lw = 2)
print 'global w = ', np.trapz(eps_f - eps_m, x), 'mm'
for i in range(20):
    args = list(s.get_samples(1).flatten())
    args.insert(1, x)
    epsfi = cb(*args) / E_f
    plt.plot(x, epsfi, color='grey')
    print 'w(',i,') = ', np.trapz(np.maximum(epsfi, eps_m) - eps_m, x) 
plt.show()
