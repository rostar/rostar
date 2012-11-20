'''
Created on Nov 19, 2012

@author: rostar
'''

from etsproxy.mayavi import mlab as m
import numpy as np
from scipy.stats import weibull_min
from stats.spirrid import make_ogrid as orthogonalize
from rostar.scratch.CB.dependent_fibers.composite_CB_model import CompositeCrackBridge
from composite_CB_modelview import CompositeCrackBridgeView
from matplotlib import pyplot as plt
from stats.spirrid.rv import RV
from reinforcement import Reinforcement, WeibullFibers
from stats.spirrid.spirrid import SPIRRID
from rf_rigid_mtrx import CBRigidMatrix

reinf1 = Reinforcement(r=0.00345,#RV('uniform', loc=0.001, scale=0.005),
                      tau=RV('uniform', loc=.5, scale=.2),
                      V_f=0.3,
                      E_f=70e3,
                      xi=WeibullFibers(shape=5., scale=0.02, L0=10.),#RV('weibull_min', shape=5., scale=.02),
                      n_int=15,
                      label='AR glass')

reinf2 = Reinforcement(r=0.00345,#RV('uniform', loc=0.002, scale=0.002),
                      tau=RV('uniform', loc=.5, scale=.1),
                      V_f=0.4,
                      E_f=200e3,
                      xi=WeibullFibers(shape=5., scale=0.03, L0=10.),
                      n_int=15,
                      label='carbon')

model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf1, reinf2],
                             Ll=100.,
                             Lr=100.)

ccb_view = CompositeCrackBridgeView(model=model)

def rigid_mtrx_determ_bond_inf_xi():
    rf = CBEMClampedFiberStress
    spirrid=SPIRRID(q=rf, sampling_type='PGrid',
                    tvars=dict(tau=tau,
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
                                        n_int=20),

def random_domain(w):
    Ef = 70e3
    Fxi = weibull_min(5., scale = 0.02)
    r = np.linspace(0., 0.005, 100)
    tau = np.linspace(0., 1., 100)
    e_arr = orthogonalize([np.arange(len(r)), np.arange(len(tau))])

    tau = tau.reshape(1, len(tau))
    r = r.reshape(len(r), 1)
    eps0 = np.sqrt(w * 2 * tau / r / Ef)
    F = Fxi.cdf(eps0)
    m.surf(e_arr[0], e_arr[1], F*30)
    m.show()

def profile(w):
    ccb_view.model.w = w
    plt.plot(ccb_view.x_arr, ccb_view.epsm_arr, label='w_eval=' + str(ccb_view.w_evaluated) + ' w_ctrl=' + str(ccb_view.model.w))
    plt.plot(ccb_view.x_arr, ccb_view.mu_epsf_arr, label='yarn')
    plt.xlabel('position [mm]')
    plt.ylabel('strain')

def sigma_c_w(w_arr, label):
    sigma_c = []
    w_err = []
    for w in w_arr:
        ccb_view.model.w = w
        sigma_c.append(ccb_view.sigma_c)
        w_err.append((ccb_view.w_evaluated - ccb_view.model.w) / (ccb_view.model.w + 1e-10))
    plt.plot(w_arr, sigma_c, lw=2, label=label, color='black')
    plt.legend(loc='best')

def sigma_f(w_arr):
    sf_arr = ccb_view.sigma_f_lst(w_arr)
    #for i, reinf in enumerate(ccb_view.model.reinforcement_lst):
    #    plt.plot(w_arr, sf_arr[:, i], label=reinf.label)
    plt.plot(w_arr, sf_arr[:,0], ls='dashed', label='$\sigma_{\mathrm{f}1}$', color='black')
    plt.plot(w_arr, sf_arr[:,1], ls='dotted', label='$\sigma_{\mathrm{f}2}$', color='black')
    plt.xlabel('crack opening w [mm]')
    plt.ylabel('stress [MPa]')
    plt.ylim(0)

#profile(.03)
sigma_c_w(np.linspace(.0, .3, 100), label='$\sigma_\mathrm{c}$')
sigma_f(np.linspace(.0, .3, 100))
plt.legend(loc='best')
plt.show()
  
#random_domain(0.2)