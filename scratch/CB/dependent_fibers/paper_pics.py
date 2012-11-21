'''
Created on Nov 19, 2012

@author: rostar
'''

from etsproxy.mayavi import mlab as m
import numpy as np
from scipy.stats import weibull_min
from stats.spirrid import make_ogrid as orthogonalize
from composite_CB_model import CompositeCrackBridge
from composite_CB_modelview import CompositeCrackBridgeView
from matplotlib import pyplot as plt
from stats.spirrid.rv import RV
from reinforcement import Reinforcement, WeibullFibers
from stats.spirrid.spirrid import SPIRRID
from paper_pics.rf_rigid_mtrx import CBRigidMatrix, CBRigidMatrixSP

reinf1 = Reinforcement(r=0.00345,#RV('uniform', loc=0.001, scale=0.005),
                      tau=RV('uniform', loc=.5, scale=.2),
                      V_f=0.3,
                      E_f=70e3,
                      xi=WeibullFibers(shape=5., scale=0.02, L0=10.),#RV('weibull_min', shape=5., scale=.02),
                      n_int=15,
                      label='AR glass')

reinf2 = Reinforcement(r=0.003,#RV('uniform', loc=0.002, scale=0.002),
                      tau=0.5,#RV('uniform', loc=.5, scale=.1),
                      V_f=0.1,
                      E_f=200e3,
                      xi=RV('weibull_min', shape=5., scale=.02),
                      n_int=15,
                      label='carbon')

model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf2],
                             Ll=1000.,
                             Lr=1000.)

ccb_view = CompositeCrackBridgeView(model=model)

r = 0.003
tau = 0.5
V_f = 0.1
E_f = 200e3
xi = 100.

def rigid_mtrx():
    cb = CBRigidMatrix()
    w_arr = np.linspace(0.0, 0.5, 100)
    plt.plot(w_arr, E_f * V_f * cb(w_arr, r, tau, E_f, V_f, xi), lw=2, label='3.1.1')
    plt.plot(w_arr, E_f * V_f * cb(w_arr, r, tau, E_f, V_f, 0.02), lw=2, label='3.1.2')
    spirrid1 = SPIRRID(q=cb, sampling_type='PGrid',
                      evars=dict(w=w_arr),
                      tvars=dict(r=r, tau=tau, E_f=E_f, V_f=V_f,
                                 xi=RV('weibull_min', shape=5.0, scale=0.02)),
                      n_int=500)
    plt.plot(w_arr, E_f * V_f * spirrid1.mu_q_arr, lw=2, label='3.1.3')
    spirrid2 = SPIRRID(q=cb, sampling_type='PGrid',
                      evars=dict(w=w_arr),
                      tvars=dict(r=RV('uniform', loc=0.002, scale=0.002),
                                 tau=RV('uniform', loc=0.3, scale=0.4),
                                 E_f=E_f, V_f=V_f, xi=xi),
                      n_int=50)
    plt.plot(w_arr, E_f * V_f * spirrid2.mu_q_arr, lw=2, label='3.1.4')
    spirrid3 = SPIRRID(q=cb, sampling_type='PGrid',
                      evars=dict(w=w_arr),
                      tvars=dict(r=RV('uniform', loc=0.002, scale=0.002),
                                 tau=RV('uniform', loc=0.3, scale=0.4),
                                 E_f=E_f, V_f=V_f, xi=0.02),
                      n_int=500)
    plt.plot(w_arr, E_f * V_f * spirrid3.mu_q_arr, lw=2, label='3.1.5')
    spirrid4 = SPIRRID(q=cb, sampling_type='PGrid',
                      evars=dict(w=w_arr),
                      tvars=dict(r=RV('uniform', loc=0.002, scale=0.002),
                                 tau=RV('uniform', loc=0.3, scale=0.4),
                                 E_f=E_f, V_f=V_f,
                                 xi=RV('weibull_min', shape=5.0, scale=0.02)),
                      n_int=20)
    plt.plot(w_arr, E_f * V_f * spirrid4.mu_q_arr, lw=2, label='3.1.6')
    plt.xlabel('w [mm]')
    plt.ylabel('$\sigma_c$ [MPa]')

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

def sigma_c_w(w_arr):
    reinf = Reinforcement(r=0.003, tau=0.5, V_f=0.1, E_f=200e3, xi=100., n_int=15, label='carbon')
    model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf],
                             Ll=1000.,
                             Lr=1000.)
    ccb_view = CompositeCrackBridgeView(model=model)
    sigma_c = []
    for w in w_arr:
        ccb_view.model.w = w
        sigma_c.append(ccb_view.sigma_c)
    plt.plot(w_arr, sigma_c, label='3.2.1')

    reinf = Reinforcement(r=0.003, tau=0.5, V_f=0.1, E_f=200e3, xi=0.02, n_int=15, label='carbon')
    model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf],
                             Ll=1000.,
                             Lr=1000.)
    ccb_view = CompositeCrackBridgeView(model=model)
    sigma_c = []
    for w in w_arr:
        ccb_view.model.w = w
        sigma_c.append(ccb_view.sigma_c)
    plt.plot(w_arr, sigma_c, label='3.2.2')

    reinf = Reinforcement(r=0.003, tau=0.5, V_f=0.1, E_f=200e3,
                          xi=RV('weibull_min', shape=5.0, scale=0.02),
                          n_int=50, label='carbon')
    model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf],
                             Ll=1000.,
                             Lr=1000.)
    ccb_view = CompositeCrackBridgeView(model=model)
    sigma_c = []
    for w in w_arr:
        ccb_view.model.w = w
        sigma_c.append(ccb_view.sigma_c)
    plt.plot(w_arr, sigma_c, label='3.2.3')

    reinf = Reinforcement(r=RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=0.3, scale=0.4),
                          V_f=0.1, E_f=200e3,
                          xi=100.,
                          n_int=10, label='carbon')
    model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf],
                             Ll=1000.,
                             Lr=1000.)
    ccb_view = CompositeCrackBridgeView(model=model)
    sigma_c = []
    for w in w_arr:
        ccb_view.model.w = w
        sigma_c.append(ccb_view.sigma_c)
    plt.plot(w_arr, sigma_c, label='3.2.4')

    reinf = Reinforcement(r=RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=0.3, scale=0.4),
                          V_f=0.1, E_f=200e3,
                          xi=RV('weibull_min', shape=70.0, scale=0.02),
                          n_int=10, label='carbon')
    model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf],
                             Ll=1000.,
                             Lr=1000.)
    ccb_view = CompositeCrackBridgeView(model=model)
    sigma_c = []
    for w in w_arr:
        ccb_view.model.w = w
        sigma_c.append(ccb_view.sigma_c)
    plt.plot(w_arr, sigma_c, label='3.2.5')

    reinf = Reinforcement(r=RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=0.3, scale=0.4),
                          V_f=0.1, E_f=200e3,
                          xi=RV('weibull_min', shape=5.0, scale=0.02),
                          n_int=10, label='carbon')
    model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf],
                             Ll=1000.,
                             Lr=1000.)
    ccb_view = CompositeCrackBridgeView(model=model)
    sigma_c = []
    for w in w_arr:
        ccb_view.model.w = w
        sigma_c.append(ccb_view.sigma_c)
    plt.plot(w_arr, sigma_c, label='3.2.6') 

def sigma_f(w_arr):
    sf_arr = ccb_view.sigma_f_lst(w_arr)
    #for i, reinf in enumerate(ccb_view.model.reinforcement_lst):
    #    plt.plot(w_arr, sf_arr[:, i], label=reinf.label)
    plt.plot(w_arr, sf_arr[:,0], ls='dashed', label='$\sigma_{\mathrm{f}1}$', color='black')
    plt.plot(w_arr, sf_arr[:,1], ls='dotted', label='$\sigma_{\mathrm{f}2}$', color='black')
    plt.xlabel('crack opening w [mm]')
    plt.ylabel('stress [MPa]')
    plt.ylim(0)

def errors(k_ratio):
    E_m = E_f*V_f/k_ratio/(1-V_f)
    xi = RV('weibull_min', shape=5., scale=.02)
    reinf = Reinforcement(r=0.003, tau=0.5, V_f=0.3, E_f=200e3, xi=xi, n_int=15, label='carbon')
    model = CompositeCrackBridge(E_m=E_m,
                             reinforcement_lst=[reinf],
                             Ll=1000.,
                             Lr=1000.)
    ccb_view = CompositeCrackBridgeView(model=model)
    sigma_c = []
    w_arr = np.linspace(0.0, 0.5, 50)
    for w in w_arr:
        ccb_view.model.w = w
        sigma_c.append(ccb_view.sigma_c)
    plt.plot(w_arr, sigma_c)

#profile(.03)
#sigma_c_w(np.linspace(.0, .5, 100))
#sigma_f(np.linspace(.0, .3, 100))
#rigid_mtrx()
errors(0.5)
plt.legend(loc='best')
plt.show()
  
#random_domain(0.2)