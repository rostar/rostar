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
    r = np.linspace(0.001, 0.005, 100)
    tau = np.linspace(0., 1., 100)
    e_arr = orthogonalize([np.arange(len(r)), np.arange(len(tau))])

    tau = tau.reshape(1, len(tau))
    r = r.reshape(len(r), 1)
    eps0 = np.sqrt(w * 2 * tau / r / Ef)
    F = Fxi.cdf(eps0)
    m.surf(e_arr[0], e_arr[1], F*30)
    m.surf(e_arr[0], e_arr[1], eps0*300)
    m.show()

def elastic_matrix(w_arr):
#    reinf = Reinforcement(r=0.003, tau=0.5, V_f=0.1, E_f=200e3, xi=100., n_int=15, label='carbon')
#    model = CompositeCrackBridge(E_m=25e3,
#                             reinforcement_lst=[reinf],
#                             Ll=1000.,
#                             Lr=1000.)
#    ccb_view = CompositeCrackBridgeView(model=model)
#    sigma_c = []
#    for w in w_arr:
#        ccb_view.model.w = w
#        sigma_c.append(ccb_view.sigma_c)
#    plt.plot(w_arr, sigma_c, label='3.2.1')

    reinf = Reinforcement(r=0.003, tau=0.5, V_f=0.1, E_f=200e3, xi=0.02, n_int=1, label='carbon')
    model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf],
                             Ll=10.,
                             Lr=10.)
    ccb_view = CompositeCrackBridgeView(model=model)
    sigma_c = []
    u = []
    for w in w_arr:
        ccb_view.model.w = w
        u.append(ccb_view.u_evaluated)
        sigma_c.append(ccb_view.sigma_c)
    plt.plot(w_arr, sigma_c, label='3.2.2')
    plt.plot(u, sigma_c, label='3.2.2')
#
#    reinf = Reinforcement(r=0.003, tau=0.5, V_f=0.1, E_f=200e3,
#                          xi=RV('weibull_min', shape=5.0, scale=0.02),
#                          n_int=50, label='carbon')
#    model = CompositeCrackBridge(E_m=25e3,
#                             reinforcement_lst=[reinf],
#                             Ll=1000.,
#                             Lr=1000.)
#    ccb_view = CompositeCrackBridgeView(model=model)
#    sigma_c = []
#    for w in w_arr:
#        ccb_view.model.w = w
#        sigma_c.append(ccb_view.sigma_c)
#    plt.plot(w_arr, sigma_c, label='3.2.3')
#
#    reinf = Reinforcement(r=RV('uniform', loc=0.002, scale=0.002),
#                          tau=RV('uniform', loc=0.3, scale=0.4),
#                          V_f=0.1, E_f=200e3,
#                          xi=100.,
#                          n_int=10, label='carbon')
#    model = CompositeCrackBridge(E_m=25e3,
#                             reinforcement_lst=[reinf],
#                             Ll=1000.,
#                             Lr=1000.)
#    ccb_view = CompositeCrackBridgeView(model=model)
#    sigma_c = []
#    for w in w_arr:
#        ccb_view.model.w = w
#        sigma_c.append(ccb_view.sigma_c)
#    plt.plot(w_arr, sigma_c, label='3.2.4')
#
#    reinf = Reinforcement(r=RV('uniform', loc=0.002, scale=0.002),
#                          tau=RV('uniform', loc=0.3, scale=0.4),
#                          V_f=0.1, E_f=200e3,
#                          xi=RV('weibull_min', shape=70.0, scale=0.02),
#                          n_int=10, label='carbon')
#    model = CompositeCrackBridge(E_m=25e3,
#                             reinforcement_lst=[reinf],
#                             Ll=1000.,
#                             Lr=1000.)
#    ccb_view = CompositeCrackBridgeView(model=model)
#    sigma_c = []
#    for w in w_arr:
#        ccb_view.model.w = w
#        sigma_c.append(ccb_view.sigma_c)
#    plt.plot(w_arr, sigma_c, label='3.2.5')
#
#    reinf = Reinforcement(r=RV('uniform', loc=0.002, scale=0.002),
#                          tau=RV('uniform', loc=0.3, scale=0.4),
#                          V_f=0.1, E_f=200e3,
#                          xi=RV('weibull_min', shape=5.0, scale=0.02),
#                          n_int=10, label='carbon')
#    model = CompositeCrackBridge(E_m=25e3,
#                             reinforcement_lst=[reinf],
#                             Ll=1000.,
#                             Lr=1000.)
#    ccb_view = CompositeCrackBridgeView(model=model)
#    sigma_c = []
#    for w in w_arr:
#        ccb_view.model.w = w
#        sigma_c.append(ccb_view.sigma_c)
#    plt.plot(w_arr, sigma_c, label='3.2.6') 

def sigma_f(w_arr):
    sf_arr = ccb_view.sigma_f_lst(w_arr)
    #for i, reinf in enumerate(ccb_view.model.reinforcement_lst):
    #    plt.plot(w_arr, sf_arr[:, i], label=reinf.label)
    plt.plot(w_arr, sf_arr[:,0], ls='dashed', label='$\sigma_{\mathrm{f}1}$', color='black')
    plt.plot(w_arr, sf_arr[:,1], ls='dotted', label='$\sigma_{\mathrm{f}2}$', color='black')
    plt.xlabel('crack opening w [mm]')
    plt.ylabel('stress [MPa]')
    plt.ylim(0)

def sigma_c_w(w_arr, r, tau, E_f, E_m, V_f, xi, n_int):
    reinf = Reinforcement(r=r, tau=tau, E_f=E_f, V_f=V_f,
                          xi=xi, n_int=n_int)
    model = CompositeCrackBridge(E_m=E_m,
                             reinforcement_lst=[reinf],
                             Ll=1000.,
                             Lr=1000.)
    ccb_view = CompositeCrackBridgeView(model=model)
    sigma_c = []
    for w in w_arr:
        ccb_view.model.w = w
        sigma_c.append(ccb_view.sigma_c)
    plt.plot(w_arr, sigma_c, lw=2, label='elastic_mtrx')

    cb = CBRigidMatrix()
    spirrid = SPIRRID(q=cb, sampling_type='PGrid',
                      evars=dict(w=w_arr),
                      tvars=dict(r=r, tau=tau, E_f=E_f, V_f=V_f,
                                 xi=xi),
                      n_int=n_int)
    plt.plot(w_arr, E_f * V_f * spirrid.mu_q_arr, lw=2, label='rigid_mtrx')

def errors(k_ratio):
    sigmax_el = []
    wmax = []
    for k in k_ratio:
        E_m = E_f * V_f / k / (1 - V_f)
        xi = WeibullFibers(shape=5., scale=0.02, L0=10.)#RV('weibull_min', shape=50., scale=.02)
        r = 0.002#RV('uniform', loc=0.002, scale=.002)
        tau = RV('weibull_min', shape=80., scale=.5)
        reinf = Reinforcement(r=r, tau=tau, V_f=0.1, E_f=200e3, xi=xi, n_int=20, label='carbon')
        model = CompositeCrackBridge(E_m=E_m,
                                 reinforcement_lst=[reinf],
                                 Ll=100.,
                                 Lr=100.)
        ccb_view = CompositeCrackBridgeView(model=model)
        sigmax_el.append(ccb_view.sigma_c_max[0])
        wmax.append(ccb_view.sigma_c_max[1])
    ccb_view.model.E_m = 10e10
    rigid_sigma = ccb_view.sigma_c_max[0]
    rigid_w = ccb_view.sigma_c_max[1]

    plt.plot(k_ratio, np.array(sigmax_el)/rigid_sigma, label='$\sigma_\mathrm{c, max, el}/\sigma_\mathrm{c, max, rigid}$', lw=2, color='red')
    plt.ylabel('$\sigma_\mathrm{c, max, el}/\sigma_\mathrm{c, max, rigid}$ and $w_\mathrm{max, el} / w_\mathrm{max, rigid}$')
    plt.xlabel('$K_\mathrm{f}/K_\mathrm{m}$')
    plt.title('effect of matrix stiffness')
    plt.plot(k_ratio, np.array(wmax)/rigid_w, label='$w_\mathrm{max, el} / w_\mathrm{max, rigid}$', lw=2, color='blue')
    plt.ylim(0, 1.3)

def profile(wr, we, r, tau, E_f, E_m, V_f, xi, n_int):
    reinf = Reinforcement(r=r, tau=tau, E_f=E_f, V_f=V_f,
                          xi=xi, n_int=n_int)
    model = CompositeCrackBridge(w=we,
                                 E_m=E_m,
                                 reinforcement_lst=[reinf],
                                 Ll=1000., Lr=1000.)
    ccb_view = CompositeCrackBridgeView(model=model)
    plt.plot(ccb_view.x_arr, ccb_view.mu_epsf_arr, lw=2, label='el_mtrx_yarn')
    plt.plot(ccb_view.x_arr, ccb_view.epsm_arr, lw=2, label='el_mtrx_mtrx')

    cb = CBRigidMatrixSP()
    x_arr = np.hstack((-np.linspace(3.*ccb_view.x_arr[-1], 0., 100),
                        np.linspace(0.0, 3.*ccb_view.x_arr[-1],100)))
    spirrid = SPIRRID(q=cb, sampling_type='PGrid',
                      evars=dict(x=x_arr),
                      tvars=dict(w=wr, r=r, tau=tau, E_f=E_f, V_f=V_f, xi=xi),
                      n_int=n_int)
    plt.plot(x_arr, spirrid.mu_q_arr, lw=2, label='rigid_mtrx_yarn')
    plt.xlabel('position [mm]')
    plt.ylabel('strain')

def sigma_c_u(w_arr, r, tau, E_f, E_m, V_f, xi, n_int):
    reinf = Reinforcement(r=r, tau=tau, E_f=E_f, V_f=V_f,
                          xi=xi, n_int=n_int)
    model = CompositeCrackBridge(E_m=E_m,
                             reinforcement_lst=[reinf],
                             Ll=30.,
                             Lr=30.)
    ccb_view = CompositeCrackBridgeView(model=model)
    sigma_c = []
    u = []
    for w in w_arr:
        ccb_view.model.w = w
        u.append(ccb_view.u_evaluated)
        sigma_c.append(ccb_view.sigma_c)
    plt.plot(w_arr, sigma_c, lw=1, color='blue')
    plt.plot(u, sigma_c, lw=1, color='red')

    cb = CBRigidMatrix()
    spirrid = SPIRRID(q=cb, sampling_type='PGrid',
                      evars=dict(w=w_arr),
                      tvars=dict(r=r, tau=tau, E_f=E_f, V_f=V_f,
                                 xi=xi),
                      n_int=n_int)
    plt.plot(w_arr, E_f * V_f * spirrid.mu_q_arr, lw=2, color='black')

elastic_matrix(np.linspace(.0, .5, 100))
#sigma_f(np.linspace(.0, .3, 100))
#rigid_mtrx()
#plt.subplot(1,3,1)
#errors(np.linspace(0.000001, 2., 20))
#plt.legend(loc='best')
#plt.subplot(1,3,2)
#sigma_c_w(w_arr=np.linspace(.0, .3, 100),
#          r=0.002, tau=RV('weibull_min', shape=5., scale=.5),
#          E_f=200e3, E_m=25e3,
#          V_f=0.9, xi=RV('weibull_min', shape=5., scale=.02),
#          n_int=50)
#plt.legend(loc='best')
#plt.subplot(1,3,3)
#profile(wr=.087, we=0.067, r=0.002, tau=RV('weibull_min', shape=5., scale=.5),
#        E_f=200e3, E_m=25e3,
#        V_f=0.05, xi=RV('weibull_min', shape=5., scale=.02),
#        n_int=50)
#for em in np.linspace(5e3, 100e3, 4):
#    sigma_c_u(w_arr=np.linspace(.0, .5, 100),
#          r=0.002, tau=RV('weibull_min', shape=5., scale=.5),
#          E_f=200e3, E_m=em,
#          V_f=0.1, xi=RV('weibull_min', shape=5., scale=.02),
#          n_int=50)
plt.legend(loc='best')
plt.show()

#random_domain(0.15)