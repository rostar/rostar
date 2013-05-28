'''
Created on 9 May 2013

@author: Q
'''
'''
Created on Nov 19, 2012

@author: rostar
'''

from etsproxy.mayavi import mlab as m
import numpy as np
from scipy.stats import weibull_min
from stats.spirrid import make_ogrid as orthogonalize
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx import CompositeCrackBridge
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx_view import CompositeCrackBridgeView
from matplotlib import pyplot as plt
from spirrid.rv import RV
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import Reinforcement, WeibullFibers

reinf = Reinforcement(r=0.01,#RV('uniform', loc=0.001, scale=0.005),
                      tau=0.1,
                      V_f=0.01,
                      E_f=200e3,
                      xi=1000.,
                      n_int=50,
                      label='reinforcement')

model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf],
                             Ll=np.infty,
                             Lr=np.infty)

ccb_view = CompositeCrackBridgeView(model=model)
w_arr = np.linspace(0.0, 1.0, 200)


def ld_rigid_vs_el_mtrx():
    w_arr = np.linspace(0.0, 0.8, 200)
    r = 0.003
    tau = 0.3
    V_f = 0.11
    E_f = 200e3
    n_int = 30
    E_m = 25e3
    xi = 0.02
    Xxi = RV('weibull_min', shape=5., scale=.02)
    Xr = RV('uniform', loc=.002, scale=.002)
    Xtau = RV('uniform', loc=.02, scale=.98)
    reinf1 = Reinforcement(r=r, tau=tau, V_f=V_f, E_f=E_f, xi=xi, n_int=n_int)
    reinf2 = Reinforcement(r=r, tau=tau, V_f=V_f, E_f=E_f, xi=Xxi, n_int=100)
    reinf3 = Reinforcement(r=Xr, tau=Xtau, V_f=V_f, E_f=E_f, xi=Xxi, n_int=15)
    model1 = CompositeCrackBridge(E_m=E_m, reinforcement_lst=[reinf1], Ll=35., Lr=35.)
    model2 = CompositeCrackBridge(E_m=E_m, reinforcement_lst=[reinf2], Ll=35., Lr=35.)
    model3 = CompositeCrackBridge(E_m=E_m, reinforcement_lst=[reinf3], Ll=35., Lr=35.)
    ccb_view1 = CompositeCrackBridgeView(model=model1)
    ccb_view2 = CompositeCrackBridgeView(model=model2)
    ccb_view3 = CompositeCrackBridgeView(model=model3)
    sigma_c1, u1, sigma_c2, u2, sigma_c3, u3 = [], [], [], [], [], []
    for w in w_arr:
        ccb_view1.model.w = w/2
        ccb_view2.model.w = w
        ccb_view3.model.w = w
        u1.append(ccb_view1.u_evaluated)
        sigma_c1.append(ccb_view1.sigma_c)
        u2.append(ccb_view2.u_evaluated)
        sigma_c2.append(ccb_view2.sigma_c)
        u3.append(ccb_view3.u_evaluated)
        sigma_c3.append(ccb_view3.sigma_c)
    plt.plot(w_arr / 2., sigma_c1, lw = 2, color='black', label='w')
    plt.plot(w_arr, sigma_c2, lw = 2, color='black')
    plt.plot(w_arr, sigma_c3, lw = 2, color='black')
    plt.plot(u1, sigma_c1, ls='dashed', lw = 2, color='black', label='u')
    plt.plot(u2, sigma_c2, ls='dashed', lw = 2, color='black')
    plt.plot(u3, sigma_c3, ls='dashed', lw = 2, color='black')
    plt.xlabel('w, u [mm]')
    plt.ylabel('$\sigma_c$ [MPa]')

def profiles_rigid_vs_el_mtrx():
    r = 0.003
    tau = 0.3
    V_f = 0.11
    E_f = 200e3
    E_m = 25e10
    Ll = 35.
    Lr = 35.
    Xxi = RV('weibull_min', shape=5., scale=.02)
    Xr = RV('uniform', loc=.002, scale=.002)
    Xtau = RV('uniform', loc=.02, scale=.98)
    reinf2 = Reinforcement(r=Xr, tau=Xtau, V_f=V_f, E_f=E_f, xi=Xxi, n_int=15)
    model2 = CompositeCrackBridge(E_m=E_m, reinforcement_lst=[reinf2], Ll=Ll, Lr=Lr)
    ccb_view2 = CompositeCrackBridgeView(model=model2)

    ccb_view2.sigma_c_max
    x2 = np.hstack((-Ll, ccb_view2.x_arr, Lr))
    mu_epsf2 = np.hstack((ccb_view2.mu_epsf_arr[0], ccb_view2.mu_epsf_arr, ccb_view2.mu_epsf_arr[-1]))
    epsm2 = np.hstack((ccb_view2.epsm_arr[0], ccb_view2.epsm_arr, ccb_view2.epsm_arr[-1]))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x2, mu_epsf2, lw=2, color='black')
    ax.plot(x2, epsm2, lw=2, color='black')
    p = ax.fill_between(x2, epsm2, mu_epsf2, facecolor='none')
    from matplotlib.patches import PathPatch
    path = p.get_paths()[0]
    p1 = PathPatch(path, fc="none", hatch="/")
    ax.add_patch(p1)
    p1.set_zorder(p.get_zorder()-0.1)
    p = ax.fill_between(x2, mu_epsf2, facecolor='none')
    path = p.get_paths()[0]
    p1 = PathPatch(path, fc="none", hatch="\\")
    ax.add_patch(p1)
    p1.set_zorder(p.get_zorder()-0.1)
    
    reinf1 = Reinforcement(r=r, tau=tau, V_f=V_f, E_f=E_f, xi=Xxi, n_int=15)
    model1 = CompositeCrackBridge(E_m=E_m, reinforcement_lst=[reinf1], Ll=Ll, Lr=Lr)
    ccb_view1 = CompositeCrackBridgeView(model=model1)
    ccb_view1.sigma_c_max
    x1 = np.hstack((-Ll, ccb_view1.x_arr, Lr))
    mu_epsf1 = np.hstack((ccb_view1.mu_epsf_arr[0], ccb_view1.mu_epsf_arr, ccb_view1.mu_epsf_arr[-1]))
    epsm1 = np.hstack((ccb_view1.epsm_arr[0], ccb_view1.epsm_arr, ccb_view1.epsm_arr[-1]))
    ax.plot(x1, mu_epsf1, lw=2, color='black')
    ax.plot(x1, epsm1, lw=2, color='black')
    
    plt.xlabel('z [mm]')
    plt.xlim(-35, 35)
    plt.ylim(0)
    plt.ylabel('$\epsilon$ [-]')

def Vf_k(k):
    Em = ccb_view.model.E_m
    Ef = ccb_view.model.reinforcement_lst[0].E_f
    return -Em*k/(Ef*k-Em*k-Ef)

def sigma_c_arr():
    Vf = Vf_k(0.1)
    reinf = Reinforcement(r=0.01, tau=0.1, V_f=Vf, E_f=200e3, xi=1000., n_int=50)
    sigma = ccb_view.sigma_c_arr(w_arr)
    plt.plot(w_arr, sigma / Vf, lw=2)
    Vf = Vf_k(0.5)
    reinf = Reinforcement(r=0.01, tau=0.1, V_f=Vf, E_f=200e3, xi=1000., n_int=50)
    sigma = ccb_view.sigma_c_arr(w_arr)
    plt.plot(w_arr, sigma / Vf, lw=2)
    Vf = Vf_k(0.8)
    reinf = Reinforcement(r=0.01, tau=0.1, V_f=Vf, E_f=200e3, xi=1000., n_int=50)
    ccb_view.model.reinforcement_lst = [reinf]
    sigma = ccb_view.sigma_c_arr(w_arr)
    plt.plot(w_arr, sigma / Vf, lw=2)

sigma_c_arr()
#ld_rigid_vs_el_mtrx()
#profiles_rigid_vs_el_mtrx()
#elastic_matrix(np.linspace(.0, .5, 100))
#sigma_f(np.linspace(.0, .3, 100))
#rigid_mtrx()
plt.legend(loc='best')
plt.show()