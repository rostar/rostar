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
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx import CompositeCrackBridge
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx_view import CompositeCrackBridgeView
from matplotlib import pyplot as plt
from spirrid.rv import RV
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import Reinforcement, WeibullFibers

reinf = Reinforcement(r=0.01, tau=0.1, V_f=0.01, E_f=200e3, xi=1000., n_int=50)
model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf],
                             Ll=np.infty,
                             Lr=np.infty)

ccb_view = CompositeCrackBridgeView(model=model)
w_arr = np.linspace(0.0, 1.0, 200)


def Vf_k(k):
    Em = ccb_view.model.E_m
    Ef = ccb_view.model.reinforcement_lst[0].E_f
    return -Em*k/(Ef*k-Em*k-Ef)

def sigma_c_arr(k):
    for ki in k:
        Vf = Vf_k(ki)
        reinf = Reinforcement(r=0.01, tau=0.1, V_f=Vf, E_f=200e3, xi=1000., n_int=50)
        ccb_view.model.reinforcement_lst = [reinf]
        sigma = ccb_view.sigma_c_arr(w_arr)
        plt.plot(w_arr, sigma / Vf, lw=2)


def profiles(k):
    for ki in k:
        Vf = Vf_k(ki)
        ccb_view.model.w = 0.5
        reinf = Reinforcement(r=0.01, tau=0.1, V_f=Vf, E_f=200e3, xi=1000., n_int=50)
        ccb_view.model.reinforcement_lst = [reinf]
        x = ccb_view.x_arr[1:-1]
        epsm = ccb_view.epsm_arr[1:-1]
        epsf = ccb_view.mu_epsf_arr[1:-1]
        plt.plot(x, epsm, lw=2)
        plt.plot(x, epsf, lw=2)


def k_influence(k_arr):
    epsf_max_lst = []
    for ki in k_arr:
        Vf = Vf_k(ki)
        ccb_view.model.w = 0.5
        reinf = Reinforcement(r=0.01, tau=0.1, V_f=Vf, E_f=200e3, xi=1000., n_int=50)
        ccb_view.model.reinforcement_lst = [reinf]
        epsf_max_lst.append(np.max(ccb_view.mu_epsf_arr))
    plt.plot(k_arr, np.array(epsf_max_lst) * reinf.E_f)
    plt.ylim(0)

def sigma_c_arr_rand(k):
    for ki in k:
        Vf = Vf_k(ki)
        reinf = Reinforcement(r=RV('uniform', loc=.005, scale=.01),
                              tau=RV('uniform', loc=.05, scale=.1),
                              V_f=Vf, E_f=200e3, xi=WeibullFibers(shape=30., sV0=0.003),
                              n_int=50)
        ccb_view.model.reinforcement_lst = [reinf]
        sigma = ccb_view.sigma_c_arr(w_arr)
        plt.plot(w_arr, sigma / Vf, lw=2)


def profiles_rand(k):
    for ki in k:
        Vf = Vf_k(ki)
        ccb_view.model.w = 0.5
        reinf = Reinforcement(r=RV('uniform', loc=.005, scale=.01),
                      tau=RV('uniform', loc=.001, scale=.2),
                      V_f=Vf, E_f=200e3, xi=WeibullFibers(shape=5., sV0=0.003),
                      n_int=50)
        ccb_view.model.reinforcement_lst = [reinf]
        sig_max, wmax = ccb_view.sigma_c_max
        ccb_view.model.w = wmax
        x = ccb_view.x_arr[1:-1]
        epsm = ccb_view.epsm_arr[1:-1]
        epsf = ccb_view.mu_epsf_arr[1:-1]
        plt.plot(x, epsm, lw=2)
        plt.plot(x, epsf, lw=2)


def k_influence_rand(k_arr):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    for i,m in enumerate([4., 8., 15.]):
        sig_max_lst = []
        wmax_lst = []
        for ki in k_arr:
            Vf = Vf_k(ki)
            print 'Vf =', Vf
            reinf = Reinforcement(r=RV('uniform', loc=.005, scale=.01),
                                  tau=RV('uniform', loc=.05, scale=.1),
                                  xi=WeibullFibers(shape=m, sV0=0.003),
                                  V_f=Vf, E_f=200e3, n_int=100)
            ccb_view.model.reinforcement_lst = [reinf]
            sig_max, wmax = ccb_view.sigma_c_max
            sig_max_lst.append(sig_max / Vf)
            wmax_lst.append(wmax)
        ax1.plot(k_arr, np.array(sig_max_lst)/sig_max_lst[0], lw=i+1)
        ax2.plot(k_arr, np.array(wmax_lst),lw=i+1)
    ax1.set_ylim(0)
    ax2.set_ylim(0)

#sigma_c_arr([0.1,0.3,0.5])
#profiles([0.1, 0.3, 0.5])
#k_influence(np.linspace(0.01, 0.9, 10))
#sigma_c_arr_rand([0.1, 0.3, 0.5])
#profiles_rand([0.1, 0.3, 0.5])
k_influence_rand(np.linspace(0.01, 0.8, 20))
plt.legend(loc='best')
plt.show()