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
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import Reinforcement, WeibullFibers, ContinuousFibers

reinf = ContinuousFibers(r=0.01, tau=0.1, V_f=0.01, E_f=200e3, xi=1000., n_int=50)
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


def k_influence(k_lst, mu_r, mu_tau, COV_r_lst, COV_tau_lst):
    k_arr = np.linspace(0.001, 0.8, 50)
    for i, cov_r in enumerate(COV_r_lst):
        epsf_max_lst = []
        loc_r = mu_r * (1 - cov_r * np.sqrt(3.0))
        scale_r = cov_r * 2 * np.sqrt(3.0) * mu_r
        loc_tau = mu_tau * (1 - COV_tau_lst[i] * np.sqrt(3.0))
        scale_tau = COV_tau_lst[i] * 2 * np.sqrt(3.0) * mu_tau
        for ki in k_arr:
            Vf = Vf_k(ki)
            ccb_view.model.w = 0.5
            reinf = ContinuousFibers(r=RV('uniform', loc=loc_r, scale=scale_r),
                      tau=RV('uniform', loc=loc_tau, scale=scale_tau),
                      V_f=Vf, E_f=200e3, xi=100.,#WeibullFibers(shape=5., sV0=0.003),
                      n_int=200)
            ccb_view.model.reinforcement_lst = [reinf]
            epsf_max_lst.append(np.max(ccb_view.mu_epsf_arr))
        plt.plot(k_arr, np.array(epsf_max_lst) * reinf.E_f)
    plt.ylim(0, 3000)
    plt.xlim(0, 0.8)

def sigma_c_arr(k_lst, mu_r, mu_tau, COV_r_lst, COV_tau_lst):
    w_arr = np.linspace(0.0,1.0,200)
    for ki in k_lst:
        Vf = Vf_k(ki)
        for i, cov_r in enumerate(COV_r_lst):
            loc_r = mu_r * (1 - cov_r * np.sqrt(3.0))
            scale_r = cov_r * 2 * np.sqrt(3.0) * mu_r
            loc_tau = mu_tau * (1 - COV_tau_lst[i] * np.sqrt(3.0))
            scale_tau = COV_tau_lst[i] * 2 * np.sqrt(3.0) * mu_tau
            reinf = ContinuousFibers(r=RV('uniform', loc=loc_r, scale=scale_r),
                          tau=RV('uniform', loc=loc_tau, scale=scale_tau),
                          V_f=Vf, E_f=200e3, xi=WeibullFibers(shape=20., sV0=0.003),
                          n_int=50)
            ccb_view.model.reinforcement_lst = [reinf]
            #sig_max, wmax = ccb_view.sigma_c_max
            #ccb_view.model.w = wmax
            sigma = ccb_view.sigma_c_arr(w_arr)
            plt.plot(w_arr, sigma / Vf, lw=2, label=str(ki))


def profiles(k_lst, mu_r, mu_tau, COV_r_lst, COV_tau_lst):
    for ki in k_lst:
        Vf = Vf_k(ki)
        ccb_view.model.w = 0.5
        for i, cov_r in enumerate(COV_r_lst):
            loc_r = mu_r * (1 - cov_r * np.sqrt(3.0))
            scale_r = cov_r * 2 * np.sqrt(3.0) * mu_r
            loc_tau = mu_tau * (1 - COV_tau_lst[i] * np.sqrt(3.0))
            scale_tau = COV_tau_lst[i] * 2 * np.sqrt(3.0) * mu_tau
            reinf = ContinuousFibers(r=RV('uniform', loc=loc_r, scale=scale_r),
                          tau=RV('uniform', loc=loc_tau, scale=scale_tau),
                          V_f=Vf, E_f=200e3, xi=100.,#WeibullFibers(shape=5., sV0=0.003),
                          n_int=50)
            ccb_view.model.reinforcement_lst = [reinf]
            #sig_max, wmax = ccb_view.sigma_c_max
            #ccb_view.model.w = wmax
            x = ccb_view.x_arr[1:-1]
            epsm = ccb_view.epsm_arr[1:-1]
            plt.plot(x, epsm, lw=2)
            plt.xlim(0, 300)

from scipy.special import gamma
from math import pi
def k_influence_rand(k_arr):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for i, m in enumerate([4., 8., 30.]):
        sig_max_lst = []
        wmax_lst = []
        for ki in k_arr:
            Vf = Vf_k(ki)
            reinf = ContinuousFibers(r=RV('uniform', loc=.005, scale=.01),
                                  tau=RV('uniform', loc=.05, scale=.1),
                                  xi=WeibullFibers(shape=m, sV0=0.003),
                                  V_f=Vf, E_f=200e3, n_int=100)
            mur = reinf.r._distr.mean
            mutau = reinf.tau._distr.mean
            muxi = reinf.xi.mean(2*mutau/mur/reinf.E_f, mur)
            g = gamma(1. + 1./(1 + m))
            sV0 = ((muxi/g)**(m+1)*(pi*mur**3*reinf.E_f)/(mutau*(m+1)))**(1./m)
            reinf.xi.sV0 = sV0
            ccb_view.model.reinforcement_lst = [reinf]
            sig_max, wmax = ccb_view.sigma_c_max
            sig_max_lst.append(sig_max / Vf)
            wmax_lst.append(wmax)
        ax1.plot(k_arr, np.array(sig_max_lst)/sig_max_lst[0])
        ax2.plot(k_arr, np.array(wmax_lst)/wmax_lst[0], label=str(m))
    ax1.set_ylim(0)
    ax2.set_ylim(0)

from mayavi import mlab
from stats.spirrid import make_ogrid as orthogonalize
def k_COV_plots(k_arr, COV_arr):
    sig_max_arr = np.zeros((len(k_arr), len(COV_arr)))
    wmax_arr = np.zeros((len(k_arr), len(COV_arr)))
    mu_r, mu_tau = 0.01, 0.1
    for i, k in enumerate(k_arr):
        for j, cov in enumerate(COV_arr):
            Vf = Vf_k(k)
            loc_r = mu_r * (1 - cov * np.sqrt(3.0))
            scale_r = cov * 2 * np.sqrt(3.0) * mu_r
            loc_tau = mu_tau * (1 - cov * np.sqrt(3.0))
            scale_tau = cov * 2 * np.sqrt(3.0) * mu_tau
            reinf = ContinuousFibers(r=RV('uniform', loc=loc_r, scale=scale_r),
                                  tau=RV('uniform', loc=loc_tau, scale=scale_tau),
                                  xi=WeibullFibers(shape=7.0, sV0=0.003),
                                  V_f=Vf, E_f=200e3, n_int=50)
            ccb_view.model.reinforcement_lst = [reinf]
            sig_max, wmax = ccb_view.sigma_c_max
            sig_max_arr[i,j] = sig_max/Vf
            wmax_arr[i,j] = wmax
    ctrl_vars = orthogonalize([np.arange(len(k_arr)), np.arange(len(COV_arr))])
    print sig_max_arr
    print wmax_arr
    mlab.surf(ctrl_vars[0], ctrl_vars[1], sig_max_arr / np.max(sig_max_arr))
    mlab.surf(ctrl_vars[0], ctrl_vars[1], wmax_arr / np.max(wmax_arr))
    mlab.show()
    
    
#k_lst, mu_r, mu_tau, COV_r_lst, COV_tau_lst = [0.1, 0.3], 0.01, 0.1, [0.0001, 0.5], [0.0001, 0.5]
#profiles(k_lst, mu_r, mu_tau, COV_r_lst, COV_tau_lst)
#sigma_c_arr(k_lst, mu_r, mu_tau, COV_r_lst, COV_tau_lst)
#k_influence_rand(np.linspace(0.01, 0.9, 30))
#k_influence(k_lst, mu_r, mu_tau, COV_r_lst, COV_tau_lst)
k_COV_plots(np.linspace(0.001, 0.9, 15), np.linspace(0.001, 0.5, 15))
plt.legend(loc='best')
plt.show()