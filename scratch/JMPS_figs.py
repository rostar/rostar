'''
Created on 15. 7. 2014

@author: admin
'''
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx_view import CompositeCrackBridge
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import ContinuousFibers
import numpy as np
from spirrid.rv import RV
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_model import SCM
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_view import SCMView
from stats.misc.random_field.random_field_1D import RandomField
import matplotlib.pyplot as plt
from scipy.special import gamma

def Gf_study():
    plt.plot([0.001, 0.05, 0.08, 0.1, 0.11, 0.12, 0.13], np.array([12., 12., 13., 13., 18., 23., 36.]) / 250., 'ro')
    plt.plot([0.001, 0.05, 0.08, 0.1, 0.11, 0.12, 0.13], np.array([12., 12., 13., 13., 18., 23., 36.]) / 250., color='red', lw=2, label='SCM 1.0%')
    plt.plot([0.001, 0.05, 0.08, 0.1, 0.11, 0.12], np.array([20., 20., 21., 23., 35., 40.]) / 250., 'bo')
    plt.plot([0.001, 0.05, 0.08, 0.1, 0.11, 0.12], np.array([20., 20., 21., 23., 35., 40.]) / 250., color='blue', lw=2, label='SCM 1.5%')
    plt.plot([0.0,0.2], np.array([0.075,0.075]),ls='dashed', color='red', label='DURAPACT < 2mm; Vf=1.0% c', lw=2)
    plt.plot([0.0,0.2], np.array([0.098,0.098]),ls='dashed', color='blue', label='DURAPACT < 2mm; Vf=1.5% c', lw=2)
    plt.text(0.025, 0.022, r'TRC < 0.4mm')
    plt.plot([0.03,0.05],[.02,.02],lw=2, color='black')
    plt.text(0.14, 0.022, r'normal concrete')
    plt.plot([0.12,0.2],[.02,.02],lw=2, color='black')
    plt.text(0.085, 0.01, r'mortar < 4mm')
    plt.plot([0.064,0.134],[.008,.008],lw=2, color='black')
    plt.legend(loc='best')
    plt.xlabel('Gf [N/mm]')
    plt.ylabel('final crack spacing [mm]')
    plt.ylim(0, 0.2)
    plt.show()

def sigmamu_shape_study():
    length = 2000.
    nx = 3000
    lacor = 1.0
    sigmamumin = 3.0
    reinf = ContinuousFibers(r=0.0035,
                          tau=RV('gamma', shape=0.2, scale=0.5),
                          V_f=0.01,
                          E_f=180e3,
                          xi=1000.,
                          label='carbon',
                          n_int=500)
    
    ccb = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf],
                                 )

    #for mm in [5., 10., 15., 20., 25., 30., 35.]:
    for mm in [5., 10., 15., 20., 30.]:
        fLc = (lacor/(lacor+length))**(1./mm)
        sm = sigmamumin / (fLc * gamma(1. + 1 / mm))
        random_field = RandomField(seed=True,
                               lacor=1.,
                               length=length,
                               nx=500,
                               nsim=1,
                               loc=.0,
                               shape=mm,
                               scale=sm,
                               distr_type='Weibull')
        
        scm = SCM(length=length,
                  nx=nx,
                  random_field=random_field,
                  CB_model=ccb,
                  load_sigma_c_arr=np.linspace(0.01, 20., 200),
                  n_BC_CB=15)
    
        scm_view = SCMView(model=scm)
        scm_view.model.evaluate()
        eps, sigma = scm_view.eps_sigma
        plt.plot(eps, sigma, label='mm = ' + str(mm) + ' sm = ' + str(sm))
    plt.xlabel('composite strain [-]')
    plt.ylabel('composite stress [MPa]')
    plt.legend(loc='best')
    plt.xlim(0)
    plt.ylim(0)
    plt.show()

def profiles():
    reinf = ContinuousFibers(r=0.0035,
                      tau=RV('uniform', loc=1., scale=3.0),
                      V_f=0.1,
                      E_f=200e3,
                      xi=RV('weibull_min', loc=0., shape=1000., scale=.01),
                      label='carbon',
                      n_int=30)

    ccb = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf],
                                 Ll=1.,
                                 Lr=1.,
                                 w=.0056)

#     ccb = CompositeCrackBridge(E_m=25e13,
#                                  reinforcement_lst=[reinf],
#                                  Ll=2.,
#                                  Lr=2.,
#                                  w=.0088)

    ccb.damage
    for i, depsf in enumerate(ccb.sorted_depsf):
        epsf_x = np.maximum(ccb._epsf0_arr[i] - depsf * np.abs(ccb._x_arr), ccb._epsm_arr)
        if i == 0:
            plt.plot(ccb._x_arr, epsf_x, color='black', label='fibers')
        else:
            plt.plot(ccb._x_arr, epsf_x, color='black')
    plt.plot(ccb._x_arr, ccb._epsm_arr, lw=2, color='blue', label='matrix')
    plt.legend(loc='best')
    plt.ylabel('matrix and fiber strain [-]')
    plt.ylabel('long. position [mm]')
    plt.show()

from quaducom.micro.resp_func.CB_clamped_rand_xi import CBClampedRandXi
from spirrid.spirrid import SPIRRID
from spirrid.rv import RV

def mechanisms():
    m = 5.15004407
    sV0 = 0.00915595
    r = 3.5e-3
    Ef = 181e3

    
    cb = CBClampedRandXi(pullout=False)
    spirrid = SPIRRID(q=cb,
                      sampling_type='LHS',
                      theta_vars=dict(E_f=Ef,
                                      sV0=sV0,
                                      V_f=1.0,
                                      r=r,
                                      m=m,
                                      tau=RV('gamma', shape=0.03684979, scale=3.75278102, loc=0.0),
                                      lm=1000.),
                      n_int=100,
                      )
    
    lm_arr = np.linspace(1000., 3., 20)
    sigma_u_hommech = []
    for lm_i in lm_arr:
        spirrid.theta_vars['lm'] = lm_i
        max_w = min(30., lm_i * 0.07)
        w_arr = np.linspace(0.0, max_w, 200)
        spirrid.eps_vars = dict(w=w_arr)
        sig_w = spirrid.mu_q_arr / r ** 2
        plt.plot(w_arr, sig_w)
        plt.show()
        sigma_u_hommech.append(np.max(sig_w))
    sigma_u_hommech = np.array(sigma_u_hommech)
    plt.plot(lm_arr, sigma_u_hommech)#/np.min(sigma_u_hommech))
    plt.xlabel('crack spacing')
    plt.ylabel('normalized strength')
    plt.show()   

if __name__ == '__main__':
    
    #profiles()
    #sigmamu_shape_study()
    mechanisms()