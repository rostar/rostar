'''
Created on 25. 7. 2014

@author: admin
'''

from enthought.traits.api import HasTraits, Array, Instance, List, Float, Int, \
    Property, cached_property
from util.traits.either_type import EitherType
from types import FloatType
from spirrid.rv import RV
from stats.pdistrib.weibull_fibers_composite_distr import WeibullFibers
import numpy as np
from matplotlib import pyplot as plt
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx_old import CompositeCrackBridge
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx_view import CompositeCrackBridgeView
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement_old import Reinforcement, ContinuousFibers, FiberBundle
#from validation import TT
from stats.misc.random_field.random_field_1D import RandomField
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_model import SCM
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_view import SCMView
from stats.pdistrib.weibull_fibers_composite_distr import WeibullFibers, fibers_MC
from scipy.interpolate import interp1d
from numpy.linalg import solve, lstsq
from etsproxy.util.home_directory import get_home_directory
import os.path
from calibration import Calibration
from CB_model_tau_weights import RandomBondCB

def valid(tau_arr, tau_weights, sV0, m):
    length = 500.
    nx = 3000
    ld = True
    w_width = False
    w_density = False
    
    random_field1 = RandomField(seed=False,
                               lacor=1.,
                               length=length,
                               nx=700,
                               nsim=1,
                               loc=.0,
                               shape=50.,
                               scale=3.4,
                               distr_type='Weibull'
                               )

    reinf1 = FiberBundle(r=3.5e-3,
                        tau=tau_arr,
                        tau_weights=tau_weights,
                        V_f=0.01,
                        E_f=200e3,
                        xi=fibers_MC(m=m, sV0=sV0),
                        label='carbon',
                        n_int=500)
 
    CB_model1 = RandomBondCB(E_m=25e3,
                                 reinforcement_lst=[reinf1],
                                 )
 
    scm1 = SCM(length=length,
              nx=nx,
              random_field=random_field1,
              CB_model=CB_model1,
              load_sigma_c_arr=np.linspace(0.01, 17., 100),
              n_BC_CB=15
              )
 
    scm_view1 = SCMView(model=scm1)
    scm_view1.model.evaluate()
    eps1, sigma1 = scm_view1.eps_sigma


    random_field2 = RandomField(seed=False,
                               lacor=1.,
                               length=length,
                               nx=700,
                               nsim=1,
                               loc=.0,
                               shape=50.,
                               scale=1.3 * 3.4,
                               distr_type='Weibull'
                               )

    reinf2 = FiberBundle(r=3.5e-3,
                        tau=tau_arr,
                        tau_weights=tau_weights,
                        V_f=0.015,
                        E_f=200e3,
                        xi=fibers_MC(m=m, sV0=sV0),
                        label='carbon',
                        n_int=500)
 
    CB_model2 = RandomBondCB(E_m=25e3,
                                 reinforcement_lst=[reinf2],
                                 )
 
    scm2 = SCM(length=length,
              nx=nx,
              random_field=random_field2,
              CB_model=CB_model2,
              load_sigma_c_arr=np.linspace(0.01, 25., 100),
              n_BC_CB=15
              )
 
    scm_view2 = SCMView(model=scm2)
    scm_view2.model.evaluate()
    eps2, sigma2 = scm_view2.eps_sigma


    if ld == True:
        plt.plot(eps1, sigma1, color='black', lw=2, label='cracks: ' + str(len(scm1.cracks_list[-1])) + ' sV0 = ' + str(sV0) + ' m = ' + str(m))
        plt.plot(eps2, sigma2, color='black', lw=2, label='cracks: ' + str(len(scm2.cracks_list[-1])))    
        plt.xlabel('composite strain [-]')
        plt.ylabel('composite stress [MPa]')
        plt.legend(loc='best')

    if w_width == True:
        plt.figure()
        plt.plot(scm_view1.model.load_sigma_c_arr, scm_view1.w_mean)
        plt.plot(scm_view1.model.load_sigma_c_arr, scm_view1.w_max)
        plt.plot(scm_view2.model.load_sigma_c_arr, scm_view2.w_mean)
        plt.plot(scm_view2.model.load_sigma_c_arr, scm_view2.w_max)
          
    if w_density == True:
        plt.figure()
        plt.plot(scm_view1.model.load_sigma_c_arr, scm_view1.w_density)
        plt.plot(scm_view2.model.load_sigma_c_arr, scm_view2.w_density)   

if __name__ == '__main__':
    def TT():
        plt.figure()
        for i in range(5):
            data = np.loadtxt("TT-4C-0" + str(i+1) + ".txt", delimiter=';')
            plt.plot(-data[:,2]/2./250. - data[:,3]/2./250.,data[:,1]/2., lw=1, color='blue')
            data = np.loadtxt("TT-6C-0" + str(i+1) + ".txt", delimiter=';')
            plt.plot(-data[:,2]/2./250. - data[:,3]/2./250.,data[:,1]/2., lw=1, color='red')
        plt.legend(loc='best')
    TT()
     
    ### CALIBRATED TAU
    w_arr = np.linspace(0.0, np.sqrt(8.), 401) ** 2
 
#===============================================================================
# read experimental data
#===============================================================================
    home_dir = get_home_directory()
    path = [home_dir, 'git',  # the path of the data file
            'rostar',
            'scratch',
            'diss_figs',
            'CB1.txt']
    filepath = os.path.join(*path)
    exp_data = np.zeros_like(w_arr)
    file1 = open(filepath, 'r')
    cb = np.loadtxt(file1, delimiter=';')
    test_xdata = -cb[:, 2] / 4. - cb[:, 3] / 4. - cb[:, 4] / 2.
    test_ydata = cb[:, 1] / (11. * 0.445) * 1000
    interp = interp1d(test_xdata, test_ydata, bounds_error=False, fill_value=0.)
    exp_data = interp(w_arr)
 
    cali = Calibration(experi_data=exp_data,
                       w_arr=w_arr,
                       tau_arr=np.logspace(np.log10(1e-5), np.log10(1), 200))
  
    sV0 = cali.sV0
    tau_arr = cali.tau_arr
    tau_weights = cali.tau_weights
    
    valid(tau_arr, tau_weights, sV0, cali.m)

    reinf = FiberBundle(r=3.5e-3,
                         tau=tau_arr,
                         tau_weights=tau_weights,
                         V_f=0.01,
                         E_f=200e3,
                         xi=fibers_MC(m=9.0, sV0=sV0),
                         label='carbon',)
 
    model = RandomBondCB(E_m=25e3,
                         reinforcement_lst=[reinf],
                         Ll=100000.,
                         Lr=100000.,
                         )
 
    ccb_view = CompositeCrackBridgeView(model=model)
 
#     def sigma_c_w(w_arr):
#         sigma_c_arr, u_arr, damage_arr = ccb_view.sigma_c_arr(w_arr, damage=True, u=True)
#         plt.plot(w_arr, sigma_c_arr, lw=2, color='black', label='w-sigma')
#         plt.xlabel('w,u [mm]')
#         plt.ylabel('$\sigma_c$ [MPa]')
#         plt.legend(loc='best')
#  
#     w = np.linspace(0.0, 8., 200)
#     sigma_c_w(w)


    plt.show()


    