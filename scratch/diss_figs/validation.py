'''
Created on 16 Feb 2014

@author: Q
'''

import numpy as np
from matplotlib import pyplot as plt
from quaducom.meso.homogenized_crack_bridge.rigid_matrix.CB_view import CBView, Model
from etsproxy.traits.api import \
    Instance, Array, List, cached_property, Property
from matplotlib import pyplot as plt
from etsproxy.traits.ui.api import ModelView
from spirrid.rv import RV
from stats.misc.random_field.random_field_1D import RandomField
import numpy as np
import copy
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_model import SCM
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_view import SCMView
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import Reinforcement, ContinuousFibers
from stats.pdistrib.weibull_fibers_composite_distr import WeibullFibers
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx import CompositeCrackBridge
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx_view import CompositeCrackBridgeView

def CB():
    model = Model(w_min=0.0, w_max=8.0, w_pts=100,
                  w2_min=0.0, w2_max=.5, w2_pts=3,
                  sV0=0.00383, m=7.0, tau_scale=0.3,
                  tau_shape=0.2, tau_loc=0.01, Ef=180e3,
                  lm=20., n_int=100)
    
    i = 0
    cb = np.loadtxt("CB" + str(i+1) + ".txt", delimiter=';')
    model.test_xdata = -cb[:,2]/4. - cb[:,3]/4. - cb[:,4]/2.
    model.test_ydata = cb[:,1] / (11. * 0.445) * 1000

    i = 1
    cb = np.loadtxt("CB" + str(i+1) + ".txt", delimiter=';')
    model.test_xdata2 = -cb[:,2]/4. - cb[:,3]/4. - cb[:,4]/2.
    model.test_ydata2 = cb[:,1] / (11. * 0.445) * 1000

    i = 2
    cb = np.loadtxt("CB" + str(i+1) + ".txt", delimiter=';')
    model.test_xdata3 = -cb[:,2]/4. - cb[:,3]/4. - cb[:,4]/2.
    model.test_ydata3 = cb[:,1] / (11. * 0.445) * 1000
    
    i = 3
    cb = np.loadtxt("CB" + str(i+1) + ".txt", delimiter=';')
    model.test_xdata4 = -cb[:,2]/4. - cb[:,3]/4. - cb[:,4]/2.
    model.test_ydata4 = cb[:,1] / (11. * 0.445) * 1000
    
    i = 4
    cb = np.loadtxt("CB" + str(i+1) + ".txt", delimiter=';')
    model.test_xdata5 = -cb[:,2]/4. - cb[:,3]/4. - cb[:,4]/2.
    model.test_ydata5 = cb[:,1] / (11. * 0.445) * 1000
    
    cb = CBView(model=model)
    cb.refresh()
    cb.configure_traits()
    
    # for i in range(5):
    #     data = np.loadtxt("CB" + str(i+1) + ".txt", delimiter=';')
    #     plt.plot(-data[:,2]/4. - data[:,3]/4. - data[:,4]/2.,data[:,1], lw=2, label="CB" + str(i+1))
    # 
    # plt.legend()
    # plt.show()
    
def TT():
    for i in range(5):
        data = np.loadtxt("TT-4C-0" + str(i+1) + ".txt", delimiter=';')
        plt.plot(-data[:,2]/2./250. - data[:,3]/2./250.,data[:,1]/2., lw=1, color='blue')
        data = np.loadtxt("TT-6C-0" + str(i+1) + ".txt", delimiter=';')
        plt.plot(-data[:,2]/2./250. - data[:,3]/2./250.,data[:,1]/2., lw=1, color='red')
    plt.legend(loc='best')

def valid():
    from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import ContinuousFibers
    from stats.pdistrib.weibull_fibers_composite_distr import WeibullFibers, fibers_MC
    length = 500.
    nx = 3000
    tau_scale = .8
    tau_shape = 0.18
    tau_loc = 0.0055
    xi_shape = 7.
    xi_scale = 0.0042
    random_field = RandomField(seed=False,
                               lacor=1.,
                               length=length,
                               nx=700,
                               nsim=1,
                               loc=.0,
                               shape=80.,
                               scale=3.4,
                               distr_type='Weibull'
                               )

    reinf = ContinuousFibers(r=3.5e-3,
                              tau=RV('gamma', loc=tau_loc, scale=tau_scale, shape=tau_shape),
                              V_f=0.01,
                              E_f=200e3,
                              xi=fibers_MC(m=xi_shape, sV0=xi_scale),
                              label='carbon',
                              n_int=500)

    CB_model = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf],
                                 )

    scm = SCM(length=length,
              nx=nx,
              random_field=random_field,
              CB_model=CB_model,
              load_sigma_c_arr=np.linspace(0.01, 17., 100),
              )

    scm_view = SCMView(model=scm)
    scm_view.model.evaluate()
    eps, sigma = scm_view.eps_sigma
    plt.plot(eps, sigma, color='black', lw=2,
             label='cracks1.0: ' + str(len(scm.cracks_list[-1]))
             + ' scale ' + str(tau_scale)
             + ' shape ' + str(tau_shape)
             + ' loc ' + str(tau_loc))
    TT()
    
    random_field = RandomField(seed=False,
                               lacor=1.,
                               length=length,
                               nx=700,
                               nsim=1,
                               loc=.0,
                               shape=80.,
                               scale=1.3 * 3.4,
                               distr_type='Weibull'
                               )
   
    reinf = ContinuousFibers(r=3.5e-3,
                              tau=RV('gamma', loc=tau_loc, scale=tau_scale, shape=tau_shape),
                              V_f=0.015,
                              E_f=200e3,
                              xi=fibers_MC(m=xi_shape, sV0=xi_scale),
                              label='carbon',
                              n_int=500)
   
    CB_model = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf],
                                 )
   
    scm = SCM(length=length,
              nx=nx,
              random_field=random_field,
              CB_model=CB_model,
              load_sigma_c_arr=np.linspace(0.01, 27., 100),
              )
   
    scm_view = SCMView(model=scm)
    scm_view.model.evaluate()
    eps, sigma = scm_view.eps_sigma
    plt.plot(eps, sigma, color='black', lw=2, label='cracks: ' + str(len(scm.cracks_list[-1])))    
    plt.xlabel('composite strain [-]')
    plt.ylabel('composite stress [MPa]')
    plt.legend(loc='best')
    plt.show()

#CB()
#TT()
valid()