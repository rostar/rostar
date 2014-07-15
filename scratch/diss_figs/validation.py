'''
Created on 16 Feb 2014

@author: Q
'''

import numpy as np
from matplotlib import pyplot as plt
from quaducom.meso.homogenized_crack_bridge.rigid_matrix.CB_view import CBView, Model
from spirrid.rv import RV
from stats.misc.random_field.random_field_1D import RandomField
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_model import SCM
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_view import SCMView
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx import CompositeCrackBridge
from quaducom.micro.resp_func.CB_clamped_rand_xi import CBClampedRandXi
from spirrid.spirrid import SPIRRID
from scipy.optimize import minimize_scalar

def CB():
    model = Model(w_min=0.0, w_max=8.0, w_pts=100,
                  w2_min=0.0, w2_max=.3, w2_pts=50,
                  sV0=0.0094, m=9.0, tau_scale=0.3,
                  tau_shape=0.2, tau_loc=0.001, Ef=180e3,
                  lm=15., n_int=100)
    
    for i in range(5):
        cb = np.loadtxt("CB" + str(i+1) + ".txt", delimiter=';')
        model.test_xdata.append(-cb[:,2]/4. - cb[:,3]/4. - cb[:,4]/2.)
        model.test_ydata.append(cb[:,1] / (11. * 0.445) * 1000)
    
    cb = CBView(model=model)
    cb.refresh()
    cb.configure_traits()
    
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
    tau_scale = .3
    tau_shape = 0.17
    tau_loc = 0.005
    xi_shape = 9.0
    xi_scale = 0.005
    ld = True
    w_width = True
    w_density = True
    
    random_field1 = RandomField(seed=False,
                               lacor=1.,
                               length=length,
                               nx=700,
                               nsim=1,
                               loc=.0,
                               shape=80.,
                               scale=3.4,
                               distr_type='Weibull'
                               )
 
    reinf1 = ContinuousFibers(r=3.5e-3,
                              tau=RV('gamma', loc=tau_loc, scale=tau_scale, shape=tau_shape),
                              V_f=0.01,
                              E_f=200e3,
                              xi=fibers_MC(m=xi_shape, sV0=xi_scale),
                              label='carbon',
                              n_int=500)
 
    CB_model1 = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf1],
                                 )
 
    scm1 = SCM(length=length,
              nx=nx,
              random_field=random_field1,
              CB_model=CB_model1,
              load_sigma_c_arr=np.linspace(0.01, 17., 100),
              n_BC_CB=12
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
                               shape=80.,
                               scale=1.3 * 3.4,
                               distr_type='Weibull'
                               )
    
    reinf2 = ContinuousFibers(r=3.5e-3,
                              tau=RV('gamma', loc=tau_loc, scale=tau_scale, shape=tau_shape),
                              V_f=0.015,
                              E_f=200e3,
                              xi=fibers_MC(m=xi_shape, sV0=xi_scale),
                              label='carbon',
                              n_int=500)
    
    CB_model2 = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf2],
                                 )
    
    scm2 = SCM(length=length,
              nx=nx,
              random_field=random_field2,
              CB_model=CB_model2,
              load_sigma_c_arr=np.linspace(0.01, 27., 100),
              n_BC_CB=12
              )
    
    scm_view2 = SCMView(model=scm2)
    scm_view2.model.evaluate()
    eps2, sigma2 = scm_view2.eps_sigma

    if ld == True:
        plt.figure()
        plt.plot(eps1, sigma1, color='black', lw=2,
                 label='cracks1.0: ' + str(len(scm1.cracks_list[-1]))
                 + ' scale ' + str(tau_scale)
                 + ' shape ' + str(tau_shape)
                 + ' loc ' + str(tau_loc))
        TT()
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




def simplified():
    cb = CBClampedRandXi()
    spirrid = SPIRRID(q=cb, sampling_type='PGrid',
                  theta_vars=dict(tau=RV('gamma', loc=0.0055, scale=0.7, shape=0.2),
                                  E_f=200e3,
                                  V_f=0.01,
                                  r=0.00345,
                                  m=7.0,
                                  sV0=0.0042,
                                  lm=1000.),
                        n_int=200)
    def sigmac(w, lm):
        spirrid.eps_vars['w'] = np.array([w])
        spirrid.theta_vars['lm'] = lm
        sigma_c = spirrid.mu_q_arr / spirrid.theta_vars['r'] ** 2
        return sigma_c

    def maxsigma(lm):
        def minfunc(w):
            res = sigmac(w, lm)
            return -res * (w < 200.) + 1e-5 * w ** 2
        w_max = minimize_scalar(minfunc, bracket=(0.0001, 0.0002))
        return w_max.x, sigmac(w_max.x, lm) / spirrid.theta_vars['V_f']

    sigmaf = []
    w_lst = []
    lcs = 1./np.linspace(8.4, 300.0, 20)
    for lcsi in lcs:
        print lcsi
        wi, sigi = maxsigma(1./lcsi)
        sigmaf.append(sigi)
        w_lst.append(wi)
    plt.plot(lcs, sigmaf)
    plt.plot(1./13.7, 1201, 'ro')
    plt.plot(1./9.7, 1373, 'bo')
    plt.errorbar(1./13.7, 1201, 104.4)
    plt.errorbar(1./9.7, 1373, 36.4)
    plt.ylim(0)
    plt.figure()
    plt.plot(lcs, w_lst)
    plt.plot(1./13.7, 0.088, 'ro')
    plt.plot(1./9.7, 0.05, 'bo')
    plt.errorbar(1./13.7, 0.088, 0.003)
    plt.errorbar(1./9.7, 0.05, 0.005)
    plt.ylim(0)
    plt.show()

#simplified()
valid()
#CB()
#TT()
plt.show()

