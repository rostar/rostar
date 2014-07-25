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
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx import CompositeCrackBridge
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx_view import CompositeCrackBridgeView
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import Reinforcement, ContinuousFibers
from validation import TT
from stats.misc.random_field.random_field_1D import RandomField
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_model import SCM
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_view import SCMView
from stats.pdistrib.weibull_fibers_composite_distr import WeibullFibers, fibers_MC


class FiberBundle(Reinforcement):
    
#===============================================================================
# Parameters
#===============================================================================        
    E_f = Float(180e3) # the elastic modulus of the fiber
    V_f = Float # volume fraction
#     tau = EitherType(klasses=[FloatType, Array, RV]) # bond stiffness
    tau = Array
    tau_weights =Array # the weights for tau, applicable when tau is an array
    xi = EitherType(klasses=[FloatType, RV, WeibullFibers]) # breaking strain
    r = EitherType(klasses=[FloatType, RV]) # fiber radius
    n_int = Int(10) # number of integration points
    
#===============================================================================
# Sampling
#===============================================================================        
    samples = Property(depends_on='r, V_f, E_f, xi, tau, n_int')
    @cached_property
    def _get_samples(self):
        if isinstance(self.tau, np.ndarray):
            tau_arr = self.tau
            stat_weights = self.tau_weights
        return 2*tau_arr/self.r/self.E_f, stat_weights, \
               np.ones_like(tau_arr), self.r*np.ones_like(tau_arr)
               
    depsf_arr = Property(depends_on='r, V_f, E_f, xi, tau, n_int')
    @cached_property
    def _get_depsf_arr(self):
        return self.samples[0]

    stat_weights = Property(depends_on='r, V_f, E_f, xi, tau, n_int')
    @cached_property
    def _get_stat_weights(self):
        return self.samples[1]

    nu_r = Property(depends_on='r, V_f, E_f, xi, tau, n_int')
    @cached_property
    def _get_nu_r(self):
        return self.samples[2]

    r_arr = Property(depends_on='r, V_f, E_f, xi, tau, n_int')
    @cached_property
    def _get_r_arr(self):
        return self.samples[3]

def valid(tau_arr, tau_weights, sV0):
    length = 500.
    nx = 3000
    tau_scale = 3.3
    tau_shape = 1110.17
    tau_loc = 0.005
    xi_shape = 9.0
    xi_scale = 0.01
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
                              tau=RV('weibull_min', loc=0.002, scale=.1, shape=1.5),
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

    if ld == True:
        plt.figure()
        plt.plot(eps1, sigma1, color='black', lw=2)
        TT()
        #plt.plot(eps2, sigma2, color='black', lw=2, label='cracks: ' + str(len(scm2.cracks_list[-1])))    
        plt.xlabel('composite strain [-]')
        plt.ylabel('composite stress [MPa]')
        plt.legend(loc='best')

#     if w_width == True:
#         plt.figure()
#         plt.plot(scm_view1.model.load_sigma_c_arr, scm_view1.w_mean)
#         plt.plot(scm_view1.model.load_sigma_c_arr, scm_view1.w_max)
#         plt.plot(scm_view2.model.load_sigma_c_arr, scm_view2.w_mean)
#         plt.plot(scm_view2.model.load_sigma_c_arr, scm_view2.w_max)
#          
#     if w_density == True:
#         plt.figure()
#         plt.plot(scm_view1.model.load_sigma_c_arr, scm_view1.w_density)
#         plt.plot(scm_view2.model.load_sigma_c_arr, scm_view2.w_density)   

if __name__ == '__main__':
    valid(np.linspace(1.0, 1.5, 10),
          np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
          10.5)
    plt.show()
    
    
    reinf = FiberBundle(r=0.0035,
                      tau=np.array([0.5]),
                      tau_weights = np.array([1.0]),
                      V_f=0.01,
                      E_f=240e3,
                      xi=fibers_MC(m=9.0, sV0=0.02))

    model = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf],
                                 Ll=1000.,
                                 Lr=1000.,
                                 )

    ccb_view = CompositeCrackBridgeView(model=model)

    def sigma_c_w(w_arr):
        sigma_c_arr, u_arr, damage_arr = ccb_view.sigma_c_arr(w_arr, damage=True, u=True)
        plt.plot(w_arr, sigma_c_arr, lw=2, color='black', label='w-sigma')
        plt.plot(w_arr, damage_arr, lw=2, color='red', label='damage')
        plt.plot(u_arr, sigma_c_arr, lw=2, label='u-sigma')
        plt.plot(ccb_view.sigma_c_max[1], ccb_view.sigma_c_max[0], 'bo')
        plt.xlabel('w,u [mm]')
        plt.ylabel('$\sigma_c$ [MPa]')
        plt.legend(loc='best')


    # TODO: check energy for combined reinf
    # energy(np.linspace(.0, .15, 100))
#    sigma_c = np.linspace(1., 7., 7)
    # profile(0.031)
    w = np.linspace(0.0, 5., 200)
    sigma_c_w(w)
    # energy(w)
    # bundle at 20 mm
    # sigma_bundle = 70e3*w/20.*np.exp(-(w/20./0.03)**5.)
    # plt.plot(w,sigma_bundle)
    # plt.plot(ccb_view.sigma_c_max[1], ccb_view.sigma_c_max[0], 'bo')
    # sigma_f(np.linspace(.0, .16, 50))
    plt.legend(loc='best')
    plt.show()


    