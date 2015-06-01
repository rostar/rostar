'''
Created on Feb 18, 2015
 
@author: rostislavrypl
'''
 
from etsproxy.traits.api import HasTraits, Array, Float, Instance, Property, cached_property, \
                                Int, Any
from quaducom.micro.resp_func.cb_short_fiber import CBShortFiber
from spirrid.spirrid import SPIRRID
from spirrid.rv import RV
from math import pi
from scipy.stats import binom, uniform
import numpy as np
from stats.pdistrib.sin2x_distr import sin2x
from spirrid.i_rf import IRF
 
 
 
class GFRCMicro(HasTraits):
    '''
    This module considers a single planar crack bridge uniformly opening under tensile load.
    The crack is bridged by a single randomly oriented and placed short filament (typically
    chopped AR-glass filament within a bundle of filaments in cementitious matrix). The matrix
    is assumed as rigid.
    '''
    FilamentCB = Instance(IRF)
    spirrid_filament = Property(depends_on='filament_CB')
    @cached_property
    def _get_spirrid_filament(self):
        resp_func = self.FilamentCB()
        spirrid = SPIRRID(q=resp_func)
        return spirrid
    
    def mean_response(self, w_arr, theta_dict):
        '''mean filament response in terms of filament force'''
        self.spirrid_filament.eps_vars = dict(w=w_arr)
        self.spirrid_filament.theta_vars = theta_dict
        self.spirrid_filament.sampling_type = 'LHS'
        self.spirrid_filament.n_int = 300
        mu = self.spirrid_filament.mu_q_arr
        return mu * theta_dict['E_f'] * theta_dict['r']**2 * pi
    
    def var_response(self, w_arr, theta_dict):
        self.spirrid_filament.eps_vars = dict(w=w_arr)
        self.spirrid_filament.theta_vars = theta_dict
        self.spirrid_filament.sampling_type = 'LHS'
        self.spirrid_filament.n_int = 300
        var = self.spirrid_filament.var_q_arr
        return var * (theta_dict['E_f'] * theta_dict['r']**2 * pi) **2


 
class GFRCMeso(HasTraits):
    '''
    This module considers a single planar crack bridge in a prismatic specimen WxHxL
    loaded in tension. The crack is bridged by a single randomly oriented and placed short
    fiber bundle (typically chopped AR-glass fibers in cementitious matrix).
    The matrix is assumed as rigid.
    '''
     
    N_fil = Int # number of filaments in the bundle 
    micro_model = Instance(GFRCMicro)
     
    spirrid_filament = Property(depends_on='micro_model')
    @cached_property
    def _get_spirrid_filament(self):
        resp_func = self.micro_model.FilamentCB()
        spirrid = SPIRRID(q=resp_func)
        return spirrid
 
    def mean_response(self, w_arr, theta_dict):
        '''mean bundle response in terms of ridging force'''
        self.spirrid_filament.eps_vars = dict(w=w_arr)
        self.spirrid_filament.theta_vars = theta_dict
        self.spirrid_filament.sampling_type = 'LHS'
        self.spirrid_filament.n_int = 50
        mu = self.spirrid_filament.mu_q_arr
        return mu * theta_dict['E_f'] * theta_dict['r']**2 * pi * self.N_fil
     
    def var_response(self, w_arr, theta_dict):
        '''to be implemented'''
        pass
     
 
class GFRCMacro(HasTraits):
    '''
    This module considers a single planar crack bridge in a prismatic specimen WxHxL
    loaded in tension. The crack is bridged by a random number of randomly oriented
    and placed short fiber bundles (typically chopped AR-glass fibers in cementitious
    matrix). The matrix is assumed as rigid.
    '''
     
    meso_model = Instance(GFRCMeso)
    w_arr = Array # array of crack openings to evaluate the bridging force for
    # geometric dimensions of the prisma
    W = Float(params=True)
    H = Float(params=True)
    L = Float(params=True)
    Vf = Float(params=True)
    Lf = Float(params=True)
    r = Float(params=True)
    N_fil = Int(params=True)
    snub = Float(params=True)
    spall = Float(params=True)
    Ef = Float(params=True)
    xi = Float(params=True)
    tau = Any(params=True)
    phi = Any(params=True)
     
    N_fib_tot = Property(depends_on='W, H, L, r, Lf, Vf')
    @cached_property
    def _get_N_fib_tot(self):
        '''
        evaluates the total number of fibers (bundles of filaments) in the prisma
        '''
        V = self.W * self.H * self.L
        Vf1 = self.Lf * pi * self.r**2
        return round(V * self.Vf / Vf1 / self.N_fil)
     
    p_intersection = Property(depends_on='L, Lf')
    @cached_property
    def _get_p_intersection(self):
        '''
        returns the probability of a randomly oriented and placed fiber to intersect a planar crack plane WxH
        boundaries are not considered here, i.e. fiber centroids are assumed to be everywhere within the prisma
        with no restrictions on their orientation. See paper Vorechovsky, Sadilek, Rypl 2012.
        '''
        return 0.5 * self.Lf / self.L
     
    N_fib_bridging = Property(depends_on='W, H, L, r, Lf, Vf')
    @cached_property
    def _get_N_fib_bridging(self):
        '''
        returns the (binomial) distribution of number of crack bridging fibers
        '''
        return binom(self.N_fib_tot, self.p_intersection)
             
    CB_response_asymptotic = Property(depends_on='+params')
    @cached_property
    def _get_CB_response_asymptotic(self):
        mean_N_fib = self.N_fib_bridging.mean()
        var_N_fib = self.N_fib_bridging.var()
        meso = self.meso_model
        theta_dict = dict(tau=self.tau,
                          phi=self.phi,
                          le=6.,#RV('uniform', loc=0.0, scale=self.Lf/2.),
                          r=self.r,
                          E_f=self.Ef,
                          snub=self.snub,
                          xi=self.xi,
                          spall=self.spall)
        mean_bundle = meso.mean_response(self.w_arr, theta_dict)
        mean_crack_bridge = mean_N_fib * mean_bundle
        var_bundle = 1.0
        var_crack_bridge = mean_N_fib * var_bundle + var_N_fib * mean_bundle**2
        return mean_crack_bridge, var_crack_bridge
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    w_arr = np.linspace(0.,9.,200)
    micro = GFRCMicro(FilamentCB=CBShortFiber)
    theta_dict = dict(tau=RV('uniform', loc=0.001, scale=0.5),
                          r=3.5e-3, E_f=70e3, le=9.0,
                          phi=0.0, snub=0.5,
                          xi=RV('weibull_min',shape=5., scale=0.02),
                          spall = 0.5)
    mean_resp = micro.mean_response(w_arr, theta_dict)
    var_resp = micro.var_response(w_arr, theta_dict)
    plt.plot(w_arr, mean_resp, lw=2)
    plt.plot(w_arr, mean_resp + np.sqrt(var_resp), lw=2, ls='dashed')
    plt.plot(w_arr, mean_resp - np.sqrt(var_resp), lw=2, ls='dashed')
    plt.show()
#     meso = GFRCMeso(micro_model=micro)
#     macro = GFRCMacro(meso_model=meso,
#                        W=40., H=40., L=200., Lf=18.,
#                       r=8e-3, Vf=.001, N_fil=100,
#                       E_f=70e3, snub=0.5, xi=1e15,
#                       tau = RV('uniform', loc=0.1, scale=0.7),
#                       spall = 0.5, w_arr=w_arr, phi=0.0)
#     mean_r, var_r = macro.CB_response_asymptotic
#     plt.plot(w_arr, mean_r, lw=2, label='tau $\sim$ $\mathcal{U}(0.1,0.8)$')
#     plt.plot(w_arr, mean_r + np.sqrt(var_r), lw=2)
#     plt.plot(w_arr, mean_r - np.sqrt(var_r), lw=2)
#     plt.xlabel('crack opening w [mm]')
#     plt.ylabel('force [N]')
#     plt.legend(loc='best')
#     plt.show()