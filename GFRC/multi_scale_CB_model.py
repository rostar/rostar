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
from util.traits.either_type import EitherType
 
 
 
class GFRCMicro(HasTraits):
    '''
    This module considers a single planar crack bridge uniformly opening under tensile load.
    The crack is bridged by a single randomly oriented and placed short filament (typically
    chopped AR-glass filament within a bundle of filaments in cementitious matrix). The matrix
    is assumed as rigid.
    '''
    #resp_func = Instance(RF)


class GFRCMeso(HasTraits):
    '''
    This module considers a single planar crack bridge in a prismatic specimen WxHxL
    loaded in tension. The crack is bridged by a single randomly oriented and placed short
    fiber bundle (typically chopped AR-glass fibers in cementitious matrix).
    The matrix is assumed as rigid.
    '''
    N_fil = Int # number of filaments in the bundle 
    micro_model = Instance(GFRCMicro)
    fiber_CB = Property(depends_on='filament_CB')
    @cached_property
    def _get_fiber_CB(self):
        resp_func = self.micro_model.resp_func()
        spirrid = SPIRRID(q=resp_func)
        return spirrid
        
    def get_MC_response(self, w_arr, theta_dict, N_fil):
        self.fiber_CB.eps_vars = dict(w=w_arr)
        self.fiber_CB.theta_vars = theta_dict
        self.fiber_CB.sampling_type = 'MCS'
        self.fiber_CB.n_int = N_fil
        return self.fiber_CB.mu_q_arr

    def get_asymptotic_response(self, w_arr, theta_dict):
        self.fiber_CB.eps_vars = dict(w=w_arr)
        self.fiber_CB.theta_vars = theta_dict
        self.fiber_CB.sampling_type = 'LHS'
        self.fiber_CB.n_int = 50
        var = self.fiber_CB.var_q_arr
        mu = self.fiber_CB.mu_q_arr
        return mu, var
    

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

    CB_response_MC_simulation = Property(depends_on='+params')
    @cached_property
    def _get_CB_response_MC_simulation(self):
        N_fib = self.N_fib_bridging.rvs(1)
        meso = self.meso_model
        phi = sin2x.ppf(np.random.rand(N_fib))
        le = uniform(loc=0., scale=self.Lf/2.).ppf(np.random.rand(N_fib))
        response = np.zeros_like(self.w_arr)
        for phi_i, le_i in zip(phi,le):
            theta_dict = dict(tau=self.tau,
                              r=self.r,
                              E_f=self.Ef,
                              le=le_i,
                              phi=phi_i,
                              snub=self.snub,
                              xi=self.xi)
            response += meso.get_MC_response(self.w_arr, theta_dict, self.N_fil) / float(N_fib)
        return response * self.Ef * self.Vf
            
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
        mean_sigmac1, var_sigmac1 = meso.get_asymptotic_response(self.w_arr, theta_dict)
        mean_response = mean_N_fib * mean_sigmac1 * self.Ef * self.Vf
        var_response = var_N_fib * mean_sigmac1 * self.Ef * self.Vf + mean_N_fib * var_sigmac1 * (self.Ef * self.Vf)**2
        return mean_response, var_response

    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    w_arr = np.linspace(0.,.5,200)
    micro = GFRCMicro(resp_func=CBShortFiber)
    meso = GFRCMeso(micro_model=micro)
    macro = GFRCMacro(meso_model=meso,
                       W=40., H=40., L=200., Lf=18.,
                      r=8e-3, Vf=.00020, N_fil=100,
                      Ef=70e3, snub=0.5, xi=1e15,
                      tau = RV('uniform', loc=0.1, scale=0.7),
                      spall = 0.5, w_arr=w_arr, phi=0.0)
    mean_r, var_r = macro.CB_response_asymptotic
    plt.plot(w_arr, mean_r * 100 * pi*(8e-3)**2 / 0.0002, lw=2, label='tau $\sim$ $\mathcal{U}(0.1,0.8)$')
    #plt.plot(w_arr, mean_r + np.sqrt(var_r), lw=2)
    #plt.plot(w_arr, mean_r - np.sqrt(var_r), lw=2)
    plt.xlabel('crack opening w [mm]')
    plt.ylabel('force [N]')
    plt.legend(loc='best')
    plt.show()