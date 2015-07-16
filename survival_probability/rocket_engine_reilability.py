'''
Created on 13. 7. 2015

@author: admin
'''

from traits.api import HasTraits, Array, Bool, Float, Int, \
                        Property, cached_property, Function, List, \
                        Instance

import pymc
from matplotlib import pyplot as plt
import numpy as np
from math import e
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from scipy.integrate.quadrature import cumtrapz


def data_generator(inspection_times, n, pressure, mr):
    # building continuous time functions for mixture ratio and pressure
    t_arr = np.linspace(0.001,inspection_times[-1],500)
    aux_times = np.hstack((0.,np.vstack((inspection_times, inspection_times + 1e-5)).T.flatten()))[:-1]
    aux_pressures = np.vstack((pressure, pressure)).T.flatten()
    aux_mr = np.vstack((mr, mr)).T.flatten()
    p_line = MFnLineArray(xdata=aux_times, ydata=aux_pressures)
    mr_line = MFnLineArray(xdata=aux_times, ydata=aux_pressures)
    pressures_t = p_line.get_values(t_arr)
    mr_t = mr_line.get_values(t_arr)
    beta_pressure = 0.3
    beta_mr = 0.8
    s_efr, s_ifr, s_wo, m_efr, m_ifr, m_wo =  1.3e26, 639.1, 15.94, 0.093, 1.302, 9.902
    h_efr = m_efr / s_efr * (t_arr/s_efr)**(m_efr-1)
    h_ifr = m_ifr / s_ifr * (t_arr/s_ifr)**(m_ifr-1)
    h_wo = m_wo / s_wo * (t_arr/s_wo)**(m_wo-1)
    h = (h_efr + h_ifr + h_wo) * np.exp(beta_pressure * pressures_t + beta_mr * mr_t)
    H = cumtrapz(h,t_arr)
    F = np.hstack((0.0,1 - np.exp(-H)))
    plt.plot(F)
    plt.show()
    
    
    
    
    return aux_times, aux_pressures


class RegressionModel(HasTraits):
    '''regression model for the liquid fuel rocket engine reliability'''
    
    inspection_times = Array
    time_intervals = Property(Array, depends_on='inspection_times')
    @cached_property
    def _get_time_intervals(self):
        t_intervals = np.diff(np.hstack((0.0, self.inspection_times)))
        return t_intervals
        
    failures = Array # number of failures per inspection interval, size equals the size of inspection times
    n = Int #number of individuals.
    k = Property(depens_on='failures,n')
    @cached_property
    def _get_k(self):
        return self.n - np.sum(self.failures)
    
    # covariates
    pressure = Array # size equals size of inspection_times
    mixture_ratio = Array # size equals size of inspection_times
    
    priors = List 
 
    log_likelihood = Function
 
    # log likelihood function for model parameters given the data D = (N, y, nu, X)
    pymc_log_likelihood = Property()
    def _get_pymc_log_likelihood(self):
        f = pymc.stochastic(self.log_likelihood, observed=True)
        return f
     
    # instance of the Model
    model = Property(Instance(pymc.Model))
    @cached_property
    def _get_model(self):
        return pymc.Model(self.pymc_log_likelihood, self.priors)
     
    samples = Property()
    def _get_samples(self):
        '''set the Markov chain Monte Carlo with specified
        No. of iterations, burn-in length and thinning'''
        iterations=10000
        burn_in=5000
        thinning=5
        sampling_engine = pymc.MCMC(self.model)
        samples = sampling_engine.sample(iter=iterations, burn=burn_in, thin=thinning)
        return sampling_engine
     
    def plotting(self):
        '''plot the results of the sampling procedure'''
        pymc.Matplot.plot(self.samples)
        plt.show()
    
if __name__ == '__main__':
    print data_generator(np.array([1.,3.,4.,7.,9.]), 100, np.array([1.4,1.,3.,2.,1.]), np.array([.7,.5,.8,.6,.9]))
#     prior_beta_pressure = pymc.Normal('beta_p',mu=1.3,tau=.1)
#     prior_beta_mixture_ratio = pymc.Normal('beta_mr',mu=1.3,tau=.1)
#     prior_beta_s_efr = pymc.Normal('beta_s_efr',mu=1.3,tau=.1)
#     prior_beta_s_ifr = pymc.Normal('beta_s_ifr',mu=1.3,tau=.1)
#     prior_beta_s_wo = pymc.Normal('beta_s_wo',mu=1.3,tau=.1)
#     prior_beta_m_efr = pymc.Normal('beta_m_efr',mu=1.3,tau=.1)
#     prior_beta_m_ifr = pymc.Normal('beta_m_ifr',mu=1.3,tau=.1)
#     prior_beta_m_wo = pymc.Normal('beta_m_wo',mu=1.3,tau=.1)
#     
#     rm = RegressionModel(inspection_times = np.array([2.,5.,9.,12.,20.]),
#                          failures = np.array([0.,2.,1.,3.,6.]),
#                          pressure = np.array([15.,20.,18.,18.,24.]),
#                          mixture_ratio = np.array([.7,.5,.3,.6,.9]),
#                          n = 100,
#                          priors = [prior_beta_pressure, prior_beta_mixture_ratio, prior_beta_s_efr,
#                                    prior_beta_s_ifr, prior_beta_s_wo, prior_beta_m_efr, prior_beta_m_ifr,
#                                    prior_beta_m_wo]
#                          )
#     
#     def parametric_likelihood(value=rm.failures, beta=rm.priors):
#         s_efr, s_ifr, s_wo, m_efr, m_ifr, m_wo, p_beta, mr_beta = beta
#         t_intervals = rm.time_intervals
#         H = np.cumsum(((t_intervals/s_efr)**m_efr + (t_intervals/s_ifr)**m_ifr + (t_intervals/s_wo)**m_wo)
#                       * e**(p_beta * rm.pressure + mr_beta * rm.mixture_ratio) * rm.time_intervals)
#         n = rm.n
#         k = n - np.sum(rm.failures)
#         loglike = (n-k) * (-H[-1]) + np.sum(rm.failures * np.log(np.hstack((-np.exp(H[0]),-np.diff(np.exp(H))))))
#         return loglike
#     
#     rm.log_likelihood = parametric_likelihood
#     
#     rm.plotting()