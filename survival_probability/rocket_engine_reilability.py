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
        return pymc.Model(self.pymc_log_likelihood, *self.priors)
     
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
    prior_beta_pressure = pymc.Normal('beta_p',mu=1.3,tau=.1)
    prior_beta_mixture_ratio = pymc.Normal('beta_mr',mu=1.3,tau=.1)
    prior_beta_s_efr = pymc.Normal('beta_s_efr',mu=1.3,tau=.1)
    prior_beta_s_ifr = pymc.Normal('beta_s_ifr',mu=1.3,tau=.1)
    prior_beta_s_wo = pymc.Normal('beta_s_wo',mu=1.3,tau=.1)
    prior_beta_m_efr = pymc.Normal('beta_m_efr',mu=1.3,tau=.1)
    prior_beta_m_ifr = pymc.Normal('beta_m_ifr',mu=1.3,tau=.1)
    prior_beta_m_wo = pymc.Normal('beta_m_wo',mu=1.3,tau=.1)
    
    rm = RegressionModel(inspection_times = np.array([2.,5.,9.,12.,20.]),
                         failures = np.array([0.,2.,1.,3.,6.]),
                         pressure = np.array([15.,20.,18.,18.,24.]),
                         mixture_ratio = np.array([.7,.5,.3,.6,.9]),
                         n = 100,
                         priors = [prior_beta_pressure, prior_beta_mixture_ratio, prior_beta_s_efr,
                                   prior_beta_s_ifr, prior_beta_s_wo, prior_beta_m_efr, prior_beta_m_ifr,
                                   prior_beta_m_wo]
                         )
    
    def parametric_likelihood(value=rm.failures, beta=rm.priors):
        s_efr, s_ifr, s_wo, m_efr, m_ifr, m_wo, p_beta, mr_beta = beta
        t_intervals = rm.time_intervals
        H = np.cumsum(((t_intervals/s_efr)**m_efr + (t_intervals/s_ifr)**m_ifr + (t_intervals/s_wo)**m_wo)
                      * e**(p_beta * rm.pressure + mr_beta * rm.mixture_ratio) * rm.time_intervals)
        n = rm.n
        k = n - np.sum(rm.failures)
        print np.log(-np.diff(np.hstack((1.,-np.exp(H)))))
        loglike = (n-k) * (-H[-1]) + np.sum(rm.failures * np.log(-np.diff(np.exp(H))))
        return loglike
    
    rm.log_likelihood = parametric_likelihood
    
    rm.plotting()