'''
Created on 21. 10. 2014

@author: admin
'''

from traits.api import HasTraits, Array, Bool, Float, Int, \
                        Property, cached_property, Function, List, \
                        Instance
import numpy as np
import pymc
from matplotlib import pyplot as plt

class SurvivalBase(HasTraits):
    
    # measured failure times (y) - continuous or grouped
    measured_failure_times = Array(Float)
    
    #number of individuals (N).
    N = Int
    def _N_default(self):
        ''' if not specified, N = len(failure_times) '''
        return len(self.failure_times)
 
    # failure times 
    failure_times = Property()
    @cached_property
    def _get_failure_times(self):
        y = np.sort(self.measured_failure_times)
        if self.N < len(y):
            raise ValueError('''the total number of individuals must be
            equal or higher than the number of measured failure times''')
        elif self.N > len(y):
            K = self.N - len(y)
            return np.hstack((y, np.repeat(1e10,K)))
        else:
            return y
     
    # True if failure time, False if the time is (right) censored
    # The assumption is that N - len(measured_survival_times) are right censored
    censoring_indicators = Property(Array(Bool), depens_on='measured_failure_times, N')
    def _get_censoring_indicators(self):
        nu_failure_times = np.repeat(True, len(self.measured_failure_times))
        nu_censored_times = np.repeat(False, self.N - len(self.measured_failure_times))
        return np.hstack((nu_failure_times, nu_censored_times))
 
    # dictionary of covariates as scalars or arrays of length N
    covariates = Array
 
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
    from scipy.stats import expon
    n = 50
    x = 2.
    beta0 = 0.3
    lambda0 = np.exp((beta0) * x)
    y = expon(scale=1./lambda0).rvs(n)
    time = np.linspace(0.,2.,1000)
    nu = 1. * (y < time[-1]) 
    sb = SurvivalBase(N=n,
                      covariates=np.array([x]),
                      priors=[pymc.Normal('beta',mu=0.,tau=1.)],
                      measured_failure_times = y,
                      )
    
    def parametric_likelihood(value=sb.failure_times, b=sb.priors):
        return np.sum(sb.censoring_indicators * sb.covariates * b) -np.sum(value * np.exp(sb.covariates * b))
    sb.log_likelihood = parametric_likelihood
    
    sb.plotting()
