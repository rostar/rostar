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
from scipy.stats import weibull_min


def data_generator(inspection_times, n, pressure, mr):
    # building continuous time functions for mixture ratio and pressure
    t_arr = np.linspace(0.001,inspection_times[-1],500)
    aux_times = np.hstack((0.,np.vstack((inspection_times, inspection_times + 1e-5)).T.flatten()))[:-1]
    aux_pressures = np.vstack((pressure, pressure)).T.flatten()
    aux_mr = np.vstack((mr, mr)).T.flatten()
    p_line = MFnLineArray(xdata=aux_times, ydata=aux_pressures)
    mr_line = MFnLineArray(xdata=aux_times, ydata=aux_mr)
    pressures_t = p_line.get_values(t_arr)
    mr_t = mr_line.get_values(t_arr)
    beta_pressure = 2.7
    beta_mr = 1.5
    s_ifr, s_wo, m_ifr, m_wo =  70., 15., 1.3, 10.
    #h_efr = m_efr / s_efr * (t_arr/s_efr)**(m_efr-1)
    h_ifr = m_ifr / s_ifr * (t_arr/s_ifr)**(m_ifr-1)
    h_wo = m_wo / s_wo * (t_arr/s_wo)**(m_wo-1)
    h = (h_ifr + h_wo) * np.exp(beta_pressure * pressures_t + beta_mr * mr_t)
    H = cumtrapz(h,t_arr, initial=0.0)
    F = np.hstack((1 - np.exp(-H), 1.0))
    PPF_line = MFnLineArray(ydata=np.hstack((t_arr,t_arr[-1]*1.1)),xdata=F)
    p = np.random.rand(n)
    np.sort(p)
    p = p[p<=F[-2]]
    samples = PPF_line.get_values(p)
    
    hist = np.histogram(samples, np.hstack((0.0,inspection_times)))
    plt.plot(np.hstack((t_arr,t_arr[-1]*1.1)),F)
    plt.plot(inspection_times, np.cumsum(hist[0])/float(n), color='red')
    plt.plot(samples, p, 'ro')
    plt.show()
    return samples


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
        iterations=20000
        burn_in=5000
        thinning=50
        sampling_engine = pymc.MCMC(self.model)
        samples = sampling_engine.sample(iter=iterations, burn=burn_in, thin=thinning)
        return sampling_engine
     
    def plotting(self):
        '''plot the results of the sampling procedure'''
        pymc.Matplot.plot(self.samples)
        plt.show()
    
if __name__ == '__main__':
    inspection_times = np.linspace(1.,9.,10)
    n = 10000
    pressures=np.random.rand(10) * 1.3
    mr = np.random.rand(10) * 1.6
    generated_cont_failure_times = data_generator(inspection_times, n, pressures, mr)
    censored_failures = np.histogram(generated_cont_failure_times, np.hstack((0.0,inspection_times)))[0]
    
    prior_beta_pressure = pymc.Normal('beta_p', mu=1., tau=.5)
    prior_beta_mixture_ratio = pymc.Normal('beta_mr', mu=3., tau=1.1)
    #prior_beta_s_efr = pymc.Uniform('beta_s_efr', lower=500., upper=5000.)
    prior_beta_s_ifr = pymc.Uniform('beta_s_ifr', lower=20., upper=1000.)
    prior_beta_s_wo = pymc.Normal('beta_s_wo', mu=10., tau=5.)
    #prior_beta_m_efr = pymc.Normal('beta_m_efr', mu=.15, tau=.05)
    prior_beta_m_ifr = pymc.Uniform('beta_m_ifr', lower=0.0, upper=3.)
    prior_beta_m_wo = pymc.Normal('beta_m_wo', mu=8., tau=2.1)
     
    rm = RegressionModel(inspection_times = inspection_times,
                         failures = censored_failures,
                         pressure = pressures,
                         mixture_ratio = mr,
                         n = n,
                         priors = [prior_beta_s_ifr, prior_beta_s_wo, prior_beta_m_ifr, prior_beta_m_wo,
                                   prior_beta_pressure, prior_beta_mixture_ratio]
                         )
     
    def parametric_likelihood(value=rm.failures, beta=rm.priors):
        s_ifr, s_wo, m_ifr, m_wo, pressure, mixture_ratio = beta
        H0 = np.hstack((0.0,(rm.inspection_times/s_ifr)**m_ifr + (rm.inspection_times/s_wo)**m_wo ))
        H = np.cumsum(np.diff(H0) * e ** (pressure * rm.pressure + mixture_ratio * rm.mixture_ratio))
        k = np.sum(value)
        S = np.exp(-H)
        loglike = (rm.n-k) * (-H[-1]) + np.sum( value * np.log(-np.diff(np.hstack((1.0,S)))))
        return loglike
     
    rm.log_likelihood = parametric_likelihood
     
    rm.plotting()
    
    
