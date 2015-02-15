from scipy.stats import weibull_min
import numpy as np
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt
from scipy.stats import uniform, norm, expon

def expon_likelihood(y, nu, x, beta):
    y = y[np.newaxis,:]
    nu = nu[np.newaxis,:]
    beta = beta[:, np.newaxis]
    L = np.exp(np.sum(nu*x*beta, axis=1)) * np.exp(-np.sum(y*np.exp(x*beta), axis=1))
    return L
 
def prior(beta):
    return norm(0.5, 1).pdf(beta)
 
def posterior(y, nu, x, beta):
    return expon_likelihood(y, nu, x, beta) * prior(beta)
 
if __name__ == '__main__':
    n = 300
    x = 2.
    beta0 = 0.3
    y = expon(scale=1./np.exp(x * beta0)).rvs(n)
    time = np.linspace(0.,5.,1000)
    nu = 1. * (y < time[-1])
 
    beta_arr = np.linspace(-1, 1, 500)
    beta_guess = np.trapz(posterior(y, nu, x, beta_arr) * beta_arr / np.trapz(posterior(y, nu, x, beta_arr), beta_arr), beta_arr)
#     plt.plot(time, 1 - np.sum(time[np.newaxis, :] > np.sort(y)[:, np.newaxis], axis=0)/float(len(y)))
#     plt.plot(time, expon(scale=1./np.exp(beta0 * x)).sf(time), label='True')
#     plt.plot(time, expon(scale=1./np.exp(beta_guess * x)).sf(time), label='eval')
    #plt.plot(beta_arr, expon_likelihood(y, nu, x, beta_arr)/ np.trapz(expon_likelihood(y, nu, x, beta_arr), beta_arr))
    #plt.plot(beta_arr, posterior(y, nu, x, beta_arr)/ np.trapz(posterior(y, nu, x, beta_arr), beta_arr))
#     plt.legend()
#     plt.show()
    # Import relevant modules
    import pymc
    import numpy as np
    
    # Some data
    n = 5*np.ones(4,dtype=int)
    x = np.array([-.86,-.3,-.05,.73])
    
    # Priors on unknown parameters
    alpha = pymc.Normal('alpha',mu=0,tau=.01)
    beta = pymc.Normal('beta',mu=0,tau=.01)
    
    # Arbitrary deterministic function of parameters
    @pymc.deterministic
    def theta(a=alpha, b=beta):
        """theta = logit^{-1}(a+b)"""
        return pymc.invlogit(a+b*x)
    
    # Binomial likelihood for data
    d = pymc.Binomial('d', n=n, p=theta, value=np.array([0.,1.,3.,5.]),\
                      observed=True)
    mymodel = pymc.Model(d, alpha, beta)
    S = pymc.MCMC(mymodel)
    S.sample(iter=10000, burn=5000, thin=2)
    pymc.Matplot.plot(S)
    plt.show()
    