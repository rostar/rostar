'''
Created on 6. 11. 2014

@author: admin
'''

import numpy as np
from scipy.stats import gamma
from scipy import stats
from matplotlib import pyplot as plt
from scipy.special import gamma as gamma_func, gammainc
from scipy.optimize import newton, roots

def params_calibration(distr):
    w_hat = 0.075
    sigmaf_hat = 404.65
    Ef = 200e3
    r = 3.5e-3
    sigmamu = 3.4
    Vf = 0.01
    muCS = 10.
    mu_sqrt_tau = sigmaf_hat / np.sqrt(2. * Ef * w_hat / r)
    mu_tau = 1.3 * r * sigmamu * (1.-Vf) / (2. * Vf * muCS)
    n_sample = 10000
    def residuum(params):
        if np.any(params) < 0.0:
            return 10. * np.abs(params)
        else:
            shape = params[0]
            scale = params[1]
            param_distr = distr(shape, scale=scale)
            mu_distr = param_distr.mean()
            p_arr = np.linspace(0.5/n_sample, 1. - 0.5/n_sample, n_sample)
            x = param_distr.ppf(p_arr)
            mu_sqrt_distr = np.sum(np.sqrt(x)) / n_sample
            resid = np.array([mu_distr - mu_tau, mu_sqrt_distr - mu_sqrt_tau])
            return resid
         
    opt = root(residuum, np.array([.2, .5]))
    params = opt.x
    if np.all(params == np.array([.2, .5])):
        print 'no solution'
        return 'no solution'
    else:
        print 'solution found', params
        return params
         
distr_dict = {}
 
for distr_name in stats.distributions.__all__:
    distr = stats.distributions.__dict__[distr_name]
    if hasattr(distr, 'numargs'):
        if (distr.numargs == 1 and distr_name != 'foldcauchy' and
        distr_name != 'foldnorm' and distr_name != 'invgauss' and
        distr_name != 'rdist' and distr_name != 'rice' and
        distr_name != 'recipinvgauss' and distr_name != 'vonmises' and
        distr_name != 'vonmises_line' and distr_name != 'bernoulli'):
            print distr_name
            try:
                params = params_calibration(distr)
                if params == 'no solution':
                    pass
                else:
                    distr_dict[distr_name] = params
            except:
                pass
             
             
print distr_dict

# GAMMA sqrt X TRANSFORMATION
# t = .7
# k = 0.2
# x = np.linspace(0.005,.5,500)
# fx = lambda x: gamma(k,scale=t).pdf(x)
# sfx = lambda x: gamma(k,scale=t).pdf(x**2) * (2*x)
# 
# 
# def MC_samples(n):
#     x_samples = gamma(k, scale=t).rvs(n)
#     return x_samples**0.5
# 
# def MC_samples2(n):
#     x_samples = gamma(k, scale=t).rvs(n)
#     return x_samples
# 
# print 'mean X = ', np.sum(MC_samples2(1000000))/1000000., 'should be ', t*k
# print 'mean sqrt(X) = ', np.sum(MC_samples(1000000))/1000000.
# print 'mean sqrt(X) analytic = ', t**0.5 * gamma_func(k+0.5)/gamma_func(k)
# 
# t_arr = np.linspace(0.001,5.0,200)
# plt.plot(t_arr, t_arr * k,label='mean X')
# plt.plot(t_arr, t_arr**0.5 * gamma_func(k+0.5)/gamma_func(k), label='mean sqrt(X)')
# plt.legend(loc='best')
# plt.figure()
# k_arr = np.linspace(0.001,1.5,200)
# plt.plot(k_arr, k_arr * t, label='mean X')
# plt.plot(k_arr, t**0.5 * gamma_func(k_arr+0.5)/gamma_func(k_arr), label='mean sqrt(X)')
# plt.legend(loc='best')
# 
# # plt.hist(MC_samples(1000000), bins=500, normed=True)
# # plt.plot(x, fx(x),label='basic')
# # plt.plot(x, sfx(x), label='sqrt2')
# # plt.xlim(0.,0.5)
# plt.legend(loc='best')
# plt.show()