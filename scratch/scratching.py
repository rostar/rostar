from scipy.stats import gamma, weibull_min
import numpy as np

var = gamma(0.051, scale=2.28).var()
mean = gamma(0.051, scale=2.28).mean()
print 'stdev = ', np.sqrt(var)
print 'mean = ', mean
print 'CoV = ', np.sqrt(var)/mean

print weibull_min(8.81, scale=0.0134).mean()
print weibull_min(200, scale=0.01267).mean()