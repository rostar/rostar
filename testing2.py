import numpy as np
import matplotlib
matplotlib.use('WxAgg')
import matplotlib.pyplot as plt
from scipy.stats import norm

loc_arr = np.linspace(0.1, 50, 1000)
std = 3.


def L(loc_arr, std, D):
    likelihood = 1.0
    for d in D:
        ith_likelihood = norm(loc=loc_arr, scale=std).pdf(d)
        likelihood *= ith_likelihood
        plt.plot(loc_arr, ith_likelihood)
    plt.show()
    return likelihood

D = norm(loc=10., scale=std).rvs(10)
D = np.hstack((D, norm(loc=20., scale=std).rvs(10)))

likeli = L(loc_arr, std, D)
c = np.trapz(likeli, loc_arr)
posterior = likeli / c

mean = np.trapz(posterior * loc_arr, loc_arr)
stdev = np.sqrt(np.trapz(posterior * loc_arr**2, loc_arr) - mean**2)

print mean, stdev

plt.plot(loc_arr, posterior)
plt.show()
