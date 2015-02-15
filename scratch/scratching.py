from scipy.stats import gamma, weibull_min
from matplotlib import pyplot as plt
from scipy.special import gamma as gamma_func
import numpy as np

data = weibull_min(5., scale=2.).ppf(np.random.rand(20))

def likel(m, s, D):
    return np.prod(weibull_min(m, scale=s).pdf(D))

