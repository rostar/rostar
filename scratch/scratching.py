import numpy as np
from scipy.optimize import fsolve
from scipy.stats import weibull_min
import time
from scipy.optimize import broyden2

val = np.random.rand(400)


def J(x):
    j = weibull_min(5.).pdf(x).reshape(1,4)
    print j
    return j


def residuum(x):
    return np.sum(weibull_min(5.).cdf(x) - val)


def residuum2(x):
    return val - weibull_min(5.).cdf(x)

#print val
#t = time.clock()
#res = fmin_bfgs(residuum, np.ones_like(val), fprime=J)
#print 'bfgs', time.clock() - t, 'sec', 'res = ', weibull_min(5.).cdf(res)
t = time.clock()
res = broyden2(residuum2, np.ones(len(val)))
print 'broy2', time.clock() - t, 'sec', 'res = ', np.sum(weibull_min(5.).cdf(res) - val)
t = time.clock()
res = fsolve(residuum2, np.ones(len(val)))
print 'fsolve', time.clock() - t, 'sec', 'res = ', np.sum(weibull_min(5.).cdf(res) - val)



#from scipy.optimize import fmin
#def rosen(x):  # The Rosenbrock function
#    return x - 1.0
#
#def rosen_der(x):
#    print x.shape
#    xm = x[1:-1]
#    xm_m1 = x[:-2]
#    xm_p1 = x[2:]
#    der = np.zeros(x.shape)
#    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
#    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
#    der[-1] = 200*(x[-1]-x[-2]**2)
#    print der.shape
#    return der
#
#from scipy.optimize import fmin_bfgs
#
#x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
#xopt = fsolve(rosen, x0, fprime=rosen_der)
#print xopt
