'''
Created on 03.07.2013

@author: acki
'''
import numpy as np
from scipy.interpolate import interp1d 
from scipy.optimize import minimize, basinhopping, brute
import os
from etsproxy.traits.api import HasTraits, Property, Callable, Array, cached_property

FILE_DIR = os.path.dirname(__file__)


class Calibration(HasTraits):

    model = Callable
    test_xdata = Array
    test_ydata = Array
    weight_arr = Array
    residuum = 5e10
    x0 = 0.0
    count = 0.0

    interpolate_experiment = Property(depends_on='test_xdata, test_ydata')
    @cached_property
    def _get_interpolate_experiment(self):
        return interp1d(self.test_xdata, self.test_ydata,
                        bounds_error=False, fill_value=0.0)

    def calibrate(self, x0, *args):
#        plt.plot(args[0], self.model(x0, args[0]))
#        plt.plot(args[0], self.interpolate_experiment(args[0]))
#        plt.show()
        def residuum(x0, *args):
            sq_err = (self.interpolate_experiment(args[0]) - self.model(x0, args[0])) ** 2
            residuum = np.sum(self.weight_arr * sq_err)
            if residuum < self.residuum:
                self.residuum = residuum
                self.x0 = x0
                plt.plot(args[0], self.interpolate_experiment(args[0]), color='black', lw=2)
                plt.plot(args[0], self.model(x0, args[0]), color='red', lw=2)
                plt.show()
            return residuum
            #else:
            #    return self.residuum * 2
        #calibrated_params = minimize(residuum, x0, args=args, method='L-BFGS-B')
        #calibrated_params = basinhopping(residuum, x0, minimizer_kwargs={'args':args,
        #                                                                 'method':'BFGS'})
        for sV0 in np.linspace(2.7755e-3, 4.0e-3, 50):
            for tau_scale in np.linspace(0.0166, .1, 50):
                self.count += 1.0
                resid = residuum([sV0, tau_scale], args[0])
                print self.count / (50*50)*100, 'percent'
                print 'current params:', [sV0, tau_scale], 'best params:', self.x0
                print 'current residuum:', resid, 'best:', self.residuum
#         for sV0 in np.linspace(2.4e-3, 4e-3, 6):
#             for ratio in np.linspace(0.87, 0.999, 6):
#                 for m in np.linspace(4.0, 5.5, 6):
#                     for a_up in np.linspace(0.018, 0.05, 6):
#                         for b_up in np.linspace(0.68, 3., 6):
#                             self.count += 1.0
#                             resid = residuum([ratio, sV0, m, a_up, b_up], args[0])
#                             print self.count / (6**5)*100, 'percent'
#                             print 'current params:', [ratio, sV0, m, a_up, b_up], 'best params:', self.x0
#                             print 'current residuum:', resid, 'best:', self.residuum   
#         for sV0 in np.linspace(1.5e-3, 4e-3, 30):
#             for m in np.linspace(3., 5., 30):
#                 for tau in np.linspace(0.01, 2.0, 30):  
#                             self.count += 1.0
#                             resid = residuum([sV0, m, tau], args[0])
#                             print self.count / (30**3)*100, 'percent'
#                             print 'current params:', [sV0, m, tau], 'best params:', self.x0
#                             print 'current residuum:', resid, 'best:', self.residuum            
            
        return
    
if __name__ == '__main__':
    from quaducom.micro.resp_func.CB_rigid_mtrx_rand_xi import CBResidualRandXi
    from spirrid.spirrid import SPIRRID
    from spirrid.rv import RV
    from matplotlib import pyplot as plt
    from scipy.special import gamma, gammainc
    from math import pi

    cb = CBResidualRandXi()
    spirrid = SPIRRID(q=cb, sampling_type='PGrid')
    
    def model_rand(params, *args):
        w = args[0]
        sV0, tau_scale = params
        E_f = 180e3
        V_f = 1.0
        r = 3.45e-3
        m = 4.5
        tau = RV('uniform', loc=0.0, scale=tau_scale)
#         a_lower = 0.0
#         tau = RV('piecewise_uniform', shape=0.0, scale=1.0)
#         tau._distr.distr_type.distribution.a_lower = a_lower
#         tau._distr.distr_type.distribution.a_upper = a_upper
#         tau._distr.distr_type.distribution.b_lower = a_upper
#         tau._distr.distr_type.distribution.b_upper = b_upper
#         tau._distr.distr_type.distribution.ratio = ratio
        n_int = 100

        spirrid.eps_vars=dict(w=w)
        spirrid.theta_vars=dict(tau=tau, E_f=E_f, V_f=V_f, r=r, m=m, sV0=sV0)
        spirrid.n_int=n_int
        if isinstance(r, RV):
            r_arr = np.linspace(r.ppf(0.001), r.ppf(0.999), 300)
            Er = np.trapz(r_arr ** 2 * r.pdf(r_arr), r_arr)
        else:
            Er = r ** 2
        sigma_c = spirrid.mu_q_arr / Er
        return sigma_c

    def model_determ(params, *args):
        w = args[0]
        sV0, tau = params
        m = 4.5
        E_f = 180e3
        V_f = 1.0
        r = 3.45e-3
        T = 2. * tau / r
        s0 = ((T * (m+1) * sV0**m)/(2. * E_f * pi * r ** 2))**(1./(m+1))
        k = np.sqrt(T/E_f)
        ef0 = k*np.sqrt(w)
        G = 1 - np.exp(-(ef0/s0)**(m+1))
        mu_int = ef0 * E_f * V_f * (1-G)
        I = s0 * gamma(1 + 1./(m+1)) * gammainc(1 + 1./(m+1), (ef0/s0)**(m+1))
        mu_broken = E_f * V_f * I / (m+1)
        return mu_int + mu_broken

    w = np.linspace(0.0, 10., 100)
    calib = Calibration(model=model_rand)
    file = open('DATA/PO01_RYP.ASC', 'r')
    calib.test_xdata = - np.loadtxt(file, delimiter=';')[:,3]
    file = open('DATA/PO01_RYP.ASC', 'r')
    calib.test_ydata = (np.loadtxt(file, delimiter=';')[:,1] + 0.035)/0.45 * 1000
    weight = np.ones_like(w)
    weight[0:5] = 0.0
    weight[5:20] = 3.0
    calib.weight_arr = weight
    
    result = calib.calibrate([(0.5, 0.95), (1.5e-3,5e-3), (3.5, 5.5)], w)
    print 'optimal params: ', result.x



    

