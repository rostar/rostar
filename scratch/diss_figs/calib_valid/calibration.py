import numpy as np
from scipy.interpolate import interp1d
import os
from etsproxy.traits.api import HasTraits, Property, Array, \
     cached_property, Float, Int, Str, implements, Bool
from spirrid.spirrid import SPIRRID
from spirrid.rv import RV
from etsproxy.util.home_directory import get_home_directory
from scipy.optimize import minimize
from spirrid.i_rf import IRF
from spirrid.rf import RF
from math import pi, e

def H(x):
    return x >= 0.0


class CBClampedRandXi(RF):
    '''
    Crack bridged by a fiber with constant
    frictional interface to rigid; free fiber end;
    '''

    implements(IRF)
    title = Str('crack bridge with rigid matrix')
    tau = Float(2.5, auto_set=False, enter_set=True, input=True,
                distr=['uniform', 'norm'])

    r = Float(0.013, auto_set=False, enter_set=True, input=True,
              distr=['uniform', 'norm'], desc='fiber radius')

    E_f = Float(72e3, auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])

    m = Float(5., auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])

    sV0 = Float(3.e-3, auto_set=False, enter_set=True, input=True,
                  distr=['uniform'])

    V_f = Float(0.0175, auto_set=False, enter_set=True, input=True,
              distr=['uniform'])

    lm = Float(np.inf, auto_set=False, enter_set=True, input=True,
              distr=['uniform'])

    w = Float(auto_set=False, enter_set=True, input=True,
               distr=['uniform'], desc='crack width',
               ctrl_range=(0.0, 1.0, 10))

    x_label = Str('crack opening [mm]')
    y_label = Str('composite stress [MPa]')

    C_code = Str('')

    pullout = Bool(True)
    
    def __call__(self, w, tau, E_f, V_f, r, m, sV0):
        '''free and fixed fibers combined
        the failure probability of fixed fibers
        is evaluated by integrating only
        between -lm/2 and lm/2.
        Only intact fibers are considered (no pullout contribution)'''
        T = 2. * tau / r + 1e-10
        k = np.sqrt(T/E_f)
        ef0cb = k*np.sqrt(w)  
        s = ((T * (m+1) * sV0**m)/(2. * E_f * pi * r ** 2))**(1./(m+1))
        Gxi_deb = 1 - np.exp(-(ef0cb/s)**(m+1))
        return ef0cb * (1-Gxi_deb) * E_f * V_f * r**2
        
        
class Model(HasTraits):

    test_xdata = Array
    test_ydata = Array
    w_min = Float(0.0, auto_set=False, enter_set=True, params=True)
    w_max = Float(2., auto_set=False, enter_set=True, params=True)
    w_pts = Int(100, auto_set=False, enter_set=True, params=True)
    Ef = Float(181e3, auto_set=False, enter_set=True, params=True)
    V_f = Float(1.0, params=True)
    r = Float(3.5e-3, params=True)

    w = Property(Array,depends_on='w_min,w_max,w_pts')
    @cached_property
    def _get_w(self):
        return np.linspace(self.w_min, self.w_max, self.w_pts)

    interpolate_experiment = Property(depends_on='test_xdata,test_ydata,w')
    @cached_property
    def _get_interpolate_experiment(self):
        interp = interp1d(self.test_xdata, self.test_ydata,
                        bounds_error=False, fill_value=0.0)
        return interp(self.w)

    def model_free(self, tau_shape, tau_scale, xi_shape, xi_scale):
        #xi_scale = 2092. / (182e3 * (pi * 3.5e-3 **2 * 70. * e)**(-1./xi_shape))
        tau_loc=0.0
        cb = CBClampedRandXi(pullout=False)
        spirrid = SPIRRID(q=cb, sampling_type='LHS')
        tau = RV('gamma', shape=tau_shape, scale=tau_scale, loc=tau_loc)
        w = self.w
        spirrid.eps_vars = dict(w=w)
        spirrid.theta_vars = dict(tau=tau, E_f=self.Ef, V_f=self.V_f,
                                  r=self.r, m=xi_shape, sV0=xi_scale)
        spirrid.n_int = 500
        sigma_c = spirrid.mu_q_arr / self.r ** 2
        plt.plot(w, sigma_c)
        return sigma_c
   
    lack = 1e10
   
    def lack_of_fit(self, params):
        #tau_loc = params[0]
        tau_shape = params[0]
        tau_scale = params[1]
        xi_shape = params[2]
        xi_scale = params[3]
        print params
        lack = np.sum((self.model_free(tau_shape, tau_scale, xi_shape, xi_scale) - self.interpolate_experiment) ** 2)
        print 'params = ', params
        print 'relative lack of fit', lack/np.trapz(self.interpolate_experiment**2,self.w)
        if lack < self.lack:
            print 'DRAW'
            self.lack = lack
            plt.ion()
            plt.cla()
            plt.plot(self.w, self.interpolate_experiment, color='black')
            plt.plot(self.w, self.model_free(tau_shape, tau_scale, xi_shape, xi_scale), color='red', lw=2)
            plt.draw()
            plt.show()
        return lack

    def eval_params(self):
        params = minimize(self.lack_of_fit, np.array([1.66955758e-01, 2.70710428e-01, 1.20000000e+01, 7.73963278e-03]),
                          method='L-BFGS-B', bounds=((0.03, 2.), (0.1, 5.), (4., 200.), (0.003, 0.02)))
        return params

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    home_dir = get_home_directory()
    
    model = Model(w_min=0.0,
                      w_max=.4,
                      w_pts=100)
    
    def average_exp():
        avg = np.zeros_like(model.w)
        for i in range(5):
            path = [home_dir, 'git',  # the path of the data file
                        'rostar',
                        'scratch',
                        'diss_figs',
                        'calib_valid',
                        'CB' + str(i+1) +'.txt']
            filepath = os.path.join(*path)
            file1 = open(filepath, 'r')
            cb = np.loadtxt(file1, delimiter=';')
            model.test_xdata = -cb[:, 2] / 4. - cb[:, 3] / 4. - cb[:, 4] / 2.
            model.test_ydata = cb[:, 1] / (11. * 0.445) * 1000
            avg += model.interpolate_experiment / 5.
        return avg


    def calibrate():
        model.w_max = 0.5
        model.test_ydata = average_exp()
        model.test_xdata = model.w
        params = model.eval_params().x
        print 'identified parameters: ', params
        #param_sets = []
        #for w_i in np.linspace(0.6, 3.0, 7):
        #    model.w_max = w_i
        #    model.test_ydata = average_exp()
        #    model.test_xdata = model.w
        #    params = model.eval_params().x
        #    print 'identified parameters: ', params
        #    param_sets.append(params)
        #print 'for the ws from ', np.linspace(0.6, 3.0, 7)
        #print 'identified params_sets: ', param_sets
        #tau_loc, tau_shape, tau_scale, xi_shape = params
        #plt.figure()
        #plt.plot(model.w, model.interpolate_experiment, color='black', lw=1, label='experiment')
        #plt.plot(model.w, model.model_free(tau_loc, tau_shape, tau_scale, xi_shape), color='blue', lw=2, label='model')
        #plt.show()
        
    def plot_experiments():
#         w_max = [0.6, 1., 1.4,  1.8,  2.2,  2.6,  3. ]
#         params_sets = np.array([[  9.76544019e-04,   1.04450297e-01,   4.75852525e-01,
#          1.12973124e+01], [  1.11997268e-03,   6.35727176e-02,   1.07691150e+00,
#          7.21863097e+00], [  1.06153607e-03,   1.03314782e-01,   4.75234392e-01,
#          1.13240343e+01], [  8.36155276e-04,   9.70497391e-02,   5.31003536e-01,
#          9.74947034e+00], [  1.95276197e-03,   2.49629657e-01,   1.01939318e-01,
#          1.00000000e+02], [  4.85766607e-04,   1.01294954e-01,   4.85650168e-01,
#          8.73975564e+00], [  5.04024488e-04,   9.00381081e-02,   4.96583046e-01,
#          7.81837633e+00]])
#         #model.w_max = w_max[0]
#         #model.model_free(*params_sets[0])
#         #for i, w_i in enumerate(w_max):
#         #    model.w_max = 10.0
        plt.plot(model.w, average_exp(), color='black', lw=1, label='test average')
#         #    plt.plot(model.w, model.model_free(*params_sets[i]), lw=2, label=str(i))
#         #for i in range(4):
        plt.plot(model.w, model.model_free(1.43142054e-03, 9.19221066e-02, 5.26535371e-01, 6.87906460), lw=2, label='model')
        plt.legend(loc='best')
        plt.xlabel('crack opening [mm]')
        plt.ylabel('fiber stress [MPa]')
        plt.xlim(0)
        plt.ylim(0)
        plt.show()
    #plot_experiments()
    calibrate()
