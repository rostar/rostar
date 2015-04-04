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
from scipy.special._ufuncs import gamma

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
        #a0 = ef0cb*E_f/T
        #print np.sum(a0>500.)/float(np.sum(a0>-0.1))
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

    def model_free(self, tau_loc, tau_shape, xi_shape):
        xi_scale = 3243. / (182e3 * (pi * 3.5e-3 **2 * 50.)**(-1./xi_shape)*gamma(1+1./xi_shape))
        #xi_scale = 1578. / (182e3 * (pi * 3.5e-3 **2 * 500. * e)**(-1./xi_shape))
        CS=9.
        mu_tau = 1.3 * self.r * 3.6 * (1.-0.01) / (2. * 0.01 * CS)
        tau_scale = (mu_tau - tau_loc)/tau_shape
        #tau_loc=0.0
        cb = CBClampedRandXi(pullout=False)
        spirrid = SPIRRID(q=cb, sampling_type='LHS')
        tau = RV('gamma', shape=tau_shape, scale=tau_scale, loc=tau_loc)
        w = self.w
        spirrid.eps_vars = dict(w=w)
        spirrid.theta_vars = dict(tau=tau, E_f=self.Ef, V_f=self.V_f,
                                  r=self.r, m=xi_shape, sV0=xi_scale)
        spirrid.n_int = 5000
        sigma_c = spirrid.mu_q_arr / self.r ** 2
        plt.plot(w, sigma_c)
        return sigma_c
   
    lack = 1e10
   
    def lack_of_fit(self, params):
        tau_loc = params[0]
        tau_shape = params[1]
        xi_shape = params[2]
        #xi_scale = params[3]
        print params
        lack = np.sum((self.model_free(tau_loc, tau_shape, xi_shape) - self.interpolate_experiment) ** 2)
        print 'params = ', params
#         print 'relative lack of fit', np.sqrt(lack)/np.sum(self.interpolate_experiment)
#         if lack < self.lack:
#             print 'DRAW'
#             self.lack = lack
#             plt.ion()
#             plt.cla()
#             plt.plot(self.w, self.interpolate_experiment, color='black')
#             plt.plot(self.w, self.model_free(tau_loc, tau_shape, xi_shape), color='red', lw=2)
#             plt.draw()
#             plt.show()
            
        return lack

    def eval_params(self):
        params = minimize(self.lack_of_fit, np.array([  1.23223954e-06 ,  1.24752674e-01 ,  3.82485448e+01]),
                          method='L-BFGS-B', bounds=((0.0, .01), (0.01, 1.), (3., 150.)))
        return params

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    plt.figure()

    home_dir = get_home_directory()
    
    model = Model(w_min=0.0,
                      w_max=.50,
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
        for w_max_i in [0.7]:#np.linspace(0.3,1.0,7):
            model.w_max = w_max_i
            model.test_ydata = average_exp()            
            model.test_xdata = model.w
            params = model.eval_params().x
            print 'identified parameters: ', params
            plt.figure()
            plt.plot(model.w, average_exp(), color='black', lw=1, label='test average')
            tau_loc, tau_shape, xi_shape = params
            plt.plot(model.w, model.model_free(tau_loc, tau_shape, xi_shape), lw=2, label='w='+str(w_max_i))
            plt.title('params: tau_loc = %.6f, tau_shape = %.4f, xi_shape = %.4f' % (tau_loc, tau_shape, xi_shape))
            plt.savefig('calibration_figsCS9/w_max%.1f.png'%w_max_i)
        
    def plot_experiments():
        model.w_max=1.
        plt.plot(model.w, average_exp(), color='black', lw=1, label='test average')
        # calibration of tau_shape, tau_scale, tau_loc=0, xi_shape, xi_scale; free deb
        #plt.plot(model.w, model.model_free(3.564e-02,  4.734,   92.17, 4.802e-02), lw=2, label='w=1.5')
        #plt.plot(model.w, model.model_free(4.53967802e-02,  2.9216601,   47.7523345, 3.53100782e-02), lw=2, label='w=1.0')
        #plt.plot(model.w, model.model_free(0.03398613 , 4.95999973 , 9.99003639 , 0.0221401 ), lw=2, label='w=0.5')
        # summary: very high strength in SCM for all cases, already at w_max=0.5, about 75% of fibers a>500mm
        ############################
        # calibration of tau_loc, tau_shape, tau_scale, xi_shape, xi_scale=f(xi_shape, L=70mm); free deb
        #plt.plot(model.w, model.model_free(6.84560881e-04, 1.34477906e-01, 3.29691498e-01, 1.13608995e+01), lw=2, label='w=0.5')
        #plt.plot(model.w, model.model_free(1.44074517e-03, 1.14672028e-01, 3.80366585e-01 ,1.14766610e+01), lw=2, label='w=1.0')
        #plt.plot(model.w, model.model_free(1.27918439e-03 ,  1.20090699e-01 ,  3.55571794e-01,   1.14373865e+01), lw=2, label='w=1.5')
        # summary: one crack in SCM, nonunique calibration
        # calibration of tau_loc, tau_sape, xi_shape, the remaining parameters computed from filament strength and crack spacing
        plt.plot(model.w, model.model_free(2.26522109e-04 ,  1.17730960e-01 ,  1.50000000e+02), lw=2, label='w=%.2f'%model.w_max)
        plt.legend(loc='best')
        plt.xlabel('crack opening [mm]')
        plt.ylabel('fiber stress [MPa]')
        plt.xlim(0)
        plt.ylim(0)
        plt.show()
    #plot_experiments()
    calibrate()
