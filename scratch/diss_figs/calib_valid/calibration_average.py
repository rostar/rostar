import numpy as np
from scipy.interpolate import interp1d
import os
from etsproxy.traits.api import HasTraits, Property, Array, \
     cached_property, Float, Int
from quaducom.micro.resp_func.CB_clamped_rand_xi import CBClampedRandXi
from spirrid.spirrid import SPIRRID
from spirrid.rv import RV
from etsproxy.util.home_directory import get_home_directory
from scipy.optimize import minimize
from matplotlib import pyplot as plt


class Model(HasTraits):

    test_xdata = Array
    test_ydata = Array
    sV0 = Float(0.007, auto_set=False, enter_set=True, params=True)
    m = Float(6., auto_set=False, enter_set=True, params=True)
    w_min = Float(0.0, auto_set=False, enter_set=True, params=True)
    w_max = Float(1., auto_set=False, enter_set=True, params=True)
    w_pts = Int(100, auto_set=False, enter_set=True, params=True)
    n_int = Int(100, auto_set=False, enter_set=True, params=True)
    tau_scale = Float(1.0, auto_set=False, enter_set=True, params=True)
    tau_loc = Float(0.005, auto_set=False, enter_set=True, params=True)
    tau_shape = Float(0.2, auto_set=False, enter_set=True, params=True)
    Ef = Float(181e3, auto_set=False, enter_set=True, params=True)
    lm = Float(13.0, auto_set=False, enter_set=True, params=True)
    V_f = Float(1.0, params=True)
    r = Float(3.5e-3, params=True)

    w = Property(Array)
    def _get_w(self):
        return np.linspace(self.w_min, self.w_max, self.w_pts)

    interpolate_experiment = Property(depends_on='test_xdata, test_ydata')
    @cached_property
    def _get_interpolate_experiment(self):
        interp = interp1d(self.test_xdata, self.test_ydata,
                        bounds_error=False, fill_value=0.0)
        return interp(self.w)
    
    average_ydata = Array

    def model_free(self, tau_shape, tau_scale, tau_loc):
        cb = CBClampedRandXi(pullout=True)
        spirrid = SPIRRID(q=cb, sampling_type='LHS')
        sV0 = self.sV0
        V_f = self.V_f
        r = self.r
        m = self.m
        tau = RV('gamma', shape=tau_shape, scale=tau_scale, loc=tau_loc)
        n_int = self.n_int
        w = self.w
        lm = 1000.
        spirrid.eps_vars = dict(w=w)
        spirrid.theta_vars = dict(tau=tau, E_f=self.Ef, V_f=V_f, r=r, m=m, sV0=sV0, lm=lm)
        spirrid.n_int = n_int
        sigma_c = spirrid.mu_q_arr / self.r ** 2
        return sigma_c

    def model_clamped(self, tau_shape, tau_scale, tau_loc):
        cb = CBClampedRandXi(pullout=True)
        spirrid = SPIRRID(q=cb, sampling_type='LHS')
        sV0 = self.sV0
        V_f = self.V_f
        r = self.r
        m = self.m
        tau = RV('gamma', shape=tau_shape, scale=tau_scale, loc=tau_loc)
        n_int = self.n_int
        w = self.w2
        lm = self.lm
        spirrid.eps_vars = dict(w=w)
        spirrid.theta_vars = dict(tau=tau, E_f=self.Ef, V_f=V_f, r=r, m=m, sV0=sV0, lm=lm)
        spirrid.n_int = n_int
        sigma_c = spirrid.mu_q_arr / self.r ** 2
        return sigma_c
   
    lack = 1e10
   
    def lack_of_fit(self, params):
        tau_shape = params[0]
        tau_scale = params[1]
        tau_loc = 0.0#params[2]
        weights = np.ones_like(np.linspace(10.0, 0.0, len(self.w)))
        lack = np.sum(weights * (self.model_free(tau_shape, tau_scale, tau_loc) - self.average_ydata) ** 2)
        print 'params = ', params
        print 'lack of fit', lack
        if lack < self.lack:
            self.lack = lack
            plt.ion()
            plt.cla()
            plt.plot(self.w, self.average_ydata, color='black')
            plt.plot(self.w, self.model_free(tau_shape, tau_scale, tau_loc), color='red', lw=2)
            plt.draw()
        return lack

    def eval_params(self):
        # params for CB1 = 0.090, 0.901, 0.0
        # params for CB2 = 0.130, 0.255, 0.0
        # params for CB3 = 0.064, 2.185, 0.0
        # params for CB4 = 0.096, 0.409, 0.0
        # params for CB5 = 0.109, 0.795, 0.0
        params = minimize(self.lack_of_fit, np.array([0.04347164, 2.83474059]),
                          method='L-BFGS-B', bounds=((0.04, 1.), (0.01, 3.)))
        return params

if __name__ == '__main__':

    model = Model()

    model.average_ydata = np.zeros_like(model.w)
    home_dir = get_home_directory()
    for i in range(5):
        path = [home_dir, 'git',  # the path of the data file
                'rostar',
                'scratch',
                'diss_figs',
                'CB'+str(i+1)+'.txt']
        filepath = os.path.join(*path)
        file1 = open(filepath, 'r')
        cb = np.loadtxt(file1, delimiter=';')
        model.test_xdata = -cb[:, 2] / 4. - cb[:, 3] / 4. - cb[:, 4] / 2.
        model.test_ydata = cb[:, 1] / (11. * 0.445) * 1000
        model.average_ydata += model.interpolate_experiment / 5.

    params = model.eval_params().x
    print 'parameters: ', params
    shape, scale = params
    loc = 0.0
    plt.plot(model.w, model.average_ydata, color='black', lw=1, label='experiment')
    plt.plot(model.w, model.model_free(shape, scale, loc), color='blue', lw=2, label='model')
    
#     CB1 = 0.090, 0.901, 0.0
#     CB2 = 0.130, 0.255, 0.0
#     CB3 = 0.064, 2.185, 0.0
#     CB4 = 0.096, 0.409, 0.0
#     CB5 = 0.109, 0.795, 0.0
#     CBs = [CB1, CB2, CB3, CB4, CB5]
#     
#     for i in range(6):
#         if i == 5:
#             shape, scale, loc = np.mean(np.array(CBs), axis=0)
#             print shape, scale, loc
#             plt.figure()
#             for i in range(5):
#                 path = [home_dir, 'git',  # the path of the data file
#                 'rostar',
#                 'scratch',
#                 'diss_figs',
#                 'CB' + str(i+1) +'.txt']
#                 filepath = os.path.join(*path)
#                 file1 = open(filepath, 'r')
#                 cb = np.loadtxt(file1, delimiter=';')
#                 model.test_xdata = -cb[:, 2] / 4. - cb[:, 3] / 4. - cb[:, 4] / 2.
#                 model.test_ydata = cb[:, 1] / (11. * 0.445) * 1000
#                 plt.plot(model.w, model.interpolate_experiment, color='black', lw=1, label='experiment ' + str(i+1))
#             plt.plot(model.w, model.model_free(shape, scale, loc), color='blue', lw=2, label='model')
#             plt.legend(loc='best')
#             plt.xlabel('crack opening [mm]')
#             plt.ylabel('fiber stress [MPa]')
#         else:
#             shape, scale, loc = CBs[i]
#             plt.figure()
#             path = [home_dir, 'git',  # the path of the data file
#                 'rostar',
#                 'scratch',
#                 'diss_figs',
#                 'CB' + str(i+1) +'.txt']
#             filepath = os.path.join(*path)
#             file1 = open(filepath, 'r')
#             cb = np.loadtxt(file1, delimiter=';')
#             model.test_xdata = -cb[:, 2] / 4. - cb[:, 3] / 4. - cb[:, 4] / 2.
#             model.test_ydata = cb[:, 1] / (11. * 0.445) * 1000
#             plt.plot(model.w, model.interpolate_experiment, color='black', lw=1, label='experiment ' + str(i+1))
#             plt.plot(model.w, model.model_free(shape, scale, loc), color='blue', lw=2, label='model')
#             plt.legend(loc='best')
#             plt.xlabel('crack opening [mm]')
#             plt.ylabel('fiber stress [MPa]')
    plt.show()
    

