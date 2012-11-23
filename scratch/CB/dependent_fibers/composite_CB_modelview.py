'''
Created on 16.11.2012

@author: Q
'''
from etsproxy.traits.ui.api import ModelView
from etsproxy.traits.api import Instance, Property, cached_property
from composite_CB_model import CompositeCrackBridge
import numpy as np
from matplotlib import pyplot as plt
from stats.spirrid.rv import RV
from reinforcement import Reinforcement, WeibullFibers
from scipy.optimize import fmin, fmin_cg, fmin_ncg, fmin_bfgs, bracket, golden, brent


class CompositeCrackBridgeView(ModelView):

    model = Instance(CompositeCrackBridge)
    results = Property(depends_on='model.E_m, model.w, model.Ll, model.Lr, model.reinforcement_lst+')
    @cached_property
    def _get_results(self):
        if self.model.w == 0.0:
            self.model.w = 1e-15
        self.model.damage
        sigma_c = np.sum(self.model._epsf0_arr * self.model.sorted_stats_weights * self.model.sorted_V_f *
                      self.model.sorted_nu_r * self.model.sorted_E_f * (1. - self.model.damage))
        Kf_broken = np.sum(self.model.sorted_V_f * self.model.sorted_nu_r * \
            self.model.sorted_stats_weights * self.model.sorted_E_f * self.model.damage)
        E_mtrx_arr = (1. - self.model.V_f_tot) * self.model.E_m + Kf_broken
        mu_epsf_arr = (sigma_c - E_mtrx_arr * self.model._epsm_arr) / (self.model.E_c - E_mtrx_arr)
        return self.model._x_arr, self.model._epsm_arr, sigma_c, mu_epsf_arr

    x_arr = Property(depends_on='model.E_m, model.w, model.Ll, model.Lr, model.reinforcement_lst+')
    @cached_property
    def _get_x_arr(self):
        return self.results[0]        

    epsm_arr = Property(depends_on='model.E_m, model.w, model.Ll, model.Lr, model.reinforcement_lst+')
    @cached_property
    def _get_epsm_arr(self):
        return self.results[1]

    sigma_c = Property(depends_on='model.E_m, model.w, model.Ll, model.Lr, model.reinforcement_lst+')
    @cached_property
    def _get_sigma_c(self):
        return self.results[2]

    mu_epsf_arr = Property(depends_on='model.E_m, model.w, model.Ll, model.Lr, model.reinforcement_lst+')
    @cached_property
    def _get_mu_epsf_arr(self):
        return self.results[3]

    w_evaluated = Property(depends_on='model.E_m, model.w, model.Ll, model.Lr, model.reinforcement_lst+')
    @cached_property
    def _get_w_evaluated(self):
        return np.trapz(self.mu_epsf_arr - self.epsm_arr, self.x_arr)

    u_evaluated = Property(depends_on='model.E_m, model.w, model.Ll, model.Lr, model.reinforcement_lst+')
    @cached_property
    def _get_u_evaluated(self):
        u_debonded = np.trapz(self.mu_epsf_arr, self.x_arr)
        u_compact = self.model.Ll * self.mu_epsf_arr[0] + self.model.Lr * self.mu_epsf_arr[-1]
        return u_debonded + u_compact

    sigma_c_max = Property(depends_on='model.E_m, model.w, model.Ll, model.Lr, model.reinforcement_lst+')
    @cached_property
    def _get_sigma_c_max(self):
        def minfunc(w):
            self.model.w = float(w)
            return - self.sigma_c
        wmax = fmin(minfunc, 0.0001, maxiter=30)
        return self.sigma_c, wmax
    
    def sigma_f_lst(self, w_arr):
        sigma_f_arr = np.zeros(len(w_arr) *
                               len(self.model.reinforcement_lst)).reshape(len(w_arr),
                                len(self.model.reinforcement_lst))
        masks = [((self.model.sorted_xi == reinf.xi) *
                          (self.model.sorted_E_f == reinf.E_f) *
                          (self.model.sorted_V_f == reinf.V_f))
                 for reinf in self.model.reinforcement_lst]
        for i, w in enumerate(w_arr):
            if w == 0.0:
                self.model.w = 1e-15
            else:
                self.model.w = w
            self.model.damage
            for j, reinf in enumerate(self.model.reinforcement_lst):
                sigma_fi = np.sum(self.model._epsf0_arr * self.model.sorted_stats_weights * self.model.sorted_nu_r *
                              self.model.sorted_E_f * (1. - self.model.damage) * masks[j])
                sigma_f_arr[i, j] = sigma_fi
        return sigma_f_arr
    
if __name__ == '__main__':

    reinf1 = Reinforcement(r=0.00345,#RV('uniform', loc=0.001, scale=0.005),
                          tau=RV('uniform', loc=.5, scale=.2),
                          V_f=0.2,
                          E_f=70e3,
                          xi=WeibullFibers(shape=5., scale=0.02, L0=10.),#RV('weibull_min', shape=5., scale=.02),
                          n_int=15,
                          label='AR glass')

    reinf2 = Reinforcement(r=0.002,#RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=.3, scale=.4),
                          V_f=0.03,
                          E_f=200e3,
                          xi=RV('weibull_min', shape=5., scale=.02),
                          n_int=15,
                          label='carbon')

    reinf3 = Reinforcement(r=0.00345,#RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=1., scale=3.),
                          V_f=0.1,
                          E_f=200e3,
                          xi=0.03,
                          n_int=20)

    model = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf2],
                                 Ll=50.,
                                 Lr=50.)

    ccb_view = CompositeCrackBridgeView(model=model)

    def profile(w):
        ccb_view.model.w = w
        plt.plot(ccb_view.x_arr, ccb_view.epsm_arr, label='w_eval=' + str(ccb_view.w_evaluated) + ' w_ctrl=' + str(ccb_view.model.w))
        plt.plot(ccb_view.x_arr, ccb_view.mu_epsf_arr, label='yarn')
        plt.xlabel('position [mm]')
        plt.ylabel('strain')

    def sigma_c_w(w_arr):
        sigma_c = []
        w_err = []
        u = []
        for w in w_arr:
            ccb_view.model.w = w
            sigma_c.append(ccb_view.sigma_c)
            w_err.append((ccb_view.w_evaluated - ccb_view.model.w) / (ccb_view.model.w))
            u.append(ccb_view.u_evaluated)
        plt.figure()
        plt.plot(w_arr, w_err, label='error in w')
        plt.legend(loc='best')
        plt.figure()
        plt.plot(w_arr, sigma_c, lw=2, label='rigid')
        plt.plot(u, sigma_c, lw=2, label='elastic')
        plt.xlabel('u [mm]')
        plt.ylabel('$\sigma_c$ [MPa]')
        plt.legend(loc='best')

    def sigma_f(w_arr):
        sf_arr = ccb_view.sigma_f_lst(w_arr)
        for i, reinf in enumerate(ccb_view.model.reinforcement_lst):
            plt.plot(w_arr, sf_arr[:, i], label=reinf.label)

    #profile(.03)
    sigma_c_w(np.linspace(.0, .45, 150))
    plt.plot(ccb_view.sigma_c_max[1], ccb_view.sigma_c_max[0], 'ro')
    #sigma_f(np.linspace(.0, .3, 50))
    plt.legend(loc='best')
    plt.show()