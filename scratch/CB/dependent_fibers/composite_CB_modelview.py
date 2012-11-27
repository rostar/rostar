'''
Created on 16.11.2012

@author: Q
'''
from etsproxy.traits.ui.api import ModelView
from etsproxy.traits.api import Instance, Property, cached_property, Array
from composite_CB_model import CompositeCrackBridge
import numpy as np
from matplotlib import pyplot as plt
from stats.spirrid.rv import RV
from reinforcement import Reinforcement, WeibullFibers
from scipy.optimize import fmin, fmin_cg, fmin_ncg, fmin_bfgs, bracket, golden, brent
from scipy.integrate import cumtrapz
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray


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
        E_mtrx = (1. - self.model.V_f_tot) * self.model.E_m + Kf_broken
        mu_epsf_arr = (sigma_c - E_mtrx * self.model._epsm_arr) / (self.model.E_c - E_mtrx)
        return self.model._x_arr, self.model._epsm_arr, sigma_c, mu_epsf_arr, E_mtrx

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
        u_compact = ((self.model.Ll - np.abs(self.x_arr[0])) * self.mu_epsf_arr[0]
                    + (self.model.Lr - np.abs(self.x_arr[-1])) * self.mu_epsf_arr[-1])
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

    Welm = Property(depends_on='model.E_m, model.w, model.Ll, model.Lr, model.reinforcement_lst+')
    @cached_property
    def _get_Welm(self):
        Km = self.results[4]
        bonded_l = self.epsm_arr[0]**2 * Km * (self.model.Ll - np.abs(self.x_arr[0]))
        bonded_r = self.epsm_arr[-1]**2 * Km * (self.model.Lr - np.abs(self.x_arr[-1]))
        return 0.5 * (np.trapz(self.epsm_arr**2 * Km, self.x_arr) + bonded_l + bonded_r) 

    Welf = Property(depends_on='model.E_m, model.w, model.Ll, model.Lr, model.reinforcement_lst+')
    @cached_property
    def _get_Welf(self):
        Kf = self.model.E_c - self.results[4]
        bonded_l = self.mu_epsf_arr[0] ** 2 * Kf * (self.model.Ll - np.abs(self.x_arr[0]))
        bonded_r = self.mu_epsf_arr[-1] ** 2 * Kf * (self.model.Lr - np.abs(self.x_arr[-1]))
        return 0.5 * (np.trapz(self.mu_epsf_arr ** 2 * Kf, self.x_arr) + bonded_l + bonded_r)

    W_el_tot = Property(depends_on='model.E_m, model.w, model.Ll, model.Lr, model.reinforcement_lst+')
    @cached_property
    def _get_W_el_tot(self):
        '''total elastic energy stored in the specimen'''
        return self.Welf + self.Welm

    W_inel_tot = Property(depends_on='model.E_m, model.w, model.Ll, model.Lr, model.reinforcement_lst+')
    @cached_property
    def _get_W_inel_tot(self):
        '''total inelastic energy dissipated during loading up to w'''
        return self.U - self.W_el_tot

    U_line = Property(depends_on='model.E_m, model.Ll, model.Lr, model.reinforcement_lst+, w_arr_energy')
    @cached_property
    def _get_U_line(self):
        '''work done by external force - mfn_line'''
        w_arr = self.w_arr_energy
        u_lst = []
        F_lst = []
        for w in w_arr:
            self.model.w = w
            u_lst.append(self.u_evaluated)
            F_lst.append(self.sigma_c)
        u_arr = np.array(u_lst)
        F_arr = np.array(F_lst)
        U_line = MFnLineArray(xdata=w_arr, ydata=np.hstack((0, cumtrapz(F_arr, u_arr))))
        return U_line

    U = Property(depends_on='model.E_m, model.Ll, model.Lr, model.reinforcement_lst+, model.w')
    @cached_property
    def _get_U(self):
        '''work done by external force U(w)'''
        return self.U_line.get_values(self.model.w)

    w_arr_energy = Array

if __name__ == '__main__':

    reinf1 = Reinforcement(r=0.00345,#RV('uniform', loc=0.001, scale=0.005),
                          tau=RV('uniform', loc=.5, scale=.2),
                          V_f=0.2,
                          E_f=70e3,
                          xi=WeibullFibers(shape=5., scale=0.02, L0=10.),#RV('weibull_min', shape=5., scale=.02),
                          n_int=15,
                          label='AR glass')

    reinf2 = Reinforcement(r=0.003,#RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=.3, scale=.4),
                          V_f=0.1,
                          E_f=200e3,
                          xi=RV('weibull_min', shape=5., scale=0.02),
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
                                 Ll=30.,
                                 Lr=30.)

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

    def energy(w_arr):
        ccb_view.w_arr_energy = w_arr
        Welm = []
        Welf = []
        Wel_tot = []
        U = []
        Winel = []
        u = []
        ccb_view.U_line
        for w in w_arr:
            ccb_view.model.w = w
            Wel_tot.append(ccb_view.W_el_tot)
            Welm.append(ccb_view.Welm)
            Welf.append(ccb_view.Welf)
            U.append(ccb_view.U)
            Winel.append(ccb_view.W_inel_tot)
            u.append(ccb_view.u_evaluated)
        plt.plot(w_arr, Welm, lw=2, label='Welm')
        plt.plot(w_arr, Welf, lw=2, label='Welf')
        plt.plot(w_arr, Wel_tot, lw=2, color='black', label='elastic strain energy')
        plt.plot(w_arr, Winel, lw=2, ls='dashed', color='black', label='inelastic energy')
        plt.plot(w_arr, U, lw=3, color='red', label='work of external force')
        plt.xlabel('w [mm]')
        plt.ylabel('W')
        plt.legend(loc='best')

    #TODO: check energy for combuned reinf
    energy(np.linspace(.0, .5, 60))
    #profile(.03)
    #sigma_c_w(np.linspace(.0, .45, 150))
    #plt.plot(ccb_view.sigma_c_max[1], ccb_view.sigma_c_max[0], 'ro')
    #sigma_f(np.linspace(.0, .3, 50))
    plt.legend(loc='best')
    plt.show()