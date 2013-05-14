'''
Created on Jul 26, 2012

@author: rostar
'''

from etsproxy.traits.api import \
    Instance, Array, List, cached_property, Property
from matplotlib import pyplot as plt
from etsproxy.traits.ui.api import ModelView
from spirrid.rv import RV
from stats.misc.random_field.random_field_1D import RandomField
import numpy as np
from scm2 import SCM2
from scm_dependent_fibers import SCM
from dependent_fibers.reinforcement import Reinforcement, WeibullFibers
from dependent_fibers.depend_CB_model import CompositeCrackBridge
from dependent_fibers.depend_CB_postprocessor import CompositeCrackBridgePostprocessor
from spirrid.rv import RV


class SCMView(ModelView):

    model = Instance(SCM)

    def crack_widths(self, sigma_c):
        # find the index of the nearest value in the load range
        idx = np.abs(self.model.load_sigma_c - sigma_c).argmin()
        # evaluate the relative strain e_rel between fibers
        # and matrix for the given load
        e_rel = self.mu_epsf_x[idx, :] - self.eps_m_x[idx, :]
        # pick the cracks that emerged at the given load
        cb_load = self.model.cb_list(sigma_c)
        if cb_load[0] is not None:
            # find the symmetry points between cracks as
            # the 0 element of their x range
            indexes = []
            for cb in cb_load:
                indexes.append(np.where(cb.position +
                                cb.x[0] == self.model.x_arr)[0])
            # add the index of the last point
            indexes.append(self.model.nx - 1)
            # list of crack widths to be filled in a loop with integrated e_rel
            crack_widths = [np.trapz(e_rel[idx:indexes[i + 1]],
                            self.model.x_arr[idx:indexes[i + 1]])
                            for i, idx in enumerate(indexes[:-1])]
            return np.array(crack_widths)
        else:
            return 0.0

    eval_w = Property(List, depends_on='model')
    @cached_property
    def _get_eval_w(self):
        return [self.crack_widths(load) for load in self.model.load_sigma_c]

    w_mean = Property(Array, depends_on='model')
    @cached_property
    def _get_w_mean(self):
        return np.array([np.mean(w) for w in self.eval_w])

    w_median = Property(Array, depends_on='model')
    @cached_property
    def _get_w_median(self):
        return np.array([np.median(w) for w in self.eval_w])

    w_stdev = Property(Array, depends_on='model')
    @cached_property
    def _get_w_stdev(self):
        return np.array([np.std(w) for w in self.eval_w])

    w_max = Property(Array, depends_on='model')
    @cached_property
    def _get_w_max(self):
        return np.array([np.max(w) for w in self.eval_w])

    x_area = Property(depends_on='model.')
    @cached_property
    def _get_x_area(self):
        return  np.ones_like(self.model.load_sigma_c)[:, np.newaxis] \
            * self.model.x_arr[np.newaxis, :]

    sigma_m_x = Property(depends_on='model.')
    @cached_property
    def _get_sigma_m_x(self):
        sigma_m_x = np.zeros_like(self.model.load_sigma_c[:, np.newaxis]
                                  * self.model.x_arr[np.newaxis, :])
        for i, q in enumerate(self.model.load_sigma_c):
            sigma_m_x[i, :] = self.model.sigma_m(q)
        return sigma_m_x

    eps_m_x = Property(Array, depends_on='model.')
    @cached_property
    def _get_eps_m_x(self):
        return self.sigma_m_x / self.model.E_m

    mu_epsf_x = Property(depends_on='model.')
    @cached_property
    def _get_mu_epsf_x(self):
        mu_epsf_x = np.zeros_like(self.model.load_sigma_c[:, np.newaxis]
                                  * self.model.x_arr[np.newaxis, :])
        for i, q in enumerate(self.model.load_sigma_c):
            mu_epsf_x[i, :] = self.model.epsf_x(q)
#         plt.plot(self.model.x_arr, self.model.epsf_x(8.45))
#         plt.plot(self.model.x_arr, self.model.epsf_x(8.75))
#         plt.show()
        return mu_epsf_x

    eps_sigma = Property(depends_on='model.')
    @cached_property
    def _get_eps_sigma(self):
        eps = np.trapz(self.mu_epsf_x, self.x_area, axis=1) / self.model.length
        eps = eps[np.isnan(eps) == False]
        if len(eps) != len(self.model.load_sigma_c):
            eps = list(eps) + [list(eps)[-1]]
            sigma = self.model.load_sigma_c[:len(eps)]
            sigma[-1] = 0.0
            return eps, sigma
        else:
            return eps, self.model.load_sigma_c

if __name__ == '__main__':
    import time
    length = 4000.
    nx = 4000
    random_field = RandomField(seed=True,
                               lacor=10.,
                                xgrid=np.linspace(0., length, 200),
                                nsim=1,
                                loc=.0,
                                shape=15.,
                                scale=6.,
                                non_negative_check=True,
                                distribution='Weibull'
                               )

    reinf = Reinforcement(r=0.00345,
                          tau=RV('weibull_min', shape=1.5, scale=.2),
                          V_f=0.0103,
                          E_f=200e3,
                          xi=WeibullFibers(shape=5., sV0=0.00618983207723),
                          n_int=50,
                          label='carbon')

    model = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf],
                                 )

    ccb_post = CompositeCrackBridgePostprocessor(model=model)

    scm = SCM(length=length,
              nx=nx,
              random_field=random_field,
              E_m = 25e3,
              reinforcement = reinf,
              load_sigma_c_min=.1,
              load_sigma_c_max=18.,
              load_n_sigma_c=200
              )

    t = time.clock()
    scm.evaluate()

    scm_view = SCMView(model=scm)
    scm_view.model.evaluate()

    def plot():
        eps, sigma = scm_view.eps_sigma
        plt.figure()
        plt.plot(eps, sigma, color='black', lw=2, label='model')
        plt.legend(loc='best')
        plt.xlabel('composite strain [-]')
        plt.ylabel('composite stress [MPa]')
        print 'time =', time.clock() - t
        plt.figure()
        plt.hist(scm_view.crack_widths(16.), bins=20, label='load = 20 MPa')
        plt.hist(scm_view.crack_widths(13.), bins=20, label='load = 15 MPa')
        plt.hist(scm_view.crack_widths(10.), bins=20, label='load = 10 MPa')
        plt.legend(loc='best')
        plt.figure()
        plt.plot(scm.load_sigma_c, scm_view.w_mean,
                 color='green', lw=2, label='mean crack width')
        plt.plot(scm.load_sigma_c, scm_view.w_median,
                 color='blue', lw=2, label='median crack width')
        plt.plot(scm.load_sigma_c, scm_view.w_mean + scm_view.w_stdev,
                 color='black', label='stdev')
        plt.plot(scm.load_sigma_c, scm_view.w_mean - scm_view.w_stdev,
                 color='black')
        plt.plot(scm.load_sigma_c, scm_view.w_max,
                 ls='dashed', color='red', label='max crack width')
        plt.legend(loc='best')
        plt.show()
    plot()
