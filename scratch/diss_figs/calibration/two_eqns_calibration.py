'''
Created on Feb 16, 2015

@author: rostislavrypl
'''
from quaducom.meso.homogenized_crack_bridge.rigid_matrix.CB_view import Model
from etsproxy.traits.api import Property, Float, cached_property, Instance, Event
from etsproxy.util.home_directory import get_home_directory
import os
import numpy as np
from scipy.optimize import fsolve
from scipy.special import gamma as gamma_func
from etsproxy.traits.ui.api import Item, View, Group, HSplit, VGroup, Tabbed
from etsproxy.traits.ui.menu import OKButton, CancelButton
from util.traits.editors.mpl_figure_editor import MPLFigureEditor
from matplotlib.figure import Figure
from etsproxy.traits.ui.api import ModelView

class _Model(Model):
    tau_scale = Property(Float, depends_on='test_xdata,test_ydata,Ef,r,V_f,CS,sigmamu, tau_scale1, m, w_hat')
    @cached_property
    def _get_tau_scale(self):
        sigmaf_hat = self.interpolate_experiment(self.w_hat)
        mu_sqrt_tau = sigmaf_hat / np.sqrt(2. * self.Ef * self.w_hat / self.r)
        mu_tau = 1.3 * self.r *self.sigmamu* (1.-self.V_f) / (2. * self.V_f * self.CS)
        def scale_res(scale):
            res = scale - ((mu_sqrt_tau) / (gamma_func(mu_tau/scale + 0.5)/gamma_func(mu_tau/scale)))**2
            return res
        scale = fsolve(scale_res, 0.52)
        return float(scale)
      
    tau_shape = Property(Float, depends_on='test_xdata,test_ydata,Ef,r,V_f,CS,sigmamu, tau_shape1, m, w_hat')
    @cached_property
    def _get_tau_shape(self):
        mu_tau = 1.3 * self.r * self.sigmamu * (1.-self.V_f) / (2. * self.V_f * self.CS)
        return mu_tau/self.tau_scale
    
    CS=Float(auto_set=False, enter_set=True, params=True)
    sigmamu=Float(auto_set=False, enter_set=True, params=True)
    w_hat=Float(auto_set=False, enter_set=True, params=True)

class _CBView(ModelView):

    def __init__(self, **kw):
        super(_CBView, self).__init__(**kw)
        self.on_trait_change(self.refresh, 'model.+params')
        self.refresh()

    model = Instance(_Model)

    figure = Instance(Figure)
    def _figure_default(self):
        figure = Figure(facecolor='white')
        return figure

    data_changed = Event

    def plot(self, fig):
        figure = fig
        figure.clear()
        axes = figure.gca()
        # plot PDF
        axes.plot(self.model.w, self.model.model_rand, lw=2.0, color='blue', \
                  label='model')
        axes.plot(self.model.w, self.model.interpolate_experiment(self.model.w), lw=1.0, color='black', \
                  label='experiment')
        axes.legend()

    def refresh(self):
        self.plot(self.figure)
        self.data_changed = True

    traits_view = View(HSplit(VGroup(Group(Item('model.tau_scale'),
                                           Item('model.tau_shape'),
                                           Item('model.m'),
                                           Item('model.sV0'),
                                           Item('model.Ef'),
                                           Item('model.w_min'),
                                           Item('model.w_max'),
                                           Item('model.w_pts'),
                                           Item('model.n_int'),
                                           Item('model.CS'),
                                           Item('model.sigmamu'),
                                           Item('model.w_hat')
                                           ),
                                      id='pdistrib.distr_type.pltctrls',
                                      label='Distribution parameters',
                                      scrollable=True,
                                      ),
                                Tabbed(Group(Item('figure',
                                            editor=MPLFigureEditor(),
                                            show_label=False,
                                            resizable=True),
                                            scrollable=True,
                                            label='Plot',
                                            ),
                                        label='Plot',
                                        id='pdistrib.figure.params',
                                        dock='tab',
                                       ),
                                dock='tab',
                                id='pdistrib.figure.view'
                                ),
                                id='pdistrib.view',
                                dock='tab',
                                title='Statistical distribution',
                                buttons=[OKButton, CancelButton],
                                scrollable=True,
                                resizable=True,
                                width=600, height=400
                        )

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    model = _Model(w_min=0.0, w_max=3.0, w_pts=200,
                  w2_min=0.0, w2_max=.5, w2_pts=200,
                  sV0=7e-3, m=7.1, tau_loc=0.0, Ef=180e3,
                  lm=20., n_int=100, CS=10.0, sigmamu=3.0,w_hat=0.05)
    
    w_arr = np.linspace(0.0,8.0,5000)
    home_dir = get_home_directory()
    sigma_f = np.zeros_like(w_arr)
    for i in range(1):
        path_i = [home_dir, 'git',  # the path of the data file
                'rostar',
                'scratch',
                'diss_figs',
                'CB'+str(i+1)+'.txt']
        filepath_i = os.path.join(*path_i)
        file_i = open(filepath_i, 'r')
        spec_i = np.loadtxt(file_i, delimiter=';')
        w_i = -spec_i[:, 2] / 4. - spec_i[:, 3] / 4. - spec_i[:, 4] / 2.
        sigma_f_i = spec_i[:, 1] / (11. * 0.445) * 1000.
        interpolator_i = interp1d(w_i, sigma_f_i,
                        bounds_error=False, fill_value=0.0)
        sigma_f += interpolator_i(w_arr)# / 5.

    model.test_xdata = w_arr
    model.test_ydata = sigma_f
    cb = _CBView(model=model)
    cb.refresh()
    cb.configure_traits()