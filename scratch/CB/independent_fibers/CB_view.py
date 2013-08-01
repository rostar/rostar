'''
Created on 03.07.2013

@author: acki
'''
import numpy as np
from scipy.interpolate import interp1d 
from scipy.optimize import minimize, basinhopping, brute
import os
from etsproxy.traits.api import HasTraits, Property, Callable, Array, cached_property
from etsproxy.pyface.image_resource import ImageResource
from etsproxy.traits.api import HasTraits, Float, Int, Event, Array, Interface, \
    Tuple, Property, cached_property, Instance, Enum, on_trait_change, Dict
from etsproxy.traits.ui.api import Item, View, Group, HSplit, VGroup, Tabbed
from etsproxy.traits.ui.menu import OKButton, CancelButton
from math import sqrt
from matplotlib.figure import Figure
from quaducom.micro.resp_func.CB_rigid_mtrx_rand_xi import CBResidualRandXi
from spirrid.spirrid import SPIRRID
from spirrid.rv import RV
from matplotlib import pyplot as plt
from scipy.special import gamma, gammainc
from math import pi
from util.traits.editors.mpl_figure_editor import MPLFigureEditor
from etsproxy.traits.ui.api import ModelView

FILE_DIR = os.path.dirname(__file__)


class Model(HasTraits):

    test_xdata = Array
    test_ydata = Array
    sV0 = Float(auto_set=False, enter_set=True, params=True)
    m = Float(auto_set=False, enter_set=True, params=True)
    w_min = Float(auto_set=False, enter_set=True, params=True)
    w_max = Float(auto_set=False, enter_set=True, params=True)
    w_pts = Int(auto_set=False, enter_set=True, params=True)
    tau_scale = Float(auto_set=False, enter_set=True, params=True)
    tau_loc = Float(auto_set=False, enter_set=True, params=True)
    tau_shape = Float(auto_set=False, enter_set=True, params=True)

    w = Property(Array)
    def _get_w(self):
        return np.linspace(self.w_min, self.w_max, self.w_pts)

    interpolate_experiment = Property(depends_on='test_xdata, test_ydata')
    @cached_property
    def _get_interpolate_experiment(self):
        return interp1d(self.test_xdata, self.test_ydata,
                        bounds_error=False, fill_value=0.0)

    model_rand = Property(Array)
    def _get_model_rand(self):
        cb = CBResidualRandXi()
        spirrid = SPIRRID(q=cb, sampling_type='PGrid')
        sV0 = self.sV0
        tau_scale = self.tau_scale
        E_f = 180e3
        V_f = 1.0
        r = 3.45e-3
        m = self.m
        tau = RV('weibull_min', shape=self.tau_shape, scale=tau_scale, loc=self.tau_loc)
        n_int = 100
        w = self.w
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

class CBView(ModelView):

    def __init__(self, **kw):
        super(CBView, self).__init__(**kw)
        self.on_trait_change(self.refresh, 'model.+params')
        self.refresh()

    model = Instance(Model)

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

    def refresh(self):
        self.plot(self.figure)
        self.data_changed = True

    traits_view = View(HSplit(VGroup(Group(Item('model.tau_scale'),
                                           Item('model.tau_shape'),
                                           Item('model.tau_loc'),
                                           Item('model.m'),
                                           Item('model.sV0'),
                                           Item('model.w_min'),
                                           Item('model.w_max'),
                                           Item('model.w_pts'),
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
    from quaducom.micro.resp_func.CB_rigid_mtrx_rand_xi import CBResidualRandXi
    from spirrid.spirrid import SPIRRID
    from spirrid.rv import RV
    from matplotlib import pyplot as plt
    from scipy.special import gamma, gammainc
    from math import pi

    model = Model(w_min=0.0, w_max=8.0, w_pts=500,
                  sV0=3.2e-3, m=5.0, tau_scale=0.04,
                  tau_shape=0.22, tau_loc=0.007)

    file1 = open('DATA/PO01_RYP.ASC', 'r')
    model.test_xdata = - np.loadtxt(file1, delimiter=';')[:,3]
    model.test_xdata = model.test_xdata - model.test_xdata[0]
    file2 = open('DATA/PO01_RYP.ASC', 'r')
    model.test_ydata = (np.loadtxt(file2, delimiter=';')[:,1] + 0.035)/0.45 * 1000
    cb = CBView(model=model)
    cb.refresh()
    cb.configure_traits()
