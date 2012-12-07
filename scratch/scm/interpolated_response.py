'''
Created on Aug 17, 2011

@author: rostar
'''

from etsproxy.traits.api import HasTraits, Float, Array, \
    Instance, cached_property, Property, Tuple
from scratch.CB.dependent_fibers.composite_CB_postprocessor import \
    CompositeCrackBridgeView
from stats.spirrid.rv import RV
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray


class InterpolatedCB(HasTraits):

    cb = Instance(CompositeCrackBridgeView)
    position = Float
    Ll = Float
    Lr = Float
    x = Array
    load_arr = Array
    load = Float

    eps_x_arr_iterpolation = Property(depends_on='cb, Ll, Lr, load_arr')
    @cached_property
    def _get_eps_x_arr_iterpolation(self):
        self.cb.model.Ll = self.Ll
        self.cb.model.Lr = self.Lr
        self.cb.apply_load(self.load_arr[-1])
        wmax = self.cb.model.w
        self.cb.apply_load(self.load_arr[0])
        w_at_crack = self.cb.model.w
        w_arr = np.linspace(w_at_crack, wmax, len(self.load_arr))
        x_points = np.array([])
        sigma_points = np.array([])
        epsm_values = np.array([])
        epsf_values = np.array([])
        for w in w_arr:
            self.cb.model.w = w
            xi = np.hstack((-self.Ll, self.cb.x_arr, self.Lr))
            x_points = np.hstack((x_points, xi))
            sigmai = np.repeat(self.cb.sigma_c, len(xi))
            sigma_points = np.hstack((sigma_points, sigmai))
            epsmi = np.hstack((self.cb.epsm_arr[0], self.cb.epsm_arr, self.cb.epsm_arr[-1]))
            epsm_values = np.hstack((epsm_values, epsmi))
            epsfi = np.hstack((self.cb.mu_epsf_arr[0], self.cb.mu_epsf_arr, self.cb.mu_epsf_arr[-1]))
            epsf_values = np.hstack((epsf_values, epsfi))
        coordinates = np.hstack((sigma_points, x_points)).reshape(2, len(x_points)).T
        interp_epsm = LinearNDInterpolator(coordinates, epsm_values)
        interp_epsf = LinearNDInterpolator(coordinates, epsf_values)
        return interp_epsm, interp_epsf

    eps_x_interpolation = Property(depends_on='cb, Ll, Lr, load')
    @cached_property
    def _get_eps_x_interpolation(self):
        self.cb.model.Ll = self.Ll
        self.cb.model.Lr = self.Lr
        self.cb.apply_load(self.load)
        x_points = np.hstack((-self.Ll, self.cb.x_arr, self.Lr))
        epsm = np.hstack((self.cb.epsm_arr[0], self.cb.epsm_arr, self.cb.epsm_arr[-1]))
        epsf = np.hstack((self.cb.mu_epsf_arr[0], self.cb.mu_epsf_arr, self.cb.mu_epsf_arr[-1]))
        interp_epsm = MFnLineArray(xdata=x_points, ydata=epsm)
        interp_epsf = MFnLineArray(xdata=x_points, ydata=epsf)
        return interp_epsm, interp_epsf

    def get_epsm_x(self, load):
        self.load = load
        return self.eps_x_interpolation[0].get_values(self.x)

    def get_epsf_x(self, load):
        self.load = load
        return self.eps_x_interpolation[1].get_values(self.x)

    def get_sigmam_x(self, load):
        return self.get_epsm_x(load) * self.cb.model.E_m

    def get_epsm_x_arr(self, load_arr):
        self.load_arr = load_arr
        sigma, x = np.meshgrid(load_arr, self.x)
        return self.eps_x_arr_iterpolation[0](sigma, x)

    def get_epsf_x_arr(self, load_arr):
        self.load_arr = load_arr
        sigma, x = np.meshgrid(load_arr, self.x)
        return self.eps_x_arr_iterpolation[1](sigma, x)

    def get_sigmam_x_arr(self, load_arr):
        return self.get_epsm_x_arr(load_arr) * self.cb.model.E_m

if __name__ == '__main__':
    from etsproxy.mayavi import mlab as m
    from stats.spirrid import make_ogrid as orthogonalize
    from matplotlib import pyplot as plt
    from scratch.CB.dependent_fibers.composite_CB_model import CompositeCrackBridge
    from scratch.CB.dependent_fibers.reinforcement import Reinforcement, WeibullFibers

    reinf = Reinforcement(r=0.00345,#RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=.3, scale=3.),
                          V_f=0.1,
                          E_f=200e3,
                          xi=100.03,
                          n_int=20)
    Ll = 17.
    Lr = 3.
    model = CompositeCrackBridge(E_m=25e3, reinforcement_lst=[reinf])
    ccb_view = CompositeCrackBridgeView(model=model)
    sigma_c_arr = np.linspace(1., 50., 50) 
    x_arr = np.linspace(-Ll, Lr, 500)
    icb = InterpolatedCB(cb=ccb_view, Ll=Ll, Lr=Lr, x=x_arr)
    epsm_arr = icb.get_epsm_x_arr(sigma_c_arr)
    epsf_arr = icb.get_epsf_x_arr(sigma_c_arr)
    sigmam_arr = icb.get_sigmam_x_arr(sigma_c_arr)
    e_arr = orthogonalize([sigma_c_arr, x_arr])
    epsm_x = icb.get_epsm_x(18.)
    epsf_x = icb.get_epsf_x(18.)
    sigmam_x = icb.get_sigmam_x(18.)
    plt.plot(icb.x, epsf_x)
    plt.plot(icb.x, epsm_x)
    plt.plot(icb.x, sigmam_x)
    plt.show()
    m.surf(e_arr[0]/np.max(e_arr[0]), e_arr[1]/np.max(e_arr[1]), epsm_arr * 100)
    m.surf(e_arr[0]/np.max(e_arr[0]), e_arr[1]/np.max(e_arr[1]), epsf_arr * 100)
    m.show()

