'''
Created on 11 May 2013

@author: Q
'''

from etsproxy.traits.api import HasTraits, Property, cached_property, \
    Instance, Array, List, Float, Int, Dict
import numpy as np
from scipy import ndimage
import types
from etsproxy.mayavi import mlab as m
from spirrid.rv import RV
from dependent_fibers.reinforcement import Reinforcement, WeibullFibers
from dependent_fibers.depend_CB_model import CompositeCrackBridge
from dependent_fibers.depend_CB_postprocessor import CompositeCrackBridgePostprocessor
from scipy.optimize import minimize

def H(x):
    return x >= 0.0

def orthogonalize_filled(args):
    '''creates meshgrid up to third dimension
    given a list of 1D arrays and floats
    '''

    array_list = []
    array_args = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            array_args.append(arg)

    if len(array_args) == 0:
        meshgrid = np.array([1.])
    elif len(array_args) == 1:
        meshgrid = [array_args[0]]
    elif len(array_args) == 2 or len(array_args) == 2:
        meshgrid = np.meshgrid(*array_args)
    else:
        raise NotImplementedError('''max number of arrays as
                                input is currently 3''')

    i = 0
    if len(meshgrid) == 0:
        meshgrid = np.array([1.])
    for arg in args:
        if isinstance(arg, np.ndarray):
            array_list.append(meshgrid[i])
            i += 1
        elif isinstance(arg, types.FloatType):
            array_list.append(np.ones_like(meshgrid[0]) * arg)
    return array_list


class NDIdxInterp(HasTraits):

    # nd array of values (measured, computed..) of
    # size orthogonalize(axes_values)
    data = Array

    # list of control input parameter values
    axes_values = List(Array(float))

    def __call__(self, *gcoords, **kw):
        '''kw: dictionary of values to interpolate for;
        len(kw) has to be equal the data dimension
        '''
        order = kw.get('order', 1)
        mode = kw.get('mode', 'nearest')

        # check if the number of dimensions to interpolate
        # in equals the number of given coordinates
        if len(self.axes_values) != len(gcoords):
            raise TypeError('''method takes {req} arguments
            ({given} given)'''.format(req=len(self.axes_values),
                                      given=len(gcoords)))
        icoords = self.get_icoords(gcoords)

        # create a meshgrid for the interpolation
        icoords = orthogonalize_filled(icoords)
        data = self.data
        # interpolate the value (linear)
        # a, b, c = [0.5, 0.5, 0.5], [0, 0, 0], [0, 1, 2]
        # icoords = [a, b, c]
        val = ndimage.map_coordinates(data, icoords, order=order, mode=mode)
        return val

    def get_icoords(self, gcoords):
        '''
        gcoords: values to be interpolated for
        this method transforms the global coords to "index" coords
        '''
        icoords = [np.interp(gcoord, axis_values, np.arange(len(axis_values)))
                   for gcoord, axis_values in zip(gcoords, self.axes_values)]
        return icoords


class InterpolatedResults(HasTraits):

    CB_model = Instance(CompositeCrackBridgePostprocessor)
    load_sigma_c_max = Float
    load_n_sigma_c = Int
    n_w = Int
    n_x = Int
    n_BC = Int
    
    load_sigma_c_arr = Property(depends_on = 'load_sigma_c_max, load_n_sigma_c')
    @cached_property
    def _get_load_sigma_c_arr(self):
        return np.linspace(0.0, self.load_sigma_c_max, self.load_n_sigma_c) 
    
    def max_sigma_w(self, Ll, Lr):
        self.CB_model.model.Ll = Ll
        self.CB_model.model.Lr = Lr
        def sigma_c_truncated(w):
            self.CB_model.model.w = w
            sigma_c = self.CB_model.sigma_c
            return sigma_c * H(self.load_sigma_c_max - sigma_c)
        def minfunc(w):
            if w < 0.0:
                return 0.0
            else:
                return - sigma_c_truncated(float(w))
        result = minimize(minfunc, 0.001)
        return self.CB_model.sigma_c, result.x
    
    BC_range = Property(depends_on = 'n_BC, CB_model')
    @cached_property
    def _get_BC_range(self):
        self.max_sigma_w(np.inf, np.inf)
        Lmax = self.CB_model.x_arr[-2]
        return np.linspace(1.0, Lmax, self.n_BC)
    
    x_arr = Property(depends_on = 'n_BC, CB_model, n_x')
    @cached_property
    def _get_x_arr(self):
        return np.linspace(-self.BC_range[-1], self.BC_range[-1], self.n_x) 

    def preinterpolate(self, sigma_c_w_x, sigma_c_cutoff, x_range):
        # values to create array grid
        axes_values = [sigma_c_cutoff, x_range]
        preinterp = NDIdxInterp(data=sigma_c_w_x, axes_values=axes_values)
        # values to interpolate for
        interp_coords = [self.load_sigma_c_arr, self.x_arr]
        return preinterp(*interp_coords, mode='constant')

    interp_grid = Property()
    @cached_property
    def _get_interp_grid(self):
        print 'evaluating mean response and adapting ranges...'
        interpolator = self.result_values
        print 'complete'
        return interpolator

    result_values = Property(Array)
    @cached_property
    def _get_result_values(self):
        L_arr = self.BC_range
        result = np.zeros((self.load_n_sigma_c, self.n_x,
                           self.n_BC, self.n_BC))
        loops_tot = len(L_arr) ** 2
        for i, ll in enumerate(L_arr):
            for j, lr in enumerate(L_arr):
                if j >= i:
                    # adapt w range
                    sigma_c_max, wmax = self.max_sigma_w(ll, lr)
                    w_arr = np.linspace(0.0, wmax, self.n_w)
                    # evaluate the result (2D (w,x) SPIRRID with adapted ranges x and w
                    epsm_w_x = self.CB_model.epsm_w_x(w_arr, self.x_arr)
                    # preinterpolate particular result for the given x and sigma ranges
                    epsm_w_x_interp = \
                    self.preinterpolate(epsm_w_x, self.load_sigma_c_arr, self.x_arr).T
                    mask = np.where(self.load_sigma_c_arr
                                    <= sigma_c_max, 1, np.NaN)[:, np.newaxis]
                    epsm_w_x_interp = epsm_w_x_interp * mask
                    eps_vars = orthogonalize([np.arange(len(w_arr))/10., self.x_arr])
#                     m.surf(eps_vars[0], eps_vars[1], epsm_w_x*1000)
#                     m.surf(eps_vars[0], eps_vars[1], epsm_w_x[:,::-1]*1100)
#                     m.surf(eps_vars[0], eps_vars[1], epsm_w_x_interp*1000)
#                     m.show()
                    # store the particular result for BC ll and lr into the result array
                    result[:, :, i, j] = epsm_w_x_interp
                    result[:, :, j, i] = epsm_w_x_interp[:,::-1]
                    current_loop = i * len(L_arr) + j + 1
                    print 'progress: %2.1f %%' % \
                    (current_loop / float(loops_tot) * 100.)
        axes_values = [self.load_sigma_c_arr, self.x_arr, self.BC_range, self.BC_range]
        return NDIdxInterp(data=result, axes_values=axes_values)

    initial_axes_values = Property(List(Array))

    @cached_property
    def _get_initial_eps_vars(self):
        return [self.adaption.load_sigma_f,
                self.adaption.x_init,
                self.adaption.BC_range,
                self.adaption.BC_range]

    def __call__(self, sigma_c_arr, x_arr, Ll, Lr):
        '''
        evaluation of matrix strain and reinforcement strain within a crack bridge
        '''
        return self.interp_grid(sigma_c_arr, x_arr, Ll, Lr)

if __name__ == '__main__':
    from stats.spirrid import make_ogrid as orthogonalize
    from matplotlib import pyplot as plt

    reinf = Reinforcement(r=0.00345,#RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=0.02, scale=20.),
                          V_f=0.15,
                          E_f=200e3,
                          xi=WeibullFibers(shape=5., sV0=0.01618983207723),
                          n_int=50,
                          label='carbon')

    model = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf],
                                 Ll=100.,
                                 Lr=100.)

    ccb_post = CompositeCrackBridgePostprocessor(model=model)
    
    ir = InterpolatedResults(CB_model = ccb_post,
                             load_sigma_c_max = 600.,
                             load_n_sigma_c = 50,
                             n_w = 20,
                             n_x = 21,
                             n_BC = 3
                             )

    ir(np.linspace(0.0, 600., 100), np.linspace(-20, 20, 100), 10., 15.)

    def plot():
        sigma = ir.load_sigma_c_arr
        x = np.linspace(-2, 2, 101)

        eps_vars = orthogonalize([np.arange(len(sigma)), np.arange(len(x))])
        # mu_q_nisp = nisp(P, x, Ll, Lr)[0]
        mu_q_isp = ir(sigma, x, 4., 4.)
        #mu_q_isp2 = ir(sigma, x, Ll, Lr)

#        plt.plot(np.arange(len(sigma)), sigma/0.0103)
#        plt.plot(np.arange(len(sigma)), np.max(mu_q_isp,axis = 0))
#        plt.show()
        # n_mu_q_arr = mu_q_nisp / np.max(np.fabs(mu_q_nisp))
        m.surf(eps_vars[0], eps_vars[1], mu_q_isp * 1000.)
        #m.surf(eps_vars[0], eps_vars[1], mu_q_isp2 * 1000.)
        m.show()

    plot()
