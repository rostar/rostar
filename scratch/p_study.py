'''
Created on Mar 6, 2012

@author: rostar
'''

from quaducom.meso.ctt.scm_numerical.ctt2 import CTT
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber import \
CBEMClampedFiberSP
import enthought.mayavi.mlab as m
import numpy as np
from stats.spirrid import make_ogrid as orthogonalize
from matplotlib import pyplot as plt
from stats.spirrid.spirrid import FunctionRandomization
from stats.misc.random_field.random_field_1D import RandomField

def ld():


    # matrix
    length = 2000
    nx = 6000
    scale = 5.
    shape = 10.
    loc = 0.0
    lacor = 0.01

    # filaments
    tau = .1
    Af = 5.31e-4
    Ef = 72e3
    Am = 50.
    Em = 30e3
    l = 0.#RV('uniform', 5.0, 20.0)
    theta = 0.0
    phi = 1.
    Ll = np.linspace(0.01, 100., 20)
    Lr = np.linspace(0.01, 100., 20)
    Nf = 1700.
    xi = 5000.#RV( 'weibull_min', scale = 0.017, shape = 5 )
    
    x = np.linspace(-30., 30., 61)
    xx = np.linspace(-30.,30.,201)
    
    rand_field = RandomField(
                                    lacor = lacor,
                                    xgrid = np.linspace(0, length, 400),
                                    nsim = 1,
                                    loc = loc,
                                    shape = shape,
                                    scale = scale,
                                    non_negative_heck = True,
                                    distribution = 'Weibull'
                               )

    rf = CBEMClampedFiberSP()
    rand = FunctionRandomization(q = rf,
         evars = dict(w = np.linspace(0.0, .25, 40),
                       x = np.linspace(-30., 30., 31),
                       Ll = Ll,
                       Lr = Lr,
                        ),
         tvars = dict(tau = tau,
                       l = l,
                       A_r = Af,
                       E_r = Ef,
                       theta = theta,
                       xi = xi,
                       phi = phi,
                       E_m = Em,
                       A_m = Am,
                       Nf = Nf,
                        ),
         n_int = 10)

    lacor = [2.0]
    tau = [0.1]
    lacor = [5.0]
    shape = [1e3, 10., 3.]
    x_lis = [x]
    for l in lacor:
        for t in tau:
            for s in shape:
                for i,x_arr in enumerate(x_lis):
                    ctt = CTT(length = length,
                      nx = nx,
                      random_field = rand_field,
                      cb_randomization = rand,
                      cb_type = 'mean',
                      force_min = 0.1, 
                      force_max = 400,
                      n_force = 500
                      )
                    ctt.cb_randomization.tvars['tau'] = t
                    ctt.cb_randomization.evars['x'] = x_arr
                    ctt.random_field.shape = s
                    ctt.random_field.lacor = l
                    ctt.evaluate()
                    eps, sig = ctt.eps_sigma
                    plt.plot(eps, sig,
                             label = 'tau = %.1f, lacor = %.1f, shape = %.1f, x = %.1f' % (t, l, s, i),
                             color = 'black', lw = 2)
    
    def plot():
        e_arr = orthogonalize([ctt.applied_force, ctt.x_arr[::10]])
        n_e_arr = [ e / np.max(np.fabs(e)) for e in e_arr ]
    
        scalar1 = ctt.matrix_strength[:,::10]
        scalar2 = ctt.sigma_m[:,::10]
    #    scalar3 = ctt.x_area
    
        n_scalar1 = scalar1 / np.max(np.fabs(scalar1))
        n_scalar2 = scalar2 / np.max(np.fabs(scalar1))
    #    n_scalar3 = scalar3 / np.max(np.fabs(scalar3))
    
        #m.surf(n_e_arr[0], n_e_arr[1], n_scalar1)
        m.surf(n_e_arr[0], n_e_arr[1], n_scalar2)
    #    m.surf(n_e_arr[0], n_e_arr[1], n_scalar3)
    
        
        m.show()
    plt.legend(loc = 'best')
    plt.show()

ld()