'''
Created on 12 Oct 2013

@author: Q
'''
'''
Created on Oct 7, 2013

@author: rostar
'''

import numpy as np
from quaducom.micro.resp_func.CB_clamped_rand_xi import CBClampedRandXi
from quaducom.micro.resp_func.CB_clamped import CBClamped
from quaducom.micro.resp_func.cb_short_fiber import CBShortFiber
from spirrid.spirrid import SPIRRID
from spirrid.rv import RV
from matplotlib import pyplot as plt
from scipy.stats import weibull_min
from stats.pdistrib.weibull_fibers_composite_distr import fibers_MC, fibers_CB_rigid, fibers_dry
from math import pi
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import ContinuousFibers
from stats.pdistrib.weibull_fibers_composite_distr import WeibullFibers, fibers_MC
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx_view import CompositeCrackBridge, CompositeCrackBridgeView


reinf = ContinuousFibers(r=0.0035,
                      tau=0.3,
                      V_f=0.05,
                      E_f=180e3,
                      xi=fibers_MC(m=4.0, sV0=0.003),
                      label='carbon',
                      n_int=500)

model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf],
                             Ll=100.,
                             Lr=100.,
                             )

ccb_view = CompositeCrackBridgeView(model=model)

def profile( w ):
    ccb_view.model.w = w
    plt.plot( ccb_view.x_arr, ccb_view.epsm_arr, lw = 2, label = 'Ll=' + str( ccb_view.model.Ll ) )
    plt.plot( ccb_view.x_arr, ccb_view.mu_epsf_arr, color = 'red', lw = 2 )
    plt.xlabel( 'position [mm]' )
    plt.ylabel( 'strain' )

def sigma_c_w( w_arr ):
    sigma_c_arr, u_arr = ccb_view.sigma_c_arr( w_arr, u = True )
    plt.plot( w_arr, sigma_c_arr, lw = 2, color = 'black', label = 'w-sigma' )
    plt.plot(u_arr, sigma_c_arr, lw=2, label='u-sigma')
    #plt.plot(ccb_view.sigma_c_max[1], ccb_view.sigma_c_max[0], 'bo')
    plt.xlabel( 'w,u [mm]' )
    plt.ylabel( '$\sigma_c$ [MPa]' )
    plt.legend( loc = 'best' )

def sigma_f( w_arr ):
    sf_arr = ccb_view.sigma_f_lst( w_arr )
    for i, reinf in enumerate( ccb_view.model.reinforcement_lst ):
        plt.plot( w_arr, sf_arr[:, i], label = reinf.label )

def energy( w_arr ):
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
        Wel_tot.append( ccb_view.W_el_tot )
        Welm.append( ccb_view.Welm )
        Welf.append( ccb_view.Welf )
        U.append( ccb_view.U )
        Winel.append( ccb_view.W_inel_tot )
        u.append( ccb_view.u_evaluated )
    #plt.plot( w_arr, Welm, lw = 2, label = 'Welm' )
    #plt.plot( w_arr, Welf, lw = 2, label = 'Welf' )
    plt.plot( w_arr, Wel_tot, lw = 2, color = 'black', label = 'elastic strain energy' )
    plt.plot( w_arr, Winel, lw = 2, ls = 'dashed', color = 'black', label = 'inelastic energy' )
    plt.plot( w_arr, U, lw = 3, color = 'red', label = 'work of external force' )
    plt.xlabel( 'w [mm]' )
    plt.ylabel( 'W' )
    plt.ylim( 0.0 )
    plt.legend( loc = 'best' )

# TODO: check energy for combined reinf
# energy(np.linspace(.0, .15, 100))
#    sigma_c = np.linspace(1., 7., 7)
#profile(0.031)
w = np.linspace(0.0, 1.0, 200)
sigma_c_w(w)
#energy(w)
# bundle at 20 mm
# sigma_bundle = 70e3*w/20.*np.exp(-(w/20./0.03)**5.)
# plt.plot(w,sigma_bundle)
# plt.plot(ccb_view.sigma_c_max[1], ccb_view.sigma_c_max[0], 'bo')
# sigma_f(np.linspace(.0, .16, 50))
plt.legend(loc= 'best' )
plt.show()