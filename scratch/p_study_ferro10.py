'''
Created on Jul 3, 2012

@author: rostar
'''
from quaducom.meso.ctt.scm_numerical.scm_model import SCM
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress import \
        CBEMClampedFiberStressSP
from stats.spirrid import make_ogrid as orthogonalize
from matplotlib import pyplot as plt
import etsproxy.mayavi.mlab as m
from stats.spirrid.rv import RV
from stats.spirrid.spirrid import SPIRRID, FunctionRandomization, MonteCarlo
from quaducom.meso.ctt.scm_numerical.interpolated_spirrid import InterpolatedSPIRRID
from stats.misc.random_field.random_field_1D import RandomField
import numpy as np

# filaments
r = 0.00345
Vf = 0.0103
tau = RV('uniform', loc = 0.02, scale = .2) # 0.5
Ef = 200e3
Em = 25e3
l = 0.0#RV( 'uniform', scale = 10., loc = 0. )
theta = 0.0
xi = 0.01#RV( 'weibull_min', scale = 0.01, shape = 5 ) # 0.017
phi = 1.
w = np.linspace(0.0, .1, 71)
x = np.linspace(-30., 30., 71)
Ll = np.linspace(0.5,30,8)
Lr = np.linspace(0.5,30,8)

length = 600.
nx = 1000
random_field = RandomField(seed = False,
                           lacor = 4.,
                            xgrid = np.linspace(0., length, 600),
                            nsim = 1,
                            loc = .0,
                            shape = 7.5,
                            scale = 7.8,
                            non_negative_check = True,
                            distribution = 'Weibull'
                           )

rf = CBEMClampedFiberStressSP()
rand = FunctionRandomization(   q = rf,
                                evars = dict(w = w,
                                             x = x,
                                             Ll = Ll,
                                             Lr = Lr,
                                             ),
                                tvars = dict(tau = tau,
                                             l = l,
                                             E_f = Ef,
                                             theta = theta,
                                             xi = xi,
                                             phi = phi,
                                             E_m = Em,
                                             r = r,
                                             V_f = Vf
                                             ),
                                n_int = 30
                                )

scm = SCM(length = length,
          nx = nx,
          random_field = random_field,
          cb_randomization = rand,
          cb_type = 'mean',
          load_sigma_c_min = 0.1,
          load_sigma_c_max = 25.,
          load_n_sigma_c = 100
          )

label = ['discrete', 'random (COV = 23%%)', 'random (COV = 52%%)']
def plot(i):
    eps, sigma = scm.eps_sigma
    mask = np.isnan(eps) == False
    eps = eps[mask]
    sigma = sigma[mask]
    print i, sigma[-1]
    sigma[-1] = 0.0
    plt.plot(eps, sigma, color = 'black', lw = 2, label = label[i])
    plt.xlabel('composite strain [-]')
    plt.ylabel('composite stress [MPa]')

tau_lst = [0.5,
           RV('uniform', loc = 0.3, scale = .4),
           RV('uniform', loc = 0.1, scale = .8)]
for i,t in enumerate(tau_lst):
    scm = SCM(length = length,
          nx = nx,
          random_field = random_field,
          cb_randomization = rand,
          cb_type = 'mean',
          load_sigma_c_min = 0.1,
          load_sigma_c_max = 20.,
          load_n_sigma_c = 100
          )
    scm.cb_randomization.tvars['tau'] = t
    scm.evaluate()
    plot(i)

plt.show()
