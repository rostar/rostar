'''
Created on 5 Jan 2014

@author: Q
'''

from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import ContinuousFibers
from stats.pdistrib.weibull_fibers_composite_distr import WeibullFibers, fibers_MC
from stats.misc.random_field.random_field_1D import RandomField
from matplotlib import pyplot as plt
import numpy as np
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_model import SCM
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_view import SCMView
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx import CompositeCrackBridge

def acor_fn():
    l = np.linspace(0,30,500)
    lrho = 10.
    R = np.exp(-(l/lrho)**2)
    plt.plot(l,R)
    lrho = 3.
    R = np.exp(-(l/lrho)**2)
    plt.plot(l,R)
    
def random_field():
    rf = RandomField(lacor=3.,
                     length=500.,
                     nx=500,
                     distribution='Weibull',
                     shape=12.,
                     scale=5.0,
                     loc=0.0)
    plt.plot(rf.xgrid, rf.random_field, lw=1, color='grey', label='Weibull')
    rf.lacor = 10.0
    plt.plot(rf.xgrid, rf.random_field, lw=2, color='black', label='Weibull')
    plt.ylim(0)
    
def mtrx_shape():
    length = 2000.
    nx = 2000
    random_field = RandomField(seed=False,
                               lacor=5.,
                               length=length,
                               nx=1000,
                               nsim=1,
                               loc=.0,
                               shape=16.5,
                               scale=3.1,
                               distribution='Weibull'
                               )

    reinf1 = ContinuousFibers(r=0.0035,
                          tau=0.03,#RV('weibull_min', loc=0.0, shape=3., scale=0.03),
                          V_f=0.01,
                          E_f=180e3,
                          xi=fibers_MC(m=5.0, sV0=10.003),
                          label='carbon',
                          n_int=500)

    CB_model = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf1],
                                 )

    scm = SCM(length=length,
              nx=nx,
              random_field=random_field,
              CB_model=CB_model,
              load_sigma_c_arr=np.linspace(0.01, 8., 100),
              )

    scm_view = SCMView(model=scm)
    scm_view.model.evaluate()

    eps, sigma = scm_view.eps_sigma
    plt.figure()
    plt.plot(eps, sigma, color='black', lw=2, label='model')
    plt.legend(loc='best')
    plt.xlabel('composite strain [-]')
    plt.ylabel('composite stress [MPa]')


if __name__ == '__main__':
    #acor_fn()
    #random_field()
    mtrx_shape()
    plt.show()