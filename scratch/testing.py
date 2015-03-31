from stats.pdistrib.weibull_fibers_composite_distr import fibers_MC
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import ContinuousFibers, ShortFibers
from matplotlib import pyplot as plt
from spirrid.rv import RV
from stats.misc.random_field.random_field_1D import RandomField
import numpy as np
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_model import SCM
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_view import SCMView
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx import CompositeCrackBridge


def ld():
    length = 1000.
    nx = 3000
    random_field = RandomField(seed=True,
                           lacor=1.,
                           length=length,
                           nx=1000,
                           nsim=1,
                           loc=.0,
                           shape=45.,
                           scale=3.360,
                           distr_type='Weibull')
    
    
    reinf_cont = ContinuousFibers(r=3.5e-3,
                              tau=RV('gamma', loc=0.0, scale=1.534, shape=.0615),
                              V_f=0.01,
                              E_f=181e3,
                              xi=fibers_MC(m=8.6, sV0=11.4e-3),
                              label='carbon',
                              n_int=100)
    
    CB_model = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf_cont],
                                 )
    scm = SCM(length=length,
              nx=nx,
              random_field=random_field,
              CB_model=CB_model,
              load_sigma_c_arr=np.linspace(0.01, 30., 200),
              n_BC_CB = 12)
    
    scm_view = SCMView(model=scm)
    scm_view.model.evaluate() 
    eps, sigma = scm_view.eps_sigma
    plt.plot(eps, sigma, color='black', lw=2, label='continuous fibers')
    
    plt.legend(loc='best')
    plt.xlabel('composite strain [-]')
    plt.ylabel('composite stress [MPa]')
    plt.show()

def TT():
    for i in range(5):
        file_i = np.loadtxt("TT-4C-0" + str(i+1) + ".txt", delimiter=';')
        plt.plot(-file_i[:,2]/2./250. - file_i[:,3]/2./250.,file_i[:,1]/2. * 20. * 100. / 1000., lw=1, color='blue')
        file_i2 = np.loadtxt("TT-6C-0" + str(i+1) + ".txt", delimiter=';')
        plt.plot(-file_i2[:,2]/2./250. - file_i2[:,3]/2./250.,file_i2[:,1]/2. * 20. * 100. / 1000., lw=1, color='red')
    plt.legend(loc='best')
    plt.xlim(0)

TT()