from stats.pdistrib.weibull_fibers_composite_distr import fibers_MC
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import ContinuousFibers, ShortFibers
from matplotlib import pyplot as plt
from spirrid.rv import RV
from stats.misc.random_field.random_field_1D import RandomField
import numpy as np
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_model import SCM
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_view import SCMView
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx import CompositeCrackBridge



length = 1000.
nx = 3000
random_field = RandomField(seed=True,
                       lacor=1.,
                       length=length,
                       nx=1000,
                       nsim=1,
                       loc=.0,
                       shape=50.,
                       scale=3.4,
                       distr_type='Weibull')

reinf_cont = ContinuousFibers(r=3.5e-3,
                          tau=RV('gamma', loc=0.0, scale=.9, shape=.09),
                          V_f=0.01,
                          E_f=200e3,
                          xi=fibers_MC(m=7., sV0=0.05),
                          label='carbon',
                          n_int=100)

reinf_short = ShortFibers(bond_law = 'plastic',
                    r=.1,
                    tau=1.5,
                    V_f=0.01,
                    E_f=200e3,
                    xi=10.,
                    snub=0.5,
                    phi=RV('sin2x', scale=1.0, shape=0.0),
                    Lf=20.,
                    label='short steel fibers',
                    n_int=50)

CB_model = CompositeCrackBridge(E_m=25e3,
                             reinforcement_lst=[reinf_cont],
                             )
scm = SCM(length=length,
          nx=nx,
          random_field=random_field,
          CB_model=CB_model,
          load_sigma_c_arr=np.linspace(0.01, 20., 200),
          n_BC_CB = 5)

scm_view = SCMView(model=scm)
scm_view.model.evaluate() 
eps, sigma = scm_view.eps_sigma
plt.plot(eps, sigma, color='black', lw=2, label='continuous fibers')

CB_model.reinforcement_lst = [reinf_cont, reinf_short]
scm = SCM(length=length,
          nx=nx,
          random_field=random_field,
          CB_model=CB_model,
          load_sigma_c_arr=np.linspace(0.01, 20., 200),
          n_BC_CB = 5)
scm_view = SCMView(model=scm)
scm_view.model.evaluate() 
eps, sigma = scm_view.eps_sigma
plt.plot(eps, sigma, color='green', lw=2, label='continuous + short fibers')

plt.legend(loc='best')
plt.xlabel('composite strain [-]')
plt.ylabel('composite stress [MPa]')
plt.show()


def plot():
    eps, sigma = scm_view.eps_sigma
    plt.plot(eps, sigma, color='black', lw=2, label='model')
    plt.legend(loc='best')
    plt.xlabel('composite strain [-]')
    plt.ylabel('composite stress [MPa]')
    plt.figure()
    plt.hist(scm_view.crack_widths(15.), bins=20, label='load = 15 MPa')
    plt.hist(scm_view.crack_widths(10.), bins=20, label='load = 10 MPa')
    plt.hist(scm_view.crack_widths(5.), bins=20, label='load = 5 MPa')
    plt.ylabel('frequency [-]')
    plt.xlabel('crack width [mm]') 
    plt.legend(loc='best')
    plt.xlim(0)
    plt.figure()
    plt.plot(scm_view.model.load_sigma_c_arr, scm_view.w_mean,
             color='green', lw=2, label='mean crack width')
    plt.plot(scm_view.model.load_sigma_c_arr, scm_view.w_median,
            color='blue', lw=2, label='median crack width')
    plt.plot(scm_view.model.load_sigma_c_arr, scm_view.w_mean + scm_view.w_stdev,
            color='black', label='stdev')
    plt.plot(scm_view.model.load_sigma_c_arr, scm_view.w_mean - scm_view.w_stdev,
            color='black')
    plt.plot(scm_view.model.load_sigma_c_arr, scm_view.w_max,
             ls='dashed', color='red', label='max crack width')
    plt.ylabel('crack width [mm]')
    plt.xlabel('composite stress [MPa]')
    plt.legend(loc='best')
    plt.figure()
    plt.plot(scm_view.model.load_sigma_c_arr, scm_view.w_density,
             color='black', lw=2, label='crack density')
    plt.legend(loc='best')
    plt.ylabel('crack density [1/mm]')
    plt.xlabel('composite stress [MPa]')
    plt.show()

