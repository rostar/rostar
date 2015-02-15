
'''
Created on 14.6.2012

@author: Q
'''

from spirrid.sampling import FunctionRandomization
from spirrid.rv import RV
from stats.misc.random_field.random_field_1D import RandomField
import numpy as np
from matplotlib import pyplot as plt
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_model import SCM
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_view import SCMView
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement_old import Reinforcement, ContinuousFibers
from stats.pdistrib.weibull_fibers_composite_distr import WeibullFibers
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx_old import CompositeCrackBridge
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx_view import CompositeCrackBridgeView
from os.path import join
from matresdev.db import SimDB
from matresdev.db.exdb import ExRun
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from scipy.optimize import fsolve, fmin


class Calibration():

    r = 0.00345
    xi = WeibullFibers(shape=4.5, sV0=3.2e-3)
    E_m = 25e3
    E_f = 180e3
    length = 2000.
    nx = 2000
    smallest_resid = 300.
    t_scale = .1
    mtrx_scale = 3.8
    mtrx_shape = 13.0

    def get_eps_extrap(self, params):
        tau = RV('weibull_min', shape=params[3], scale=params[4])
        random_field = RandomField(seed=True,
                                   lacor=5.,
                                    xgrid=np.linspace(0., self.length, 500),
                                    nsim=1,
                                    loc=.0,
                                    shape=params[1],
                                    scale=params[2],
                                    distribution='Weibull'
                                   )

        reinf = ContinuousFibers(r=self.r,
                              tau=tau,
                              V_f=0.0166,
                              E_f=params[0],
                              xi=self.xi,
                              n_int=50,
                              label='carbon')

        scm = SCM(length=self.length,
                  nx=self.nx,
                  random_field=random_field,
                  E_m=self.E_m,
                  reinforcement=reinf,
                  load_sigma_c_min=.1,
                  load_sigma_c_max=25.,
                  load_n_sigma_c=80
                  )

        scm_view = SCMView(model=scm)
        scm_view.model.evaluate()
        return scm_view.eps_sigma

    def get_eps(self, params):
        tau = RV('weibull_min', shape=params[3], scale=params[4])
        random_field = RandomField(seed=True,
                                   lacor=5.,
                                    xgrid=np.linspace(0., self.length, 500),
                                    nsim=1,
                                    loc=.0,
                                    shape=params[1],
                                    scale=params[2],
                                    distribution='Weibull'
                                   )

        reinf = ContinuousFibers(r=self.r,
                              tau=tau,
                              V_f=0.0111,
                              E_f=params[0],
                              xi=self.xi,
                              n_int=50,
                              label='carbon')

        scm = SCM(length=self.length,
                  nx=self.nx,
                  random_field=random_field,
                  E_m=self.E_m,
                  reinforcement=reinf,
                  load_sigma_c_min=.1,
                  load_sigma_c_max=16.,
                  load_n_sigma_c=80
                  )

        scm_view = SCMView(model=scm)
        scm_view.model.evaluate()
        return scm_view.eps_sigma

    def experiment(self, e):
        eps = V13.ex_type.eps_asc[108:]
        sig = V13.ex_type.sig_c_asc[108:]
        line = MFnLineArray(xdata=eps, ydata=sig)
        vect_line = np.vectorize(line.get_value)
        return vect_line(e)

    def residuum(self, params):
        try:
            eps_opt, sigma = self.get_eps(params)
            squares = (self.experiment(eps_opt) - sigma) ** 2
            weight = np.ones(len(squares))
            weight[50:70] = 5.
            resid = np.sum((self.experiment(eps_opt) - sigma) ** 2 * weight)
            print 'params:', params
            print 'RESIDUUM =', resid
            if resid < self.smallest_resid:
                self.smallest_resid = resid
                plt.figure()
                plt.plot(V13.ex_type.eps_asc, V13.ex_type.sig_c_asc, color='black')
                plt.plot(V23.ex_type.eps_asc, V23.ex_type.sig_c_asc, color='black')
                plt.plot(V33.ex_type.eps_asc, V33.ex_type.sig_c_asc, color='black')
                plt.plot(eps_opt, sigma, color='red',
                         label='%.2f, %.2f' %(params[0], params[1]))
                plt.plot(V2small.ex_type.eps_asc, V2small.ex_type.sig_c_asc, color='black')
                plt.plot(V3small.ex_type.eps_asc, V3small.ex_type.sig_c_asc, color='black')
                try:
                    eps_extrap, sigma_extrap = self.get_eps_extrap(params)
                    plt.plot(eps_extrap, sigma_extrap, color='red')
                except:
                    print 'extrapolation failed'
                    pass
                plt.xlim(0., 0.008)
                plt.legend(loc='best')
                plt.show()
            return resid
        except:
            print 'FAILED'
            return self.smallest_resid * 1.5

    def get_parameters(self):
        opt_params = fmin(self.residuum, np.array([195e3, 11.0, 4.0, 2.0, 0.1]))
        print opt_params


if __name__ == '__main__':
    calib = Calibration()
    simdb = SimDB()

    # specify the path to the data file.
    SH4V1_6cm = join(simdb.exdata_dir,
                     'tensile_tests',
                     'dog_bone',
                     '2012-04-12_TT-12c-6cm-0-TU_SH4',
                     'TT-12c-6cm-0-TU-SH4-V1.DAT')

    SH4V2_6cm = join(simdb.exdata_dir,
                     'tensile_tests',
                     'dog_bone',
                     '2012-04-12_TT-12c-6cm-0-TU_SH4',
                     'TT-12c-6cm-0-TU-SH4-V2.DAT')

    SH4V3_6cm = join(simdb.exdata_dir,
                     'tensile_tests',
                     'dog_bone',
                     '2012-04-12_TT-12c-6cm-0-TU_SH4',
                     'TT-12c-6cm-0-TU-SH4-V3.DAT')
    SH3V1_6cm = join(simdb.exdata_dir,
                     'tensile_tests',
                     'dog_bone',
                     '2012-03-20_TT-12c-6cm-0-TU_SH3',
                     'TT-12c-6cm-TU-0-SH3-V1.DAT')

    SH3V2_6cm = join(simdb.exdata_dir,
                     'tensile_tests',
                     'dog_bone',
                     '2012-03-20_TT-12c-6cm-0-TU_SH3',
                     'TT-12c-6cm-TU-0-SH3-V2.DAT')

    SH3V3_6cm = join(simdb.exdata_dir,
                     'tensile_tests',
                     'dog_bone',
                     '2012-03-20_TT-12c-6cm-0-TU_SH3',
                     'TT-12c-6cm-TU-0-SH3-V3.DAT')

    SH3V1_4cm = join(simdb.exdata_dir,
                     'tensile_tests',
                     'dog_bone',
                     '2012-03-20_TT-12c-4cm-0-TU_SH3',
                     'TT-12c-4cm-TU-0-SH3-V1.DAT')
    SH3V2_4cm = join(simdb.exdata_dir,
                     'tensile_tests',
                     'dog_bone',
                     '2012-03-20_TT-12c-4cm-0-TU_SH3',
                     'TT-12c-4cm-TU-0-SH3-V2.DAT')
    SH3V3_4cm = join(simdb.exdata_dir,
                     'tensile_tests',
                     'dog_bone',
                     '2012-03-20_TT-12c-4cm-0-TU_SH3',
                     'TT-12c-4cm-TU-0-SH3-V3.DAT')

    # construct the experiment
    V14 = ExRun(data_file=SH4V1_6cm)
    V24 = ExRun(data_file=SH4V2_6cm)
    V34 = ExRun(data_file=SH4V3_6cm)
    V13 = ExRun(data_file=SH3V1_6cm)
    V23 = ExRun(data_file=SH3V2_6cm)
    V33 = ExRun(data_file=SH3V3_6cm)
    V1small = ExRun(data_file=SH3V1_4cm)
    V2small = ExRun(data_file=SH3V2_4cm)
    V3small = ExRun(data_file=SH3V3_4cm)

    def plot():

        # access the response values.
        for V in [V13, V23, V33]:
            eps = V.ex_type.eps_asc * 100.
            sig_c = V.ex_type.sig_c_asc
            plt.plot(eps, sig_c, color='black', lw=2)
        for V in [V2small, V3small]:
            eps = V.ex_type.eps_asc * 100.
            sig_c = V.ex_type.sig_c_asc
            plt.plot(eps, sig_c, color='black', lw=2)
        plt.xlim(0.0, 0.8)
        plt.ylim(0.0, 27.)

#    # filaments
#        r = 0.00345
#        xi = WeibullFibers(shape=4.0, sV0=.0025)
#        E_m = 25e3
#        E_f = 180e3
#        mtrx_scale = 5.0
#        mtrx_shape = 13.0
#        length = 1000.
#        nx = 1000
#        tau = RV('uniform', loc=0.01, scale=0.1)
#        random_field = RandomField(seed=True,
#                                   lacor=5.,
#                                    xgrid=np.linspace(0., length, 500),
#                                    nsim=1,
#                                    loc=.0,
#                                    shape=mtrx_shape,
#                                    scale=mtrx_scale,
#                                    distribution='Weibull'
#                                   )
#
#        reinf = ContinuousFibers(r=r,
#                              tau=tau,
#                              V_f=0.0111,
#                              E_f=E_f,
#                              xi=xi,
#                              n_int=200,
#                              label='carbon')
#
#        CB_model = CompositeCrackBridge(E_m=E_m,
#                                        reinforcement_lst=[reinf],
#                                        )
#
#        scm = SCM(length=length,
#                  nx=nx,
#                  random_field=random_field,
#                  CB_model=CB_model,
#                  load_sigma_c_arr=np.linspace(0.0, 30, 60)
#                  )
#
#        scm_view = SCMView(model=scm)
#        scm_view.model.evaluate()
#
#        eps1, sigma1 = scm_view.eps_sigma
#        plt.plot(np.array(eps1) * 100., sigma1, lw=2, ls='dashed', color='red')
#                 #label='no of cracks = ' + str(len(scm_view1.crack_widths(12.))))
#        plt.legend(loc='best')
##        plt.xlabel('composite strain [-]')
##        plt.ylabel('composite stress [MPa]')
#
#        #random_field.scale *= 1.05
#        reinf.V_f = 0.0166
#
#        scm = SCM(length=length,
#                  nx=nx,
#                  random_field=random_field,
#                  load_sigma_c_arr=np.linspace(0.0,30,60)
#                  )
#
#        scm_view = SCMView(model=scm)
#        scm_view.model.evaluate()
#        eps2, sigma2 = scm_view.eps_sigma
#        plt.plot(np.array(eps2) * 100., sigma2, lw=2, color='red')
#        plt.legend(loc='best')
#        plt.xlabel('composite strain [%]')
#        plt.ylabel('composite stress [MPa]')
##        plt.legend(loc='best')
#        #plt.figure()
#        #plt.hist(scm_view.crack_widths(15.), bins=20, label='load = 15 MPa')
#        #plt.hist(scm_view1.crack_widths(22.), bins=20, label='load = 22 MPa')
#        #plt.legend(loc='best')
        plt.show()

    plot()

    #calib.get_parameters()
