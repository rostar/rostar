
'''
Created on 14.6.2012

@author: Q
'''
import numpy as np
from matplotlib import pyplot as plt
from etsproxy.traits.api import \
    Instance, Array, List, cached_property, Property
from etsproxy.traits.ui.api import ModelView
from stats.spirrid.sampling import FunctionRandomization
from stats.spirrid.rv import RV
from stats.misc.random_field.random_field_1D import RandomField
import numpy as np
from matplotlib import pyplot as plt
from quaducom.meso.ctt.scm_numerical.scm_model import SCM
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress import \
CBEMClampedFiberStressSP
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress_residual \
import CBEMClampedFiberStressResidualSP
from etsproxy.mayavi import mlab
from quaducom.meso.ctt.scm_numerical.scm_view import SCMView
from os.path import join
from matresdev.db import SimDB
from matresdev.db.exdb import ExRun

if __name__ == '__main__':

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

    # access the response values.
    for V in [V14, V24, V34]:
        eps = V.ex_type.eps_asc
        sig_c = V.ex_type.sig_c_asc
        plt.plot(eps, sig_c, color = 'red')
    for V in [V13, V23, V33]:
        eps = V.ex_type.eps_asc
        sig_c = V.ex_type.sig_c_asc
        plt.plot(eps, sig_c, color = 'blue')
    for V in [V1small, V2small, V3small]:
        eps = V.ex_type.eps_asc
        sig_c = V.ex_type.sig_c_asc
        plt.plot(eps, sig_c, color = 'green')
    plt.xlim(0.0, 0.008)


# filaments
    r = 0.00345
    Vf = 0.0166
    tau = RV('weibull_min', shape=1.0, scale=.04)
    Ef = 200e3
    Em = 25e3
    l = 5.0#RV('uniform', scale=6., loc=.5)
    theta = 0.0
    xi = RV('weibull_min', scale=0.02, shape=5)
    phi = 1.
    Pf = RV('uniform', scale=1., loc=0.)
    m = 5.0
    s0 = 0.017

    length = 550.
    nx = 1500
    random_field = RandomField(seed=True,
                               lacor=10.,
                                xgrid=np.linspace(0., length, 600),
                                nsim=1,
                                loc=.0,
                                shape=15.,
                                scale=6.5,
                                non_negative_check=True,
                                distribution='Weibull'
                               )

#    rf = CBEMClampedFiberStressSP()
#    rand = FunctionRandomization(q=rf,
#                                 tvars=dict(tau=tau,
#                                            l=l,
#                                            E_f=Ef,
#                                            theta=theta,
#                                            xi=xi,
#                                            phi=phi,
#                                            E_m=Em,
#                                            r=r,
#                                            V_f=Vf
#                                                 ),
#                                    n_int=20
#                                    )

    rf = CBEMClampedFiberStressResidualSP()
    rand = FunctionRandomization(q=rf,
                                 tvars=dict(tau=tau,
                                            l=l,
                                            E_f=Ef,
                                            theta=theta,
                                            Pf=Pf,
                                         phi=phi,
                                         E_m=Em,
                                         r=r,
                                         V_f=Vf,
                                         m=m,
                                         s0=s0
                                         ),
                                    n_int=20
                                    )

    scm = SCM(length=length,
              nx=nx,
              random_field=random_field,
              cb_randomization=rand,
              cb_type='mean',
              load_sigma_c_min=.1,
              load_sigma_c_max=27.,
              load_n_sigma_c=200,
              n_w=60,
              n_x=81,
              n_BC=5
              )

    scm_view1 = SCMView(model=scm)
    scm_view1.model.evaluate()

    eps, sigma = scm_view1.eps_sigma
    plt.plot(eps, sigma, color='black', lw=2,
             label='no of cracks = ' + str(len(scm_view1.crack_widths(12.))))
    plt.legend(loc='best')
    plt.xlabel('composite strain [-]')
    plt.ylabel('composite stress [MPa]')
    plt.legend(loc='best')

    # filaments
    r = 0.00345
    Vf = 0.0111
    tau = RV('weibull_min', shape=1.0, scale=.04)
    Ef = 200e3
    Em = 25e3
    l = 5.#RV('uniform', scale=6., loc=.5)
    theta = 0.0
    xi = RV('weibull_min', scale=0.02, shape=5)
    phi = 1.
    Pf = RV('uniform', scale=1., loc=0.)
    m = 5.0
    s0 = 0.017

    length = 550.
    nx = 1500
    random_field = RandomField(seed=True,
                               lacor=10.,
                                xgrid=np.linspace(0., length, 600),
                                nsim=1,
                                loc=.0,
                                shape=15.,
                                scale=6.5,
                                non_negative_check=True,
                                distribution='Weibull'
                               )

#    rf = CBEMClampedFiberStressSP()
#    rand = FunctionRandomization(q=rf,
#                                 tvars=dict(tau=tau,
#                                            l=l,
#                                            E_f=Ef,
#                                            theta=theta,
#                                            xi=xi,
#                                            phi=phi,
#                                            E_m=Em,
#                                            r=r,
#                                            V_f=Vf
#                                                 ),
#                                    n_int=20
#                                    )

    rf = CBEMClampedFiberStressResidualSP()
    rand = FunctionRandomization(q=rf,
                                 tvars=dict(tau=tau,
                                            l=l,
                                            E_f=Ef,
                                            theta=theta,
                                            Pf=Pf,
                                         phi=phi,
                                         E_m=Em,
                                         r=r,
                                         V_f=Vf,
                                         m=m,
                                         s0=s0
                                         ),
                                    n_int=20
                                    )

    scm = SCM(length=length,
              nx=nx,
              random_field=random_field,
              cb_randomization=rand,
              cb_type='mean',
              load_sigma_c_min=.1,
              load_sigma_c_max=25.,
              load_n_sigma_c=200,
              n_w=60,
              n_x=81,
              n_BC=5
              )

    scm_view = SCMView(model=scm)
    scm_view.model.evaluate()

    eps, sigma = scm_view.eps_sigma
    plt.plot(eps, sigma, color='black', lw=2,
             label='no of cracks = ' + str(len(scm_view.crack_widths(12.))))
    plt.legend(loc='best')
    plt.xlabel('composite strain [-]')
    plt.ylabel('composite stress [MPa]')
    plt.legend(loc='best')
    plt.figure()
    plt.hist(scm_view.crack_widths(15.), bins=20, label='load = 15 MPa')
    plt.hist(scm_view1.crack_widths(22.), bins=20, label='load = 22 MPa')
    plt.legend(loc='best')

    plt.show()
