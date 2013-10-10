'''
Created on Oct 7, 2013

@author: rostar
'''

from os.path import join
from matresdev.db import SimDB
from matresdev.db.exdb import ExRun
from matplotlib import pyplot as plt

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
        plt.plot(eps, sig_c, color='black')
    for V in [V2small, V3small]:
        eps = V.ex_type.eps_asc * 100.
        sig_c = V.ex_type.sig_c_asc
        plt.plot(eps, sig_c, color='black')
    plt.xlim(0.0, 0.8)
    plt.ylim(0.0, 27.)
    plt.show()

plot()

    def cdf_MC(self, e, depsf, r, lcs):
        '''weibull_fibers_cdf_mc'''
        Ll, Lr, m, sV0 = self.Ll, self.Lr, self.m, self.sV0
        s = ((depsf*(m+1.)*sV0**m*self.V0)/(pi*r**2.))**(1./(m+1.))
        a0 = (e+1e-15)/depsf
        expLfree = (e/s) ** (m + 1) * (1.-(1.-al/a0)**(m+1.))
        expLfixed = a0 / Ll * (e/s) ** (m + 1) * (1.-(1.-Ll/a0)**(m+1.))
        maskL = al < Ll
        expL = expLfree * maskL + expLfixed * (maskL == False)
        expRfree = (e/s) ** (m + 1) * (1.-(1.-ar/a0)**(m+1.))
        expRfixed = a0 / Lr * (e/s) ** (m + 1) * (1.-(1.-Lr/a0)**(m+1.))
        maskR = ar < Lr
        expR = expRfree * maskR + expRfixed * (maskR == False)
        return 1. - np.exp(- expL - expR)

    def ef0_break_CB(self, depsf, r, sV0, m, Pf):
        '''weibull_fibers_cdf_cb_rigid'''
        s = ((depsf*(m+1)*sV0**m)/(2.*pi*r**2))**(1./(m+1))
        return s * (-np.log(1.-Pf)) ** (1./(m+1))
    
    def __call__(self, w, tau, E_f, V_f, r, m, sV0, Pf, lcs):
        #strain and debonded length of intact fibers
        T = 2. * tau / r
        ef0_inf = np.sqrt(T * w / E_f)
        ef0_BC = w / lcs + lcs * T / 4. / E_f
        a0 = ef0_inf * E_f / T
        mask = a0 < lcs / 2.0
        # strain at fiber breakage
        depsf = T / E_f
        ef0_break = self.ef0_break_CB(depsf, r, sV0, m, Pf)
        # debonded length at fiber breakage
        a_break = ef0_break * E_f / T
        #mean pullout length of broken fibers
        mu_Lpo = a_break / (m + 1)
        # strain carried by broken fibers
        ef0_residual = T / E_f * mu_Lpo
        ef0_CB_xi = ef0_residual * H(ef0_inf - ef0_break) + ef0_inf * H(ef0_break - ef0_inf)
        return ef0_CB_xi * E_f * V_f * r**2

