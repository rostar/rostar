'''
Created on Sep 20, 2012

@author: rostar
'''

import numpy as np
from stats.spirrid.rv import RV
from matplotlib import pyplot as plt
from etsproxy.traits.api import HasTraits, cached_property, \
    Float, Property, Instance, List, Int
from scipy.integrate import cumtrapz
from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
from scipy.optimize import brentq


class Reinforcement(HasTraits):

    #r = Float
    V_f = Float
    E_f = Float
    #xi = Instance(rv_continuous)
    #tau = Instance(rv_continuous)
    n_int = Int

    depsf_arr = Property(depends_on='n_int')
    @cached_property
    def _get_depsf_arr(self):
        weights = 1.0
        if isinstance(self.tau, RV):
            tau = self.tau.ppf(
                np.linspace(.5 / self.n_int, 1. - .5 / self.n_int, self.n_int))
            weights *= 1. / self.n_int
        else:
            tau = self.tau
        if isinstance(self.r, RV):
            r = self.r.ppf(
                np.linspace(.5 / self.n_int, 1. - .5 / self.n_int, self.n_int))
            weights *= 1. / self.n_int
        else:
            r = self.r

        if isinstance(tau, np.ndarray) and isinstance(r, np.ndarray):
            r = r.reshape(1, self.n_int)
            tau = tau.reshape(self.n_int, 1)
        return 2. * tau / r / self.E_f, weights


class CompositeCrackBridge(HasTraits):

    reinforcement_lst = List(Instance(Reinforcement))
    w = Float
    E_m = Float(25e3)
    
    V_f_tot = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_V_f_tot(self):
        V_f_tot = 0.0
        for reinf in self.reinforcement_lst:
            V_f_tot += reinf.V_f
        return V_f_tot

    sorted_depsf = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_depsf(self):
        depsf_arr = np.array([])
        for reinf in self.reinforcement_lst:
            depsf_arr = np.hstack((depsf_arr, reinf.depsf_arr[0].flatten()))
        sorted_depsf = np.sort(depsf_arr)[::-1]
        return sorted_depsf

    weights = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_weights(self):
        weights = [reinf.depsf_arr[1] for reinf in self.reinforcement_lst]
        return weights

    def depsm_depsf(self, depsf):
        bonded_f_stiffness = 0.0
        for reinf in self.reinforcement_lst:
            jCDF = np.sum(depsf >= reinf.depsf_arr[0]) * reinf.depsf_arr[1]
            bonded_f_stiffness += reinf.V_f * (1. - jCDF) * reinf.E_f
        E_mtrx = (1. - self.V_f_tot) * self.E_m + bonded_f_stiffness
        mean_acting_depsm = 0.0
        for reinf in self.reinforcement_lst:
            sum_depsf = np.sum(reinf.depsf_arr[0][reinf.depsf_arr[0] <= depsf]) * reinf.depsf_arr[1]
            mean_acting_depsm += sum_depsf * reinf.E_f * reinf.V_f
        return mean_acting_depsm / E_mtrx

    eps_m_x = Property(depends_on='w, reinforcement+')
    @cached_property
    def _get_eps_m_x(self):
        um = [0.0]
        epsm = [0.0]
        depsm = [0.0]
        depsm[0] += self.depsm_depsf(self.sorted_depsf[0])
        x_lst = [0.0]
        for depsf in self.sorted_depsf:
            depsm.append(self.depsm_depsf(depsf))
            demi = depsm[-2]
            emi = epsm[-1]
            x = x_lst[-1]
            umi = um[-1]
            dx = (-depsf * x - demi * x + (demi ** 2 * x ** 2 +
                2 * depsf * umi - 2 * depsf * emi * x +
                depsf * self.w + 4 * demi * umi - 4 * demi * emi * x +
                2 * demi * self.w) ** (0.5)) / (depsf + 2 * demi)
            x_lst.append(x_lst[-1] + dx)
            epsm.append(epsm[-1] + depsm[-2] / 2. * dx + depsm[-1] / 2. * dx)
            um.append(um[-1] + epsm[-2] / 2. * dx + epsm[-1] / 2. * dx)
        return np.array(epsm), np.array(x_lst)

    eps_y_x = Property(depends_on='w, reinforcement+')
    @cached_property
    def _get_eps_y_x(self):
        epsy = np.zeros_like(self.eps_m_x[0])
        epsm = self.eps_m_x[0]
        x_arr = self.eps_m_x[1]
        for i, depsf in enumerate(self.sorted_depsf):
            for reinf in self.reinforcement_lst:
                if len(np.where(depsf == reinf.depsf_arr[0])[0]) != 0:
                    weight = reinf.depsf_arr[1]
                    Vf_ratio = reinf.V_f / self.V_f_tot
            epsf = depsf * x_arr[i + 1] + epsm[i + 1] - x_arr * depsf
            epsy += np.maximum(epsf, epsm) * weight * Vf_ratio
        return epsy
    
    eps_y_x_force_equil = Property(depends_on='w, reinforcement+')
    @cached_property
    def _get_eps_y_x_force_equil(self):
        E_fibers = 0.0
        for reinf in self.reinforcement_lst:
            E_fibers += reinf.E_f * reinf.V_f
        Ec = (self.E_m * (1. - self.V_f_tot) + E_fibers)
        sigma_c = self.eps_m_x[0][-1] * Ec
        return (sigma_c - self.E_m * (1. - self.V_f_tot) * self.eps_m_x[0]) / E_fibers

if __name__ == '__main__':

    reinf1 = Reinforcement(r = 0.003,#r=RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=1., scale=5.),
                          V_f=0.125,
                          E_f=200e3,
                          n_int=200)
    reinf2 = Reinforcement(r=0.003,#RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=10., scale=4.),
                          V_f=0.05,
                          E_f=200e3,
                          n_int=200)

    ccb = CompositeCrackBridge(w=0.4,
                               reinforcement_lst=[reinf1, reinf2])
    plt.plot(ccb.eps_m_x[1], ccb.eps_m_x[0], label='matrix1')
    plt.plot(ccb.eps_m_x[1], ccb.eps_y_x, label='yarn1')
    plt.plot(ccb.eps_m_x[1], ccb.eps_y_x_force_equil, label='forces')
    #ccb.reinforcement_lst = [reinf1]
    #plt.plot(ccb.eps_m_x[1], ccb.eps_m_x[0], label='2')
    plt.legend(loc='best')
    plt.show()
    