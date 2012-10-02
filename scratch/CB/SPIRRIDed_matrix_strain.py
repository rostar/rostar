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

    #r = Float(0.00345)
    V_f = Float(0.4)
    E_f = Float(200e3)
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

    sorted_depsf = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_depsf(self):
        depsf_arr = np.array([])
        weights_arr = np.array([])
        for reinf in self.reinforcement_lst:
            depsf_arr = np.hstack((depsf_arr, reinf.depsf_arr[0].flatten()))
            weights_arr = np.hstack((weights_arr,
                        np.ones_like(reinf.depsf_arr[0].flatten())*
                        reinf.depsf_arr[1]))
        indices = np.searchsorted(depsf_arr)
        sorted_weights = weights_arr[indices]
        sorted = np.sort(depsf_arr)[::-1]
        return sorted

    weight = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_weight(self):
        return self.reinforcement_lst[0].depsf_arr[1]

    def depsm_depsf(self, depsf):
        jCDF = np.sum(depsf >= self.sorted_depsf) * self.weight
        reinf = self.reinforcement_lst[0]
        E_mtrx = reinf.V_f * (1. - jCDF) * reinf.E_f + (1. - reinf.V_f) * self.E_m
        sum_depsf = np.sum(self.sorted_depsf[self.sorted_depsf <= depsf]) * self.weight
        return sum_depsf * reinf.E_f * reinf.V_f / E_mtrx

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
            dem = depsm[-1]
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

    eps_f_x = Property(depends_on='w, reinforcement+')
    @cached_property
    def _get_eps_f_x(self):
        epsy = np.zeros_like(self.eps_m_x[0])
        epsm = self.eps_m_x[0]
        x_arr = self.eps_m_x[1]
        for reinf in self.reinforcement_lst:
            self.sorted_depsf


if __name__ == '__main__':

    reinf1 = Reinforcement(r=RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=1., scale=5.),
                          V_f=0.125,
                          E_m=25e3,
                          E_f=200e3,
                          n_int=5)
    reinf2 = Reinforcement(r = 0.003,#r=RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=1., scale=5.),
                          V_f=0.125,
                          E_m=25e3,
                          E_f=200e3,
                          n_int=400)
    reinf3 = Reinforcement(r=RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=1., scale=5.),
                          V_f=0.125,
                          E_m=25e3,
                          E_f=200e3,
                          n_int=20)

    ccb = CompositeCrackBridge(w=0.4,
                               reinforcement_lst=[reinf1])
    plt.plot(ccb.eps_m_x[1], ccb.eps_m_x[0], label='1')
    ccb.reinforcement_lst[0] = reinf2
    plt.plot(ccb.eps_m_x[1], ccb.eps_m_x[0], label='2')
    ccb.reinforcement_lst[0] = reinf3
    plt.plot(ccb.eps_m_x[1], ccb.eps_m_x[0], label='3')
    plt.legend(loc='best')
    plt.show()
    