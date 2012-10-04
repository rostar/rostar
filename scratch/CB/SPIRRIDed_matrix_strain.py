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
    Ll = Float
    Lr = Float
    
    V_f_tot = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_V_f_tot(self):
        V_f_tot = 0.0
        for reinf in self.reinforcement_lst:
            V_f_tot += reinf.V_f
        return V_f_tot

    E_c = Property(depends_on='reinforcement_lst+')
    def _get_E_c(self):
        E_fibers = 0.0
        for reinf in self.reinforcement_lst:
            E_fibers += reinf.V_f * reinf.E_f
        return self.E_m * (1. - self.V_f_tot) + E_fibers

    sorted_depsf = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_depsf(self):
        depsf_arr = np.array([])
        for reinf in self.reinforcement_lst:
            depsf_arr = np.hstack((depsf_arr, reinf.depsf_arr[0].flatten()))
        sorted_depsf = np.sort(depsf_arr)[::-1]
        return sorted_depsf

    sorted_weights = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_weights(self):
        depsf_arr = np.array([])
        weights_arr = np.array([])
        for reinf in self.reinforcement_lst:
            depsf_arr = np.hstack((depsf_arr, reinf.depsf_arr[0].flatten()))
            weight_stats = np.ones(len(reinf.depsf_arr[0].flatten())) * reinf.depsf_arr[1]
            Vf_ratio = reinf.V_f * reinf.E_f / (self.E_c - (1. - self.V_f_tot) * self.E_m)
            weight_reinf_ratio = np.ones(len(reinf.depsf_arr[0].flatten())) * Vf_ratio
            weights_arr = np.hstack((weights_arr, weight_stats * weight_reinf_ratio))
        sorted_weights = weights_arr[np.argsort(depsf_arr)[::-1]]
        return sorted_weights


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

    def epsy_arr(self, epsm_arr, epsy_crack):
        sigma_c = epsy_crack * (self.E_c - self.E_m * (1. - self.V_f_tot))
        epsy_arr = (sigma_c - self.E_m * (1. - self.V_f_tot) * epsm_arr)\
            / (self.E_c - self.E_m * (1. - self.V_f_tot))
        return epsy_arr

    def double_sided(self, depsf, xi, demi, emi, umi):
        depsm = self.depsm_depsf(depsf)
        dx = (-depsf * xi - demi * xi + (demi ** 2 * xi ** 2 +
            2 * depsf * umi - 2 * depsf * emi * xi +
            depsf * self.w + 4 * demi * umi - 4 * demi * emi * xi +
            2 * demi * self.w) ** (0.5)) / (depsf + 2 * demi)
        epsm = emi + demi / 2. * dx + depsm / 2. * dx
        um = umi + emi / 2. * dx + epsm / 2. * dx
        return dx, depsm, epsm, um
    
    def one_sided(self):
        print 'ja'

    profile = Property(depends_on='w, Ll, Lr, reinforcement+')
    @cached_property
    def _get_profile(self):
        um_short, epsm_short, x_short = [0.0], [0.0], [0.0]
        um_long, epsm_long, x_long = [0.0], [0.0], [0.0]
        depsm_short = [self.depsm_depsf(self.sorted_depsf[0])]
        depsm_long = [self.depsm_depsf(self.sorted_depsf[0])]
        epsy_crack = 0.0
        Lmin = min(self.Ll, self.Lr)
        Lmax = max(self.Ll, self.Lr)
        for i, depsf in enumerate(self.sorted_depsf):
            if x_short[-1] < Lmin and x_long[-1] < Lmax:
                '''double sided pullout'''
                dx, depsm, epsm, um = self.double_sided(depsf, x_short[-1], depsm_short[-1],
                                                   epsm_short[-1], um_short[-1])
                if x_short[-1] + dx < Lmin:
                    depsm_short.append(depsm)
                    depsm_long.append(depsm)
                    x_short.append(x_short[-1] + dx)
                    x_long.append(x_long[-1] + dx)
                    epsm_short.append(epsm)
                    epsm_long.append(epsm)
                    um_short.append(um)
                    um_long.append(um)
                    epsy_crack += self.sorted_weights[i] * (epsm_short[-1] + x_short[-1] * depsf)
                else:
                    dx = Lmin - x_short[-1]
                    x_short.append(x_short[-1] + dx)
                    epsm_short.append(epsm_short[-1] + depsm_short[-1] * dx)
                    um_short.append(um_short[-1] + epsm_short[-2] * dx)
                    if Lmax == Lmin:
                        x_long.append(x_long[-1] + dx)
                        epsm_long.append(epsm_long[-1] + depsm_long[-1] * dx)
                        um_long.append(um_long[-1] + epsm_long[-2] * dx)
                        c = 2. * (epsm_long[-1] * x_long[-1] - um_long[-1])
                        h = (self.w - c - depsf * x_long[-1] ** 2) / (2. * x_long[-1])
                        epsy_crack += self.sorted_weights[i] * (epsm_short[-1] + x_short[-1] * depsf + h)
                    else:
                        clamped = [x_short[-1], epsm_short[-1], um_short[-1]]
                        dx, depsm, epsm, um = self.one_sided(depsf, x_long[-1], depsm_long[-1],
                                            epsm_long[-1], um_long[-1], clamped)
                        depsm_long.append(depsm)
                        x_long.append(x_long[-1] + dx)
                        epsm_long.append(epsm)
                        um_long.append(um)
                        epsy_crack += self.sorted_weights[i] * (epsm_short[-1] + x_short[-1] * depsf)                        

            elif x_short[-1] == Lmin and x_long[-1] < Lmax:
                '''one sided pullout'''
                dx, depsm, epsm, um = self.double_sided(depsf, x_short[-1], depsm_short[-1],
                                                   epsm_short[-1], um_short[-1])
            
            elif x_short[-1] == Lmin and x_long[-1] == Lmax:
                '''clamped fibers'''
                c1 = epsm_long[-1] * x_long[-1] - um_long[-1]
                c2 = epsm_short[-1] * x_short[-1] - um_short[-1]
                c3 = depsf * x_long[-1] ** 2 / 2.
                c4 = depsf * x_short[-1] ** 2 / 2.
                c5 = (depsf * (x_long[-1] - x_short[-1]) + (epsm_long[-1] - epsm_short[-1])) * x_short[-1]
                h = (self.w - c1 - c2 - c3 - c4 - c5) / (x_long[-1] + x_short[-1])
                epsy_crack += self.sorted_weights[i] * (depsf * x_long[-1] + epsm_long[-1] + h)


        x_arr = np.hstack((-np.array(x_short)[::-1], np.array(x_long)))
        epsm_arr = np.hstack((np.array(epsm_short)[::-1], np.array(epsm_long)))
        epsy_arr = self.epsy_arr(epsm_arr, epsy_crack)
        print 'w = ', np.trapz(epsy_arr - epsm_arr, x_arr)
        return x_arr, epsm_arr, epsy_arr

if __name__ == '__main__':

    reinf1 = Reinforcement(r=RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=1., scale=.5),
                          V_f=0.15,
                          E_f=200e3,
                          n_int=20)
    reinf2 = Reinforcement(r=RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=10., scale=2.),
                          V_f=0.05,
                          E_f=70e3,
                          n_int=20)
    reinf3 = Reinforcement(r=RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=20., scale=2.),
                          V_f=0.25,
                          E_f=50e3,
                          n_int=30)

    ccb = CompositeCrackBridge(w=0.4,
                               reinforcement_lst=[reinf1, reinf2, reinf3],
                               Ll=2.,
                               Lr=2.)

    def profile(w):
        ccb.w = w
        plt.plot(ccb.profile[0], ccb.profile[1], label='matrix1')
        plt.plot(ccb.profile[0], ccb.profile[2], label='yarn')
        print 'w = ', np.trapz(ccb.profile[2] - ccb.profile[1], ccb.profile[0])
        plt.legend(loc='best')
        plt.show()
        
    def eps_w(w_arr):
        eps = []
        for w in w_arr:
            print 'w_ctrl=', w
            ccb.w = w
            eps.append(np.max(ccb.profile[2]))
        plt.plot(w_arr, eps, label='ld')
        plt.legend()
        plt.show()

    #profile(4.16)
    eps_w(np.linspace(3., 5.2, 10))
            
        
    