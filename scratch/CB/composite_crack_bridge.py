'''
Created on Sep 20, 2012

The module evaluates fibers and matrix strain in the vicinity of a crack bridge.
Fiber diameter and bond coefficient can be set as random variables.
Reinforcement types can be combined by creating a list of Reinforcement instances
and defining it as the reinforcement_lst Trait in the CompositeCrackBridge class.
TODO:   - add breaking strain
        - define r, xi, tau as EitherType Traits
@author: rostar
'''

import numpy as np
from stats.spirrid.rv import RV
from matplotlib import pyplot as plt
from etsproxy.traits.api import HasTraits, cached_property, \
    Float, Property, Instance, List, Int


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
    E_m = Float
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
        dx = (-depsf * xi - demi * xi + (demi ** 2 * xi ** 2
            + (4 * umi - 4 * emi * xi + 2 * self.w) * demi + ( 2 *
            umi - 2 * emi * xi + self.w) * depsf) ** (0.5)) / (depsf + 2 * demi)
        epsm = emi + demi / 2. * dx + depsm / 2. * dx
        um = umi + emi / 2. * dx + epsm / 2. * dx
        return dx, depsm, epsm, um

    def one_sided(self, depsf, xi, demi, emi, umi, clamped):
        w = self.w
        depsm = self.depsm_depsf(depsf)
        xs = clamped[0]
        ems = clamped[1]
        ums = clamped[2]
        c2 = ems * xs - ums
        dx = ((-xi - xs) * depsf + ((xi + xs) ** 2 * demi ** 2 + 4 * (depsf *
            xs ** 2 - xs * emi + umi - c2 - emi * xi + w + xs * ems) * demi +
            2 * depsf ** 2 * xs ** 2 + 2 * (-xs * emi + xs * ems + umi + w -
            c2 - emi * xi) * depsf) ** (0.5) - demi * xi - xs * demi) / (depsf
            + 2 * demi)
        epsm = emi + demi / 2. * dx + depsm / 2. * dx
        um = umi + emi / 2. * dx + epsm / 2. * dx
        return dx, depsm, epsm, um

    def clamped(self, depsf, xs, xl, ems, eml, ums, uml):
        c1 = eml * xl - uml
        c2 = ems * xs - ums
        c3 = depsf * xl ** 2 / 2.
        c4 = depsf * xs ** 2 / 2.
        c5 = (depsf * (xl - xs) + (eml - ems)) * xs
        h = (self.w - c1 - c2 - c3 - c4 - c5) / (xl + xs)
        return depsf * xl + eml + h

    profile = Property(depends_on='w, Ll, Lr, reinforcement+')
    @cached_property
    def _get_profile(self):
        um_short, epsm_short, x_short = [0.0], [0.0], [0.0]
        um_long, epsm_long, x_long = [0.0], [0.0], [0.0]
        depsm_short = [self.depsm_depsf(self.sorted_depsf[0])]
        depsm_long = [self.depsm_depsf(self.sorted_depsf[0])]
        epsy_crack = 0.0
        ff = epsy_crack
        Lmin = min(self.Ll, self.Lr)
        Lmax = max(self.Ll, self.Lr)
        for i, depsf in enumerate(self.sorted_depsf):
            if x_short[-1] < Lmin and x_long[-1] < Lmax:
                '''double sided pullout'''
                dx, depsm, epsm, um = self.double_sided(depsf, x_short[-1], depsm_short[-1],
                                                   epsm_short[-1], um_short[-1])
                if x_short[-1] + dx < Lmin:
                    # dx increment does not reach the boundary
                    depsm_short.append(depsm)
                    depsm_long.append(depsm)
                    x_short.append(x_short[-1] + dx)
                    x_long.append(x_long[-1] + dx)
                    epsm_short.append(epsm)
                    epsm_long.append(epsm)
                    um_short.append(um)
                    um_long.append(um)
                    epsy_crack += self.sorted_weights[i] * (epsm_short[-1] + x_short[-1] * depsf)# * ((epsm_short[-1] + x_short[-1] * depsf) < 0.2)
                else:
                    # boundary reached at shorter side
                    deltax = Lmin - x_short[-1]
                    x_short.append(Lmin)
                    epsm_short.append(epsm_short[-1] + depsm_short[-1] * deltax)
                    um_short.append(um_short[-1] + (epsm_short[-2] + epsm_short[-1]) * deltax / 2.)

                    short_side = [x_short[-1], epsm_short[-1], um_short[-1]]
                    dx, depsm, epsm, um = self.one_sided(depsf, x_long[-1], depsm_long[-1],
                                            epsm_long[-1], um_long[-1], short_side)

                    if x_long[-1] + dx >= Lmax:
                        # boundary reached at longer side
                        deltax = Lmax - x_long[-1]
                        x_long.append(Lmax)
                        epsm_long.append(epsm_long[-1] + depsm_long[-1] * deltax)
                        um_long.append(um_long[-1] + (epsm_long[-2] + epsm_long[-1]) * deltax / 2.)
                        epsy_crack_clamped = self.clamped(depsf, x_short[-1], x_long[-1], epsm_short[-1],
                             epsm_long[-1], um_short[-1], um_long[-1])
                        epsy_crack += self.sorted_weights[i] * epsy_crack_clamped# * (epsy_crack_clamped < 0.2)
                    else:
                        depsm_long.append(depsm)
                        x_long.append(x_long[-1] + dx)
                        epsm_long.append(epsm)
                        um_long.append(um)
                        epsy_crack += self.sorted_weights[i] * (epsm_long[-1] + x_long[-1] * depsf)#*((epsm_long[-1] + x_long[-1] * depsf) < 0.2)                     

            elif x_short[-1] == Lmin and x_long[-1] < Lmax:
                #one sided pullout
                clamped = [x_short[-1], epsm_short[-1], um_short[-1]]
                dx, depsm, epsm, um = self.one_sided(depsf, x_long[-1], depsm_long[-1],
                                    epsm_long[-1], um_long[-1], clamped)
                if x_long[-1] + dx < Lmax:
                    depsm_long.append(depsm)
                    x_long.append(x_long[-1] + dx)
                    epsm_long.append(epsm)
                    um_long.append(um)
                    epsy_crack += self.sorted_weights[i] * (epsm_long[-1] + x_long[-1] * depsf)# * ((epsm_long[-1] + x_long[-1] * depsf) < 0.2)
                else:
                    dx = Lmax - x_long[-1]
                    x_long.append(Lmax)
                    epsm_long.append(epsm_long[-1] + depsm_long[-1] * dx)
                    um_long.append(um_long[-1] + (epsm_long[-2] + epsm_long[-1]) * dx / 2.)
                    epsy_crack_clamped = self.clamped(depsf, x_short[-1], x_long[-1], epsm_short[-1],
                                 epsm_long[-1], um_short[-1], um_long[-1])
                    epsy_crack += self.sorted_weights[i] * epsy_crack_clamped# * (epsy_crack_clamped<0.2)

            elif x_short[-1] == Lmin and x_long[-1] == Lmax:
                #clamped fibers
                epsy_crack_clamped = self.clamped(depsf, x_short[-1], x_long[-1], epsm_short[-1],
                             epsm_long[-1], um_short[-1], um_long[-1])
                epsy_crack += self.sorted_weights[i] * epsy_crack_clamped# * (epsy_crack_clamped<0.2)
        x_arr = np.hstack((-np.array(x_short)[::-1], np.array(x_long)))
        epsm_arr = np.hstack((np.array(epsm_short)[::-1], np.array(epsm_long)))
        epsy_arr = self.epsy_arr(epsm_arr, epsy_crack)
        #print 'w = ', np.trapz(epsy_arr - epsm_arr, x_arr)
        return x_arr, epsm_arr, epsy_arr

if __name__ == '__main__':

    reinf1 = Reinforcement(r=0.00345,#RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=1., scale=3.),
                          V_f=0.05,
                          E_f=200e3,
                          n_int=100)
    reinf2 = Reinforcement(r=0.00345,#r=RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=10., scale=2.),
                          V_f=0.05,
                          E_f=70e3,
                          n_int=20)
    reinf3 = Reinforcement(r=0.00345,#r=RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=20., scale=2.),
                          V_f=0.25,
                          E_f=50e3,
                          n_int=30)

    ccb = CompositeCrackBridge(E_m=25e3,
                               w=0.4,
                               reinforcement_lst=[reinf1],
                               Ll=5.,
                               Lr=40.)

    def profile(w):
        ccb.w = w
        plt.plot(ccb.profile[0], ccb.profile[1], label='matrix1')
        plt.plot(ccb.profile[0], ccb.profile[2], label='yarn')
        plt.legend(loc='best')
        plt.show()

    def eps_w(w_arr):
        eps = []
        for w in w_arr:
            #print 'w_ctrl=', w
            ccb.w = w
            eps.append(np.max(ccb.profile[2]))
        plt.plot(w_arr, eps, label='ld')
        #plt.legend()
        #plt.show()

    profile(4.)
    #eps_w(np.linspace(0., 4., 50))