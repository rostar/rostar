'''
Created on Sep 20, 2012

The module evaluates fibers and matrix strain in the vicinity of a crack bridge.
Fiber diameter and bond coefficient can be set as random variables.
Reinforcement types can be combined by creating a list of Reinforcement instances
and defining it as the reinforcement_lst Trait in the CompositeCrackBridge class.
TODO:   - add breaking strain
@author: rostar
'''

import numpy as np
from stats.spirrid.rv import RV
from matplotlib import pyplot as plt
from etsproxy.traits.api import HasTraits, cached_property, \
    Float, Property, Instance, List, Int, Array
from types import FloatType
from util.traits.either_type import EitherType
from scipy.stats import uniform


class Reinforcement(HasTraits):

    r = EitherType(klasses=[FloatType, RV])
    V_f = Float
    E_f = Float
    xi = EitherType(klasses=[FloatType, RV])
    tau = EitherType(klasses=[FloatType, RV])
    n_int = Int

    depsf_arr = Property(depends_on='r, V_f, E_F, xi, tau, n_int')
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

    sorted_filaments = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_filaments(self):
        depsf_arr = np.array([])
        V_f_arr = np.array([])
        E_f_arr = np.array([])
        xi_arr = np.array([])
        weight_arr = np.array([])
        for reinf in self.reinforcement_lst:
            n_int = len(reinf.depsf_arr[0].flatten())
            depsf_arr = np.hstack((depsf_arr, reinf.depsf_arr[0].flatten()))
            weight_arr = np.hstack((weight_arr, np.repeat(reinf.depsf_arr[1], n_int)))
            V_f_arr = np.hstack((V_f_arr, np.repeat(reinf.V_f, n_int)))
            E_f_arr = np.hstack((E_f_arr, np.repeat(reinf.E_f, n_int)))
            xi_arr = np.hstack((xi_arr, np.repeat(reinf.xi, n_int)))
        argsort = np.argsort(depsf_arr)[::-1]
        return depsf_arr[argsort], V_f_arr[argsort], E_f_arr[argsort], xi_arr[argsort],  weight_arr[argsort]
    
    sorted_depsf = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_depsf(self):
        return self.sorted_filaments[0]

    sorted_V_f = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_V_f(self):
        return self.sorted_filaments[1]
    
    sorted_E_f = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_E_f(self):
        return self.sorted_filaments[2]
    
    sorted_xi = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_xi(self):
        return self.sorted_filaments[3]

    sorted_stats_weights = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_stats_weights(self):
        return self.sorted_filaments[4]

    sorted_Vf_weights = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_Vf_weights(self):
        return self.sorted_V_f * self.sorted_E_f / (self.E_c - (1. - self.V_f_tot) * self.E_m)

    sorted_xi_cdf = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_xi_cdf(self):
        sorted_xi_cdf = []
        for xi in self.sorted_xi:
            if isinstance(xi, FloatType):
                sorted_xi_cdf.append(lambda x: 1.0 * (xi <= x))
            elif isinstance(xi, RV):
                sorted_xi_cdf.append(xi._distr.cdf)
        return np.array(sorted_xi_cdf)

    def vect_xi_cdf(self, epsy):
        return np.array([ff(epsy[i]) for i, ff in enumerate(self.sorted_xi_cdf)])

    damage = Array
    def _damage_default(self):
        return np.zeros_like(self.sorted_depsf)

    def depsm_depsf(self, depsf):
        intact_bonded_fibers = np.sum(self.sorted_V_f * self.sorted_stats_weights *
                                      self.sorted_E_f * (depsf <= self.sorted_depsf) * (1. - self.damage))
        broken_fibers = np.sum(self.sorted_V_f * self.sorted_stats_weights *
                               self.sorted_E_f * self.damage)
        add_m_stiffness = intact_bonded_fibers + broken_fibers
        E_mtrx = (1. - self.V_f_tot) * self.E_m + add_m_stiffness
        mean_acting_depsm = np.sum(self.sorted_depsf * (self.sorted_depsf <= depsf) *
                                   self.sorted_stats_weights * self.sorted_E_f *
                                   self.sorted_V_f * (1. - self.damage))
        return mean_acting_depsm / E_mtrx, (1. - self.V_f_tot) * self.E_m + broken_fibers

    def epsy_arr(self, epsm_arr, epsy_crack, E_mtrx_arr):
        sigma_c = epsy_crack * (self.E_c - self.E_m * (1. - self.V_f_tot))
        epsy_arr = (sigma_c - E_mtrx_arr * epsm_arr) / (self.E_c - E_mtrx_arr)
        return epsy_arr

    def double_sided(self, depsf, xi, demi, emi, umi):
        dx = (-depsf * xi - demi * xi + (demi ** 2 * xi ** 2
            + (4 * umi - 4 * emi * xi + 2 * self.w) * demi + ( 2 *
            umi - 2 * emi * xi + self.w) * depsf) ** (0.5)) / (depsf + 2 * demi)
        depsm, E_mtrx = self.depsm_depsf(depsf)
        epsm = emi + demi / 2. * dx + depsm / 2. * dx
        um = umi + emi / 2. * dx + epsm / 2. * dx
        return dx, depsm, epsm, um, E_mtrx

    def one_sided(self, depsf, xi, demi, emi, umi, clamped):
        w = self.w
        
        depsm, E_mtrx = self.depsm_depsf(depsf)
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
        return dx, depsm, epsm, um, E_mtrx

    def clamped(self, depsf, xs, xl, ems, eml, ums, uml):
        c1 = eml * xl - uml
        c2 = ems * xs - ums
        c3 = depsf * xl ** 2 / 2.
        c4 = depsf * xs ** 2 / 2.
        c5 = (depsf * (xl - xs) + (eml - ems)) * xs
        h = (self.w - c1 - c2 - c3 - c4 - c5) / (xl + xs)
        return depsf * xl + eml + h

    iters = Int(5)

    profile = Property(depends_on='w, Ll, Lr, reinforcement+')
    @cached_property
    def _get_profile(self):
        damage = []
        for j in range(self.iters):
            um_short, epsm_short, x_short = [0.0], [0.0], [0.0]
            um_long, epsm_long, x_long = [0.0], [0.0], [0.0]
            depsm_short = [self.depsm_depsf(self.sorted_depsf[0])[0]]
            depsm_long = [self.depsm_depsf(self.sorted_depsf[0])[0]]
            E_mtrx_glob = 0.0 
            epsy_crack = np.zeros_like(self.sorted_depsf)
            Lmin = min(self.Ll, self.Lr)
            Lmax = max(self.Ll, self.Lr)
            for i, depsf in enumerate(self.sorted_depsf):
                if x_short[-1] < Lmin and x_long[-1] < Lmax:
                    '''double sided pullout'''
                    dx, depsm, epsm, um, E_mtrx = self.double_sided(depsf, x_short[-1], depsm_short[-1],
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
                        E_mtrx_glob = E_mtrx
                        epsy_crack[i] = (epsm_short[-1] + x_short[-1] * depsf)
                    else:
                        # boundary reached at shorter side
                        deltax = Lmin - x_short[-1]
                        x_short.append(Lmin)
                        epsm_short.append(epsm_short[-1] + depsm_short[-1] * deltax)
                        um_short.append(um_short[-1] + (epsm_short[-2] + epsm_short[-1]) * deltax / 2.)
                        E_mtrx_glob = E_mtrx
    
                        short_side = [x_short[-1], epsm_short[-1], um_short[-1]]
                        dx, depsm, epsm, um, E_mtrx = self.one_sided(depsf, x_long[-1], depsm_long[-1],
                                                epsm_long[-1], um_long[-1], short_side)
    
                        if x_long[-1] + dx >= Lmax:
                            # boundary reached at longer side
                            deltax = Lmax - x_long[-1]
                            x_long.append(Lmax)
                            epsm_long.append(epsm_long[-1] + depsm_long[-1] * deltax)
                            um_long.append(um_long[-1] + (epsm_long[-2] + epsm_long[-1]) * deltax / 2.)
                            E_mtrx_glob = E_mtrx
                            epsy_crack_clamped = self.clamped(depsf, x_short[-1], x_long[-1], epsm_short[-1],
                                 epsm_long[-1], um_short[-1], um_long[-1])
                            epsy_crack[i] = epsy_crack_clamped
                        else:
                            depsm_long.append(depsm)
                            x_long.append(x_long[-1] + dx)
                            epsm_long.append(epsm)
                            um_long.append(um)
                            E_mtrx_glob = E_mtrx
                            epsy_crack[i] = (epsm_long[-1] + x_long[-1] * depsf)                     
    
                elif x_short[-1] == Lmin and x_long[-1] < Lmax:
                    #one sided pullout
                    clamped = [x_short[-1], epsm_short[-1], um_short[-1]]
                    dx, depsm, epsm, um, E_mtrx = self.one_sided(depsf, x_long[-1], depsm_long[-1],
                                        epsm_long[-1], um_long[-1], clamped)
                    if x_long[-1] + dx < Lmax:
                        depsm_long.append(depsm)
                        x_long.append(x_long[-1] + dx)
                        epsm_long.append(epsm)
                        um_long.append(um)
                        E_mtrx_glob = E_mtrx
                        epsy_crack[i] = (epsm_long[-1] + x_long[-1] * depsf)
                    else:
                        dx = Lmax - x_long[-1]
                        x_long.append(Lmax)
                        epsm_long.append(epsm_long[-1] + depsm_long[-1] * dx)
                        um_long.append(um_long[-1] + (epsm_long[-2] + epsm_long[-1]) * dx / 2.)
                        E_mtrx_glob = E_mtrx
                        epsy_crack_clamped = self.clamped(depsf, x_short[-1], x_long[-1], epsm_short[-1],
                                     epsm_long[-1], um_short[-1], um_long[-1])
                        epsy_crack[i] = epsy_crack_clamped
    
                elif x_short[-1] == Lmin and x_long[-1] == Lmax:
                    #clamped fibers
                    epsy_crack_clamped = self.clamped(depsf, x_short[-1], x_long[-1], epsm_short[-1],
                                 epsm_long[-1], um_short[-1], um_long[-1])
                    epsy_crack[i] = epsy_crack_clamped
            x_arr = np.hstack((-np.array(x_short)[::-1], np.array(x_long)))
            self.damage = self.vect_xi_cdf(epsy_crack)
            damage.append(np.sum(self.damage))
        #plt.plot(damage)
        #plt.show()
        epsm_arr = np.hstack((np.array(epsm_short)[::-1], np.array(epsm_long)))
        epsy_arr = self.epsy_arr(epsm_arr,
                                 np.sum(epsy_crack * self.sorted_stats_weights * self.sorted_Vf_weights * (1. - self.damage)),
                                 E_mtrx_glob)
        print 'w = ', np.trapz(epsy_arr - epsm_arr, x_arr)
        return x_arr, epsm_arr, epsy_arr, np.sum(epsy_crack * self.sorted_stats_weights * self.sorted_Vf_weights * self.sorted_E_f * (1. - self.damage))


if __name__ == '__main__':

    reinf1 = Reinforcement(r=0.00345,#r=RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=.05, scale=.3),
                          V_f=0.01,
                          E_f=200e3,
                          xi=RV('weibull_min', shape=5., scale=.02),
                          n_int=100)
    
    reinf2 = Reinforcement(r=0.00345,#r=RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=1.5, scale=.3),
                          V_f=0.05,
                          E_f=200e3,
                          xi=RV('weibull_min', shape=5., scale=99.03),
                          n_int=200)

    reinf3 = Reinforcement(r=0.00345,#RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=1., scale=3.),
                          V_f=0.1,
                          E_f=200e3,
                          n_int=20)

    ccb = CompositeCrackBridge(E_m=25e3,
                               reinforcement_lst=[reinf1],
                               Ll=100.,
                               Lr=10.)

    def profile(w):
        ccb.w = w
        ccb.iters = 15
        plt.plot(ccb.profile[0], ccb.profile[1], label='matrix1')
        plt.plot(ccb.profile[0], ccb.profile[2], label='yarn')
        plt.legend(loc='best')
        plt.show()

    def eps_w(w_arr):
#        eps = []
#        ccb.iters = 1
#        for w in w_arr:
##            print 'w_ctrl=', 
#            ccb.w = w
#            ccb.damage = np.zeros_like(ccb.sorted_E_f)
#            eps.append(ccb.profile[3])
#        plt.plot(w_arr, eps, label=str(ccb.iters))
#
#        eps = []
#        ccb.iters = 2
#        for w in w_arr:
##            print 'w_ctrl=', 
#            ccb.w = w
#            ccb.damage = np.zeros_like(ccb.sorted_E_f)
#            eps.append(ccb.profile[3])
#        plt.plot(w_arr, eps, label=str(ccb.iters))
#     
#        eps = []
#        ccb.iters = 3
#        for w in w_arr:
##            print 'w_ctrl=', 
#            ccb.w = w
#            ccb.damage = np.zeros_like(ccb.sorted_E_f)
#            eps.append(ccb.profile[3])
#        plt.plot(w_arr, eps, label=str(ccb.iters))
           
        eps = []
        ccb.iters = 3
        for w in w_arr:
#            print 'w_ctrl=', 
            ccb.w = w
            ccb.damage = np.zeros_like(ccb.sorted_E_f)
            eps.append(ccb.profile[3])
        plt.plot(w_arr, eps, color='black', lw=2, label=str(ccb.iters))
        plt.legend(loc='best')
        plt.show()
    
    #profile(.4)
    
    eps_w(np.linspace(0., 1.5, 50))
