'''
Created on Sep 20, 2012

The CompositeCrackBridge class has a method for evaluating fibers and matrix
strain in the vicinity of a crack bridge.
Fiber diameter and bond coefficient can be set as random variables.
Reinforcement types can be combined by creating a list of Reinforcement
instances and defining it as the reinforcement_lst Trait in the
CompositeCrackBridge class.

@author: rostar
'''
import numpy as np
from stats.spirrid.rv import RV
from matplotlib import pyplot as plt
from etsproxy.traits.api import HasTraits, cached_property, \
    Float, Property, Instance, List
from types import FloatType
from reinforcement import Reinforcement, WeibullFibers
from scipy.optimize import fsolve, broyden2
import time as t

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
    @cached_property
    def _get_E_c(self):
        E_fibers = 0.0
        for reinf in self.reinforcement_lst:
            E_fibers += reinf.V_f * reinf.E_f
        return self.E_m * (1. - self.V_f_tot) + E_fibers

    sorted_theta = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_theta(self):
        '''sorts the integral points by bond in descending order'''
        depsf_arr = np.array([])
        V_f_arr = np.array([])
        E_f_arr = np.array([])
        xi_arr = np.array([])
        stat_weights_arr = np.array([])
        V_f_weights_arr = np.array([])
        for reinf in self.reinforcement_lst:
            n_int = len(np.hstack((np.array([]), reinf.depsf_arr)))
            depsf_arr = np.hstack((depsf_arr, reinf.depsf_arr))
            V_f_arr = np.hstack((V_f_arr, np.repeat(reinf.V_f, n_int)))
            E_f_arr = np.hstack((E_f_arr, np.repeat(reinf.E_f, n_int)))
            xi_arr = np.hstack((xi_arr, np.repeat(reinf.xi, n_int)))
            stat_weights_arr = np.hstack((stat_weights_arr,
                                          np.repeat(reinf.stat_weights, n_int)))
            V_f_weights_arr = np.hstack((V_f_weights_arr, reinf.V_f_weights))
        argsort = np.argsort(depsf_arr)[::-1]
        return depsf_arr[argsort], V_f_arr[argsort], E_f_arr[argsort], \
                xi_arr[argsort],  stat_weights_arr[argsort], \
                V_f_weights_arr[argsort]

    sorted_depsf = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_depsf(self):
        return self.sorted_theta[0]

    sorted_V_f = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_V_f(self):
        return self.sorted_theta[1]

    sorted_E_f = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_E_f(self):
        return self.sorted_theta[2]

    sorted_xi = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_xi(self):
        return self.sorted_theta[3]

    sorted_stats_weights = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_stats_weights(self):
        return self.sorted_theta[4]

    sorted_V_f_weights = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_V_f_weights(self):
        return self.sorted_theta[5]

    sorted_E_f_ratio = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_E_f_ratio(self):
        '''stiffness of a reinforcement type with
        respect to total reinforcement stiffness'''
        return self.sorted_V_f * self.sorted_E_f / (self.E_c -
                (1. - self.V_f_tot) * self.E_m)

    sorted_xi_cdf = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_xi_cdf(self):
        '''breaking strain: CDF for random and Heaviside for discrete values'''
        methods = []
        masks = []
        for reinf in self.reinforcement_lst:
            masks.append(self.sorted_xi == reinf.xi)
            if isinstance(reinf.xi, FloatType):
                methods.append(lambda x: 1.0 * (reinf.xi <= x))
            elif isinstance(reinf.xi, RV):
                methods.append(reinf.xi._distr.cdf)
            elif isinstance(reinf.xi, WeibullFibers):
                methods.append(reinf.xi.weibull_fibers_Pf)
        return methods, masks

    def vect_xi_cdf(self, epsy, x_short, x_long):
        Pf = np.zeros_like(self.sorted_depsf)
        methods, masks = self.sorted_xi_cdf
        for i, method in enumerate(methods):
            if method.__name__ == 'weibull_fibers_Pf':
                Pf += method(epsy * masks[i], self.sorted_depsf,
                             x_short=x_short, x_long=x_long)
            else:
                Pf += method(epsy * masks[i])
        return Pf

    def dem_depsf(self, depsf, damage):
        '''evaluates the deps_m given deps_f
        at that point and the damage array'''
        Kf = self.sorted_V_f * self.sorted_V_f_weights * \
            self.sorted_stats_weights * self.sorted_E_f
        Kf_intact_bonded = np.sum(Kf * (depsf <= self.sorted_depsf)
                                         * (1. - damage))
        Kf_broken = np.sum(Kf * damage)
        Kf_add = Kf_intact_bonded + Kf_broken
        Km = (1. - self.V_f_tot) * self.E_m
        Kc = Km + Kf_add
        mean_acting_depsm = np.sum(self.sorted_depsf * (self.sorted_depsf < depsf) *
                                   self.sorted_stats_weights * self.sorted_E_f *
                                   self.sorted_V_f * self.sorted_V_f_weights * (1. - damage))
        return mean_acting_depsm / Kc, (1. - self.V_f_tot) * self.E_m + Kf_broken

    def dem_init(self, damage):
        '''evaluates the initial slope of eps_m given the damage array'''
        Kf_broken = np.sum(self.sorted_V_f * self.sorted_V_f_weights *
                           self.sorted_stats_weights * self.sorted_E_f *
                           damage)
        Kc = (1. - self.V_f_tot) * self.E_m + Kf_broken
        mean_acting_depsm = np.sum(self.sorted_depsf * self.sorted_stats_weights *
                                   self.sorted_E_f * self.sorted_V_f *
                                   self.sorted_V_f_weights * (1. - damage))
        return mean_acting_depsm / Kc

    def epsy_arr(self, epsm_arr, epsy_crack, E_mtrx_arr):
        mu_epsf0 = np.sum(epsy_crack * self.sorted_stats_weights * self.sorted_E_f_ratio *\
                self.sorted_V_f_weights * (1 - self.damage))
        sigma_c = mu_epsf0 * (self.E_c - self.E_m * (1. - self.V_f_tot))
        epsy_arr = (sigma_c - E_mtrx_arr * epsm_arr) / (self.E_c - E_mtrx_arr)
        return epsy_arr

    def double_sided(self, defi, x0, demi, em0, um0, damage):
        dxi = (-defi * x0 - demi * x0 + (defi * x0 ** 2 * demi
            + demi ** 2 * x0 ** 2 - 2 * defi * em0 * x0 + 2 *
            defi * um0 + defi * self.w - 2 * demi * em0 * x0 +
            2 * demi * um0 + demi * self.w) ** (.5)) / (defi + demi)
        dem, E_mtrx = self.dem_depsf(defi, damage)
        emi = em0 + demi * dxi
        umi = um0 + (em0 + emi) * dxi / 2.
        return dxi, dem, emi, umi, E_mtrx

    def one_sided(self, defi, x0, demi, em0, um0, clamped, damage):
        w = self.w
        xs = clamped[0]
        ums = clamped[1]
        dxi = (-xs * demi - demi * x0 - defi * xs - defi * x0 + (2 *
                demi * x0 * defi * xs + demi * x0 ** 2 * defi + 2 *
                demi ** 2 * x0 * xs + 3 * defi * xs ** 2 * demi - 2 *
                demi * xs * em0 - 2 * demi * em0 * x0 - 2 * defi *
                xs * em0 - 2 * defi * em0 * x0 + demi ** 2 * x0 ** 2 +
                2 * defi ** 2 * xs ** 2 + xs ** 2 * demi ** 2 + 2 *
                demi * um0 + 2 * demi * ums + 2 * demi * w + 2 * defi *
                um0 + 2 * defi * ums + 2 * defi * w) ** (0.5)) / (demi + defi)
        dem, E_mtrx = self.dem_depsf(defi, damage)
        emi = em0 + demi * dxi
        umi = um0 + (em0 + emi) * dxi / 2.
        return dxi, dem, emi, umi, E_mtrx

    def clamped(self, defi, xs, xl, ems, eml, ums, uml):
        c1 = eml * xl - uml
        c2 = ems * xs - ums
        c3 = defi * xl ** 2 / 2.
        c4 = defi * xs ** 2 / 2.
        c5 = (defi * (xl - xs) + (eml - ems)) * xs
        h = (self.w - c1 - c2 - c3 - c4 - c5) / (xl + xs)
        return defi * xl + eml + h

    def damage_residuum(self, iter_damage):
        um_short, em_short, x_short = [0.0], [0.0], [0.0]
        um_long, em_long, x_long = [0.0], [0.0], [0.0]
        dem_short = [self.dem_init(iter_damage)]
        dem_long = [self.dem_init(iter_damage)]
        epsy_crack = np.zeros_like(self.sorted_depsf)
        Lmin = min(self.Ll, self.Lr)
        Lmax = max(self.Ll, self.Lr)
        for i, defi in enumerate(self.sorted_depsf):
            if x_short[-1] < Lmin and x_long[-1] < Lmax:
                '''double sided pullout'''
                dxi, dem, emi, umi, E_mtrx = self.double_sided(defi,
                                    x_short[-1], dem_short[-1],
                                    em_short[-1], um_short[-1], iter_damage)

                if x_short[-1] + dxi < Lmin:
                    # dx increment does not reach the boundary
                    dem_short.append(dem)
                    dem_long.append(dem)
                    x_short.append(x_short[-1] + dxi)
                    x_long.append(x_long[-1] + dxi)
                    em_short.append(emi)
                    em_long.append(emi)
                    um_short.append(umi)
                    um_long.append(umi)
                    epsy_crack[i] = (em_short[-1] + x_short[-1] * defi)
                else:
                    # boundary reached at shorter side
                    deltax = Lmin - x_short[-1]
                    x_short.append(Lmin)
                    em_short.append(em_short[-1] + dem_short[-1] * deltax)
                    um_short.append(um_short[-1] + (em_short[-2] + em_short[-1]) * deltax / 2.)
                    short_side = [x_short[-1], um_short[-1]]
                    dxi, dem, emi, umi, E_mtrx = self.one_sided(defi, x_long[-1], dem_long[-1],
                                            em_long[-1], um_long[-1], short_side, iter_damage)

                    if x_long[-1] + dxi >= Lmax:
                        # boundary reached at longer side
                        deltax = Lmax - x_long[-1]
                        x_long.append(Lmax)
                        em_long.append(em_long[-1] + dem_long[-1] * deltax)
                        um_long.append(um_long[-1] + (em_long[-2] + em_long[-1]) * deltax / 2.)
                        epsy_crack_clamped = self.clamped(defi, x_short[-1], x_long[-1], em_short[-1],
                             em_long[-1], um_short[-1], um_long[-1])
                        epsy_crack[i] = epsy_crack_clamped
                    else:
                        dem_long.append(dem)
                        x_long.append(x_long[-1] + dxi)
                        em_long.append(emi)
                        um_long.append(umi)
                        epsy_crack[i] = (em_long[-1] + x_long[-1] * defi)

            elif x_short[-1] == Lmin and x_long[-1] < Lmax:
                #one sided pullout
                clamped = [x_short[-1], um_short[-1]]
                dxi, dem, emi, umi, E_mtrx = self.one_sided(defi, x_long[-1], dem_long[-1],
                                    em_long[-1], um_long[-1], clamped, iter_damage)
                if x_long[-1] + dxi < Lmax:
                    dem_long.append(dem)
                    x_long.append(x_long[-1] + dxi)
                    em_long.append(emi)
                    um_long.append(umi)
                    epsy_crack[i] = (em_long[-1] + x_long[-1] * defi)
                else:
                    dxi = Lmax - x_long[-1]
                    x_long.append(Lmax)
                    em_long.append(em_long[-1] + dem_long[-1] * dxi)
                    um_long.append(um_long[-1] + (em_long[-2] + em_long[-1]) * dxi / 2.)
                    epsy_crack_clamped = self.clamped(defi, x_short[-1], x_long[-1], em_short[-1],
                                 em_long[-1], um_short[-1], um_long[-1])
                    epsy_crack[i] = epsy_crack_clamped

            elif x_short[-1] == Lmin and x_long[-1] == Lmax:
                #clamped fibers
                epsy_crack_clamped = self.clamped(defi, x_short[-1], x_long[-1], em_short[-1],
                             em_long[-1], um_short[-1], um_long[-1])
                epsy_crack[i] = epsy_crack_clamped
        residuum = self.vect_xi_cdf(epsy_crack, x_short=x_short, x_long=x_long) - iter_damage
        return residuum

    damage = Property(depends_on='w, Ll, Lr, reinforcement+')
    @cached_property
    def _get_damage(self):
        if self.w == 0.:
            damage = np.zeros_like(self.sorted_depsf)
        else:
            ff = t.clock()
            try:
                damage = broyden2(self.damage_residuum, 0.2 * np.ones_like(self.sorted_depsf), maxiter=20)
            except:
                print 'broyden2 does not converge fast enough: switched to fsolve for this step'
                damage = fsolve(self.damage_residuum, 0.2 * np.ones_like(self.sorted_depsf))
            print 'damage =', np.sum(damage) / len(damage), 'iteration time =', t.clock() - ff, 'sec' 
        return damage 

    results = Property(depends_on='w, Ll, Lr, reinforcement+')
    @cached_property
    def _get_results(self):
        um_short, em_short, x_short = [0.0], [0.0], [0.0]
        um_long, em_long, x_long = [0.0], [0.0], [0.0]
        dem_short = [self.dem_init(self.damage)]
        dem_long = [self.dem_init(self.damage)]
        epsy_crack = np.zeros_like(self.sorted_depsf)
        Lmin = min(self.Ll, self.Lr)
        Lmax = max(self.Ll, self.Lr)
        for i, defi in enumerate(self.sorted_depsf):
            if x_short[-1] < Lmin and x_long[-1] < Lmax:
                '''double sided pullout'''
                dxi, dem, emi, umi, E_mtrx = self.double_sided(defi, x_short[-1], dem_short[-1],
                                                   em_short[-1], um_short[-1], self.damage)

                if x_short[-1] + dxi < Lmin:
                    # dx increment does not reach the boundary
                    dem_short.append(dem)
                    dem_long.append(dem)
                    x_short.append(x_short[-1] + dxi)
                    x_long.append(x_long[-1] + dxi)
                    em_short.append(emi)
                    em_long.append(emi)
                    um_short.append(umi)
                    um_long.append(umi)
                    epsy_crack[i] = (em_short[-1] + x_short[-1] * defi)
                    E_mtrx_glob = E_mtrx
                else:
                    # boundary reached at shorter side
                    deltax = Lmin - x_short[-1]
                    x_short.append(Lmin)
                    em_short.append(em_short[-1] + dem_short[-1] * deltax)
                    um_short.append(um_short[-1] + (em_short[-2] + em_short[-1]) * deltax / 2.)
                    short_side = [x_short[-1], um_short[-1]]
                    dxi, dem, emi, umi, E_mtrx = self.one_sided(defi, x_long[-1], dem_long[-1],
                                            em_long[-1], um_long[-1], short_side, self.damage)
                    E_mtrx_glob = E_mtrx
                    if x_long[-1] + dxi >= Lmax:
                        # boundary reached at longer side
                        deltax = Lmax - x_long[-1]
                        x_long.append(Lmax)
                        em_long.append(em_long[-1] + dem_long[-1] * deltax)
                        um_long.append(um_long[-1] + (em_long[-2] + em_long[-1]) * deltax / 2.)
                        epsy_crack_clamped = self.clamped(defi, x_short[-1], x_long[-1], em_short[-1],
                             em_long[-1], um_short[-1], um_long[-1])
                        epsy_crack[i] = epsy_crack_clamped
                    else:
                        dem_long.append(dem)
                        x_long.append(x_long[-1] + dxi)
                        em_long.append(emi)
                        um_long.append(umi)
                        epsy_crack[i] = (em_long[-1] + x_long[-1] * defi)

            elif x_short[-1] == Lmin and x_long[-1] < Lmax:
                #one sided pullout
                clamped = [x_short[-1], um_short[-1]]
                dxi, dem, emi, umi, E_mtrx = self.one_sided(defi, x_long[-1], dem_long[-1],
                                    em_long[-1], um_long[-1], clamped, self.damage)
                if x_long[-1] + dxi < Lmax:
                    dem_long.append(dem)
                    x_long.append(x_long[-1] + dxi)
                    em_long.append(emi)
                    um_long.append(umi)
                    E_mtrx_glob = E_mtrx
                    epsy_crack[i] = (em_long[-1] + x_long[-1] * defi)
                else:
                    dxi = Lmax - x_long[-1]
                    x_long.append(Lmax)
                    em_long.append(em_long[-1] + dem_long[-1] * dxi)
                    um_long.append(um_long[-1] + (em_long[-2] + em_long[-1]) * dxi / 2.)
                    E_mtrx_glob = E_mtrx
                    epsy_crack_clamped = self.clamped(defi, x_short[-1], x_long[-1], em_short[-1],
                                 em_long[-1], um_short[-1], um_long[-1])
                    epsy_crack[i] = epsy_crack_clamped

            elif x_short[-1] == Lmin and x_long[-1] == Lmax:
                #clamped fibers
                epsy_crack_clamped = self.clamped(defi, x_short[-1], x_long[-1], em_short[-1],
                             em_long[-1], um_short[-1], um_long[-1])
                epsy_crack[i] = epsy_crack_clamped
        x_arr = np.hstack((-np.array(x_short)[::-1], np.array(x_long)))
        em_arr = np.hstack((np.array(em_short)[::-1], np.array(em_long)))
        ey_arr = self.epsy_arr(em_arr, epsy_crack, E_mtrx_glob)
        return x_arr, em_arr, ey_arr, np.sum(epsy_crack *
                                            self.sorted_stats_weights *
                                            self.sorted_E_f_ratio *
                                            self.sorted_V_f_weights *
                                            self.sorted_E_f * (1. - self.damage))

    x_arr = Property(depends_on='w, Ll, Lr, reinforcement+')
    @cached_property
    def _get_x_arr(self):
        return self.results[0]

    em_arr = Property(depends_on='w, Ll, Lr, reinforcement+')
    @cached_property
    def _get_em_arr(self):
        return self.results[1]

    ey_arr = Property(depends_on='w, Ll, Lr, reinforcement+')
    @cached_property
    def _get_ey_arr(self):
        return self.results[2]

    max_norm_stress = Property(depends_on='w, Ll, Lr, reinforcement+')
    @cached_property
    def _get_max_norm_stress(self):
        return self.results[3]

    w_evaluated = Property(depends_on='w, Ll, Lr, reinforcement+')
    @cached_property
    def _get_w_evaluated(self):
        return  np.trapz(self.ey_arr - self.em_arr, self.x_arr)

if __name__ == '__main__':

    reinf1 = Reinforcement(r=RV('uniform', loc=0.001, scale=0.005),
                          tau=RV('uniform', loc=.5, scale=.5),
                          V_f=0.2,
                          E_f=200e3,
                          xi=WeibullFibers(shape=5., scale=0.02, L0=10.),#RV('weibull_min', shape=5., scale=.02),
                          n_int=15)

    reinf2 = Reinforcement(r=RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=.5, scale=.1),
                          V_f=0.3,
                          E_f=200e3,
                          xi=RV('weibull_min', shape=10., scale=.03),
                          n_int=15)

    reinf3 = Reinforcement(r=0.00345,#RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=1., scale=3.),
                          V_f=0.1,
                          E_f=200e3,
                          xi=0.03,
                          n_int=20)

    ccb = CompositeCrackBridge(E_m=25e3,
                               reinforcement_lst=[reinf1, reinf2],
                               Ll=3.,
                               Lr=5.)

    def profile(w):
        ccb.w = w
        plt.plot(ccb.x_arr, ccb.em_arr, label='w_eval=' + str(ccb.w_evaluated) + ' w_ctrl=' + str(ccb.w))
        plt.plot(ccb.x_arr, ccb.ey_arr, label='yarn')
        plt.xlabel('position [mm]')
        plt.ylabel('strain')

    def eps_w(w_arr, label):
        eps = []
        w_err = []
        for w in w_arr:
            ccb.w = w
            eps.append(ccb.max_norm_stress)
            w_err.append((ccb.w_evaluated - ccb.w) / (ccb.w + 1e-10))
        plt.figure()
        plt.plot(w_arr, w_err, label='error in w')
        plt.legend(loc='best')
        plt.figure()
        plt.plot(w_arr, eps, lw=2, label=label)
        plt.legend(loc='best')

    def bundle(w_arr, L):
        from scipy.stats import weibull_min
        yy = w_arr / L * (1. - weibull_min(5., scale=0.02).cdf(w_arr / L))
        plt.plot(w_arr, yy * 200e3, lw=4, color='red', ls='dashed', label='bundle')

    #profile(.03)
    eps_w(np.linspace(.0, .3, 50), label='ld')
    #bundle(np.linspace(0, 0.65, 30), 20.)
    plt.legend(loc='best')
    plt.show()
