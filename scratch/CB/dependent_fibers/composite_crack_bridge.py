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
from reinforcement import Reinforcement
from scipy.optimize import fsolve


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
        weight_arr = np.array([])
        for reinf in self.reinforcement_lst:
            n_int = len(np.hstack((np.array([]), reinf.depsf_arr[0])).flatten())
            depsf_arr = np.hstack((depsf_arr, reinf.depsf_arr[0])).flatten()
            weight_arr = np.hstack((weight_arr, np.repeat(reinf.depsf_arr[1], n_int)))
            V_f_arr = np.hstack((V_f_arr, np.repeat(reinf.V_f, n_int)))
            E_f_arr = np.hstack((E_f_arr, np.repeat(reinf.E_f, n_int)))
            xi_arr = np.hstack((xi_arr, np.repeat(reinf.xi, n_int)))
        argsort = np.argsort(depsf_arr)[::-1]
        return depsf_arr[argsort], V_f_arr[argsort], E_f_arr[argsort], xi_arr[argsort],  weight_arr[argsort]

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

    sorted_Vf_weights = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_Vf_weights(self):
        return self.sorted_V_f * self.sorted_E_f / (self.E_c -
                (1. - self.V_f_tot) * self.E_m)

    sorted_xi_cdf = Property(depends_on='reinforcement_lst+')
    @cached_property
    def _get_sorted_xi_cdf(self):
        '''breaking strain: CDF for random and Heaviside for discrete values'''
        sorted_xi_cdf = []
        for xi in self.sorted_xi:
            if isinstance(xi, FloatType):
                sorted_xi_cdf.append(lambda x: 1.0 * (xi <= x))
            elif isinstance(xi, RV):
                sorted_xi_cdf.append(xi._distr.cdf)
        return np.array(sorted_xi_cdf)

    def vect_xi_cdf(self, epsy):
        return np.array([ff(epsy[i]) for i, ff in enumerate(self.sorted_xi_cdf)])

    def depsm_depsf(self, depsf, damage):
        '''evaluates the deps_m given deps_f at that point and the damage array'''
        intact_bonded_fibers = np.sum(self.sorted_V_f * self.sorted_stats_weights *
                                      self.sorted_E_f * (depsf <= self.sorted_depsf) *
                                      (1. - damage))
        broken_fibers = np.sum(self.sorted_V_f * self.sorted_stats_weights *
                               self.sorted_E_f * damage)
        add_m_stiffness = intact_bonded_fibers + broken_fibers
        E_mtrx = (1. - self.V_f_tot) * self.E_m + add_m_stiffness
        mean_acting_depsm = np.sum(self.sorted_depsf * (self.sorted_depsf <= depsf) *
                                   self.sorted_stats_weights * self.sorted_E_f *
                                   self.sorted_V_f * (1. - damage))
        return mean_acting_depsm / E_mtrx, (1. - self.V_f_tot) * self.E_m + broken_fibers

    def depsm_depsf2(self, depsf, damage):
        '''evaluates the deps_m given deps_f at that point and the damage array'''
        intact_bonded_fibers = np.sum(self.sorted_V_f * self.sorted_stats_weights *
                                      self.sorted_E_f * (depsf < self.sorted_depsf) *
                                      (1. - damage))
        broken_fibers = np.sum(self.sorted_V_f * self.sorted_stats_weights *
                               self.sorted_E_f * damage)
        add_m_stiffness = intact_bonded_fibers + broken_fibers
        E_mtrx = (1. - self.V_f_tot) * self.E_m + add_m_stiffness
        mean_acting_depsm = np.sum(self.sorted_depsf * (self.sorted_depsf <= depsf) *
                                   self.sorted_stats_weights * self.sorted_E_f *
                                   self.sorted_V_f * (1. - damage))
        print np.sum(self.sorted_depsf * (self.sorted_depsf < depsf) *
                                   self.sorted_stats_weights * self.sorted_E_f *
                                   self.sorted_V_f * (1. - damage))

    def epsy_arr(self, epsm_arr, epsy_crack, E_mtrx_arr):
        sigma_c = epsy_crack * (self.E_c - self.E_m * (1. - self.V_f_tot))
        epsy_arr = (sigma_c - E_mtrx_arr * epsm_arr) / (self.E_c - E_mtrx_arr)
        return epsy_arr

    def double_sided(self, depsf, xi, demi, emi, umi, damage):
        dx = (-depsf * xi - demi * xi + (demi ** 2 * xi ** 2
            + (4 * umi - 4 * emi * xi + 2 * self.w) * demi + ( 2 *
            umi - 2 * emi * xi + self.w) * depsf) ** (0.5)) / (depsf + 2 * demi)
        depsm, E_mtrx = self.depsm_depsf(depsf, damage)
        epsm = emi + demi / 2. * dx + depsm / 2. * dx
        um = umi + emi / 2. * dx + epsm / 2. * dx
        return dx, depsm, epsm, um, E_mtrx

    def one_sided(self, depsf, xi, demi, emi, umi, clamped, damage):
        w = self.w
        depsm, E_mtrx = self.depsm_depsf(depsf, damage)
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

    ii=0
    def damage_residuum(self, iter_damage):
        self.ii += 1
        #print self.ii
        um_short, epsm_short, x_short = [0.0], [0.0], [0.0]
        um_long, epsm_long, x_long = [0.0], [0.0], [0.0]
        depsm_short = [self.depsm_depsf(self.sorted_depsf[0], iter_damage)[0]]
        depsm_long = [self.depsm_depsf(self.sorted_depsf[0], iter_damage)[0]]
        if self.ii == 9 or self.ii == 30:
           self.depsm_depsf2(self.sorted_depsf[-1], iter_damage)
        epsy_crack = np.zeros_like(self.sorted_depsf)
        Lmin = min(self.Ll, self.Lr)
        Lmax = max(self.Ll, self.Lr)
        for i, depsf in enumerate(self.sorted_depsf):
            if x_short[-1] < Lmin and x_long[-1] < Lmax:
                '''double sided pullout'''
                dx, depsm, epsm, um, E_mtrx = self.double_sided(depsf, x_short[-1], depsm_short[-1],
                                                   epsm_short[-1], um_short[-1], iter_damage)

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
                    epsy_crack[i] = (epsm_short[-1] + x_short[-1] * depsf)
                else:
                    # boundary reached at shorter side
                    deltax = Lmin - x_short[-1]
                    x_short.append(Lmin)
                    epsm_short.append(epsm_short[-1] + depsm_short[-1] * deltax)
                    um_short.append(um_short[-1] + (epsm_short[-2] + epsm_short[-1]) * deltax / 2.)
                    short_side = [x_short[-1], epsm_short[-1], um_short[-1]]
                    dx, depsm, epsm, um, E_mtrx = self.one_sided(depsf, x_long[-1], depsm_long[-1],
                                            epsm_long[-1], um_long[-1], short_side, iter_damage)

                    if x_long[-1] + dx >= Lmax:
                        # boundary reached at longer side
                        deltax = Lmax - x_long[-1]
                        x_long.append(Lmax)
                        epsm_long.append(epsm_long[-1] + depsm_long[-1] * deltax)
                        um_long.append(um_long[-1] + (epsm_long[-2] + epsm_long[-1]) * deltax / 2.)
                        epsy_crack_clamped = self.clamped(depsf, x_short[-1], x_long[-1], epsm_short[-1],
                             epsm_long[-1], um_short[-1], um_long[-1])
                        epsy_crack[i] = epsy_crack_clamped
                    else:
                        depsm_long.append(depsm)
                        x_long.append(x_long[-1] + dx)
                        epsm_long.append(epsm)
                        um_long.append(um)
                        epsy_crack[i] = (epsm_long[-1] + x_long[-1] * depsf)                     

            elif x_short[-1] == Lmin and x_long[-1] < Lmax:
                #one sided pullout
                clamped = [x_short[-1], epsm_short[-1], um_short[-1]]
                dx, depsm, epsm, um, E_mtrx = self.one_sided(depsf, x_long[-1], depsm_long[-1],
                                    epsm_long[-1], um_long[-1], clamped, iter_damage)
                if x_long[-1] + dx < Lmax:
                    depsm_long.append(depsm)
                    x_long.append(x_long[-1] + dx)
                    epsm_long.append(epsm)
                    um_long.append(um)
                    epsy_crack[i] = (epsm_long[-1] + x_long[-1] * depsf)
                else:
                    dx = Lmax - x_long[-1]
                    x_long.append(Lmax)
                    epsm_long.append(epsm_long[-1] + depsm_long[-1] * dx)
                    um_long.append(um_long[-1] + (epsm_long[-2] + epsm_long[-1]) * dx / 2.)
                    epsy_crack_clamped = self.clamped(depsf, x_short[-1], x_long[-1], epsm_short[-1],
                                 epsm_long[-1], um_short[-1], um_long[-1])
                    epsy_crack[i] = epsy_crack_clamped

            elif x_short[-1] == Lmin and x_long[-1] == Lmax:
                #clamped fibers
                epsy_crack_clamped = self.clamped(depsf, x_short[-1], x_long[-1], epsm_short[-1],
                             epsm_long[-1], um_short[-1], um_long[-1])
                epsy_crack[i] = epsy_crack_clamped
        #if self.ii == 9 or self.ii == 30:
#        if self.ii == 1:
#           print self.depsm_depsf(self.sorted_depsf[0], iter_damage)[0]
        self.DP.append(np.sum(self.vect_xi_cdf(epsy_crack) - iter_damage))
        return self.vect_xi_cdf(epsy_crack) - iter_damage
    
    DP = List
    
    damage = Property(depends_on='w, Ll, Lr, reinforcement+')
    @cached_property
    def _get_damage(self):
        return fsolve(self.damage_residuum, np.zeros_like(self.sorted_depsf),
                      xtol=0.01)
        
    profile = Property(depends_on='w, Ll, Lr, reinforcement+')
    @cached_property
    def _get_profile(self):
        um_short, epsm_short, x_short = [0.0], [0.0], [0.0]
        um_long, epsm_long, x_long = [0.0], [0.0], [0.0]
        depsm_short = [self.depsm_depsf(self.sorted_depsf[0], self.damage)[0]]
        depsm_long = [self.depsm_depsf(self.sorted_depsf[0], self.damage)[0]]
        E_mtrx_glob = 0.0
        #plt.plot(self.DP)
        #plt.show()
        epsy_crack = np.zeros_like(self.sorted_depsf)
        Lmin = min(self.Ll, self.Lr)
        Lmax = max(self.Ll, self.Lr)
        for i, depsf in enumerate(self.sorted_depsf):
            if x_short[-1] < Lmin and x_long[-1] < Lmax:
                '''double sided pullout'''
                dx, depsm, epsm, um, E_mtrx = self.double_sided(depsf, x_short[-1], depsm_short[-1],
                                                   epsm_short[-1], um_short[-1], self.damage)
                
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
                                            epsm_long[-1], um_long[-1], short_side, self.damage)

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
                                    epsm_long[-1], um_long[-1], clamped, self.damage)
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
        epsm_arr = np.hstack((np.array(epsm_short)[::-1], np.array(epsm_long)))
        epsy_arr = self.epsy_arr(epsm_arr,
                             np.sum(epsy_crack * self.sorted_stats_weights * self.sorted_Vf_weights * (1. - self.damage)),
                             E_mtrx_glob) 
        return x_arr, epsm_arr, epsy_arr, np.sum(epsy_crack *
                                                 self.sorted_stats_weights *
                                                 self.sorted_Vf_weights *
                                                 self.sorted_E_f * (1. - self.damage))

    w_evaluated = Property(depends_on='w, Ll, Lr, reinforcement+')
    @cached_property
    def _get_w_evaluated(self):
        return  np.trapz(self.profile[2] - self.profile[1], self.profile[0])
        

if __name__ == '__main__':

    reinf1 = Reinforcement(r=0.00345,#RV('uniform', loc=0.002, scale=0.002),
                          tau=0.5,#RV('uniform', loc=.5, scale=.001),
                          V_f=0.2,
                          E_f=200e3,
                          xi=RV('weibull_min', shape=5., scale=.02),
                          n_int=25)
    
    reinf2 = Reinforcement(r=0.00345,#r=RV('uniform', loc=0.002, scale=0.002),
                          tau=0.5,#RV('uniform', loc=.0001, scale=.0001),
                          V_f=0.2,
                          E_f=200e3,
                          xi=RV('weibull_min', shape=10., scale=.04),
                          n_int=30)

    reinf3 = Reinforcement(r=0.00345,#RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=1., scale=3.),
                          V_f=0.1,
                          E_f=200e3,
                          n_int=20)

    ccb = CompositeCrackBridge(E_m=25e3,
                               reinforcement_lst=[reinf1],
                               Ll=10.,
                               Lr=10.)

    def profile(w):
        ccb.w = w
        plt.plot(ccb.profile[0], ccb.profile[1], label='w_eval='+str(ccb.w_evaluated)+' w_ctrl='+str(ccb.w))
        plt.plot(ccb.profile[0], ccb.profile[2], label='yarn')
        plt.xlabel('position [mm]')
        plt.ylabel('strain')

    def eps_w(w_arr, label):           
        eps = []
        w_err = []
        for w in w_arr:
            ccb.w = w
            eps.append(ccb.profile[3])
            #w_err.append((ccb.w_evaluated - ccb.w) / (ccb.w + 1e-10))
        #plt.figure()
        #plt.plot(w_arr, w_err)
        #plt.figure()
        plt.plot(w_arr, eps, lw=2, label=label)
        
    def bundle(w_arr, L):
        from scipy.stats import weibull_min
        yy = w_arr / L * (1. - weibull_min(5., scale=0.02).cdf(w_arr/L))
        plt.plot(w_arr, yy * 200e3, lw = 4, color = 'red', ls='dashed', label='bundle')
    
    def plot3d():
        import mayavi.mlab as m
        from stats.spirrid import make_ogrid
        from mathkit.mfn.mfn_line.mfn_line import MFnLineArray
        w_arr = np.linspace(0, .8, 100)
        x_arr = np.linspace(-10., 10., 100)
        profile = np.zeros((100, 100))
        for i, w in enumerate(w_arr):
            ccb.w = w
            line = MFnLineArray(xdata=ccb.profile[0], ydata=ccb.profile[2])
            for j, x in enumerate(x_arr):
                profile[i, j] = line.get_value(x)
        e = make_ogrid([np.arange(len(w_arr)), np.arange(len(x_arr))])
        m.surf(e[0], e[1], profile*500)    
        m.show()
    
    profile(.001)
    
    #eps_w(np.linspace(.0, .5, 30), label='discrete')
    reinf1 = Reinforcement(r=0.00345,#RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=.49999, scale=.00001),
                          V_f=0.2,
                          E_f=200e3,
                          xi=RV('weibull_min', shape=5., scale=.02),
                          n_int=25)
    ccb = CompositeCrackBridge(E_m=25e3,
                               reinforcement_lst=[reinf1],
                               Ll=10.,
                               Lr=10.)
    profile(0.001)
    #eps_w(np.linspace(.0, .5, 30), label='random')
    #bundle(np.linspace(0, 0.65, 30), 20.)
    plt.legend(loc='best')
    plt.show()
    #plot3d()