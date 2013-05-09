'''
Created on Oct 26, 2012
compares the model with analytical results for
1) fiber bundle
2) composite with deterministic bond and random strength
@author: rostar
'''

from depend_CB_model import CompositeCrackBridge
from depend_CB_postprocessor import CompositeCrackBridgeView
from reinforcement import Reinforcement
from spirrid.rv import RV
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    # AR-glass
    reinf1 = Reinforcement(r=0.001,#RV('uniform', loc=0.012, scale=0.002),
                          tau=0.1,#RV('uniform', loc=.3, scale=.1),
                          V_f=0.2,
                          E_f=72e3,
                          xi=RV('weibull_min', shape=5., scale=.02),
                          n_int=50)

    # carbon
    reinf2 = Reinforcement(r=RV('uniform', loc=0.002, scale=0.002),
                          tau=RV('uniform', loc=.6, scale=.1),
                          V_f=0.05,
                          E_f=200e3,
                          xi=RV('weibull_min', shape=10., scale=.015),
                          n_int=15)

    # instance of CompCrackBridge with matrix E and BC
    model = CompositeCrackBridge(E_m=25e3,
                               reinforcement_lst=[reinf1],
                               Ll=10.,
                               Lr=20.)

    ccb_view = CompositeCrackBridgeView(model=model)

    def bundle_comparison(w_arr, L, shape, scale, E):
        '''bundle (Weibull fibers) response for comparison with the CB model'''
        from scipy.stats import weibull_min
        eps = w_arr / L * (1. - weibull_min(shape, scale=scale).cdf(w_arr / L))
        plt.plot(w_arr / L, eps * E, lw=4, color='red', ls='dashed', label='FB model')

        bundle = Reinforcement(r=0.13, tau=0.00001, V_f=0.9999, E_f=E,
                          xi=RV('weibull_min', shape=shape, scale=scale),
                          n_int=50)
        ccb = CompositeCrackBridge(E_m=25e3,
                                   reinforcement_lst=[bundle],
                                   Ll=L / 2.,
                                   Lr=L / 2.)
        ccb_view.model = ccb
        eps = []
        for w in w_arr:
            ccb.w = w
            eps.append(ccb_view.sigma_c / E)
        plt.plot(w_arr / L, np.array(eps) * E, color='blue', lw=2, label='CB model')
        plt.legend(loc='best')

    def analytical_comparison():
        '''for the case when tau is deterministic,
        there is an analytical solution.
        The differences are caused by the additional matrix
        stiffness due to broken fibers, which are in the CB model
        added to matrix stiffness. As the matrix E grows and the V_f
        decreases, the solutions tend to get closer'''
        tau, E_f, E_m, V_f = 0.1, 72e3, 25e3, 0.2
        r, shape, scale = 0.001, 5., 0.02
        # analytical solution for damage controlled test
        ctrl_damage = np.linspace(0.0, .99, 100)

        def crackbridge(w, tau, E_f, E_m, V_f, r, omega):
            Kf = E_f * V_f * (1 - omega)
            Km = E_m * (1 - V_f) + E_f * V_f * omega 
            Kc = Kf + Km
            T = 2. * tau * V_f * (1. - omega) / r
            c = np.sqrt(Kc * T / Km / Kf)
            return c * np.sqrt(w) * (1 - omega)

        def w_omega(tau, E_f, E_m, V_f, r, omega, shape, scale):
            Kf = E_f * V_f * (1 - omega)
            Km = E_m * (1 - V_f) + E_f * V_f * omega
            Kc = Kf + Km
            T = 2. * tau * V_f * (1. - omega) / r
            return (-np.log(1. - omega)) ** (2. / shape) \
                    * scale ** 2 * Km * Kf / Kc / T

        w_lst = [w_omega(tau, E_f, E_m, V_f, r, omega, shape, scale)
                 for omega in ctrl_damage]
        epsf = crackbridge(np.array(w_lst), tau, E_f, E_m, V_f, r, ctrl_damage)
        plt.plot(np.array(w_lst), epsf * E_f * V_f, color='red',
                lw=4, ls='dashed', label='analytical')

        reinf = Reinforcement(r=r, tau=tau, V_f=V_f, E_f=E_f,
                              xi=RV('weibull_min', shape=shape, scale=scale),
                              n_int=20)

        ccb = CompositeCrackBridge(E_m=E_m, reinforcement_lst=[reinf],
                                   Ll=1000., Lr=1000.)

        stress = []
        w_arr = np.linspace(0.0, np.max(w_lst), 100)
        for w in w_arr:
            ccb_view.model.w = w
            stress.append(ccb_view.sigma_c)
        plt.plot(w_arr, stress, color='blue', lw=2, label='CB model')
        plt.legend(loc='best')

    bundle_comparison(np.linspace(0, 0.65, 30), 20., 5., 0.02, 70e3)
    #analytical_comparison()
    plt.show()
