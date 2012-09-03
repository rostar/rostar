'''
Created on Sep 30, 2011

The example implements micromechanical responses of a crack opening
in a composite material where a single discrete crack is bridged
by:
a) single homogeneous filament
b) yarn with random filament properties
The diagram shows a comparison between the formulation with discrete
filament strength and length dependent strength with residual filament
stress due to pullout of the broken filament from the matrix.
BC - fixed displacement at both ends (Ll and Lr) of the filament.
@author: rostar
'''

from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress import \
    CBEMClampedFiberStress
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress_residual import \
    CBEMClampedFiberStressResidual, CBEMClampedFiberStressResidualSP
from iter_cb import CBEMClampedFiberStressResidualSPIter
from stats.spirrid.spirrid import SPIRRID
from stats.spirrid.rv import RV
from matplotlib import pyplot as plt
import numpy as np
from iteration_CB import CBIter
from iteration_CB2 import CBIter2

if __name__ == '__main__':
    # filaments
    r = 0.00345
    V_f = 0.0103
    tau = RV('uniform', loc = 0.05, scale = 2.)
    E_f = 200e3
    E_m = 25e3
    l = 0.0
    theta = 0.0
    phi = 1.
    Ll = 50.
    Lr = 50.
    s0 = 10.0205
    m = 5.0
    Pf = 0.5#RV('uniform', loc=0., scale=1.0)

    w = 0.4
    x = np.linspace(-40, 40, 1000)

    cb = CBEMClampedFiberStressResidualSP()
    s = SPIRRID(q = cb,
         sampling_type = 'PGrid',
         evars = dict(x = x),
         tvars = dict(w = w, tau = tau, l = l, E_f = E_f, theta = theta, m = m, phi = phi,
                    E_m = E_m, r = r, V_f = V_f, Ll = Ll, Lr = Lr, s0 = s0, Pf = Pf),
         n_int = 100)

    n_int = s.n_int
    pi = np.linspace(0.5 / n_int, 1. - 0.5 / n_int, n_int)

    def iter():
        load = np.ones(len(x)) * np.max(s.mu_q_arr) * V_f
        sigma_m = (load - s.mu_q_arr * V_f) / (1. - V_f)
        # initial spirrid guess
        eps_f = s.mu_q_arr / E_f
        eps_m = sigma_m / E_m
        plt.plot(x, eps_f, lw = 1, color = 'black',
                 ls = 'solid', label = 'initial SPIRRID guess')
        plt.plot(x, eps_m, lw = 1, color = 'black',
                 ls = 'solid')

        w_means = []
        w_stdevs = []
        w_global = []
       
        cb_iter = CBEMClampedFiberStressResidualSPIter()
        w_err = []
        for i in pi:
            t = tau.ppf(i)
            eps_ff = cb_iter(w, x, t, l, E_f, E_m, theta, Pf,
                     phi, Ll, Lr, V_f, r, s0, m) / E_f
            eps_ff = np.maximum(eps_ff, eps_m)
            w_err.append(np.trapz(eps_ff - eps_m, x))
        w_err = np.array(w_err)
        w_means.append(w_err.mean())
        w_stdevs.append(w_err.std())
        w_global.append(np.trapz(eps_f - eps_m, x))

        iters = 5

        for j in range(iters):
            cbi = CBIter(eps_m = eps_m, x_arr = x, q_init = cb.q)
            si = SPIRRID(q = cbi,
                         sampling_type = 'PGrid',
                         evars = dict(x = x),
                         tvars = dict(w = w, tau = tau, l = l, E_f = E_f, theta = theta, m = m, phi = phi,
                        E_m = E_m, r = r, V_f = V_f, Ll = Ll, Lr = Lr, s0 = s0, Pf = Pf),
            n_int = 100)
    
            load = np.ones(len(x)) * np.max(si.mu_q_arr * E_f) * V_f
            sigma_m = (load - si.mu_q_arr * E_f * V_f) / (1. - V_f)
            eps_f = si.mu_q_arr
            eps_m = sigma_m / E_m
            w_err = []
            for i, n in enumerate(pi):
                t = tau.ppf(n)
                q = cbi.q[i]
                Tf = 2. * t / r
                e_x = (q - Tf * np.abs(x)) / E_f
                eps_ff = np.maximum(e_x, eps_m)
                w_err.append(np.trapz(eps_ff - eps_m, x))
            w_err = np.array(w_err)
            w_means.append(w_err.mean())
            w_stdevs.append(w_err.std())
            print 'iter =', j
            w_global.append(np.trapz(eps_f - eps_m, x))

        plt.plot(x, si.mu_q_arr, color = 'blue', label = 'after %i iterations' % iters)
        plt.plot(x, eps_m, color = 'blue')
        plt.legend()
        plt.title('fibers and matrix mean strain')
        plt.xlabel('x position [mm]')
        plt.ylabel('strain [-]')
        plt.figure()
        plt.plot(w_means, label = 'mean of filaments')
        plt.plot(w_global, label = 'global')
        plt.title('mean of w for individual filaments')
        plt.xlabel('number of iterations')
        plt.ylabel('mean w')
        plt.legend()
        plt.figure()
        plt.plot(np.array(w_stdevs) / np.array(w_means))
        plt.title('COV of w for individual filaments')
        plt.xlabel('number of iterations')
        plt.ylabel('stdev w')
        plt.legend()
        plt.show()

    def iter2():
        load = np.ones(len(x)) * np.max(s.mu_q_arr) * V_f
        sigma_m = (load - s.mu_q_arr * V_f) / (1. - V_f)
        # initial spirrid guess
        eps_f = s.mu_q_arr / E_f
        eps_m = sigma_m / E_m
        plt.plot(x, eps_f, lw = 1, color = 'black',
                 ls = 'solid', label = 'initial SPIRRID guess')
        plt.plot(x, eps_m, lw = 1, color = 'black',
                 ls = 'solid')

        w_means = []
        w_stdevs = []
       
        cb_iter = CBEMClampedFiberStressResidualSPIter()
        w_err = []
        for i in pi:
            t = tau.ppf(i)
            eps_ff = cb_iter(w, x, t, l, E_f, E_m, theta, Pf,
                     phi, Ll, Lr, V_f, r, s0, m) / E_f
            eps_ff = np.maximum(eps_ff, eps_m)
            w_err.append(np.trapz(eps_ff - eps_m, x))
        w_err = np.array(w_err)
        w_means.append(w_err.mean())
        w_stdevs.append(w_err.std())

        iters = 20

        for i in range(iters):
            print i
            cbi = CBIter2(eps_m = eps_m, x_arr = x, q_i = cb.q)
            si = SPIRRID(q = cbi,
                         sampling_type = 'PGrid',
                         evars = dict(x = x),
                         tvars = dict(w = w, tau = tau, l = l, E_f = E_f, theta = theta, m = m, phi = phi,
                        E_m = E_m, r = r, V_f = V_f, Ll = Ll, Lr = Lr, s0 = s0, Pf = Pf),
            n_int = 100)

            load = np.ones(len(x)) * np.max(si.mu_q_arr * E_f) * V_f
            sigma_m = (load - si.mu_q_arr * E_f * V_f) / (1. - V_f)
            eps_f = si.mu_q_arr
            eps_m = sigma_m / E_m
            w_err = []
            for i, n in enumerate(pi):
                t = tau.ppf(n)
                q = cbi.q[i]
                Tf = 2. * t / r
                e_x = (q - Tf * np.abs(x)) / E_f
                eps_ff = np.maximum(e_x, eps_m)
                w_err.append(np.trapz(eps_ff - eps_m, x))
            w_err = np.array(w_err)
            w_means.append(w_err.mean())
            w_stdevs.append(w_err.std())

        plt.plot(x, si.mu_q_arr, color = 'blue', label = 'after %i iterations' % iters)
        plt.plot(x, eps_m, color = 'blue')
        plt.legend()
        plt.title('fibers and matrix mean strain')
        plt.xlabel('x position [mm]')
        plt.ylabel('strain [-]')
        plt.figure()
        plt.plot(w_means)
        plt.title('mean of w for individual filaments')
        plt.xlabel('number of iterations')
        plt.ylabel('mean w')
        plt.figure()
        plt.plot(np.array(w_stdevs) / np.array(w_means))
        plt.title('COV of w for individual filaments')
        plt.xlabel('number of iterations')
        plt.ylabel('stdev w')
        plt.legend()
        plt.show()

    iter()
#    iter2()
