
from cb_adapt_Vf import CBEMClampedFiberStressVf
from cb_adapt_Vf_w_controled import CBEMClampedFiberStressVfw
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber_stress import CBEMClampedFiberStress
from stats.spirrid.spirrid import SPIRRID
from stats.spirrid.rv import RV
from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':

    #----------------------------------------------------
    # controled by BC points and without residual stress
    # ---------------------------------------------------
    
    # filaments
    r = 0.00345
    V_f = 0.1
    tau = 0.5#RV('uniform', loc=0.05, scale=2.)
    E_f = 200e3
    E_m = 25e3
    l = 0.0
    theta = 0.0
    phi = 1.
    Ll = 50.
    Lr = 50.
    xi = RV('weibull_min', shape=5., scale=.02)

    w = np.linspace(0, 1.0, 200)

    def no_res_stress_CB():
        cb = CBEMClampedFiberStressVf()
        s = SPIRRID(q=cb,
             sampling_type='LHS',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=500)

        sb = [0]
        w_eval = [0]
        for vf in np.linspace(0.1, 0.0005, 100):
            s.tvars['V_f'] = vf
            #plt.plot(w, s.mu_q_arr)
            sb.append(np.max(s.mu_q_arr))
            w_eval.append(w[np.argmax(s.mu_q_arr)])
        plt.plot(w_eval, sb, label='BC')

    #-----------------------------------------------
    # controled by crack opening no residual stress
    # ----------------------------------------------

    def no_res_stress_w():
        cb = CBEMClampedFiberStressVfw()
        s = SPIRRID(q=cb,
             sampling_type='LHS',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=500)

        sb = [0]
        w_eval = [0]
        for vf in np.linspace(0.1, 0.0005, 100):
            s.tvars['V_f'] = vf
            sb.append(np.max(s.mu_q_arr))
            w_eval.append(w[np.argmax(s.mu_q_arr)])
        plt.plot(w_eval, sb, label='w')

    def no_adaption():
        cb = CBEMClampedFiberStress()
        s = SPIRRID(q=cb,
             sampling_type='LHS',
             evars=dict(w=w),
             tvars=dict(tau=tau, l=l, E_f=E_f, theta=theta, xi=xi, phi=phi,
                        E_m=E_m, r=r, V_f=V_f, Ll=Ll, Lr=Lr),
             n_int=500)
        plt.plot(w, s.mu_q_arr * V_f / 100., label='no adaption')

no_res_stress_CB()
no_res_stress_w()
no_adaption()
plt.legend()
plt.show()