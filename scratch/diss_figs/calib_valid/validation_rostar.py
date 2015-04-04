'''
Created on 16 Feb 2014

@author: Q
'''

import numpy as np
from quaducom.meso.homogenized_crack_bridge.rigid_matrix.CB_view import CBView, Model
from spirrid.rv import RV
from stats.misc.random_field.random_field_1D import RandomField
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_model import SCM
from quaducom.meso.scm.numerical.interdependent_fibers.scm_interdependent_fibers_view import SCMView
from quaducom.meso.homogenized_crack_bridge.elastic_matrix.hom_CB_elastic_mtrx import CompositeCrackBridge
from quaducom.micro.resp_func.CB_clamped_rand_xi import CBClampedRandXi
from spirrid.spirrid import SPIRRID
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from math import pi, e
from scipy.special import gamma

def CB():
    model = Model(w_min=0.0, w_max=8.0, w_pts=100,
                  w2_min=0.0, w2_max=.3, w2_pts=50,
                  sV0=0.0094, m=9.0, tau_loc=0.00, Ef=180e3,
                  lm=15., n_int=100)
    
    for i in range(5):
        cb = np.loadtxt("CB" + str(i+1) + ".txt", delimiter=';')
        model.test_xdata.append(-cb[:,2]/4. - cb[:,3]/4. - cb[:,4]/2.)
        model.test_ydata.append(cb[:,1] / (11. * 0.445) * 1000)
    
    cb = CBView(model=model)
    cb.refresh()
    cb.configure_traits()
    
def TT():
    for i in range(5):
        file_i = np.loadtxt("TT-4C-0" + str(i+1) + ".txt", delimiter=';')
        plt.plot(-file_i[:,2]/2./250. - file_i[:,3]/2./250.,file_i[:,1]/2., lw=1, color='blue')
        file_i2 = np.loadtxt("TT-6C-0" + str(i+1) + ".txt", delimiter=';')
        plt.plot(-file_i2[:,2]/2./250. - file_i2[:,3]/2./250.,file_i2[:,1]/2., lw=1, color='red')
    plt.legend(loc='best')
    plt.xlim(0)


def valid(CS, param_set, w_max):
    plt.figure()
    #TT()
    from quaducom.meso.homogenized_crack_bridge.elastic_matrix.reinforcement import ContinuousFibers
    from stats.pdistrib.weibull_fibers_composite_distr import fibers_MC
    length = 250.
    nx = 5000
    tau_loc, tau_shape, xi_shape = param_set
    E_f = 182e3 
    mu_tau = 1.3 * 3.5e-3 * 3.6 * (1.-0.01) / (2. * 0.01 * CS)
    tau_scale = (mu_tau - tau_loc)/tau_shape
    xi_scale = 3243. / (182e3 * (pi * 3.5e-3 **2 * 50.)**(-1./xi_shape)*gamma(1+1./xi_shape))
    #xi_scale = 1578. / (182e3 * (pi * 3.5e-3 **2 * 500. * e)**(-1./xi_shape))
    n_CB = 8
    ld = False
    w_width = False
    w_density = False
    
    random_field1 = RandomField(seed=False,
                               lacor=1.,
                               length=length,
                               nx=800,
                               nsim=1,
                               loc=.0,
                               shape=45.,
                               scale=3.6,
                               distr_type='Weibull'
                               )
 
    reinf1 = ContinuousFibers(r=3.5e-3,
                              tau=RV('gamma', loc=tau_loc, scale=tau_scale, shape=tau_shape),
                              V_f=0.01,
                              E_f=E_f,
                              xi=fibers_MC(m=xi_shape, sV0=xi_scale),
                              label='carbon',
                              n_int=500)
 
    CB_model1 = CompositeCrackBridge(E_m=25e3,
                                 reinforcement_lst=[reinf1],
                                 )
 
    scm1 = SCM(length=length,
              nx=nx,
              random_field=random_field1,
              CB_model=CB_model1,
              load_sigma_c_arr=np.linspace(0.01, 35., 100),
              n_BC_CB=n_CB
              )
 
    scm_view1 = SCMView(model=scm1)
    scm_view1.model.evaluate()
    eps1, sigma1 = scm_view1.eps_sigma
    eps11, sigma11 = scm_view1.eps_sigma_altern
    cr_lst1 = scm_view1.model.cracking_stress_lst
    interp1 = interp1d(sigma1, eps1)
    plt.plot(eps11, sigma11, color='red', lw=2,
                label='Vf=1.0, crack spacing = ' + str(length/float(len(cr_lst1))))
    #plt.plot(eps11, sigma11, color='black', ls='dashed', lw=2, label='altern eps')
    plt.plot(interp1(np.array(cr_lst1) - np.max(sigma1)/100.), np.array(cr_lst1) - np.max(sigma1)/100., 'ro')
    
    if len(cr_lst1) > 2:
         
        random_field2 = RandomField(seed=False,
                                   lacor=1.,
                                   length=length,
                                   nx=700,
                                   nsim=1,
                                   loc=.0,
                                   shape=45.,
                                   scale=4.38,
                                   distr_type='Weibull'
                                   )
                  
        reinf2 = ContinuousFibers(r=3.5e-3,
                                  tau=RV('gamma', loc=tau_loc, scale=tau_scale, shape=tau_shape),
                                  V_f=0.015,
                                  E_f=E_f,
                                  xi=fibers_MC(m=xi_shape, sV0=xi_scale),
                                  label='carbon',
                                  n_int=500)
                  
        CB_model2 = CompositeCrackBridge(E_m=25e3,
                                     reinforcement_lst=[reinf2],
                                     )
                  
        scm2 = SCM(length=length,
                  nx=nx,
                  random_field=random_field2,
                  CB_model=CB_model2,
                  load_sigma_c_arr=np.linspace(0.01, 40., 100),
                  n_BC_CB=n_CB
                  )
                  
        scm_view2 = SCMView(model=scm2)
        scm_view2.model.evaluate()
        eps2, sigma2 = scm_view2.eps_sigma
        cr_lst2 = scm_view2.model.cracking_stress_lst
        interp2 = interp1d(sigma2, eps2)
        
        plt.plot(eps2, sigma2, color='blue', lw=2,
                                    label='Vf=1.5, crack spacing = ' + str(length/float(len(cr_lst2))))
        plt.plot(interp2(np.array(cr_lst2) - np.max(sigma2)/100.), np.array(cr_lst2) - np.max(sigma2)/100., 'bo')
    plt.xlabel('composite strain [-]')
    plt.ylabel('composite stress [MPa]')
    plt.xlim(0., np.max(eps1) * 1.15)
    plt.legend(loc='best')
    plt.savefig('validation_figsCS%1.f/w_max%.1f.png'%(CS,w_max))
    plt.close()



    if ld == True:
        plt.plot(eps1, sigma1, color='red', lw=2,
                label='Vf=1.0, crack spacing = ' + str(length/float(len(cr_lst1))))
        #plt.plot(eps11, sigma11, color='black', ls='dashed', lw=2, label='altern eps')
        plt.plot(interp1(np.array(cr_lst1) - np.max(sigma1)/100.), np.array(cr_lst1) - np.max(sigma1)/100., 'ro')
        plt.plot(eps2, sigma2, color='blue', lw=2,
                                    label='Vf=1.5, crack spacing = ' + str(length/float(len(cr_lst2))))
        plt.plot(interp2(np.array(cr_lst2) - np.max(sigma2)/100.), np.array(cr_lst2) - np.max(sigma2)/100., 'bo')
        plt.xlabel('composite strain [-]')
        plt.ylabel('composite stress [MPa]')
        plt.xlim(0., np.max(eps1) * 1.1)
        plt.legend(loc='best')

    if w_width == True:
        plt.figure()
        plt.plot(scm_view1.model.load_sigma_c_arr, scm_view1.w_mean)
        plt.plot(scm_view1.model.load_sigma_c_arr, scm_view1.w_max)
#         plt.plot(scm_view2.model.load_sigma_c_arr, scm_view2.w_mean)
#         plt.plot(scm_view2.model.load_sigma_c_arr, scm_view2.w_max)
         
    if w_density == True:
        plt.figure()
        plt.plot(scm_view1.model.load_sigma_c_arr , scm_view1.w_density)
#         plt.plot(scm_view2.model.load_sigma_c_arr, scm_view2.w_density)


def simplified():
    cb = CBClampedRandXi()
    spirrid = SPIRRID(q=cb, sampling_type='PGrid',
                  theta_vars=dict(tau=RV('gamma', loc=0.0055, scale=0.7, shape=0.2),
                                  E_f=200e3,
                                  V_f=0.01,
                                  r=0.00345,
                                  m=7.0,
                                  sV0=0.0042,
                                  lm=1000.),
                        n_int=200)
    def sigmac(w, lm):
        spirrid.eps_vars['w'] = np.array([w])
        spirrid.theta_vars['lm'] = lm
        sigma_c = spirrid.mu_q_arr / spirrid.theta_vars['r'] ** 2
        return sigma_c

    def maxsigma(lm):
        def minfunc(w):
            res = sigmac(w, lm)
            return -res * (w < 200.) + 1e-5 * w ** 2
        w_max = minimize_scalar(minfunc, bracket=(0.0001, 0.0002))
        return w_max.x, sigmac(w_max.x, lm) / spirrid.theta_vars['V_f']

    sigmaf = []
    w_lst = []
    lcs = 1./np.linspace(8.4, 300.0, 20)
    for lcsi in lcs:
        print lcsi
        wi, sigi = maxsigma(1./lcsi)
        sigmaf.append(sigi)
        w_lst.append(wi)
    plt.plot(lcs, sigmaf)
    plt.plot(1./13.7, 1201, 'ro')
    plt.plot(1./9.7, 1373, 'bo')
    plt.errorbar(1./13.7, 1201, 104.4)
    plt.errorbar(1./9.7, 1373, 36.4)
    plt.ylim(0)
    plt.figure()
    plt.plot(lcs, w_lst)
    plt.plot(1./13.7, 0.088, 'ro')
    plt.plot(1./9.7, 0.05, 'bo')
    plt.errorbar(1./13.7, 0.088, 0.003)
    plt.errorbar(1./9.7, 0.05, 0.005)
    plt.ylim(0)
    plt.show()

if __name__ == '__main__':
    #simplified()
    #valid()
    #CB()
    #TT()
    cs15 = [[1e-6,0.12263, 45.27],
            [0.,0.1099, 14.35],
            [1e-6,0.1039, 10.89],
            [0.,0.119, 29.07],
            [1.13e-4,0.1229, 150.],
            [1.46e-4,0.1213, 150.],
            [1.89e-4,0.1193, 150],
            [2.27e-4,0.117, 150.],]
    cs14 = [[0.,0.1217, 41.19],
            [0.,0.1203, 42.59],
            [0.,0.1203, 42.55],
            [76e-6,0.1155, 43.30],
            [22.5e-6,0.11, 52.61],
            [26.7e-6,0.1062, 40.74],
            [30.1e-6,0.105, 40.74],
            [33.3e-6,0.104, 41.61],]
    cs13 = [[0.0, 0.1184, 44.14],
            [0.,0.1180, 44.7],
            [0.0, 0.1185, 43.51],
            [2.94e-4, 0.0935, 17.88],
            [4.60e-4,0.0887, 17.43],
            [4.75e-4,0.0883, 17.51],
            [4.98e-4,0.0887, 17.51],
            [5.19e-4,0.0873, 17.71],]
    cs12 = [[0.,0.1154, 47.0],
            [0.,0.1159, 49.8],
            [146e-6,0.0888, 11.41],
            [612e-6,0.0756, 10.54],
            [763.e-6,0.0726, 10.47],
            [756.e-6,0.0727, 10.50],
            [761.e-6,0.0726, 10.49],
            [768.e-6,0.0726, 10.56],]
    cs11 = [[0.,0.1122, 49.3],
            [0.,0.1137, 48.74],
            [0., 0.1151, 47.65],
            [986e-6,0.0610, 7.28],
            [1107e-6,0.0592, 7.30],
            [1074e-6,0.0597, 7.30],
            [1060e-6,0.0599, 7.28],
            [1049e-6,0.0601, 7.30],]
    cs10 = [[0.,0.1068, 52.49],
            [1e-6, 0.0792, 6.25],
            [923e-6, 0.0549, 5.50],
            [1374e-6, 0.0493, 5.51],
            [1459e-6, 0.0483, 5.53],
            [1403e-6, 0.0490, 5.51],
            [1369e-6, 0.0493, 5.47],
            [1343e-6, 0.0496, 5.47],]
    cs9 = [[0., 0.1068, 55.15],
            [367e-6, 0.0571, 4.45],
            [1528e-6, 0.0418, 4.36],
            [1807e-6, 0.0394, 4.42],
            [1589e-6, 0.0495, 6.82],
            [1743e-6, 0.0399, 4.39],
            [1679e-6, 0.0404, 4.35],
            [1634e-6, 0.0407, 3.31],]
    CS=9.
    w_max = np.linspace(0.3,1.0,8)
    for i, param_set in enumerate(cs9):
        valid(CS, param_set,w_max[i])
