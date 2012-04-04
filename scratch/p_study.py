'''
Created on Mar 6, 2012

@author: rostar
'''

from quaducom.meso.ctt.scm_numerical.ctt2 import CTT
from quaducom.micro.resp_func.cb_emtrx_clamped_fiber import \
CBEMClampedFiberSP
import enthought.mayavi.mlab as m
import numpy as np
from stats.spirrid import make_ogrid as orthogonalize
from matplotlib import pyplot as plt
from stats.spirrid.spirrid import FunctionRandomization
from stats.misc.random_field.random_field_1D import RandomField
from stats.spirrid.rv import RV

def pmatrix_shape():
    rf = CBEMClampedFiberSP()
    rand = FunctionRandomization(q = rf,
         evars = dict(w = np.linspace(0, .5, 40),
                       x = np.linspace(-50., 50., 141),
                       Ll = np.linspace(0.01, 50., 12),
                       Lr = np.linspace(0.01, 50., 12)),
         tvars = dict(tau = 0.1, l = 0.0, A_r = 5.31e-4, E_r = 72e3, theta = 0.0,
                       xi = 1e20, phi = 1.0, E_m = 30e3, A_m = 50., Nf = 1700.),
                                n_int = 10)

    rand_field = RandomField( lacor = 30.0, xgrid = np.linspace(0, 3000, 1000),
                        nsim = 1, loc = 0.0, shape = 10.0, scale = 5.0,
                        non_negative_check = True, distribution = 'Weibull')

    shape_list = [1000., 10., 3.]
    for i, s in enumerate(shape_list):
        ctt = CTT(length = 3000., nx = 6000,
              random_field = rand_field,
              cb_randomization = rand, cb_type = 'mean',
              force_min = 0.1, force_max = 400, n_force = 500)
        ctt.random_field.shape = s
        ctt.evaluate()
        eps, sig = ctt.eps_sigma
        plt.plot(eps, sig, lw = 2, color = 'black', label = 'line = %.1f' %i)
    plt.title('var shape param')
    plt.legend(loc = 'best')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylim(0,8)
    plt.show()
    
def pmatrix_lacor():
    rf = CBEMClampedFiberSP()
    rand = FunctionRandomization(q = rf,
         evars = dict(w = np.linspace(0, .5, 40),
                       x = np.linspace(-40., 40., 141),
                       Ll = np.linspace(0.01, 50., 14),
                       Lr = np.linspace(0.01, 50., 14)),
         tvars = dict(tau = 0.1, l = 0.0, A_r = 5.31e-4, E_r = 72e3, theta = 0.0,
                       xi = 1e20, phi = 1.0, E_m = 30e3, A_m = 50., Nf = 1700.),
                                n_int = 10)

    rand_field = RandomField( lacor = 5.0, xgrid = np.linspace(0, 3000, 1000),
                        nsim = 1, loc = 0.0, shape = 7.0, scale = 5.0,
                        non_negative_check = True, distribution = 'Weibull')

    def subplot(x, stress, strength):
        plt.figure()
        plt.plot(x, strength[-1,:], lw = 2, color = 'black')
        plt.plot(x, stress[-1,:], lw = 2, color = 'black')
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.ylim(0,8)
        plt.xlim(0,500)
        plt.fill_between(x, stress[-1,:], color = 'lightgrey')
        plt.show()
    
    strength_list = []
    stress_list = []    
    acor_list = [0.1, 5., 20.]
    for i, a in enumerate(acor_list):
        ctt = CTT(length = 3000., nx = 6000,
              random_field = rand_field,
              cb_randomization = rand, cb_type = 'mean',
              force_min = 0.1, force_max = 400, n_force = 500)
        ctt.random_field.lacor = a
        ctt.evaluate()
        eps, sig = ctt.eps_sigma
        plt.plot(eps, sig, lw = 2, color = 'black', label = 'line = %.1f' %i)
        strength_list.append(ctt.matrix_strength)
        stress_list.append(ctt.sigma_m)
        
    plt.title('var lacor')
    plt.legend(loc = 'best')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylim(0,8)
    plt.show()
    for i, strength in enumerate(strength_list):
        subplot(ctt.x_arr, stress_list[i], strength)
  
def ld():

    version = 'test'
    if version == 'test':
        x = np.linspace(-40., 40., 141)
        w = np.linspace(0, 1.0, 40)
        Ll = np.linspace(0.01, 100., 10)
        Lr = np.linspace(0.01, 100., 10)
        n_force = 500
        nx = 6000
        length = 3000 
    else:
        x = np.linspace(-30., 30., 101)
        w = np.linspace(0, 0.25, 50)
        Ll = np.linspace(0.01, 100., 20)
        Lr = np.linspace(0.01, 100., 20)
        n_force = 500
        nx = 6000
        length = 1200
    

    # matrix
    scale = 5.
    shape = 10.
    loc = 0.0
    # filaments
    tau = .05
    Af = 5.31e-4
    Ef = 72e3
    Am = 50.
    Em = 30e3
    l = 0.#RV('uniform', 5.0, 20.0)
    theta = 0.0
    phi = 1.
    Nf = 1700.
    xi = 1e20#RV( 'weibull_min', scale = 0.017, shape = 5 )
    


    rf = CBEMClampedFiberSP()
    rand = FunctionRandomization(q = rf,
         evars = dict(w = w,
                       x = x,
                       Ll = Ll,
                       Lr = Lr,
                        ),
         tvars = dict(tau = tau,
                       l = l,
                       A_r = Af,
                       E_r = Ef,
                       theta = theta,
                       xi = xi,
                       phi = phi,
                       E_m = Em,
                       A_m = Am,
                       Nf = Nf,
                        ),
         n_int = 10)

    tau = [.05]
           #RV('weibull_min', shape = 1.1, scale = 2., loc = .01),
           #RV('weibull_min', shape = 1.1, scale = 10., loc = .01),
           #]
    acor = [0.1, 6., 30.]
    xlist = [np.linspace(-50., 50., 141)]
    L = [np.linspace(0.01, 50., 12)]
    for i, t in enumerate(tau):
        for lc in acor:
            rand_field = RandomField(
                                lacor = lc,
                                xgrid = np.linspace(0, length, 1400),
                                nsim = 1,
                                loc = loc,
                                shape = shape,
                                scale = scale,
                                non_negative_check = True,
                                distribution = 'Weibull'
                           )
            ctt = CTT(length = length,
              nx = nx,
              random_field = rand_field,
              cb_randomization = rand,
              cb_type = 'mean',
              force_min = 0.1, 
              force_max = 400,
              n_force = n_force
              )
            ctt.cb_randomization.tvars['tau'] = tau[i]
            ctt.cb_randomization.evars['x'] = xlist[i]
            ctt.cb_randomization.evars['Ll'] = L[i]
            ctt.cb_randomization.evars['Lr'] = L[i]
            ctt.evaluate()
            eps, sig = ctt.eps_sigma
            plt.plot(eps, sig, lw = 2, color = 'black', label = 'line = %.1f' %i)
        
#            Er = ctt.cb_randomization.tvars['E_r']
#            Ar = ctt.cb_randomization.tvars['A_r'] * Nf
#            Am = ctt.cb_randomization.tvars['A_m']
#            Ac = Ar+Am
            #plt.plot(eps, eps * Er * Ar / Ac, label = 'ref')
            #plt.plot(eps, eps * Er * Ar / (Ac) + 5. * 1.337 / 4., color = 'black', lw = 2., ls = 'dashed', label = 'comp')
            
#            e_arr = orthogonalize([ctt.applied_force, ctt.x_arr[::10]])
#            n_e_arr = [ e / np.max(np.fabs(e)) for e in e_arr ]
#        
#            scalar1 = ctt.matrix_strength[:,::5]
#            scalar2 = ctt.sigma_m[:,::5]
#        
#            n_scalar1 = scalar1 / np.max(np.fabs(scalar1))
#            n_scalar2 = scalar2 / np.max(np.fabs(scalar1))
#        
#            m.surf(n_e_arr[0], n_e_arr[1], n_scalar1)
#            m.surf(n_e_arr[0], n_e_arr[1], n_scalar2)
    
        

    plt.legend(loc = 'best')
    plt.show()
#    m.show()

def crack_distr():
    from scipy.optimize import fmin
    from scipy.stats import weibull_min
    rf = CBEMClampedFiberSP()
    rand = FunctionRandomization(q = rf,
         evars = dict(w = np.linspace(0, .5, 60),
                       x = np.linspace(-40., 40., 141),
                       Ll = np.linspace(0.01, 50., 14),
                       Lr = np.linspace(0.01, 50., 14)),
         tvars = dict(tau = 0.1, l = 0.0, A_r = 5.31e-4, E_r = 72e3, theta = 0.0,
                       xi = 1e20, phi = 1.0, E_m = 30e3, A_m = 50., Nf = 1700.),
                                n_int = 10)

    rand_field = RandomField( lacor = 5.0, xgrid = np.linspace(0, 3000, 1000),
                        nsim = 1, loc = 0.0, shape = 7.0, scale = 5.0,
                        non_negative_check = True, distribution = 'Weibull')

    ctt = CTT(length = 3000., nx = 6000,
    random_field = rand_field,
    cb_randomization = rand, cb_type = 'mean',
    force_min = 0.1, force_max = 400, n_force = 500)
    ctt.random_field.lacor = 20.
    ctt.evaluate()
    
    def cracking():
        fig = plt.figure()
        ax1 = fig.add_subplot(111) 
        ax2 = ax1.twinx()            
        
        ax1.plot(ctt.P_list, range(len(ctt.P_list)), color= 'black', lw = 2,
                 label = 'No. of cracks')
        ax2.plot(ctt.P_list, ctt.length/np.arange(2,len(ctt.P_list)+2),
                 color = 'black', lw = 2, ls = 'dashed', label = 'mean CS')
        
        ax1.set_title('emerging cracks')
        ax1.set_ylabel('No. of cracks')
        ax1.set_xlabel('applied load [N]')
        ax2.set_ylabel('mean crack spacing [mm]')
        ax1.legend(loc = 'lower left')
        ax2.legend(loc = 'lower right')
    
    w_list = []
    def plot():
        for i, crack in enumerate(ctt.cb_list):
            w_list.append(crack.get_crack_width()[-1])
        plt.hist(w_list, bins = 40, normed = True, color = 'lightgrey', label = 'numerical solution')
        plt.title('crack width distribution')
        plt.ylabel('PMF')
        plt.xlabel('crack width [mm]')
        plt.xlim(0)

    plot()
    def maxlike( v, *args ):
        data = args
        shape, scale = v
        r = sum( np.log( weibull_min.pdf( data, shape, scale = scale, loc = 0. ) ) )
        return - r

    # optimizing method for finding the parameters
    # of the fitted Weibull distribution
    def maximum_likelihood( data ):
        params = fmin( maxlike, [2., 5.], args = ( data ) )
        moments = weibull_min( params[0], scale = params[1], loc = 0. ).stats( 'mv' )
        print ' #### max likelihood #### '
        print 'mean = ', moments[0]
        print 'var = ', moments[1]
        print 'Weibull shape = ', params[0]
        print 'Weibull scale = ', params[1]
        # plot the Weibull fit according to maximum likelihood 
        e = np.linspace( 0., 0.3 * ( np.max( data ) - np.min( data ) ) + np.max( data ), 100 )
        plt.plot( e, weibull_min.pdf( e, params[0], scale = params[1], loc = 0. ),
             color = 'black', linewidth = 3, label = 'Weibull max. likelihood fit' )
    
    maximum_likelihood(np.array(w_list))
    #cracking()    
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(loc = 'best')
    plt.show()

#pmatrix_lacor()
crack_distr()