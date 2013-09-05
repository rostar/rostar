'''
Created on 03.07.2013

@author: acki
'''
import numpy as np
from scipy.interpolate import interp1d
from quaducom.micro.resp_func.CB_rigid_mtrx import CBResidual
import numpy as np
from spirrid.spirrid import SPIRRID
from spirrid.rv import RV
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import leastsq
import os
FILE_DIR = os.path.dirname( __file__ )
def TensileTest( Dataplot, optimization, ModelPlot, paras, dataname_data, dataname_model, sto, pullout_test, piei, Ratio, E_f, r, m, sV0, Pf, n_int, weighting1_paras, weighting3_paras , tau_first_guessing, sto_guessings ):
    l_of_all_data = ['DATA/PO01_RYP.ASC', 'DATA/PO02_RYP.ASC', 'DATA/PO03_RYP.ASC']
    Data_to_plot = np.array( pullout_test )
    def flf( po_no ):
            fl_arr = np.array( [10.5, 0, 4] )
            return fl_arr[po_no - 1]
    fitting_length = flf( Data_to_plot )
    ##################
    #WEIGHTING
    def weighting( weights_list ):
            basis = np.zeros( 300 )
            for element in weights_list:
                x, y, z = element
                basis += ( np.linspace( 0, flf( number ), 300 ) >= x ) * ( np.linspace( 0, flf( number ), 300 ) <= y ) * z
            return basis
    
    if optimization == True:
        number = Data_to_plot[0]
        weighting1 = weighting( weighting1_paras )
        weighting3 = weighting( weighting3_paras )
        w_list = [ weighting1, weighting3 ]
        if Data_to_plot[0] == 3:
            weighting_arr = w_list[1]
        else:
            weighting_arr = w_list[0]
      
    
    # Data into arrays
    w_x = np.linspace( 0, 18, 300 )
    plotlist = []
    for i in Data_to_plot:
        plotlist.append( l_of_all_data[i - 1] )
    mean = []
    F_w_list = []
    for i, dataname in enumerate( plotlist ):
        f = open( dataname, 'r' )
        time = []
        F = []
        w1 = []
        w2 = []
        for line in f:
                            #1###
                            index1 = line.index( ';' ) 
                            time.append( float( line[:index1] ) )
                            line = line[index1 + 1:]
                            #2###
                            index2 = line.index( ';' )
                            F.append( np.float( line[:index2] ) )
                            line = line[index2 + 1:]
                            #3###
                            index3 = line.index( ';' )
                            w1.append( np.float( line[:index3] ) )
                            line = line[index3 + 1:]
                            #4###
                            w2.append( np.float( line ) )
                            
        F = np.array( F )
        w1 = np.array( w1 )
        w2 = np.array( w2 )
        F = F - F[0] + 0.035
        if Data_to_plot[0] == 3:
            F = F - 0.02
        w1 = np.abs( w1 ) - np.abs( w1[0] )
        w2 = np.abs( w2 ) - np.abs( w2[0] )
        w_mid = ( w1 + w2 ) / 2
        if Data_to_plot[0] == 1:
            w_mid = w2
        string = '%d'.format( i )
        # plt.plot( w2, F )
        f.close()
        F_w_item = interp1d( w_mid, F, bounds_error = False, fill_value = -1. )
        F_w_list.append( F_w_item )
        if Dataplot == True:
            plt.plot( w_mid, F, linewidth = '2' )
            np.savetxt( os.path.join( FILE_DIR, dataname_data ), [w_mid, F], delimiter = ',' )
    
    
    
    
    
    
    def residuum( iter_arr, i, w_fitted , weight ):
        if sto == True:
            if np.any( iter_arr[0] < 0 ) or iter_arr[1] < 0 or iter_arr[0] > iter_arr[1]:
                print 'warning: Invalid iter value', iter_arr
                return np.sum( np.repeat( np.infty, 300 ) )
            else: 
                print 'normal', iter_arr
                test = np.sum( ( ( F_w_list[i]( w_fitted ) - Model( iter_arr, w_fitted ) ) * weight ) ** 2 ) 
                return   test
        else:
            print 'normal', iter_arr
            test = np.sum( ( ( F_w_list[i]( w_fitted ) - Model( iter_arr, w_fitted ) ) * weight ) ** 2 ) 
            return   test
            
    ####################################MODEL##########################################
    def Model( itertuple, w ):
        if sto == False:
            tauI = itertuple[0]
        if sto == True:
            a_up, b_up = itertuple

        def CB_composite_stress( w, tau, E_f, V_f, r, m, sV0, Pf, n_int ):
            cb = CBResidual()
            spirrid = SPIRRID( q = cb,
                        sampling_type = 'PGrid',
                        eps_vars = dict( w = w ),
                        theta_vars = dict( tau = tau, E_f = E_f, V_f = V_f, r = r,
                                   m = m, sV0 = sV0, Pf = Pf ),
                        n_int = n_int )
            if isinstance( r, RV ):
                r_arr = np.linspace( r.ppf( 0.001 ), r.ppf( 0.999 ), 300 )
                Er = np.trapz( r_arr ** 2 * r.pdf( r_arr ), r_arr )
            else:
                Er = r ** 2
            sigma_c = spirrid.mu_q_arr / Er
            return sigma_c * 0.445 / 1000
        if sto == True:
        #distr
            a_low = 0.
            ratio = Ratio
            b_low = a_up
            np.savetxt( os.path.join( FILE_DIR, 'distr_par.txt' ), [[a_low, a_up, b_low, b_up, ratio]], delimiter = ',' )
            tau = RV( 'piecewise_uniform', shape = 0, scale = 1 )
        else:
            tau = tauI
        V_f = 1.
        #general
        res = CB_composite_stress( w, tau, E_f, V_f, r, m, sV0, Pf, n_int )
        if piei == True:
            plt.plot( w, res )
            plt.plot( w_mid, F, linewidth = '2' )
            plt.show()
        return res
    
    if ModelPlot == True:
        w = np.linspace( 0, fitting_length[i], 300 )
        for p in paras:
            sigma = Model( p, w )
            plt.plot( w, sigma )
            np.savetxt( os.path.join( FILE_DIR, dataname_model ), [w, sigma], delimiter = ',' )
    
    
    if optimization == True:
        if sto == True:
            #Sto
            a_up_first_guessing, b_up_first_guessing = sto_guessings
            for i, F_w in enumerate( F_w_list ):
                w_fitted = np.linspace( 0, fitting_length[i], 300 )
                weight = np.array( weighting_arr )
                res = minimize( residuum, args = [i, w_fitted, weight], x0 = [a_up_first_guessing, b_up_first_guessing], method = 'Nelder-Mead' )
                var1, var2 = res.x
                sigma_c = Model( [ var1, var2 ], w_fitted )
                plt.plot( w_fitted, sigma_c )
        else:
            #Det
            for i, F_w in enumerate( F_w_list ):
                w_fitted = np.linspace( 0, fitting_length[i], 300 )
                weight = np.array( weighting_arr )
                res = minimize( residuum, args = [i, w_fitted, weight], x0 = [ tau_first_guessing], method = 'Nelder-Mead' )
                var1 = res.x
                sigma_c = Model( [ var1[0] ], w_fitted )
                plt.plot( w_fitted, sigma_c )
                
                
                
#################Plots##################
pullout_test = [1] # 1 or 3
Dataplot = True
optimization = False
############################
#Weightings
#zwischen x und y mit z wichten. [[x,y,z]]
weighting1_paras = [[0.5, 4, 1], [4, 8, .2]]
weighting3_paras = [[0.2, 3.5, 1], [0, 0.2, 0.1]]
############################
det_tau_first_guessing = 0.3
sto_guessings = [0.02, 20.]
ModelPlot = True
#Parameterliste anpassen bei det oder Sto
#STO 
#paras = [[0.01257187, .5], [0.01257187, 0.3]]
#Det
paras = [[0.05, 2.]]

dataname_data = 'Data_sto1'
dataname_model = 'model_sto1'
plot_in_every_iter = False

#general
E_f = 180e3
r = 3.45 * 1e-3
m = 3.5
sV0 = 2.e-3
Pf = RV( 'uniform', loc = 0., scale = 1.0 )
n_int = 40
ratio = .8



if len( paras[0] ) > 1:
    sto = True
else:
    sto = False
TensileTest( Dataplot, optimization, ModelPlot, paras, dataname_data,
             dataname_model, sto, pullout_test, plot_in_every_iter,
             ratio, E_f, r, m, sV0, Pf, n_int, weighting1_paras,
             weighting3_paras, det_tau_first_guessing, sto_guessings )
plt.xlim( 0, 8 )
plt.ylim( 0, .5 )
plt.show()
