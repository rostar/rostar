'''
Created on Jun 17, 2010

@author: rostislav
'''

from enthought.traits.api import HasTraits, Array, Float
from scipy.special import gamma
from scipy.optimize import leastsq
from numpy import linspace, array, max, sqrt, min
from matplotlib import pyplot as plt
from scipy.stats import weibull_min
from math import e, pi
from T05.adapter_tests.carbon.carbon400tex_twist_length_adapter_vs_UB import lengths, A_strength


class ParamsExtraction( HasTraits ):

    # tested parameters    
    tested_l = Array
    tested_fu = Array
    E = Float
    l_bundle = Float
    
    # define function for calculating the Weibull scaling
    def powerlaw( self, length, scale, shape ):
        return scale * gamma( 1. + 1. / shape ) * ( self.tested_l[0] / length ) ** ( 1. / shape )
    
    def error_func( self, p ):
        return self.tested_fu - self.powerlaw( self.tested_l, p[0], p[1] )
    
    def weibull_params( self ):
        iparams = [self.tested_fu[0], 3.0]
        return leastsq( self.error_func, iparams, full_output = 1 )
    
    # stress-strain curve of a perfect clamped bundle consisting of
    # fibers with two parameters Weibull strength distribution given
    # by the evaluated parameters
    def values( self, eps ):
        out = self.weibull_params()
        scale0 = out[0][0]
        shape = out[0][1]
        scale = ( scale0 * ( self.tested_l[0] / self.l_bundle ) ** ( 1. / shape ) /self.E)  
        return eps * self.E * ( 1. - weibull_min.cdf( eps, shape, scale = scale ) )

    def bundle_reduction(self):
        out = self.weibull_params()
        shape = out[0][1]
        return (shape**(1./shape)*e**(1./shape)*gamma(1.+1./shape))**(-1)


if __name__ == '__main__':

#################################
######### PLOTTING ##############
#################################
    
    pe = ParamsExtraction( tested_l = array( [2.5, 5.0] ), 
                          tested_fu = array( [0.17, 0.155] )/((7.8e-3/2.)**2*pi),
                          E = 72e3,
                          l_bundle = 35.0,
                          d = 7.8e-3 )

    def weibull_params_fit():
        br = pe.bundle_reduction()
        plt.figure()
        out = pe.weibull_params()
        params = out[0]
        covar = out[1]
        scale = params[0]
        shape = params[1]
        shapeErr = sqrt( covar[0][0] )
        scaleErr = sqrt( covar[1][1] )
    
        x = linspace( .1, 550., 1000 )
        y = pe.powerlaw( x, scale, shape )
        
        plt.subplot( 2, 1, 1 )

        plt.plot( x, y, color = 'red', label = 'filament'  )
        plt.plot( x, y * br, color = 'blue', label = 'bundle-theory' )
        plt.plot( pe.tested_l, pe.tested_fu, 'ro', label = 'filament measurements' )
        plt.plot( lengths, A_strength[:,0], 'bo', label = 'bundle measurements' )
        shape_ytick = min( y ) + 0.9 * ( max( y ) - min( y ) )
        scale_ytick = min( y ) + 0.8 * ( max( y ) - min( y ) )
        xtick = min( x ) + 0.5 * max( x ) - min( x )
        plt.text( xtick, shape_ytick, 'shape: %5.2f +/- %5.2f' % ( shape, shapeErr ) )
        plt.text( xtick, scale_ytick, 'scale for $l_0$ = ' + str( pe.tested_l[0] ) + 'mm: %5.2f +/- %5.2f' % ( scale, scaleErr ) )
        plt.title( 'best fit Weibull scaling' )
        plt.xlabel( 'length' )
        plt.ylabel( 'strength' )
        #plt.legend(loc = 'best')
        #plt.xlim( x[0] - ( x[-1] - x[0] ) * 0.05, x[-1] + ( x[-1] - x[0] ) * 0.05 )
        #plt.ylim( y[-1] + ( y[-1] - y[0] ) * 0.05, y[0] - ( y[-1] - y[0] ) * 0.05 )
        
        plt.subplot( 2, 1, 2 )
        plt.loglog( x, y, color = 'red', label = 'filament' )
        plt.loglog( x, y*br, color = 'blue', label = 'bundle-theory' )
        plt.loglog( pe.tested_l, pe.tested_fu, 'ro', label = 'filament measurements' )
        plt.loglog( lengths, A_strength[:,0], 'bo', label = 'bundle measurements' )
        plt.xlabel( 'log length' )
        plt.ylabel( 'log strength' )
        plt.xlim( x[0] - ( x[-1] - x[0] ) * 0.01, x[-1] + ( x[-1] - x[0] ) * 0.05 )
        plt.ylim( y[-1]*br + ( y[-1]*br - y[0]*br ) * 0.05, y[0] - ( y[-1] - y[0] ) * 0.05 )
        plt.legend(loc = 'best')
        
    def bundle_ld():
        plt.figure()
        eps = linspace( 0, 0.06, 100 )
        sigma = pe.values( eps )
        plt.plot( eps, sigma, linewidth = 2 )
        plt.title( 'stress-strain of an asymptotic bundle of length %.1f' %pe.l_bundle )
        plt.xlabel( 'strain [-]' )
        plt.ylabel( 'stress [MPa]' )

    
    bundle_ld()
    weibull_params_fit()
    plt.show()
    
    
        
