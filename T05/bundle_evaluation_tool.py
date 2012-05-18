'''
Created on Jun 17, 2010

BundleEvaluationTool defines the statistical and length dependent properties
and dependencies between filament and yarn. Given yarn or filament tests at
various lengths, the tool can evaluate SE curves for both; the strength distribution
for the filaments and l-d curves for yarns.

@author: Rostislav Rypl
'''

from enthought.traits.api import HasTraits, Array, Float, Property, cached_property
from scipy.special import gamma
from scipy.optimize import leastsq
from numpy import linspace, array, max, sqrt, min
from matplotlib import pyplot as plt
from scipy.stats import weibull_min
from math import e, pi
from T05.adapter_tests.carbon.carbon400tex_twist_length_adapter_vs_UB import lengths as A_lengths, A_strength, B_strength


class BundleEvaluationTool( HasTraits ):

    # tested parameters    
    lengths = Array
    strengths = Array
    Ef = Float
    ref_length = Float(100.)
    
    # define function for calculating the Weibull scaling
    def powerlaw( self, length, scale, shape ):
        return scale * gamma( 1. + 1. / shape ) * ( self.ref_length / length ) ** ( 1. / shape )
    
    def error_func( self, p ):
        return self.strengths - self.powerlaw( self.lengths, p[0], p[1] )
    
    weibull_params = Property(depends_on = 'strengths, lengths' )
    @cached_property
    def _get_weibull_params( self ):
        iparams = [1000., 3.0]
        return leastsq( self.error_func, iparams, full_output = 1 )
    
    # stress-strain curve of a perfect clamped bundle consisting of
    # fibers with two parameters Weibull strength distribution given
    # by the evaluated parameters
    def values( self, eps, bundle_length ):
        out = self.weibull_params
        scale0 = out[0][0]
        shape = out[0][1]
        scale = ( scale0 * ( self.ref_length / bundle_length ) ** ( 1. / shape ) /self.E)  
        return eps * self.E * ( 1. - weibull_min.cdf( eps, shape, scale = scale ) )

    def bundle_reduction(self):
        out = self.weibull_params
        shape = out[0][1]
        return 1./(shape**(1./shape)*e**(1./shape)*gamma(1.+1./shape))


if __name__ == '__main__':

#################################
######### PLOTTING ##############
#################################
    
    # predicting the SE curve for a bundle from filament data
    filament_tests = False
    # predicting the SE curve for a filament from bundle data
    bundle_tests = True
    if filament_tests:
        pe = BundleEvaluationTool( lengths = array( [25., 50.] ), 
                                   strengths = array( [3557., 3243.] ),
                                   E = 130e3
                                   )
        br = pe.bundle_reduction()
    elif bundle_tests:
        pe = BundleEvaluationTool( lengths = A_lengths[1:], 
                                   strengths = A_strength[1:,0],
                                   E = 130e3
                                   )
        br = 1./pe.bundle_reduction()
        
    def SE():
        plt.figure()
        out = pe.weibull_params
        params = out[0]
        covar = out[1]
        scale = params[0]
        shape = params[1]
        shapeErr = sqrt( covar[0][0] )
        scaleErr = sqrt( covar[1][1] )
    
        x = linspace( .2, 550., 1000 )
        y = pe.powerlaw( x, scale, shape )
        
        plt.subplot( 2, 1, 1 )

        if filament_tests:
            plt.plot( x, y, color = 'red', label = 'filament'  )
            plt.plot( x, y * br, color = 'blue', label = 'bundle-theory' )
            plt.plot( pe.lengths, pe.strengths, 'ro', label = 'filament measurements' )
        
            # additional data
            plt.plot( A_lengths, A_strength[:,0], 'bo', label = 'bundle measurements' )
        elif bundle_tests:
            plt.plot( x, y, color = 'blue', label = 'bundle'  )
            plt.plot( x, y * br, color = 'red', label = 'filament-theory' )
            plt.plot( pe.lengths, pe.strengths, 'bo', label = 'bundle measurements' )
        
            # additional data
            plt.plot( array([25.,50.]), array([3557., 3243.]), 'ro', label = 'filament measurements' )
        shape_ytick = min( y ) + 0.9 * ( max( y ) - min( y ) )
        scale_ytick = min( y ) + 0.8 * ( max( y ) - min( y ) )
        xtick = min( x ) + 0.5 * max( x ) - min( x )
        plt.text( xtick, shape_ytick, 'shape: %5.2f +/- %5.2f' % ( shape, shapeErr ) )
        plt.text( xtick, scale_ytick, 'scale for $l_0$ = %5.f' %pe.ref_length
                  + 'mm: %5.f +/- %5.2f' % ( scale, scaleErr ) )
        plt.title( 'best fit Weibull scaling' )
        plt.xlabel( 'length' )
        plt.ylabel( 'strength' )
        #plt.legend(loc = 'best')
        #plt.xlim( x[0] - ( x[-1] - x[0] ) * 0.05, x[-1] + ( x[-1] - x[0] ) * 0.05 )
        #plt.ylim( y[-1] + ( y[-1] - y[0] ) * 0.05, y[0] - ( y[-1] - y[0] ) * 0.05 )
        
        plt.subplot( 2, 1, 2 )
        if filament_tests:
            plt.loglog( x, y, color = 'red', label = 'filament' )
            plt.loglog( x, y*br, color = 'blue', label = 'bundle-theory' )
            plt.loglog( pe.lengths, pe.strengths, 'ro', label = 'fil tests' )
            plt.loglog( A_lengths, A_strength[:,0], 'bo', label = 'bundle tests' )
            plt.loglog( A_lengths, B_strength[:,0], 'ko', label = 'UB' )
        elif bundle_tests:
            plt.loglog( x, y, color = 'blue', label = 'bundle' )
            plt.loglog( x, y*br, color = 'red', label = 'filament-theory' )
            plt.loglog( pe.lengths, pe.strengths, 'bo', label = 'bundle tests' )
            plt.loglog( array([25.,50.]), array([3557., 3243.]), 'ro', label = 'fil tests' )
            plt.loglog( A_lengths, B_strength[:,0], 'ko', label = 'UB' )
        plt.xlabel( 'length' )
        plt.ylabel( 'strength' )
        #plt.xlim( x[0] - ( x[-1] - x[0] ) * 0.01, x[-1] + ( x[-1] - x[0] ) * 0.05 )
        #plt.ylim( y[-1]*br + ( y[-1]*br - y[0]*br ) * 0.05, y[0] - ( y[-1] - y[0] ) * 0.05 )
        plt.legend(loc = 'best')
        
    def bundle_ld():
        bundle_length = 35.
        plt.figure()
        eps = linspace( 0, 0.04, 100 )
        sigma = pe.values( eps, bundle_length )
        plt.plot( eps, sigma, linewidth = 2 )
        plt.title( 'stress-strain of an asymptotic bundle of length %.1f' %bundle_length )
        plt.xlabel( 'strain [-]' )
        plt.ylabel( 'stress [MPa]' )

    
    #bundle_ld() 
    SE()
    plt.show()
    
    
        
