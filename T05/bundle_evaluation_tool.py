'''
Created on Jun 17, 2010

BundleEvaluationTool defines the statistical and length dependent properties
and dependencies between filament and yarn. Given yarn or filament tests at
various lengths, the tool can evaluate SE curves for both; the strength distribution
for the filaments and l-d curves for yarns.

@author: Rostislav Rypl
'''

from etsproxy.traits.api import HasTraits, Array, Float, Property, cached_property
from scipy.special import gamma
from scipy.optimize import leastsq
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import weibull_min
from math import e, pi
from T05.adapter_tests.carbon.carbon400tex_twist_length_adapter_vs_UB import lengths as A_lengths, A_strength, B_strength
from T05.adapter_tests.AR_glass.resin_vs_adapter import l, fh, fa
import matplotlib

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
        scale = ( scale0 * ( self.ref_length / bundle_length ) ** ( 1. / shape ) /self.Ef)  
        return eps * self.Ef * ( 1. - weibull_min.cdf( eps, shape, scale = scale ) )

    def bundle_reduction(self):
        out = self.weibull_params
        shape = out[0][1]
        return 1./(shape**(1./shape)*e**(1./shape)*gamma(1.+1./shape))


if __name__ == '__main__':

#################################
######### PLOTTING ##############
#################################
    
    # predicting the SE curve for a bundle from filament data
    filament_tests = True
    # carbon tested at Textechno/ITA
    fil_lengths_carbon = np.array([25.,50.])
    fil_strengths_carbon = np.array([3557., 3243.])
    Ecarbon = 200e3
    # AR-glass tested at Textechno/ITA
    fil_lengths_glass = np.array([20., 100.])
    fil_strengths_glass = np.array([2157., 1790.])
    Eglass = 70e3
    #yarn tests on AR-glass
    fh = np.array(fh) / 0.445
    fa = np.array(fa) / 0.445
    lengths_glass = l
    strengths_glass_adapter = fa
    strengths_glass_resin = fh
    
    # filament test carbon Dresden (50 and 100 mm) and ITA (25 and 49.99 mm)
    fil_lengths_carbon = np.array([25., 49.99, 50., 100.])
    fil_strengths_carbon = np.array([3557., 3243., 3295., 2969.])
    Ecarbon = 200e3

    # filament tests Basalt Dresden - tested in Textechno
    fil_lengths_basalt = np.array([50., 100.])
    fil_strengths_basalt = np.array([2928., 2466.])

    # filament tests AR-Glas Dresden - tested in Textechno
    fil_lengths_glas = np.array([50., 100.])
    fil_strengths_glas = np.array([2125., 1792.])

    # predicting the SE curve for a filament from bundle data
    bundle_tests = False
    if filament_tests:
        pe = BundleEvaluationTool( lengths = np.array([25., 50., 100.]), 
                                   strengths = np.array([3557., 3295., 2969.]),
                                   Ef = Ecarbon
                                   )
        br = pe.bundle_reduction()
    elif bundle_tests:
        pe = BundleEvaluationTool( lengths = lengths_glass[3:], 
                                   strengths = strengths_glass_adapter[3:],
                                   Ef = Eglass
                                   )
        br = 1./pe.bundle_reduction()

    def SE():
        plt.figure()
        out = pe.weibull_params
        params = out[0]
        covar = out[1]
        scale = params[0]
        shape = params[1]
        print scale, shape
        shapeErr = np.sqrt( covar[0][0] )
        scaleErr = np.sqrt( covar[1][1] )
    
        x = np.linspace( .2, 550., 1000 )
        y = pe.powerlaw( x, scale, shape )
        
#        plt.subplot( 1, 2, 1 )
#
#        if filament_tests:
#            plt.plot( x, y, color = 'black', lw = 1, label = 'filament scaling'  )
#            plt.plot( x, y * br, color = 'black', lw = 2, ls = 'dashed', label = 'bundle model' )
#            plt.plot( pe.lengths, pe.strengths, 'k^', label = 'filament tests' )
#            plt.xticks(fontsize = 16)
#            plt.yticks(fontsize = 16)
#        
#            # additional data
#            # carbon
#            #plt.plot( A_lengths, A_strength[:,0], 'ko', label = 'Statimat 4U adapter' )
#            #plt.plot( A_lengths, B_strength[:,0], 's', color = 'grey', label = 'standard method' )
#            # glass
#            plt.plot( lengths_glass, strengths_glass_adapter, 'ko', label = 'Statimat 4U adapter' )
#            plt.plot( lengths_glass, strengths_glass_resin, 's', color = 'grey', label = 'resin porters' )
#            plt.grid(True,which="both",ls="-", color = 'grey')
#            plt.xlim(10)
#        elif bundle_tests:
#            plt.plot( x, y, color = 'blue', label = 'bundle'  )
#            plt.plot( x, y * br, color = 'red', label = 'filament-theory' )
#            plt.plot( pe.lengths, pe.strengths, 'bo', label = 'bundle measurements' )
#            plt.xticks(fontsize = 16)
#            plt.yticks(fontsize = 16)
#        
#            # additional data
#            plt.plot( np.min( y ) + 0.9 * ( np.max( y ) - np.min( y ) ))
#        scale_ytick = np.min( y ) + 0.8 * ( np.max( y ) - np.min( y ) )
#        xtick = np.min( x ) + 0.5 * np.max( x ) - np.min( x )
#        plt.ylim(500,3000)
#        #plt.text( xtick, shape_ytick, 'shape: %5.2f +/- %5.2f' % ( shape, shapeErr ) )
#        #plt.text( xtick, scale_ytick, 'scale for $l_0$ = %5.f' %pe.ref_length
#        #          + 'mm: %5.f +/- %5.2f' % ( scale, scaleErr ) )
#        plt.title( 'Weibull scaling' , fontsize = 16)
#        plt.xlabel( 'length [mm]' , fontsize = 16)
#        plt.ylabel( 'strength [MPa]' , fontsize = 16)
#        #plt.legend(loc = 'best')
#        #plt.xlim( x[0] - ( x[-1] - x[0] ) * 0.05, x[-1] + ( x[-1] - x[0] ) * 0.05 )
#        #plt.ylim( y[-1] + ( y[-1] - y[0] ) * 0.05, y[0] - ( y[-1] - y[0] ) * 0.05 )
#        plt.legend(loc = 'best')
        
        logax = plt.subplot( 1, 1, 1 )
        if filament_tests:
            logax.loglog( x, y, color = 'black', label = 'extrapolated filament strength', subsy = [2, 3, 4, 5, 6, 7, 8, 9] )
            logax.loglog( x, y*br, color = 'black', lw = 2, ls = 'dashed', label = 'theoretical bundle strength' )
#            logax.loglog( pe.lengths, pe.strengths, 'k^', label = 'Filamenttests AR-Glas - Dresden' )
            logax.loglog( pe.lengths[:2], pe.strengths[:2], 'k^', label = 'filament tests' )
#            logax.loglog( pe.lengths[2:], pe.strengths[2:], 'r^', label = 'Filamenttests - Dresden' )
            #glass
#            logax.loglog( lengths_glass, strengths_glass_adapter, 'ko', label = 'Statimat 4U adapter' )
#            logax.loglog( lengths_glass, strengths_glass_resin, 's', color = 'grey', label = 'resin porters' )
            #carbon
            plt.loglog( A_lengths, A_strength[:,0], 'ko', label = 'yarn - Statimat 4U adapter' )
            plt.loglog( A_lengths, B_strength[:,0], 's', color = 'grey', label = 'yarn - Statimat 4U bollards' )
#            plt.loglog( [200., 400.], [1518., 1364.], 'ro', label = 'Dresden/Textechno S4U-Adpater' )
            #plt.loglog( [50., 100., 200., 400.], [1728., 1633., 1570., 1490.], 'ro', label = 'Dresden/Textechno S4U-Adpater basalt' )
            #plt.loglog( [200., 400.], [1649., 1528.], 'ko', label = 'Dresden/Dresden Capstan Grips basalt' )
            #plt.loglog( [50., 100., 200., 400.], [1284, 1158, 1284, 1227], 'ro', label = 'Dresden/Textechno S4U-Adpater ARglas' )
            #plt.loglog( [200., 400.], [1238, 1284], 'ko', label = 'Dresden/Dresden Capstan Grips ARglas' )            
            plt.xticks(fontsize = 16)
            plt.yticks(fontsize = 16)
            plt.grid(True,which="both",ls="-", color = 'grey')
        elif bundle_tests:
            plt.loglog( x, y, color='red', ls='dashed', label='Buendel' )
            plt.loglog( x, y*br, color = 'black', ls='dashed', label = 'Filament' )
            plt.loglog( pe.lengths, pe.strengths, 'ro')
            plt.loglog( np.array([20.,100.]), np.array([2157., 1790.]), 'ko')
            #plt.loglog( A_lengths, B_strength[:,0], 'ko', label = 'UB' )
            plt.xticks(fontsize = 16)
            plt.grid(True, which="both", ls="-", color='grey')
            plt.yticks([1000, 2000, 3000], fontsize=16)
        plt.xlabel( 'length [mm]', fontsize = 16 )
        plt.title('Weibull scaling log-log', fontsize = 16)
        yax = logax.yaxis
        yminor = yax.get_ticklocs(minor = True)
        ymajor = yax.get_ticklocs()
        yax.set_ticklabels(np.array(yminor,dtype = 'int'), minor = True, fontsize = 16)
        yax.set_ticklabels(np.array(ymajor,dtype = 'int'), fontsize = 16)
        xax = logax.xaxis
        xmajor = xax.get_ticklocs()
        xax.set_ticklabels((10,100,1000), fontsize = 16)
        plt.ylim(500,5000)
        plt.xlim(10,1000)
        #plt.ylabel( 'strength' )
        #plt.xlim( x[0] - ( x[-1] - x[0] ) * 0.01, x[-1] + ( x[-1] - x[0] ) * 0.05 )
        #plt.ylim( y[-1]*br + ( y[-1]*br - y[0]*br ) * 0.05, y[0] - ( y[-1] - y[0] ) * 0.05 )
        plt.legend(loc = 'best')

    def bundle_ld():
        bundle_length = 35.
        plt.figure()
        eps = np.linspace(0, 0.04, 100)
        sigma = pe.values(eps, bundle_length)
        plt.plot(eps, sigma, linewidth=2)
        plt.title('stress-strain of an asymptotic bundle of length %.1f' %bundle_length)
        plt.xlabel('strain [-]')
        plt.ylabel('stress [MPa]')

    #bundle_ld() 
    SE()
    plt.show()
