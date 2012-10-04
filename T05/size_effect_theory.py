'''
Created on Sep 22, 2009

@author: rostislav
'''

from etsproxy.traits.api import HasTraits, Float, Int, Property
from numpy import linspace, argmax, sqrt
from numpy.random import rand
from math import exp
from scipy.stats import weibull_min, norm
from matplotlib import pyplot as plt


class SizeEffect( HasTraits ):
    '''
    Size effect depending on the yarn length
    '''

    l_b = Float( 0.1, auto_set = False, enter_set = True, # [m]
                  desc = 'yarn total length',
                  modified = True )
    m_f = Float( 5, auto_set = False, enter_set = True, # [-]
                desc = 'Weibull shape parameter for filaments',
                modified = True )
    Nf = Int( 24000, auto_set = False, enter_set = True, # [-]
                desc = 'number of filaments in yarn',
                modified = True )
    l_rho = Float( 0.02, auto_set = False, enter_set = True, # [m]
                desc = 'autocorrelation length for fiber strength dispersion',
                modified = True )
    s_rho = Float( 2500, auto_set = False, enter_set = True, # [m]
            desc = 'scale parameter for autocorrelation length',
            modified = True )

    # these parameters are called plot, but the model should not 
    # interfere with the view (@todo resolve)
    l_plot = Float( 80., auto_set = False, enter_set = True, # [m]
                desc = 'maximum yarn length',
                modified = True )

    min_plot_length = Float( 0.0001, auto_set = False, enter_set = True, # [m]
                desc = 'minimum yarn length',
                modified = True )

    n_points = Int( 100, auto_set = False, enter_set = True,
                desc = 'points to plot',
                modified = True )

    # autocorrelation length function
    def fl( self, l ):
        return ( self.l_rho / ( self.l_rho + l ) ) ** ( 1. / self.m_f )
    '''second option'''
    #    return (l/self.l_rho + self.l_rho/(self.l_rho + l))**(-1./self.m_f)

    # scale parameter depending on length
    def s( self, l ):
        return self.fl( l ) * self.s_rho

    def mu_f( self, l ):
        return weibull_min( self.m_f, scale = self.s( l ), loc = 0.0 ).stats( 'm' )

    def mu_b( self, l ):
        return self.m_f ** ( -1.0 / self.m_f ) * self.s( l ) * exp( -1.0 / self.m_f )

class FilBun( SizeEffect ):

    nf = Int( 50 )
    K = Float( 72e9 * 0.89e-6 )
    weib = Property
    def _get_weib( self ):
        return weibull_min( self.m_f, scale = self.s( 0.2 ) / self.K )
    def filaments( self ):
        no = linspace( 0.001, 0.999, self.nf )
        strains = self.weib.ppf( no )
        return strains


if __name__ == '__main__':
    se = SizeEffect()
    fb = FilBun()
    
    def size_effect():
        
        l = linspace( 0.001, 50, 100000 )
        mu_f = se.mu_f( l )
        mu_b = se.mu_b( l )
        chob = se.mu_b( se.l_b )
        #plt.plot( l, mu_f, linewidth = 2, color = 'black', label = 'filament' )
        plt.plot( 0.05, se.mu_b( 0.05 ), 'ro' )
        plt.errorbar( 0.05, se.mu_b( 0.05 ), yerr = se.mu_b( 0.05 ) * 0.2, ecolor = 'red',
                     linewidth = 2 )
       # plt.plot( l, mu_b, linewidth = 2, color = 'black',
       #           ls = '--', label = 'bundle size effect' )
        #plt.plot( [se.l_b, 50], [chob, chob], linewidth = 2, color = 'red', label = 'yarn strength' )
        #plt.plot( [se.l_b, se.l_b], [10, chob], linestyle = 'dotted',
        #          linewidth = 1, color = 'black' )
        plt.plot( 1.0, se.mu_b( 1.0 ) * 1.3, 'ro' )
        plt.errorbar( 1.0, se.mu_b( 1.0 ) * 1.3, yerr = se.mu_b( 0.05 ) * 0.1, ecolor = 'red',
                     elinewidth = 2 )
#        plt.plot( l, l * 10e15, lw = 2, ls = '--',
#                  color = 'red', label = 'yarn size effect' )
        plt.xscale( 'log' )
        plt.yscale( 'log' )
        plt.legend( loc = 'best' )
        plt.xlabel( 'length log(l) $\mathrm{[m]}$', size = 'large' )
        plt.ylabel( 'strength log($\sigma_u$) $\mathrm{[N/mm^2]}$', size = 'large' )
        plt.title( 'STL evaluation', fontsize = 20 )
        plt.xlim( ( 0.001, max( l ) * .9 ) )
        plt.ylim( ( min( mu_b ) * 0.95, max( mu_f ) * 0.7 ) )
    def fil():

        strains = fb.filaments()
        x = linspace( 0., max( strains ) * 1.2, 1000 )
        for s in strains:
            plt.plot( [0., s, s], [0., fb.K * s, 0.], color = 'black' )
        plt.plot( [0., strains[-1], strains[-1]], [0., fb.K * strains[-1], 0.],
                 color = 'black', label = 'single filaments' )
        #plt.plot([0.0,0.0001],[0.0,0.0], color = 'black', label = 'filament' )
        #plt.plot( fb.weib.stats( 'm' ), fb.K * fb.weib.stats( 'm' ), 'ro' )
        #plt.plot( [0., fb.weib.stats( 'm' )], 2 * [fb.K * fb.weib.stats( 'm' )],
        #          color = 'red', linestyle = '--', label = 'mean filament strength' )
        bundle = fb.K * x * ( 1. - fb.weib.cdf( x ) )
        plt.plot( x, bundle, linewidth = 3, color = 'red',
                 label = 'bundle' )
        idmx = argmax( bundle )
        #plt.plot( x[idmx], max( bundle ), 'ro' )
        #plt.errorbar( x[idmx], max( bundle ), yerr = 0.3 * max( bundle ), lw = 2, color = 'red' )
        #plt.plot( [0., x[idmx]], [max( bundle ), max( bundle )],
        #          linestyle = '--', color = 'blue', label = 'bundle strength' )
        plt.legend( loc = 'upper left' )
        plt.xlabel( 'strain [-]', size = 'large' )
        plt.ylabel( 'stress [$\mathrm{N/mm^2}$]', size = 'large' )
        plt.title( 'l-d at gauge length 200 mm', fontsize = 20 )
        plt.xlim( ( 0., max( x ) * 1.1 ) )
        plt.ylim( ( 0., 2500 ) )
    
    def weibull():
        sigma = linspace( 0, 2300., 300 )
        CDF = weibull_min( se.m_f, scale = se.s( 0.2 ) ).cdf( sigma )
        plt.plot( sigma, CDF, lw = 2, color = 'black', label = 'filament: Weibull min. distr.' )
        plt.xlabel( 'stress [$\mathrm{N/mm^2}$]', size = 'large' )
        plt.ylabel( 'prob. of failure', size = 'large' )
        plt.title( 'CDF at gauge length 200 mm', fontsize = 20 )
        #plt.legend( loc = 'lower right' )
        
    def normal():
        sigma = linspace( 0, 2300., 300 )
        c = exp( -1. / se.m_f )
        mu = se.s( 0.2 ) * se.m_f ** ( -1. / se.m_f ) * c
        stdev = mu / c * sqrt( c * ( 1 - c ) ) * 1. / sqrt( 600. )
        CDF = norm( mu , stdev ).cdf( sigma )
        plt.plot( sigma, CDF, lw = 2, color = 'red', label = 'bundle: Gauss normal distr.' )
        plt.ylim( -0.01, 1.01 )
        #plt.xlabel( 'stress [$\mathrm{N/mm^2}$]', size = 'large' )
        #plt.ylabel( 'prob. of failure', size = 'large' )
        #plt.title( 'CDF at gauge length 200 mm', fontsize = 20 )
        plt.legend( loc = 'lower right' )


    #weibull()
    #normal()
    #fil()
    #size_effect()
    plt.show()


