'''
Created on May 16, 2012

--- module for evaluation of the difference in variances ---

includes:   hypothesis testing, p-value evaluation,
            confidence interval evaluation 

general
assumptions: 1) independent populations
             2) (close to) normal distribution

choices
regarding
the test:    1) testing the variance sigma against a threshold sigma_0
                (xi^2 test)
             2) testing two variances from two normal populations
                (F-test)

@author: rostar
'''

from etsproxy.traits.api import HasTraits, Array, Property, Float, cached_property
import numpy as np
from scipy.stats import chi2, f

class TwoSampleVarianceTest(HasTraits):
    ''' xi-test and F-test: tests the null hypothesis (sigma^2 = sigma_0^2)
        against an alternative hypothesis (sigma^2 != sigma_0^2) at a defined
        level of significance alpha (type I error) = P(H0 rejected|H0 true)
        additionally gives informations about p-values and confidence intervals
    '''
    
    data1 = Array
    data2 = Array
    var0 = Float
    
    y1 = Property(depends_on = 'data1')
    @cached_property
    def _get_y1(self):
        ''' mean of data1 '''
        return np.mean(self.data1)

    y2 = Property(depends_on = 'data2')
    @cached_property
    def _get_y2(self):
        ''' mean of data2 '''
        return np.mean(self.data2)   

    S1 = Property(depends_on = 'data1')
    @cached_property
    def _get_S1(self):
        ''' stdev of data1 '''
        return np.std(self.data1)

    S2 = Property(depends_on = 'data2')
    @cached_property
    def _get_S2(self):
        ''' stdev of data2 '''
        return np.std(self.data2)
    
    n1 = Property(depends_on = 'data1')
    @cached_property
    def _get_n1(self):
        ''' number of replications of data1 '''
        return len(self.data1)

    n2 = Property(depends_on = 'data2')
    @cached_property
    def _get_n2(self):
        ''' number of replications of data2 '''
        return len(self.data2)    
       
    # comparing variance from data1 to a threshold value - chi2 test
    def var_threshold(self, alpha):
        SS = (self.n1 - 1)*self.S1
        chi20 = SS/self.var0
        n1 = self.n1
        # hypothesis testing
        H1a = chi2.ppf(1 - alpha/2., n1 - 1) < chi20 or chi2.ppf(alpha/2., n1 - 1) > chi20
        H1b = chi2.ppf(alpha/2., n1 - 1) > chi20
        H1c = chi2.ppf(1 - alpha/2., n1 - 1) < chi20
        # p-value
        p1a = np.max(np.array([chi2.sf(chi20, n1 - 1), 1 - chi2.sf(chi20, n1 - 1)]))
        p1b = chi2.sf(chi20, n1 - 1)
        p1c = 1 - chi2.sf(chi20, n1 - 1)
        
        # confidence intervals: the minimum level of significance
        # alpha for which the null hypothesis is rejected 
        c1 = (n1-1)*SS/chi2.ppf(1 - alpha/2., n1 - 1)        
        c2 = (n1-1)*SS/chi2.ppf(alpha/2., n1 - 1)
        return H1a, H1b, H1c, p1a, p1b, p1c, (c1,c2)
    
    # comparing variance to variance F test
    def var_var(self, alpha):
        f0 = self.S1**2/self.S2**2
        n1, n2 = self.n1, self.n2
        # hypothesis testing
        H1a = f.ppf(1 - alpha/2., n1 - 1, n2 - 1) < f0 or f.ppf(alpha/2., n1 - 1, n2 - 1) > f0
        H1b = f.ppf(alpha/2., n1 - 1, n2 - 1) < f0
        H1c = f.ppf(1 - alpha/2., n1 - 1, n2 - 1) > f0
        # p-value
        p1a = np.max(np.array([f.sf(f0, n1 - 1, n2 - 1), 1 - f.sf(f0, n1 - 1, n2 - 1)]))
        p1b = f.sf(f0, n1 - 1, n2 - 1)
        p1c = 1 - f.sf(f0, n1 - 1, n2 - 1)
        
        # confidence intervals: the minimum level of significance
        # alpha for which the null hypothesis is rejected 
        c1 = self.S1**2/self.S1**2 * f.ppf(alpha/2., n2 - 1, n1 - 1)     
        c2 = self.S1**2/self.S1**2 * f.ppf(1 - alpha/2., n2 - 1, n1 - 1)     
        return H1a, H1b, H1c, p1a, p1b, p1c, (c1,c2)
        
    # evaluate the hypothesis testing   
    def evaluate(self, alpha, compare = 'threshold'):
        ''' choices for the test: 'threshold','variance' '''
        print 'at the level of significance ', alpha, ':'
        if compare == 'threshold':
            H1a, H1b, H1c, p1a, p1b, p1c, CI = self.var_threshold(alpha)
            print 'H1 var1 != var0 is ', H1a
            print 'H1 var1 > var0 is ', H1b
            print 'H1 var1 < var0 is ', H1c        
            print 'probability of type I error for var1 != var0:', p1a
            print 'probability of type I error for var1 > var0:', p1b
            print 'probability of type I error for var1 < var0:', p1c
            print 'CI (%.1f%%) for var1 - var0:' %(100.-100.*alpha), CI
        elif compare == 'variance':
            H1a, H1b, H1c, p1a, p1b, p1c, CI = self.var_var(alpha)
            print 'H1 var1 != var2 is ', H1a
            print 'H1 var1 > var2 is ', H1b
            print 'H1 var1 < var2 is ', H1c        
            print 'probability of type I error for var1 != var2:', p1a
            print 'probability of type I error for var1 > var2:', p1b
            print 'probability of type I error for var1 < var2:', p1c
            print 'CI (%.1f%%) for var1 - var2:' %(100.-100.*alpha), CI
        else:
            raise ValueError('choose compare = threshold or variance')

   
if __name__ == '__main__':
    
    means = TwoSampleVarianceTest()
    means.var0 = 100.
    means.data1 = np.array([460., 415,486,488,493,443,512,512,569,487,
                              458,426,454,387,464,524,551,521,462,579])
    means.data2 = np.array([542.,503,578,533,572,537,543,538,582,563,
                              572,541,541,561,526,535,474,536,568,503])
    means.sigma1 = np.sqrt(np.var(means.data1))
    means.sigma2 = np.sqrt(np.var(means.data2))
    means.evaluate(alpha = 0.05, compare = 'variance')
    