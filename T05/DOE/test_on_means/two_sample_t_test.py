'''
Created on May 16, 2012

--- module for evaluation of the difference in means ---

includes:   hypothesis testing, p-value evaluation,
            confidence interval evaluation 

general
assumptions: 1) independent populations
             2) (close to) normal distribution

choices
regarding
stdev:       1) equal stdev: t-test, equal (similar) standard deviation
             2) different stdev: t-test, generally different standard deviation
             3) known stdev: Z-test, the standard deviation is known exactly
                (to be defined globally in sigma1 and sigma2)

@author: rostar
'''

from enthought.traits.api import HasTraits, Array, Property, Float, cached_property
import numpy as np
from scipy.stats import t, norm

class TwoSampleMeanTest(HasTraits):
    ''' t-test and Z-test: tests the null hypothesis (mu1 = mu2) against an alternative
        hypothesis (mu1 > or < or != mu2) at a defined level of significance
        alpha (type I error) = P(H0 rejected|H0 true) additionally gives informations
        about p-values and confidence intervals
        TODO: test mean against a threshold value
    '''
    
    data1 = Array
    data2 = Array
    mean_0 = Float
    
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
       
    # the case when the stdevs of both samples are close to being equal
    def equal_stdev(self, alpha):
        n1, n2, y1, y2 = self.n1, self.n2, self.y1, self.y2
        Sp = np.sqrt( ((n1 - 1)*self.S1**2 +
                       (n2 - 1)*self.S2**2) / (n1 + n2 - 2) )
        t0 = (y1 - y2) / (Sp * np.sqrt(1./n1 + 1./n2))
        
        # hypothesis testing
        H1a = t.ppf(1 - alpha/2., n1 + n2 -2) < np.abs(t0)
        H1b = t.ppf(1 - alpha, n1 + n2 -2) < t0
        H1c = t.ppf(alpha, n1 + n2 -2) > t0
        # p-value
        p1a = t.sf(np.abs(t0), n1 + n2 -2) * 2
        p1b = t.sf(t0, n1 + n2 -2)
        p1c = t.cdf(t0, n1 + n2 -2)
        c1 = y1 - y2 - t.ppf(1 - alpha/2., n1 + n2 -2) * Sp * np.sqrt(1./n1+1./n2)        
        c2 = y1 - y2 + t.ppf(1 - alpha/2., n1 + n2 -2) * Sp * np.sqrt(1./n1+1./n2)
        return H1a, H1b, H1c, p1a, p1b, p1c, (c1,c2)
    
    # the case when stdevs of the samples differ
    def different_stdev(self, alpha):
        t0 = (self.y1 - self.y2) / (np.sqrt(self.S1**2/self.n1 +
                                            self.S2**2/self.n2))
        # hypothesis testing
        n1, n2, y1, y2, S1, S2 = self.n1, self.n2, self.y1, self.y2, self.S1, self.S2
        df = int((S1**2/n1+S2**2/n2)**2/((S1**2/n1)**2/(n1-1)+(S2**2/n2)**2/(n2-1)))
        H1a = t.ppf(1 - alpha/2., df) < np.abs(t0)
        H1b = t.ppf(1 - alpha, df) < t0
        H1c = t.ppf(alpha, df) > t0
        # p-value
        p1a = t.sf(np.abs(t0), df) * 2
        p1b = t.sf(t0, df)
        p1c = t.cdf(t0, df)
        c1 = y1 - y2 - t.ppf(1 - alpha/2., df) * np.sqrt(S1**2/n1+S2**2/n2)        
        c2 = y1 - y2 + t.ppf(1 - alpha/2., df) * np.sqrt(S1**2/n1+S2**2/n2)
        return H1a, H1b, H1c, p1a, p1b, p1c, (c1,c2)
    
    sigma1 = Float
    sigma2 = Float
    
    # the case when stdevs of the samples are known exactly
    def known_stdev(self, alpha):
        n1, n2, y1, y2 = self.n1, self.n2, self.y1, self.y2
        z0 = (y1 - y2) / (np.sqrt(self.sigma1**2/n1 + self.sigma2**2/n2))
        # hypothesis testing
        H1a = norm.ppf(1 - alpha/2.) < np.abs(z0)
        H1b = norm.ppf(1 - alpha) < z0
        H1c = norm.ppf(alpha) > z0
        # p-value
        p1a = norm.sf(np.abs(z0)) * 2
        p1b = norm.sf(z0)
        p1c = norm.cdf(z0)
        c1 = y1 - y2 - norm.ppf(1 - alpha/2.) * np.sqrt(self.sigma1**2/n1+self.sigma2**2/n2)        
        c2 = y1 - y2 + norm.ppf(1 - alpha/2.) * np.sqrt(self.sigma1**2/n1+self.sigma2**2/n2)        
        return H1a, H1b, H1c, p1a, p1b, p1c, (c1,c2)
    
    # evaluate the hypothesis testing   
    def evaluate(self, alpha, stdev = 'different'):
        ''' choices for stdev: 'equal','different'; 'known'
        confidence intervals: the minimum level of significance
        alpha for which the null hypothesis is rejected '''
        n1 = self.n1
        n2 = self.n2
        print 'at the level of significance ', alpha, ':'
        if stdev == 'equal':
            H1a, H1b, H1c, p1a, p1b, p1c, CI = self.equal_stdev(alpha)
        elif stdev == 'different':
            H1a, H1b, H1c, p1a, p1b, p1c, CI = self.different_stdev(alpha)
        elif stdev == 'known':
            H1a, H1b, H1c, p1a, p1b, p1c, CI = self.known_stdev(alpha)
        else:
            raise ValueError('choose stdev = equal, different or known')
        print 'H1 mu1 != mu2 is ', H1a
        print 'H1 mu1 > mu2 is ', H1b
        print 'H1 mu1 < mu2 is ', H1c        
        print 'probability of type I error for mu1 != mu2:', p1a
        print 'probability of type I error for mu1 > mu2:', p1b
        print 'probability of type I error for mu1 < mu2:', p1c
        print 'CI (%.1f%%) for mu1 - mu2:' %(100-100*alpha), CI
   
if __name__ == '__main__':
    
    means = TwoSampleMeanTest()
    means.data1 = np.array([460., 415,486,488,493,443,512,512,569,487,
                              458,426,454,387,464,524,551,521,462,579])
    means.data2 = np.array([542.,503,578,533,572,537,543,538,582,563,
                              572,541,541,561,526,535,474,536,568,503])
    means.sigma1 = np.sqrt(np.var(means.data1))
    means.sigma2 = np.sqrt(np.var(means.data2))
    means.evaluate(alpha = 0.05, stdev = 'different')
    