'''
Created on May 16, 2012

--- module for evaluation of the difference in means ---

includes:   hypothesis testing2, p-value evaluation,
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

from etsproxy.traits.api import HasTraits, Array, Property, Float, cached_property
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

        # hypothesis testing2
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
        # hypothesis testing2
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

    # the case when stdevs of the samples are known exactly
    def known_stdev(self, alpha, stdev1, stdev2):
        n1, n2, y1, y2 = self.n1, self.n2, self.y1, self.y2
        z0 = (y1 - y2) / (np.sqrt(stdev1 ** 2. / n1 + stdev2 ** 2. / n2))
        # hypothesis testing2
        H1a = norm.ppf(1 - alpha / 2.) < np.abs(z0)
        H1b = norm.ppf(1 - alpha) < z0
        H1c = norm.ppf(alpha) > z0
        # p-value
        p1a = norm.sf(np.abs(z0)) * 2
        p1b = norm.sf(z0)
        p1c = norm.cdf(z0)
        c1 = y1 - y2 - norm.ppf(1 - alpha / 2.) * np.sqrt(stdev1 ** 2. / n1 + stdev2 ** 2. / n2)
        c2 = y1 - y2 + norm.ppf(1 - alpha / 2.) * np.sqrt(stdev1 ** 2. / n1 + stdev2 ** 2. / n2)
        return H1a, H1b, H1c, p1a, p1b, p1c, (c1, c2)

    # the case when stdevs of the samples differ and n, y, S are given explicitly
    def different_stdev_explicite(self, alpha, y1, y2, S1, S2, n1, n2):
        t0 = (y1 - y2) / (np.sqrt(S1 ** 2 / n1 + S2 ** 2 / n2))
        # hypothesis testing2
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
        CI = (c1, c2)
        print 'at the level of significance ', alpha, ':'
        print 'H1 mu1 != mu2 is ', H1a
        print 'H1 mu1 > mu2 is ', H1b
        print 'H1 mu1 < mu2 is ', H1c
        print 'probability of type I error for mu1 != mu2:', p1a
        print 'probability of type I error for mu1 > mu2:', p1b
        print 'probability of type I error for mu1 < mu2:', p1c
        print 'CI (%.1f%%) for mu1 - mu2:' %(100-100*alpha), CI, CI/y1

    # evaluate the hypothesis testing2
    def evaluate(self, alpha, stdev='different'):
        ''' choices for stdev: 'equal','different'; 'known'
        confidence intervals: the minimum level of significance
        alpha for which the null hypothesis is rejected '''
        n1 = self.n1
        n2 = self.n2
        print 'mean [COV] data1 = ', self.y1, '[', self.S1/self.y1, ']'
        print 'mean [COV] data2 = ', self.y2, '[', self.S2/self.y2, ']'
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
        print 'CI (%.1f%%) for mu1 - mu2:' %(100-100*alpha), CI, CI/self.data1.mean()
   
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    tests = np.loadtxt('tests.csv', delimiter=';') / 0.445
    a50 = tests[1:21,0]
    a70 = tests[1:21,1]
    a110 = tests[1:21,2]
    a160 = tests[1:21,3]
    a230 = tests[1:21,4]
    a340 = tests[1:21,5]
    a500 = tests[1:21,6]
    r50 = tests[22:43,0]
    r70 = tests[22:43,1]
    r110 = tests[22:43,2]
    r160 = tests[22:43,3]
    r230 = tests[22:43,4]
    r340 = tests[22:43,5]
    r500 = tests[22:43,6]

    means = TwoSampleMeanTest()
#    means.data1 = a500
#    means.data2 = r500
#    means.sigma1 = means.S1
#    means.sigma2 = means.S2
#    means.evaluate(alpha=0.05, stdev='different')

#    E-glass adapter vs resin
    A_800tex = 0.3137 # mm2
    A_1200tex = 0.4706 # mm2
    A_n = 20.
    R_n = 15.
    # lengths
    l_adapter = np.array([50., 70., 110., 160., 230., 340.,  500.])
    l_resin = np.array([50., 110., 160., 230., 500.])
    
    # strength and COV resin blocks
    f_resin_800tex = np.array([364.20, 374.82, 321.19, 319.84, 296.71]) / A_800tex
    COV_resin_800tex = np.array([8.7, 11.58, 15.94, 19.25, 19.80])
    
    f_resin_1200tex = np.array([583.30, 536.72, 545.99, 501.61, 475.09]) / A_1200tex
    COV_resin_1200tex = np.array([9.71, 11.29, 12.85, 16.22, 20.88])
    
    # strength and COV Adapter
    f_adapter_800tex = np.array([458.1, 462.98, 460.85, 399.63, 402.48, 381.1, 359.55]) / A_800tex
    COV_adapter_800tex = np.array([9.32, 7.9, 6.23, 12.67, 15.96, 17.46, 18.57])
    
    f_adapter_1200tex = np.array([715.39, 721.47, 694.6, 661.01, 650.4, 606.91, 541.33, ]) / A_1200tex
    COV_adapter_1200tex = np.array([6.08, 5.25, 8.23, 12.22, 14.21, 13.51, 15.87])
    
    f_adapter_carbon_400tex = np.array([2164.49438202, 2092.58426966, 1936.17977528, 1837.30337079, 1578.42696629])
    COV_adapter_carbon_400tex = np.array([3.74, 3.2, 3.63, 5.23, 6.35])
    
    f_UB_carbon_400tex = np.array([1952.35955056, 1932.58426966, 1786.96629213, 1693.48314607, 1565.84269663])
    COV_UB_carbon_400tex = np.array([3.71, 2.99, 2.36, 4.24, 6.13])

    means.different_stdev_explicite(0.05, f_adapter_carbon_400tex[4], f_UB_carbon_400tex[4],
                                    f_adapter_carbon_400tex[4] * COV_adapter_carbon_400tex[4] / 100.,
                                    f_UB_carbon_400tex[4] * COV_UB_carbon_400tex[4] / 100., A_n, R_n)


