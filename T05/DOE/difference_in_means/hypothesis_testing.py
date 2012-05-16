'''
Created on May 16, 2012

simple module for hypothesis testing of the difference in means by:
    - t-test
    - p-values

@author: rostar
'''

from enthought.traits.api import HasTraits, Instance, Property, Float, cached_property
from diff_means_data_input import DiffMeansDataInput
import numpy as np
from scipy.stats import t, norm

class TTest(HasTraits):
    
    data = Instance(DiffMeansDataInput, ())

    alpha = Float(0.5)
    ''' level of significance (type I error) = P(H0 rejected|H0 true) '''

    t0 = Property(depends_on = 'data.data1, data.data2')
    @cached_property
    def _get_t0(self):
        S1 = np.sqrt( np.var(self.data.data1) )
        S2 = np.sqrt( np.var(self.data.data2) )
        n1 = len(self.data.data1)
        n2 = len(self.data.data2)
        mu1 = np.mean(self.data.data1)
        mu2 = np.mean(self.data.data2)
        Sp = np.sqrt( ((n1 - 1)*S1**2 + (n2 - 1)*S2**2) / (n1 + n2 - 2) )
        t0 = (mu1 - mu2) / (Sp * np.sqrt(1./n1 + 1./n2))
        return t0
    
    def evaluate(self):
        n1 = len(self.data.data1)
        n2 = len(self.data.data2)
        print 'at the level of significance ', self.alpha, ':'
        res0 = t.ppf(1 - self.alpha/2., n1 + n2 -2) > np.abs(self.t0)
        print 'mu1 = mu2 is ',res0
        res1 = t.ppf(1 - self.alpha/2., n1 + n2 -2) < np.abs(self.t0)
        print 'mu1 != mu2 is ',res1
        res2 = t.ppf(1-self.alpha, n1 + n2 -2) < self.t0
        print 'mu1 > mu2 is ',res2
        res3 = -t.ppf(1-self.alpha, n1 + n2 -2) > self.t0
        print 'mu1 < mu2 is ',res3        
    
if __name__ == '__main__':
    tt = TTest()
    tt.data.data1 = norm.rvs( loc = 20, size = 10)
    tt.data.data2 = norm.rvs(loc = 20, size = 10)
    tt.evaluate()
    