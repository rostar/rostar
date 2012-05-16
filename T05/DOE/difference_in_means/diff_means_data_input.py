'''
Created on May 16, 2012

input data for testing the difference in means by:
    - hypothesis testing
    - confidence intervals

@author: rostar
'''

from enthought.traits.api import HasTraits, Array

class DiffMeansDataInput(HasTraits):
    
    data1 = Array
    data2 = Array
