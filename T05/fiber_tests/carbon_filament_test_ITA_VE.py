'''
Created on May 3, 2012

@author: rch
'''


import math

def carbon():
    # cross section
    d = 7.8e-3
    r = d / 2.0

    A = math.pi * r ** 2

    length1 = 2.5 #mm

    F_25 = 0.17 #N
    F_25_std = 0.0238 #N

    # reduction factor for a bundle from Coleman diagram
    bundle_factor = 0.7
    
    print 'A', A
    print 'F_25', F_25
    print 'F_cov_25', F_25_std / F_25
    sig_25 = F_25 / A

    print 'sig_25_filament', sig_25
    print 'sig_25_bundle', sig_25 * bundle_factor

    length2 = 50 #mm
    
    F_50 = 0.155
    F_50_std = 0.0195
    
    print 'F_50', F_50
    print 'F_cov_50', F_50_std / F_50
    sig_50 = F_50 / A
    print 'sig_50_filament', sig_50
    
    print 'sig_50_bundle', sig_50 * bundle_factor

carbon()

