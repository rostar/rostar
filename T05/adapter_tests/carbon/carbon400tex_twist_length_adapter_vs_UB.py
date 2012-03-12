'''
Created on 12.3.2012
Tensile tests on carbon 400 tex yarns with variable length and twist ratio.
The tests shall be compared with the classical method (same parameter matrix)
@author: Q
'''

import numpy as np
from matplotlib import pyplot as plt

# ADAPTER
def twist_length():
    lengths = np.array([35., 50., 130., 250., 500.])
    twists = np.array([0.,10.,20.,30.,40.])
    length35 = np.array([120.4,117.8,115.0,114.2, 112.4])
    err_length35 = np.array([120.4 * 0.0374, 117.8 * 0.0406, 
                             115.0 * 0.0459 , 114.2 * 0.0314, 112.4 * 0.0363])
    length70 = np.array([116.4, 118.1, 116.1, 114.5, 111.5])
    err_length70 = np.array([116.4 * 0.032, 118.1 * 0.0321,
                             116.1* 0.0296, 114.5 * 0.0348, 111.5 * 0.0338])
    length130 = np.array([107.7, 114.5, 115.7, 115.0, 112.2])
    err_length130 = np.array([107.7 * 0.0363, 114.5 * 0.0484,
                             115.7 * 0.0345, 115.0 * 0.0355, 112.2 * 0.058])
    length250 = np.array([0., 0.0, 0.0, 0.0, 0.0])
    err_length250 = np.array([0., 0.0, 0.0, 0.0, 0.0])
    length500 = np.array([0., 0.0, 0.0, 0.0, 0.0])
    err_length500 = np.array([0., 0.0, 0.0, 0.0, 0.0])
    
    results = np.array([length35, length70, length130])
    COV = np.array([err_length35, err_length70, err_length130])
    
    def twist():
        plt.figure()
        plt.errorbar(x = twists, y = results[0], \
               yerr = COV[0], color = 'red')
        plt.plot(twists, results[0], color = 'red', lw = 2,
                 label = 'l = 35 mm')
    
        plt.errorbar(x = twists, y = results[1], \
               yerr = COV[1], color = 'blue')
        plt.plot(twists, results[1], color = 'blue', lw = 2, label = 'l = 70 mm')
    
        plt.errorbar(x = twists, y = results[2], \
               yerr = COV[2], color = 'green')
        plt.plot(twists, results[2], color = 'green', lw = 2, label = 'l = 130 mm')
        
        plt.legend(loc = 'best')
        plt.ylabel('strength cN/tex')
        plt.xlabel('twist [t/m]')
        plt.xlim(-10, 50)
        plt.ylim(50,140)
        plt.title('Strength of twisted yarns')

    def length():
        plt.figure()
        plt.plot(lengths[:3], results[:,0],
                        lw = 2, color = 'red', label = 'twist 0')
        plt.errorbar(x = lengths[:3], y = results[:,0], \
               yerr = COV[:,0], color = 'red')
        plt.plot(lengths[:3], results[:,1],
                        lw = 2, color = 'blue', label = 'twist 10')
        plt.errorbar(x = lengths[:3], y = results[:,1], \
               yerr = COV[:,1], color = 'blue')
        plt.plot(lengths[:3], results[:,2],
                        lw = 2, color = 'green', label = 'twist 20')
        plt.errorbar(x = lengths[:3], y = results[:,2], \
               yerr = COV[:,2], color = 'green')
        plt.plot(lengths[:3], results[:,3],
                        lw = 2, color = 'magenta', label = 'twist 30')
        plt.errorbar(x = lengths[:3], y = results[:,3], \
               yerr = COV[:,3], color = 'magenta')
        plt.plot(lengths[:3], results[:,4],
                        lw = 2, color = 'yellow', label = 'twist 40')
        plt.errorbar(x = lengths[:3], y = results[:,4], \
               yerr = COV[:,4], color = 'yellow')
        plt.title('Length dependent strength')
        plt.ylim(50, 140)
        plt.xlim(0,150)
        plt.ylabel('strength cN/tex')
        plt.xlabel('length [mm]')
        plt.legend(loc = 'best')
    
    twist()
    length()

twist_length()
plt.show()