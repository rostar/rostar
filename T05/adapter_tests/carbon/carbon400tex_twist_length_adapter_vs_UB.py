'''
Created on 12.3.2012
Tensile tests on carbon 400 tex yarns with variable length and twist ratio.
The tests shall be compared with the classical method (same parameter matrix)
A stands for Adapter tests
B stands for classical tests
@author: Q
'''

import numpy as np
from stats.spirrid import make_ogrid
from scipy.interpolate import RectBivariateSpline as Spl
from enthought.mayavi import mlab as m
from matplotlib import pyplot as plt

# DATA
lengths = np.array([35., 50., 130., 250., 500.])
twists = np.array([0., 10., 20., 30., 40.])

# Adapter tests
A35 = np.array([120.4, 117.8, 115.0, 114.2, 112.4])
A35cv = np.array([3.74, 4.06, 4.59, 3.14, 3.63])
A70 = np.array([116.4, 118.1, 116.1, 114.5, 111.5])
A70cv = np.array([3.2, 3.21, 2.96, 3.48, 3.38])
A130 = np.array([107.7, 114.5, 115.7, 115.0, 112.2])
A130cv = np.array([3.63, 4.84, 3.45, 3.55, 5.8])
A250 = np.array([102.2, 114.3, 113.7, 116.4, 116.7])
A250cv = np.array([5.23, 3.98, 3.38, 3.74, 3.0])
A500 = np.array([87.8, 100.6, 110.3, 115.7, 117.4])
A500cv = np.array([6.35, 5.74, 4.0, 6.0, 3.17])

A_strength = np.array([A35, A70, A130, A250, A500])
A_COV = np.array([A35cv, A70cv, A130cv, A250cv, A500cv])

# Umlenkbolzen tests 
B35 = np.array([108.6, 100., 100., 100., 110.9])
B35cv = np.array([3.71, 2., 2., 2., 3.21])
B70 = np.array([100., 100., 100., 100., 100.])
B70cv = np.array([2., 2., 2., 2., 2.])
B130 = np.array([99.4, 100., 108.4, 100., 111.2])
B130cv = np.array([2.36, 2., 4.79, 2., 3.65])
B250 = np.array([100., 100., 100., 100., 100.])
B250cv = np.array([2., 2., 2., 2., 2.])
B500 = np.array([87.1, 100., 100., 100., 111.9])
B500cv = np.array([6.13, 2., 2., 2., 3.08])

B_strength = np.array([B35, B70, B130, B250, B500])
B_COV = np.array([B35cv, B70cv, B130cv, B250cv, B500cv])

def plot_results(strength_arr, COV_arr):
    '''contains plotting methods'''

    stdev = strength_arr * COV_arr / 100. 

# DATA
lengths = np.array([35., 50., 130., 250., 500.])
twists = np.array([0., 10., 20., 30., 40.])

# Adapter tests
A35 = np.array([120.4, 117.8, 115.0, 114.2, 112.4])
A35cv = np.array([3.74, 4.06, 4.59, 3.14, 3.63])
A70 = np.array([116.4, 118.1, 116.1, 114.5, 111.5])
A70cv = np.array([3.2, 3.21, 2.96, 3.48, 3.38])
A130 = np.array([107.7, 114.5, 115.7, 115.0, 112.2])
A130cv = np.array([3.63, 4.84, 3.45, 3.55, 5.8])
A250 = np.array([102.2, 114.3, 113.7, 116.4, 116.7])
A250cv = np.array([5.23, 3.98, 3.38, 3.74, 3.0])
A500 = np.array([87.8, 100.6, 110.3, 115.7, 117.4])
A500cv = np.array([6.35, 5.74, 4.0, 6.0, 3.17])

A_strength = np.array([A35, A70, A130, A250, A500])
A_COV = np.array([A35cv, A70cv, A130cv, A250cv, A500cv])

# Umlenkbolzen tests 
B35 = np.array([108.6, 100., 100., 100., 110.9])
B35cv = np.array([3.71, 2., 2., 2., 3.21])
B70 = np.array([100., 100., 100., 100., 100.])
B70cv = np.array([2., 2., 2., 2., 2.])
B130 = np.array([99.4, 100., 108.4, 100., 111.2])
B130cv = np.array([2.36, 2., 4.79, 2., 3.65])
B250 = np.array([100., 100., 100., 100., 100.])
B250cv = np.array([2., 2., 2., 2., 2.])
B500 = np.array([87.1, 100., 100., 100., 111.9])
B500cv = np.array([6.13, 2., 2., 2., 3.08])

B_strength = np.array([B35, B70, B130, B250, B500])
B_COV = np.array([B35cv, B70cv, B130cv, B250cv, B500cv])

plt.ylim(0, ylim = 4)
plt.loglog(lengths, A_strength[0])
plt.show()

def plot_results(strength_arr, COV_arr):
    '''contains plotting methods'''

    stdev = strength_arr * COV_arr / 100. 

    def twist():
        '''plots strength depending on twist'''
        plt.figure()
        plt.errorbar(x = twists, y = strength_arr[0], \
               yerr = stdev[0], color = 'red')
        plt.plot(twists, strength_arr[0], color = 'red', lw = 2,
                 label = 'l = 35 mm')
        plt.errorbar(x = twists, y = strength_arr[1], \
               yerr = stdev[1], color = 'blue')
        plt.plot(twists, strength_arr[1], color = 'blue', lw = 2,
                 label = 'l = 70 mm')
        plt.errorbar(x = twists, y = strength_arr[2], \
               yerr = stdev[2], color = 'green')
        plt.plot(twists, strength_arr[2], color = 'green', lw = 2,
                 label = 'l = 130 mm')
        plt.errorbar(x = twists, y = strength_arr[3], \
               yerr = stdev[3], color = 'black')
        plt.plot(twists, strength_arr[3], color = 'black', lw = 2,
                 label = 'l = 250 mm')
        plt.errorbar(x = twists, y = strength_arr[4], \
               yerr = stdev[4], color = 'brown')
        plt.plot(twists, strength_arr[4], color = 'brown', lw = 2,
                 label = 'l = 500 mm')
        
        plt.legend(loc = 'best')
        plt.ylabel('strength cN/tex')
        plt.xlabel('twist [t/m]')
        plt.xlim(-10, 50)
        plt.ylim(80, 140)
        plt.title('Strength of twisted yarns')

#    def length():
#        '''plots strength depending on length'''
#        plt.figure()
#
#        plt.plot(lengths, strength_arr[:,0], 
#                 lw = 2, color = 'red', label = 'twist 0')
#        plt.errorbar(x = lengths, y = strength_arr[:,0],
#                     yerr = stdev[:,0], color = 'red')
#        plt.plot(lengths, strength_arr[:,1],
#
#        plt.errorbar(x = lengths, y = strength_arr[:,0],
#               yerr = stdev[:,0], color = 'red')
#        plt.plot(lengths, strength_arr[:,1],
#                        lw = 2, color = 'blue', label = 'twist 10')
#        plt.errorbar(x = lengths, y = strength_arr[:,1],
#               yerr = stdev[:,1], color = 'blue')
#        plt.plot(lengths, strength_arr[:,2],
#
#        plt.errorbar(x = lengths, y = strength_arr[:,1],
#               yerr = stdev[:,1], color = 'blue')
#        plt.plot(lengths, strength_arr[:,2],
#                        lw = 2, color = 'green', label = 'twist 20')
#        plt.errorbar(x = lengths, y = strength_arr[:,2],
#               yerr = stdev[:,2], color = 'green')
#        plt.plot(lengths, strength_arr[:,3],
#
#        plt.errorbar(x = lengths, y = strength_arr[:,2],
#               yerr = stdev[:,2], color = 'green')
#        plt.plot(lengths, strength_arr[:,3],
#                        lw = 2, color = 'magenta', label = 'twist 30')
#        plt.errorbar(x = lengths, y = strength_arr[:,3],
#               yerr = stdev[:,3], color = 'magenta')
#        plt.plot(lengths, A_strength[:,4],
#
#        plt.errorbar(x = lengths, y = strength_arr[:,3],
#               yerr = stdev[:,3], color = 'magenta')
#        plt.plot(lengths, A_strength[:,4],
#                        lw = 2, color = 'yellow', label = 'twist 40')
#        plt.errorbar(x = lengths, y = strength_arr[:,4],
#               yerr = stdev[:,4], color = 'yellow')
#
#        plt.errorbar(x = lengths, y = strength_arr[:,4],
#               yerr = stdev[:,4], color = 'yellow')
#        plt.title('Length dependent strength')
#        plt.ylim(50, 140)
#        plt.xlim(0,520)
#
#        plt.xlim(0,520)
#        plt.ylabel('strength cN/tex')
#        plt.xlabel('length [mm]')
#        plt.legend(loc = 'best')
    
    def plot_3d(cov, smooth):
        
        if smooth == True:
            sm_lengths = np.linspace(lengths[0], lengths[-1], 200)
            sm_twists = np.linspace(twists[0], twists[-1], 200)
            strength_spl = Spl(lengths, twists, strength_arr)
            sm_variables = make_ogrid([np.array(sm_lengths),
                                         np.array(sm_twists)])
            sm_strength = strength_spl(sm_variables[0],
                                       sm_variables[1]) / np.max(strength_arr)
            
            m.surf(sm_variables[0] / np.max(sm_variables[0]),
                   sm_variables[1] / np.max(sm_variables[1]),
                   sm_strength)
            
            if cov == True:
                stdev_spl = Spl(lengths, twists, stdev)
                sm_stdev = stdev_spl(sm_variables[0],
                                     sm_variables[1]) / np.max(strength_arr)
                m.surf(sm_variables[0] / np.max(sm_variables[0]),
                       sm_variables[1] / np.max(sm_variables[1]),
                       sm_strength + sm_stdev, color = (0, 0, 0), opacity = .3)
                m.surf(sm_variables[0] / np.max(sm_variables[0]),
                       sm_variables[1] / np.max(sm_variables[1]),
                       sm_strength + sm_stdev, color = (0, 0, 0), opacity = .3)
            
        else:
            variables = make_ogrid([np.array(lengths), np.array(twists)])
            m.surf(variables[0], variables[1], strength_arr)
            
            if cov == True:
                m.surf(variables[0], variables[1],
                       strength_arr + stdev, color = (0, 0, 0), opacity = .3)
                m.surf(variables[0], variables[1],
                       strength_arr - stdev, color = (0, 0, 0), opacity = .3)
    def plot_3d(cov, smooth):
        
        if smooth == True:
            sm_lengths = np.linspace(lengths[0], lengths[-1], 200)
            sm_twists = np.linspace(twists[0], twists[-1], 200)
            strength_spl = Spl(lengths, twists, strength_arr)
            sm_variables = make_ogrid([np.array(sm_lengths),
                                         np.array(sm_twists)])
            sm_strength = strength_spl(sm_variables[0],
                                       sm_variables[1]) / np.max(strength_arr)
            
            m.surf(sm_variables[0] / np.max(sm_variables[0]),
                   sm_variables[1] / np.max(sm_variables[1]),
                   sm_strength)
            
            if cov == True:
                stdev_spl = Spl(lengths, twists, stdev)
                sm_stdev = stdev_spl(sm_variables[0],
                                     sm_variables[1]) / np.max(strength_arr)
                m.surf(sm_variables[0] / np.max(sm_variables[0]),
                       sm_variables[1] / np.max(sm_variables[1]),
                       sm_strength + sm_stdev, color = (0, 0, 0), opacity = .3)
                m.surf(sm_variables[0] / np.max(sm_variables[0]),
                       sm_variables[1] / np.max(sm_variables[1]),
                       sm_strength + sm_stdev, color = (0, 0, 0), opacity = .3)
            
        else:
            variables = make_ogrid([np.array(lengths), np.array(twists)])
            m.surf(variables[0], variables[1], strength_arr)
            
            if cov == True:
                m.surf(variables[0], variables[1],
                       strength_arr + stdev, color = (0, 0, 0), opacity = .3)
                m.surf(variables[0], variables[1],
                       strength_arr - stdev, color = (0, 0, 0), opacity = .3)

    # SETTINGS
    #twist()
    #length()
    plot_3d(cov = False, smooth = False)

    # SETTINGS
    #twist()
    #length()
    plot_3d(cov = False, smooth = False)

plot_results(B_strength, B_COV)
plot_results(A_strength, A_COV)
plt.show()
m.show()

plot_results(B_strength, B_COV)
plot_results(A_strength, A_COV)
plt.show()
m.show()
