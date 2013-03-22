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
from etsproxy.mayavi import mlab as m
from matplotlib import pyplot as plt

# DATA
lengths = np.array([35., 70., 130., 250., 500.])
twists = np.array([0.,10.,20.,30.,40.])

# Adapter tests
A35 = np.array([120.4,117.8,115.0,114.2, 112.4])
A35cv = np.array([3.74, 4.06, 4.59, 3.14, 3.63])
A70 = np.array([116.4, 118.1, 116.1, 114.5, 111.5])
A70cv = np.array([3.2, 3.21, 2.96, 3.48, 3.38])
A130 = np.array([107.7, 114.5, 115.7, 115.0, 112.2])
A130cv = np.array([3.63, 4.84, 3.45, 3.55, 5.8])
A250 = np.array([102.2, 114.3, 113.7, 116.4, 116.7])
A250cv = np.array([5.23, 3.98, 3.38, 3.74, 3.0])
A500 = np.array([87.8, 100.6, 110.3, 115.7, 117.4])
A500cv = np.array([6.35, 5.74, 4.0, 6.0, 3.17])

A_strength = np.array([A35, A70, A130, A250, A500]) * 16./0.89
A_COV = np.array([A35cv, A70cv, A130cv, A250cv, A500cv])

# Umlenkbolzen tests 
B35 = np.array([108.6,114.1,113.1,112.4, 110.9])
B35cv = np.array([3.71, 2.93, 4.09, 3.22, 3.21])
B70 = np.array([107.5,111.8,114.0,114.2, 113.7])
B70cv = np.array([2.99, 3.04, 3.04, 3.92, 2.84])
B130 = np.array([99.4,110.3,108.4, 112.5, 111.2])
B130cv = np.array([2.36, 3.75, 4.79, 3.13, 3.65])
B250 = np.array([94.2,106.8,111.8,112.7, 113.4])
B250cv = np.array([4.24, 5.15, 2.98, 2.32, 2.66])
B500 = np.array([87.1,102.1,111.4,113.6, 111.9])
B500cv = np.array([6.13, 4.12, 2.19, 3.41, 3.08])

B_strength = np.array([B35, B70, B130, B250, B500]) * 16./0.89
B_COV = np.array([B35cv, B70cv, B130cv, B250cv, B500cv])
Rel_strength = A_strength / B_strength - 1.0

if __name__ == '__main__':

    plt.plot(lengths, A_strength[:, 0], 'ko')
    plt.errorbar(x=lengths, y=A_strength[:,0], label='Adapter',
           yerr = A_COV[:,0] * A_strength[:,0]/100., lw = 2, color = 'red')
    plt.plot(lengths, B_strength[:,0], 'ko')
    plt.errorbar(x = lengths, y = B_strength[:,0], label = 'Statimat + UB',
           yerr = B_COV[:,0] * B_strength[:,0]/100., lw=2, color = 'black', ls = 'solid')

    #plt.plot(lengths, A_strength[:,1],
    #                lw = 2, color = 'red', label = '10 twists')
    #
    #plt.plot(lengths, B_strength[:,1],
    #                lw = 2, color = 'red', ls = 'dashed')
    #
    #
    #plt.plot(lengths, A_strength[:,2],
    #                lw = 2, color = 'green', label = '20 twists')
    #
    #plt.plot(lengths, B_strength[:,2],
    #                lw = 2, color = 'green', ls = 'dashed')
    #
    #plt.plot(lengths, A_strength[:,3],
    #                lw = 2, color = 'blue', label = '30 twists')
    #
    #plt.plot(lengths, B_strength[:,3],
    #                lw = 2, color = 'blue', ls = 'dashed')
    
#    plt.plot(lengths, A_strength[:,4],
#                    lw = 2, color = 'black', label = 'Adapter')
#    plt.errorbar(x = lengths, y = A_strength[:,4],
#           yerr = A_COV[:,4] * A_strength[:,4]/100., color = 'black')
#    plt.plot(lengths, B_strength[:,4],
#                    lw = 2, color = 'black', ls = 'dashed', label = 'Umlenkbolzen')
#    plt.errorbar(x = lengths, y = B_strength[:,4],
#           yerr = B_COV[:,4] * B_strength[:,4]/100., color = 'black', ls = 'dashed')    
    
    plt.ylim(0, 2300)
    plt.xlim(0,520)
    plt.ylabel('Festigkeit [MPa]', fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.legend(loc = 'lower right')
    plt.title('Toho Tenax Carbon 400 tex', fontsize = 16)
    plt.xlabel('Laenge [mm]', fontsize = 16)
    
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
            plt.plot(twists, strength_arr[1], color = 'blue',lw = 2,
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
            
            plt.title('carbon 400tex UMLENKBOLZEN')
            plt.xlim(-10, 50)
            plt.ylim(1000,2500)
            plt.ylabel('strength MPa')
            plt.xlabel('twist [t/m]')
            plt.legend(loc = 'best')
    
        def length():
            '''plots strength depending on length'''
            #plt.figure()
            plt.semilogx(lengths, strength_arr[:,0],
                            lw = 2, color = 'red', label = 'twist 0')
            plt.errorbar(x = lengths, y = strength_arr[:,0],
                   yerr = stdev[:,0], color = 'red')
            plt.plot(lengths, strength_arr[:,1],
                            lw = 2, color = 'blue', label = 'twist 10')
            plt.errorbar(x = lengths, y = strength_arr[:,1],
                   yerr = stdev[:,1], color = 'blue')
            plt.plot(lengths, strength_arr[:,2],
                            lw = 2, color = 'green', label = 'twist 20')
            plt.errorbar(x = lengths, y = strength_arr[:,2],
                   yerr = stdev[:,2], color = 'green')
            plt.plot(lengths, strength_arr[:,3],
                            lw = 2, color = 'black', label = 'twist 30')
            plt.errorbar(x = lengths, y = strength_arr[:,3],
                   yerr = stdev[:,3], color = 'black')
            plt.plot(lengths, strength_arr[:,4],
                            lw = 2, color = 'brown', label = 'twist 40')
            plt.errorbar(x = lengths, y = strength_arr[:,4],
                   yerr = stdev[:,4], color = 'brown')
            plt.title('carbon 400tex UMLENKBOLZEN')
            plt.ylim(1000, 2500)
            plt.xlim(0,520)
            plt.ylabel('strength MPa')
            plt.xlabel('length [mm]')
            plt.legend(loc = 'best')
        
        def plot_3d(cov, smooth):
            
            if smooth == True:
                sm_lengths = np.linspace(lengths[0],lengths[-1], 200)
                sm_twists = np.linspace(twists[0],twists[-1],200)
                strength_spl = Spl(lengths, twists, strength_arr)
                sm_variables = make_ogrid([np.array(sm_lengths),
                                             np.array(sm_twists)])
                sm_strength = strength_spl(sm_variables[0],
                                           sm_variables[1])/np.max(strength_arr)
                
                m.surf(sm_variables[0]/np.max(sm_variables[0]),
                       sm_variables[1]/np.max(sm_variables[1]),
                       sm_strength)
                
                if cov == True:
                    stdev_spl = Spl(lengths, twists, stdev)
                    sm_stdev = stdev_spl(sm_variables[0],
                                         sm_variables[1])/np.max(strength_arr)
                    m.surf(sm_variables[0]/np.max(sm_variables[0]),
                           sm_variables[1]/np.max(sm_variables[1]),
                           sm_strength + sm_stdev, color = (0,0,0), opacity = .3)
                    m.surf(sm_variables[0]/np.max(sm_variables[0]),
                           sm_variables[1]/np.max(sm_variables[1]),
                           sm_strength + sm_stdev, color = (0,0,0), opacity = .3)
                
            else:
                variables = make_ogrid([np.array(lengths), np.array(twists)])
                m.surf(variables[0], variables[1], strength_arr)
                
                if cov == True:
                    m.surf(variables[0], variables[1],
                           strength_arr + stdev, color = (0,0,0), opacity = .3)
                    m.surf(variables[0], variables[1],
                           strength_arr - stdev, color = (0,0,0), opacity = .3)
    
        # SETTINGS
        #twist()
        #length()
        #plot_3d(cov = False, smooth = False)
    
    #plot_results(B_strength, B_COV)
    #plot_results(A_strength, A_COV)
    #plot_results(Rel_strength, A_COV)
    
    plt.show()
    #m.show()
