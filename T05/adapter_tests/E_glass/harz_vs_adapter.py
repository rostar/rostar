'''
Created on Mar 25, 2013

@author: rostar
'''

import numpy as np

A_800tex = 0.3137 # mm2 E_glass
A_1200tex = 0.4706 # mm2 E_glass
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

from matplotlib import pyplot as plt

if __name__ == '__main__':

    def e_glass_800tex():
        plt.plot(l_resin, f_resin_800tex, 'ko')
        plt.errorbar(l_resin, f_resin_800tex, color='orange',
                     yerr=np.array(f_resin_800tex) / 100. * COV_resin_800tex,
                         lw = 2, label = 'Harzeinbettung/ITA')

        plt.plot(l_adapter, f_adapter_800tex, 'ko')
        plt.errorbar(l_adapter, f_adapter_800tex, color='red',
                 yerr = np.array(f_adapter_800tex) / 100. * COV_adapter_800tex,
                     lw = 2, label = 'Adapter/Textechno')
        plt.title('E-Glas 800 tex', fontsize=16)

    def e_glass_1200tex():
        plt.plot(l_resin, f_resin_1200tex, 'ko')
        plt.errorbar(l_resin, f_resin_1200tex, color='black', ls='dashed',
                     yerr = np.array(f_resin_1200tex) / 100. * COV_resin_1200tex,
                         lw = 2, label = 'Harzeinbettung/ITA')
        plt.plot(l_adapter, f_adapter_1200tex, 'ko')
        plt.errorbar(l_adapter, f_adapter_1200tex, color='black',
                     yerr = np.array(f_adapter_1200tex) / 100. * COV_adapter_1200tex,
                         lw = 2, label = 'Adapter/Textechno')
        plt.title('E-Glas 1200 tex', fontsize=16)


    #e_glass_800tex()
    e_glass_1200tex()
    plt.grid()
    plt.xlim(0, 800)
    plt.ylim(0)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.legend(loc = 'lower right')
    plt.xlabel('Laenge [mm]', fontsize = 16)
    plt.ylabel('Festigkeit [MPa]', fontsize = 16)
    plt.show()
