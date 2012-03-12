'''
Created on Jan 13, 2012

@author: rostar
'''


# lengths
l = [50., 70., 110, 160, 230, 340, 500]

# strength and COV resin blocks
fh = [485.09, 450.50, 421.08, 390.33, 400.07, 387.80, 360.14]
cvh = [10.31, 13.29, 12.30, 18.42, 16.80, 11.80, 19.77 ]

# strength and COV Adapter
fa = [542.90, 544.49, 550.47, 541.30, 519.61, 512.26, 483.74]
cva = [5.05, 7.52, 4.23, 7.23, 9.32, 9.29, 6.56]

# comparison of strength and COV Adapter from coil
fs = [546.53, 571.99, 565.17, 576.55, 554.90, 527.80, 534.35]
cvs = [6.65, 2.96, 3.79, 6.66, 6.25, 12.14, 16.61]

# comparison with former results at 1200 tex glass with adapter - optimized params
lengths = [50., 70, 100, 150, 200, 250, 350, 500]
com_f = [579.36, 570.41, 578.92, 572.17, 580.79, 567.55, 574.63, 550.96]
com_cv = [2.74, 2.78, 4.31, 3.57, 4.47, 4.20, 4.63, 5.32]

from matplotlib import pyplot as plt
import numpy as np

harz = True
adapter = True
# results from the same coil with adapter - samples taken directly from the coil
vergleich = True
# results from former tests with adapter - with optimized params for the material
vergleich2 = True
loglog = False
# if False, stress is plotted
force = False

if force == False:
    fh = np.array(fh) / 0.445
    fa = np.array(fa) / 0.445
    fs = np.array(fs) / 0.445
    com_f = np.array(com_f) / 0.445

if harz == True:
    if loglog == True:
        plt.loglog(l, fh, 'bo')
    else:
        plt.plot(l, fh, 'bo')
    plt.errorbar(l, fh, color = 'blue', yerr = np.array(fh) / 100. * cvh,
                 lw = 2, label = 'Harz rand. [20], 9.1.TT')

if adapter == True:
    if loglog == True:
        plt.loglog(l, fa, 'ro')
    else:
        plt.plot(l, fa, 'ro')
    plt.errorbar(l, fa, color = 'red', yerr = np.array(fa) / 100. * cva,
                 lw = 2, label = 'S4U AOP rand. [20], 9.1.TT')

if vergleich == True:
    if loglog == True:
        plt.loglog(l, fs, 'ko')
    else:
        plt.plot(l, fs, 'ko')
    plt.errorbar(l, fs, color = 'black', yerr = np.array(fs) / 100. * cvs,
                 lw = 1, ls = '--', label = 'S4U AOP von der Spule [10], 9.1.TT')

if vergleich2 == True:
    if loglog == True:
        plt.loglog(lengths, com_f, 'go')
    else:
        plt.plot(lengths, com_f, 'go')
    plt.errorbar(lengths, com_f, color = 'green', yerr = np.array(com_f) / 100. * com_cv,
                 lw = 1, ls = '--', label = 'S4U AOP von der Spule [20] 2.8. TT')


plt.grid()
plt.xlim(0, 600)
plt.ylim(0)
plt.legend(loc = 'best')
plt.title('AR-Glas 1200 tex')
plt.xlabel('Laenge [mm]')
if force == True:
    plt.ylabel('Kraft [N]')
else:
    plt.ylabel('Spannung [MPa]')
plt.show()
