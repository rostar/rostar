'''
Created on Mar 2, 2012

@author: rostar
'''
import numpy as np
from matplotlib import pyplot as plt

cor = 400 / 100. / 0.89 * 4.

def screening():
    plt.figure()
    err = np.array([106.6 * 0.03, 110.6 * 0.043, 82.9 * 0.051, 110.1 * 0.018]) * cor
    plt.errorbar(x = [1, 2, 3, 4], y = np.array([106.6, 110.6, 82.9, 110.1]) * cor, \
           yerr = err, color = 'black')
    plt.plot(1, 106.6 * cor, 'ro', label = 'V vs V')
    plt.plot(2, 110.6 * cor, 'bo', label = 'V vs V + UB')
    plt.plot(3, 82.9 * cor, 'go', label = 'V vs Al')
    plt.plot(4, 110.1 * cor, 'ko', label = 'V vs Al + UB')
    plt.legend(loc = 'best')
    plt.ylabel('MPa')
    plt.title('optimum screening without adapter, length 135')
    plt.ylim(0)
    plt.xlim(0, 5)

def SE():
    plt.figure()
    l = np.array([35, 135, 500])
    s_UB = np.array([106.9, 110.6, 93.8])
    s_adapter = np.array([121.8, 110.1, 98.4])
    err_UB = np.array([106.9 * 0.045, 110.6 * 0.03, 93.8 * 0.091])
    plt.errorbar(x = l, y = s_UB * cor, \
           yerr = err_UB * cor, color = 'black')
    err_adapter = np.array([121.8 * 0.031, 110.1 * 0.035, 98.4 * 0.079])
    plt.errorbar(x = l, y = s_adapter * cor, \
           yerr = err_adapter * cor, color = 'black')
    plt.plot(l, s_adapter * cor, color = 'red', lw = 2, label = 'Adapter')
    plt.plot(l, s_UB * cor, color = 'blue', lw = 2, label = 'V vs V + UB')
    plt.legend(loc = 'best')
    plt.ylabel('MPa')
    plt.xlabel('length')
    plt.xlim(0, 600)
    plt.ylim(0)
    plt.title('size effect Adapter vs UB 0 twist')

def SEtwist():
    plt.figure()
    s_UB = np.array([110.6, 107.9])
    err_UB = np.array([110.6 * 0.03, 107.9 * 0.034])
    s_adapter = np.array([110.1, 115.6, 110.5])
    err_adapter = np.array([110.1 * 0.035, 115.6 * 0.028, 110.5 * 0.062])
    plt.errorbar(x = [0, 20], y = s_UB * cor, \
           yerr = err_UB * cor, color = 'black')
    plt.errorbar(x = [0, 20, 40], y = s_adapter * cor, \
           yerr = err_adapter * cor, color = 'black')
    plt.plot([0, 20, 40], s_adapter * cor, color = 'red', lw = 2, label = 'Adapter')
    plt.plot([0, 20], s_UB * cor, color = 'blue', lw = 2, label = 'V vs V + UB')
    plt.legend(loc = 'best')
    plt.ylabel('MPa')
    plt.xlabel('twist')
    plt.xlim(-10, 50)
    plt.ylim(0)
    plt.title('size effect Adapter vs UB with twist, l = 135 mm')

screening()
SE()
SEtwist()
plt.show()
