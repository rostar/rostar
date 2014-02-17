'''
Created on 16 Feb 2014

@author: Q
'''

import numpy as np
from matplotlib import pyplot as plt
from quaducom.meso.homogenized_crack_bridge.rigid_matrix.CB_view import CBView, Model

def CB():
    model = Model(w_min=0.0, w_max=8.0, w_pts=100,
                  w2_min=0.0, w2_max=.5, w2_pts=3,
                  sV0=0.00383, m=7.0, tau_scale=0.3,
                  tau_shape=0.2, tau_loc=0.01, Ef=180e3,
                  lm=20., n_int=100)
    
    i = 0
    cb = np.loadtxt("CB" + str(i+1) + ".txt", delimiter=';')
    model.test_xdata = -cb[:,2]/4. - cb[:,3]/4. - cb[:,4]/2.
    model.test_ydata = cb[:,1] / (11. * 0.45) * 1300

    i = 1
    cb = np.loadtxt("CB" + str(i+1) + ".txt", delimiter=';')
    model.test_xdata2 = -cb[:,2]/4. - cb[:,3]/4. - cb[:,4]/2.
    model.test_ydata2 = cb[:,1] / (11. * 0.45) * 1300

    i = 2
    cb = np.loadtxt("CB" + str(i+1) + ".txt", delimiter=';')
    model.test_xdata3 = -cb[:,2]/4. - cb[:,3]/4. - cb[:,4]/2.
    model.test_ydata3 = cb[:,1] / (11. * 0.45) * 1300
    
    i = 3
    cb = np.loadtxt("CB" + str(i+1) + ".txt", delimiter=';')
    model.test_xdata4 = -cb[:,2]/4. - cb[:,3]/4. - cb[:,4]/2.
    model.test_ydata4 = cb[:,1] / (11. * 0.45) * 1300
    
    i = 4
    cb = np.loadtxt("CB" + str(i+1) + ".txt", delimiter=';')
    model.test_xdata5 = -cb[:,2]/4. - cb[:,3]/4. - cb[:,4]/2.
    model.test_ydata5 = cb[:,1] / (11. * 0.45) * 1300
    
    cb = CBView(model=model)
    cb.refresh()
    cb.configure_traits()
    
    # for i in range(5):
    #     data = np.loadtxt("CB" + str(i+1) + ".txt", delimiter=';')
    #     plt.plot(-data[:,2]/4. - data[:,3]/4. - data[:,4]/2.,data[:,1], lw=2, label="CB" + str(i+1))
    # 
    # plt.legend()
    # plt.show()
    
def TT():
    for i in range(5):
        data = np.loadtxt("TT-4C-0" + str(i+1) + ".txt", delimiter=';')
        plt.plot(-data[:,2]/2./250. - data[:,3]/2./250.,data[:,1]/2., lw=1, label="MC" + str(i+1))
#     data = np.loadtxt("TT-4C-05.txt", delimiter=';')
#     plt.plot(-data[:,2],data[:,1], label="1")
#     plt.plot(-data[:,3],data[:,1], label="2")
#     plt.plot(-data[:,4],data[:,1], label="3")
    plt.legend(loc='best')
    plt.show()

CB()
#TT()