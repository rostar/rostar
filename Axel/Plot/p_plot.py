'''
Created on 24.07.2013

@author: acki
'''
import numpy as np
import os
from matplotlib import pyplot as plt
FILE_DIR = os.path.dirname( __file__ )

'''general'''
#E_f = 170e3
#r = 3.45 * 1e-3  
#m = 4.8
#sV0 = 3.e-3
#Pf = RV( 'uniform', loc = 0., scale = 1.0 )

'''distr tau'''
#a_low=0
#Paras 1:a_up=b_low=0.018 ,b_up= .82
#Paras 2:a_up=b_low=0.023 ,b_up= .7


def plot1_sto():
    wData, sigmaData = np.loadtxt( os.path.join( FILE_DIR, 'Data1.txt' ), delimiter = ',' )
    wModel, sigmaModel = np.loadtxt( os.path.join( FILE_DIR, 'model1.txt' ), delimiter = ',' )
    plt.plot( wData, sigmaData / 0.445 * 1000, 'k', linewidth = 1 , label = 'experiment' )
    plt.plot( wModel, sigmaModel / 0.445 * 1000, 'k--' , linewidth = 1, label = 'model' )

def plot2_sto():
    wData, sigmaData = np.loadtxt( os.path.join( FILE_DIR, 'Data2.txt' ), delimiter = ',' )
    wModel, sigmaModel = np.loadtxt( os.path.join( FILE_DIR, 'model2.txt' ), delimiter = ',' )
    plt.plot( wData[:-63], sigmaData[:-63] / 0.445 * 1000, 'k', linewidth = 1 , label = 'experiment' )
    plt.plot( wModel[:-150], sigmaModel[:-150] / 0.445 * 1000, 'k--' , linewidth = 1, label = 'model' )
    
def plot1_det():
    wData, sigmaData = np.loadtxt( os.path.join( FILE_DIR, 'Data_det1.txt' ), delimiter = ',' )
    wModel, sigmaModel = np.loadtxt( os.path.join( FILE_DIR, 'model_det1.txt' ), delimiter = ',' )
    plt.plot( wData[:-63], sigmaData[:-63] / 0.445 * 1000, 'k', linewidth = 1 , label = 'experiment' )
    plt.plot( wModel, sigmaModel / 0.445 * 1000, 'k--' , linewidth = 1, label = 'model' )

def plot2_det():
    wData, sigmaData = np.loadtxt( os.path.join( FILE_DIR, 'Data_det2.txt' ), delimiter = ',' )
    wModel, sigmaModel = np.loadtxt( os.path.join( FILE_DIR, 'model_det2.txt' ), delimiter = ',' )
    plt.plot( wData[:-63], sigmaData[:-63] / 0.445 * 1000, 'k', linewidth = 1 , label = 'experiment' )
    plt.plot( wModel, sigmaModel / 0.445 * 1000, 'k--' , linewidth = 1, label = 'model' ) 

plot1_sto()
#plot2_sto()
plot1_det()
#plot2_det()
plt.legend()
plt.xlabel( 'w [mm]', fontsize = '14' )
plt.ylabel( 'sigma [Mpa]', fontsize = '14' )
plt.grid()
plt.xlim( 0, 11 )
plt.ylim( 0, 1200 )
plt.show()
