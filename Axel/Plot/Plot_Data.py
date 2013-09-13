from etsproxy.traits.api import \
    HasTraits, Instance, Int, Array, List, \
    cached_property, Property, Float, Bool, String
from operator import attrgetter
import numpy as np

import copy
from spirrid.rv import RV
from matplotlib import pyplot as plt
import time as t
import mayavi
import numpy
import pickle
import os
class Plot_view( HasTraits ):
        foldername = String
        
        def open_folder( self ):
            name = 'Data/{}'.format( self.foldername )
            os.chdir( name )
        def plot_e_s( self, pc, ps ):
                if pc:
                    plt.figure()
                    print os.listdir( os.curdir )
                    combined_es = open( 'combined_es.pkl', 'rb' )
                    combined_e_s_arr = pickle.load( combined_es )
                    combined_es.close()
                    plt.plot( combined_e_s_arr[0], combined_e_s_arr[1], linewidth = 2., linestyle = '--', \
                               color = 'k', label = 'combined' )
                if ps:
                    solo_es = open( 'solo_es.pkl', 'rb' )
                    solo_e_s_arr = pickle.load( solo_es )
                    solo_es.close()
                    plt.plot( solo_e_s_arr[0], solo_e_s_arr[1], linewidth = 2., \
                              linestyle = '-.', color = 'b' , label = 'solo' )
                    plt.legend( loc = 'best' )
                    plt.xlabel( 'composite strain [-]' )
                    plt.ylabel( 'composite stress [MPa]' )
                return 0

        def plot_s_w( self, pc, ps ):
                if pc:
                    plt.figure()
                    combined_sw = open( 'combined_sw.pkl', 'rb' )
                    combined_sw_arr = pickle.load( combined_sw )
                    combined_sw.close()
                    plt.plot( combined_sw_arr[0], combined_sw_arr[1] )
                if ps:
                    solo_sw = open( 'solo_sw.pkl', 'rb' )
                    solo_sw_arr = pickle.load( solo_sw )
                    solo_sw.close()
                    plt.plot( solo_sw_arr[0], solo_sw_arr[1] )
                return 0
if __name__ == '__main__':
    # Control
    pc = True
    ps = True
    plot_es = True
    plot_sw = False
    plot_folder = 'C2SF1'
    ######
    
    
    
    ini = Plot_view( foldername = plot_folder )
    ini.open_folder()
    if plot_es:ini.plot_e_s( pc, ps )
    if plot_sw:ini.plot_s_w( pc, ps )
    plt.legend( loc = 'best' )
    plt.show()
                
