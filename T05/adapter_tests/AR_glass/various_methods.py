'''
Created on Jun 7, 2010

@author: rostislav
'''

from matplotlib import pyplot as p
from numpy import array

def glass():

    jana_lengths = array([9.2, 23.9, 58.8, 128.5, 308.4, 738.3])
    jana_mean = array([824.79, 795.67, 737.92, 693.19, 625.44, 487.45])
    jana_stdev = array([126.33, 121.29, 122.93, 101.18, 80.95, 82.52 ])

    # tests with spools
    sp_std = array([42.3, 55.3, 95.6])
    sp_length = array([400., 550., 750.])
    sp_strength = array([ 766.33, 704.53, 614.67])
    p.plot(sp_length, sp_strength, 'x--', color = 'black', linewidth = 2,
            label = 'capstan grips [10] ITA')
    p.errorbar(elinewidth = 2, x = jana_lengths, y = jana_mean, \
           yerr = jana_stdev, color = 'magenta')
    p.plot(jana_lengths, jana_mean, color = 'magenta', linewidth = 3,
            label = 'JANA')
    p.errorbar(elinewidth = 2, x = sp_length, y = sp_strength, \
           yerr = sp_std, color = 'black', fmt = 'x')

    # tests with additional clamps
    clamp_std = array([52.2, 78.51, 84.84])
    clamp_length = array([0., 50., 100.])
    clamp_strength = array([868.47, 951.13 , 1038.03])
    p.plot(clamp_length, clamp_strength, 's--', color = 'black', linewidth = 2,
            label = 'aluminium blocks p[5] ITA')
    p.errorbar(elinewidth = 2, x = clamp_length, y = clamp_strength, \
           yerr = clamp_std, color = 'black', fmt = 's')


    # tests in resin-blocks
    resin_std = array([108.65, 112.67, 56.14])
    resin_length = array([100., 200., 300.])
    resin_strength = array([1055.92, 913.18 , 928.72 ])
    p.plot(resin_length, resin_strength, '^--', color = 'black', linewidth = 2,
            label = 'epoxy resin [10] ITA')
    p.errorbar(elinewidth = 2, x = resin_length, y = resin_strength, \
           yerr = resin_std, color = 'black', fmt = '^')

    # tests with statimat
    statimat_std = array([39.87, 31.48, 64.22, 51.16])
    statimat_length = array([135., 300., 500., 800])
    statimat_strength = array([795.16, 729.07, 702.93, 677.61])
    p.plot(statimat_length, statimat_strength, 'o', \
           color = 'blue', linestyle = '--', linewidth = 2,
           label = 'S4U UB')
    p.errorbar(elinewidth = 2, x = statimat_length, y = statimat_strength, \
               yerr = statimat_std, color = 'blue', fmt = 'o')

    cs_area = 0.89 # mm^2
    # tests with statimat
    statimat_4ux_force_std = array([41.64, 29.63, 37.17, 27.55 ])
    statimat_4ux_std = statimat_4ux_force_std / cs_area
    # the last one I did not note [rch]
    statimat_4ux_length = array([50., 150., 300., 500])
    # force [N]
    statimat_4ux_force = array([962.61,
                                 904.59,
                                 881,
                                 867])
    statimat_4ux_strength = statimat_4ux_force / cs_area


    optimized_params_lengths = array([50., 70, 100, 150, 200, 250, 350, 500])
    optimized_params_strength = array([920.04, 912.50, 913.92, 911.31, 922.06, 898.67, 893.16, 878.78]) / 0.89
    optimized_params_cov = array([4.30, 4.44, 3.14, 2.89, 4.76, 4.39, 5.17, 4.40])


    p.errorbar(linewidth = 2, x = optimized_params_lengths, y = optimized_params_strength, \
               yerr = optimized_params_strength * optimized_params_cov / 100., color = 'green', fmt = 'o',
               ls = '-', label = 'S4U AOP [20] TT')

    p.plot(statimat_4ux_length, statimat_4ux_strength, 'o', \
           color = 'red', linestyle = '--', linewidth = 3,
           label = 'Adapter first try [10] TT')
    p.errorbar(elinewidth = 2, x = statimat_4ux_length, y = statimat_4ux_strength, \
               yerr = statimat_4ux_std, color = 'red', fmt = 'o')

    p.plot(50, 998.54 / 0.89, 'go')

    p.ylim(ymin = 0)
    p.xlabel('test length [mm]')
    p.ylabel('breaking stress [MPa]')
    p.grid()
    p.legend(loc = 'best')
    p.title('AR-glass 2400 tex')

def carbon():
    # tests with spools
    sp_length = array([400., 550., 750.])
    sp_strength = array([ 1440.56, 1454.78, 1423.45])
    p.plot(sp_length, sp_strength, 'x--', color = 'black', lw = 2, label = 'capstan grips [10] ITA')

    # tests with additional clamps
    clamp_length = array([0., 50., 100.])
    clamp_strength = array([1955.75, 1758.22, 1687.74])
    p.plot(clamp_length, clamp_strength, 's--', color = 'blue', lw = 2, label = 'aluminium blocks [5] ITA')

    # tests in resin-blocks
    resin_length = array([100., 200., 300.])
    resin_strength = array([1874.99, 1712.21, 1606.77])
    p.plot(resin_length, resin_strength, '^--', color = 'black', lw = 2, label = 'epoxy resin [10] ITA')

    # tests with statimat
    statimat_std = array([65.75, 51.58, 65.68, 65.95])
    statimat_length = array([135., 300., 500., 800])
    statimat_strength = array([1890.84, 1748.41, 1569.08, 1592.72])
    statimat_shift = statimat_length - 130

    optimized_params_lengths = array([50., 70, 100, 150, 200, 250, 350, 500])
    optimized_params_strength = array([1720.37, 1737.90, 1714.25, 1526.31, 1477.59, 1381.55, 1269.31, 1246.36]) / 0.894
    optimized_params_cov = array([5.16, 3.13, 4.37, 7.14, 8.69, 9.23, 7.12, 10.81])

    p.xlabel('test length [mm]')
    p.ylabel('breaking stress [MPa]')

    p.errorbar(linewidth = 2, x = statimat_length, y = statimat_strength, \
               yerr = statimat_std, color = 'r', fmt = 'o', ls = '-', label = 'S4U UB ITA')
    p.errorbar(linewidth = 2, x = statimat_shift, y = statimat_strength, \
               yerr = statimat_std, color = 'red', fmt = 'o', ls = '--', label = 'S4U UB shift 130mm')
    p.errorbar(linewidth = 2, x = optimized_params_lengths, y = optimized_params_strength, \
               yerr = optimized_params_strength * optimized_params_cov / 100., color = 'green', fmt = 'o',
               ls = '-', label = 'S4U AOP [20] TT')

    p.plot(array([200, 500]), array([1718., 1383.]) / 0.89,
            'bo', label = 'ITA Spule', ls = '-', color = 'black')
    p.plot(array([200, 500]), array([1589., 1066.]) / 0.89,
           'bo', label = 'transportierte Spule', ls = '--', color = 'black')

    p.ylim(ymin = 0)
    p.grid()
    p.legend(loc = 'best')
    p.title('carbon 1600 tex')

#carbon()
glass()
p.show()
