from stats.misc.random_field.random_field_nD import RandomField
import numpy as np
import matplotlib
matplotlib.use('WxAgg')
import matplotlib.pyplot as plt



m = 4.
n_sim = 10
lrho = 5.0
L_arr = np.arange(10)[1:]*lrho

for Li in L_arr:
    rf = RandomField(seed=False,
                     distr_type='Weibull',
                     lacor_arr=np.array([lrho]),
                     nDgrid=[np.linspace(0.0, Li, int(Li/lrho*100))]
                    )
    print rf.nDgrid
    reevaluate=True
    for sim in np.arange(n_sim):
        rf.reevaluate = True
        plt.plot(rf.nDgrid[0], rf.random_field)
    plt.show()