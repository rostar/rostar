import numpy as np
from scipy.optimize import fsolve
from scipy.stats import weibull_min
from matplotlib import pyplot as plt
from scipy.optimize import broyden2
from scipy.special import gammainc
from scipy.interpolate import griddata
from mayavi import mlab as m
from math import pi
import time as t

x = np.linspace(-2,2,100)
y = 4 - x**2

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x,y)
p = ax.fill_between(x, y, facecolor='none')

from matplotlib.patches import PathPatch
path = p.get_paths()[0]
p1 = PathPatch(path, fc="none", hatch="/")
ax.add_patch(p1)
p1.set_zorder(p.get_zorder()-0.1)

pp = ax.fill_between(x, y, facecolor='none')
path = p.get_paths()[0]
p1 = PathPatch(path, fc="none", hatch="\\")
ax.add_patch(p1)
p1.set_zorder(p.get_zorder()-0.1)
plt.show()