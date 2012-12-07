import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from etsproxy.mayavi import mlab as m
from stats.spirrid import make_ogrid as orthogonalize


x1 = np.linspace(-20, 20, 50)
y1 = np.repeat(1., len(x1))
x2 = np.linspace(-20.2, 20.2, 70)
y2 = np.repeat(20., len(x2))
x3 = np.linspace(-20.9, 20.9, 90)
y3 = np.repeat(30., len(x3))
x = np.hstack((x1,x2,x3))
y = np.hstack((y1,y2,y3))
points = np.hstack((x,y)).reshape(2, len(x)).T
z = np.cos(np.abs(y)) * np.sin(x)
i = LinearNDInterpolator(points, z)
print i([0.,1.],[1.,1.])
xx = np.linspace(-20, 20, 100)
yy = np.linspace(0, 30, 100)
XX, YY = np.meshgrid(xx,yy)
e_arr = orthogonalize([xx, yy])
m.surf(e_arr[0], e_arr[1], i(XX, YY))
m.show()
