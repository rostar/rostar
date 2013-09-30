'''
Created on Aug 12, 2013

strain profiles in two fibers with different tau in a multiply cracked composite

@author: rostar
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import root

cs = 8.
taul = 0.04
tauk = 0.3
r = 0.0035
Ef = 240e3
sigmaf = 1000.
N = 14
depsfl = 2. * taul / r / Ef
depsfk = 2. * tauk / r / Ef

epsf0l_lst = []
def residuum(vect):
    if np.any(vect < 0.0):
        return np.array(vect) * 1e10
    else:
        l_arr = vect[:len(vect)/2]
        k_arr = vect[len(vect)/2:]
        res = []
        epsf0l = [0.0]
        epsf0k = [0.0]
        for i, l in enumerate(l_arr):
            k = k_arr[i]
            if i == 0:
                epsfl = l * depsfl
                epsf0l.append(epsfl)
                epsfk = k * depsfk
                epsf0k.append(epsfk)
                res.append((np.abs(epsfl) + np.abs(epsfk))/2. * Ef - sigmaf)
                
                wl = l * epsfl / 2. + depsfl * (l-l_arr[i+1]) * l_arr[i+1]
                wk = k * epsfk / 2. + depsfk * (k-k_arr[i+1]) * k_arr[i+1]
                res.append(np.abs(wl - wk) * sigmaf)
                
            elif i > 0 and i < len(l_arr) - 1:
                epsfl = epsf0l[i] + depsfl * cs - 2. * depsfl * l
                epsf0l.append(epsfl)
                epsfk = epsf0k[i] + depsfk * cs - 2. * depsfk * k
                epsf0k.append(epsfk)
                res.append((np.abs(epsfl) + np.abs(epsfk))/2. * Ef - sigmaf)
                wl = epsfl * (cs - l + l_arr[i+1]) - depsfl / 2. * ((cs - l)**2 + l_arr[i+1]**2)
                wk = epsfk * (cs - k + k_arr[i+1]) - depsfk / 2. * ((cs - k)**2 + k_arr[i+1]**2)
                res.append(np.abs(wl - wk) * sigmaf)
            
            else:
                epsfl = epsf0l[i] + depsfl * cs - 2. * depsfl * l
                epsf0l.append(epsfl)
                epsfk = epsf0k[i] + depsfk * cs - 2. * depsfk * k
                epsf0k.append(epsfl)
                res.append((np.abs(epsfl) + np.abs(epsfk))/2. * Ef - sigmaf)
                wl = (epsfl - (cs - l)/2. * depsfl) * 2. * (cs - l)
                wk = (epsfk - (cs - k)/2. * depsfk) * 2. * (cs - k)
                res.append(np.abs(wl - wk) * sigmaf)
        return np.array(res)

result = root(residuum, np.ones(N) * 4., method='krylov')
print result.x
print residuum(result.x)
residuum([ 44.07025239, 1.97554396,   1.21636174,   2.98268599,  21.77892428,
   4.60733681,   1.77084082,   2.9111265, ])

l1_arr = np.hstack((0.0, result.x[1:N/2]))
l2_arr = np.hstack((result.x[0], cs - l1_arr[1:]))
l_diffs = np.array(zip(l1_arr, l2_arr)).flatten()
fact = np.ones(N-1)
fact[1::2] *= -1.
l_taus = depsfl * l_diffs[1:] * fact
epsfl = np.hstack((0.0, np.cumsum(l_taus)))
l_arr = np.cumsum(l_diffs)

k1_arr = np.hstack((0.0, result.x[N/2 + 1:]))
k2_arr = np.hstack((result.x[N/2], cs - k1_arr[1:]))
k_diffs = np.array(zip(k1_arr, k2_arr)).flatten()
fact = np.ones(N-1)
fact[1::2] *= -1.
k_taus = depsfk * k_diffs[1:] * fact
epsfk = np.hstack((0.0, np.cumsum(k_taus)))
k_arr = np.cumsum(k_diffs)

plt.plot(k_arr - k_arr[-1], epsfk, label='high tau', lw=2)
plt.plot(l_arr - l_arr[-1], epsfl, label='low tau', lw=2)
plt.legend()
plt.title('strain profiles of two fibers with different tau at the clamping')
plt.xlabel('position [mm]')
plt.ylabel('fiber strain [-]')
plt.show()

