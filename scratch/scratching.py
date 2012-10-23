import numpy as np

f1 = lambda x: x**2

f_arr = np.array([f1, f1])

def func(f_arr, a):
    return np.array([ff(a[i]) for i, ff in enumerate(f_arr)])

print func(f_arr, np.array([2,3])) 