'''
Created on Oct 14, 2011

@author: rostar
'''
import numpy as np
import math

start_length = 35.0
stop_length = 500.0
n_lengths = 5

lengths = np.logspace(math.log(start_length, 10), math.log(stop_length, 10), n_lengths)

print lengths

lengths = [35, 70, 130, 250, 500]

twist = [0, 10, 20, 30, 40]
n_tests = 20

# get the unique test combinations
combinations = []
for t in twist:
    for l in lengths:
        # (twist, length, loading rate)
        combinations.append((t, l, l*0.1))

# multiply them by the number of replications
ordered_tests = n_tests * combinations

# randomize the tests
np.random.shuffle(ordered_tests)

print ordered_tests
