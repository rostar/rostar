'''
Created on Oct 14, 2011

@author: rostar
'''
import numpy as np
import math

start_length = 50.0
stop_length = 500.0
n_lengths = 7

lengths = np.logspace(math.log(start_length, 10), math.log(stop_length, 10), n_lengths)

print lengths

#lengths = [50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
test = ['Statimat', 'resin']
n_tests = 30

# get the unique test combinations
combinations = []
for t in test:
    for l in lengths:
        combinations.append((t, l))

# multiply them by the number of replications
ordered_tests = n_tests * combinations

# randomize the tests
np.random.shuffle(ordered_tests)

print ordered_tests
