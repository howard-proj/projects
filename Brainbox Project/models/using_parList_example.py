from parList import (
    rand_par_list,
    lin_par_list
)

'''
The output of the functions is a list of lists of paramaters.

The paramaters order is [windowSize, sampleProportion, cynthiaWindow, threshold]
'''

iterations = 4
random_list = rand_par_list(iterations)
linear_list = lin_par_list(iterations)

print(random_list)
print(linear_list)
