'''
    Tao You
    5/13/2021
    --This file contains upperlimit try-outs for modeling dark matter contribution
    to AMS positron excess
'''

import numpy as np
import os

# Define function that finds the upperlimit
def upperlimindex(chisq_arr):
    '''
    Upperlimit defined as the sigma_v that causes a worsening of the chisquare values of >= 2.7 from the
    minimum. This function takes in an array of sigma_v's, a corresponding array of chisquare values (which
    is two dimensional accounting for both the dark matter channel as well as sigma_v's, finds the minimum of
    chisquare value for each channel and returns an array of chisquare value minimums
    '''
    # chi_arr is the array of chisquare values for single channel
    index_min = np.argsort(chisq_arr)[0]
    print(index_min)
    index_before = index_min - 1
    index_after = index_min + 1

    # Calculate parabola from these three points

    min_chisq = min(chisq_arr)
    proxy = np.argwhere(chisq_arr >= min_chisq + 2.7)

    if len(proxy) > 0:
        lim_index = proxy[0, 0]
        return lim_index
    else:
        return np.argwhere(chisq_arr == max(chisq_arr))[0,0]

os.chdir("sigma_v vs chisquare")
os.chdir("sigma_v vs chisquare16")
table = np.loadtxt('sigma_v_vs_chisquare_mass=3039.txt')
chisquare_arr = table[:,1]
sigma_arr = table[:,0]
print(np.shape(chisquare_arr))
print(upperlimindex(chisquare_arr))
print(sigma_arr[upperlimindex(chisquare_arr)])