'''
    Tao You
    5/13/2021
    --This file test the upperlimit method for a single dark matter mass
'''

import numpy as np
import math
import AMS_DM_model as DM
from iminuit import Minuit
import os
import shutil
import time_package
import matplotlib.pyplot as pl

# Determine the channels we would like to fit
#e 7, mu 10, tau 13, bb 16, tt 17, WW 20, ZZ 23, gamma 25, h 26
channel_arr = np.array([13])

# Make array of cross sections and masses
mass_arr = np.logspace(1.0, 4.0, 1) # GeV
sigma_arr = np.logspace(-29.0 ,20.0 ,5)
sigma_exp_arr = np.log10(sigma_arr)

# Extract AMS-02 data
tabledata = np.loadtxt('positron_ams02_19.dat')
pos=tabledata[:,6]*np.power(tabledata[:,2],3.)*tabledata[:,13]/1.e4 #rescale the flux into 1/GeV/cm$^2$/s/sr
epos= tabledata[:,2] #flux in 1/GeV/m$^2$/s/sr
x_errorup_pos_val = tabledata[:,1]
x_errordown_pos_val = tabledata[:,0]
x_errorup_pos = np.subtract(x_errorup_pos_val, epos)
x_errordown_pos = np.subtract(epos, x_errordown_pos_val)
errortot_pos = np.power(tabledata[:,2],3.)*np.sqrt(tabledata[:,7]*tabledata[:,7]+tabledata[:,12]*tabledata[:,12])*tabledata[:,13]/1.e4

# Load ATNF PWN data
ATNF = np.loadtxt('ATNF_model_data.txt')
model_vec_pulsar = ATNF[:, 1]
model_vec_sec = ATNF[:, 2]
model_vec_tot = ATNF[:, 3]

# Define function to be minimized
def f_BPL_DM(par3, par4):
    DM_mass = par3
    sigma_v_normalization = np.power(10, par4)
    chisq = 0
    for t in range(24, len(epos), 1):
        model_PWN = model_vec_pulsar[t]
        model_DM = sigma_v_normalization*DM.DM_spectrum(epos[t], DM_mass, 13)[0]
        model_sec = model_vec_sec[t]
        model_tot = model_PWN + model_sec + model_DM
        chisq = chisq + np.power((model_tot - pos[t]) / errortot_pos[t], 2.)
    return chisq

# Define an empty array of chisquare values from the the cross section array
chisquare_arr = np.zeros((len(channel_arr), len(mass_arr), len(sigma_exp_arr)))
upperlim_arr = np.zeros((len(channel_arr), len(mass_arr)))

for i, DMchannel in enumerate(channel_arr):
    for j, mass in enumerate(mass_arr):
        for k, sigma_exp in enumerate(sigma_exp_arr):
            m = Minuit(f_BPL_DM, par3=mass, par4=sigma_exp)
            m.errors['par3'] = 1e-4
            m.errors['par4'] = 1e-4
            m.limits['par3'] = (1e1, 1e4)
            m.limits['par4'] = (-3, 6)
            m.errordef = 1
            m.migrad()
            # print('value', m.values)
            # print('error', m.errors)
            print('fval', m.fval)
            chisquare_arr[i, j, k] = m.fval

print(chisquare_arr)

# # Output files
#
# # Check to see if the output file already exists
# if os.path.isdir("sigma_v vs chisquare"):
#     shutil.rmtree("sigma_v vs chisquare")
#
# # make output directory
# Output = "sigma_v vs chisquare"
# os.mkdir(Output)
# os.chdir(Output)
#
# for i, channel in enumerate(chisquare_arr):
#     outF = open("sigma_v_vs_chisquare%d.txt"%channel_arr[i], "w")
#     for j, mass in enumerate(chisquare_arr[i]):
#         for k, chisq in enumerate(chisquare_arr[i,j]):
#             outF.write("%.3f %.3e \n"%(sigma_exp_arr[k], chisq))
#     outF.close()
#
# # exit "Output" directory
#
# os.chdir("../")