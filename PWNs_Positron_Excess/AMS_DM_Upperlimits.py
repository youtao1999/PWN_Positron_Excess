'''
    Tao You
    5/13/2021
    --This file contains upperlimit try-outs for modeling dark matter contribution
    to AMS positron excess
'''

import numpy as np
import AMS_DM_model as DM
import AMS_model as PWN
from iminuit import Minuit
import os
import shutil
from scipy.interpolate import interp2d
import matplotlib.pyplot as pl
import time_package


# Determine the channels we would like to fit
#e 7, mu 10, tau 13, bb 16, tt 17, WW 20, ZZ 23, gamma 25, h 26
channel_arr = np.array([16])

# Make array of cross sections and masses
mass_arr = np.logspace(1.0, 4.0, 2) # GeV
# mass_arr = np.array([1000.])
sigma_arr = np.logspace(-29.0 ,-20.0 ,5)
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

# Interpolating 2D x = epos, y = gamma_array, z = totalflux
gamma_array = np.arange(1.4, 2.4, 0.1)
x = epos
y = gamma_array
z = np.zeros((len(gamma_array), len(epos)))

for i in range(len(gamma_array)):
    # os.chdir(Output)
    flux_array = np.loadtxt("fluxtot_gamma%d.txt" % i)
    z[i, :] = flux_array[:, 1]
    # os.chdir("../")

funcinterpolate = interp2d(x, y, z)

# Define function to be minimized
def f_BPL_DM(par0,par1,par2,par3,par4):
    eta = pow(10., par0)
    gamma = par1
    normalization_sec = par2
    DM_mass = par4
    sigma_v_normalization = np.power(10,par3)
    chisq = 0
    for t in range(24, len(epos), 1):
        model_PWN = eta * funcinterpolate(epos[t], gamma)
        model_DM = np.power(epos[t],3.)*sigma_v_normalization*DM.DM_spectrum(epos[t], DM_mass, DMchannel)[0]/np.power(10.,-26)
        model_sec = PWN.flux_secondary(epos[t], normalization_sec)
        model_tot = model_PWN + model_sec + model_DM
        chisq = chisq + np.power((model_tot - pos[t]) / errortot_pos[t], 2.)
    return chisq

# Define an empty array of chisquare values from the the cross section array
chisquare_arr = np.zeros((len(channel_arr), len(mass_arr), len(sigma_exp_arr)))
upperlim_arr = np.zeros((len(channel_arr), len(mass_arr)))


for i, DMchannel in enumerate(channel_arr):
    for j, mass in enumerate(mass_arr):
        for k, sigma_exp in enumerate(sigma_exp_arr):
            m = Minuit(f_BPL_DM, par0=0.06, par1=1.44, par2=1.26, par3=sigma_exp, par4=mass)
            m.errors['par0'] = 0.06
            m.errors['par1'] = 0.03
            m.errors['par2'] = 0.02
            m.errors['par3'] = 1e-4
            m.errors['par4'] = 1e-4
            m.limits['par0'] = (-10.0, 10.)
            m.limits['par1'] = (0., 5.)
            m.limits['par2'] = (0.1, 10.)
            m.limits['par3'] = (-29, 20)
            m.limits['par4'] = (95., 105.)
            m.fixed['par3'] = True
            m.fixed['par4'] = True
            m.errordef = 1
            m.migrad()
            print('value', m.values)
            print('error', m.errors)
            print('fval', m.fval)
            chisquare_arr[i, j, k] = m.fval

print(chisquare_arr)

# Define function that finds the upperlimit
def upperlimindex(chisq_arr):
    '''
    Upperlimit defined as the sigma_v that causes a worsening of the chisquare values of >= 2.7 from the
    minimum. This function takes in an array of sigma_v's, a corresponding array of chisquare values (which
    is two dimensional accounting for both the dark matter channel as well as sigma_v's, finds the minimum of
    chisquare value for each channel and returns an array of chisquare value minimums
    '''
    # chi_arr is the array of chisquare values for single channel
    min_chisq = min(chisq_arr)
    proxy = np.argwhere(chisq_arr >= min_chisq + 2.7)
    if len(proxy) > 0:
        lim_index = proxy[0, 0]
        return lim_index
    else:
        return np.argwhere(chisq_arr == max(chisq_arr))[0,0]

# Now traverse chisquare_arr to calculate upperlimits

for i, channel in enumerate(channel_arr):
    for j, mass in enumerate(mass_arr):
        chisq_arr = chisquare_arr[i, j]
        index = upperlimindex(chisq_arr)
        upperlim_arr[i, j] = sigma_arr[index]

print("Upperlimit sigma: ", upperlim_arr)

# Output files

# Check to see if the output file already exists
if os.path.isdir("Cross section upperlimit vs dark matter mass"):
    shutil.rmtree("Cross section upperlimit vs dark matter mass")

# make output directory
Output = "Cross section upperlimit vs dark matter mass"
os.mkdir(Output)
os.chdir(Output)

for i, channel in enumerate(channel_arr):
    outF = open("Sigma_v_upperlim_vs_mass_channel%d.txt"%channel, "w")
    for u, upperlim in enumerate(upperlim_arr[i]):
        outF.write("%.3f %.3e \n"%(mass_arr[u], upperlim))
    outF.close()

# exit "Output" directory

os.chdir("../")