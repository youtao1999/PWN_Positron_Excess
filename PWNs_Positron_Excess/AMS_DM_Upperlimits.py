'''
    Tao You
    5/13/2021
    --This file contains upperlimit try-outs for modeling dark matter contribution
    to AMS positron excess
'''

import numpy as np
import math
import AMS_DM_model as DM
import AMS_model as PWN
from iminuit import Minuit
import os
import shutil
import matplotlib.pyplot as pl

# Determine the channels we would like to fit
#e 7, mu 10, tau 13, bb 16, tt 17, WW 20, ZZ 23, gamma 25, h 26
channel_arr = np.array([16])

# Make array of cross sections
sigma_arr = np.logspace(10^-29,10^20,5)
sigma_exp_arr = np.log10(sigma_arr)

# Define necessary constants for PWN component
kpc = 3.09e18
kyr = 1000*365*24*3600
E_range = np.power(10.,np.arange(math.log10(1.),math.log10(1e5)+0.1,0.1)) #GeV
d = 0.5*kpc #kpc
T = 100*kyr #age in kyrs
E_c = 1000 #cutoff energy in GeV
gamma = 1.8
tau_0 = 10*kyr #kyr
E_dot = 1e35 #spin-down luminosity in erg/s
eta = 0.1 # flux conversion rate
delta = 0.408
k_0 = kpc**2*0.0967/(1.0e6*365*24*3600)
alpha = 2
b_0 = 1e-16

# Extract AMS-02 data
tabledata = np.loadtxt('positron_ams02_19.dat')
pos=tabledata[:,6]*np.power(tabledata[:,2],3.)*tabledata[:,13]/1.e4 #rescale the flux into 1/GeV/cm$^2$/s/sr
epos= tabledata[:,2] #flux in 1/GeV/m$^2$/s/sr
x_errorup_pos_val = tabledata[:,1]
x_errordown_pos_val = tabledata[:,0]
x_errorup_pos = np.subtract(x_errorup_pos_val, epos)
x_errordown_pos = np.subtract(epos, x_errordown_pos_val)
errortot_pos = np.power(tabledata[:,2],3.)*np.sqrt(tabledata[:,7]*tabledata[:,7]+tabledata[:,12]*tabledata[:,12])*tabledata[:,13]/1.e4

# Define function to be minimized
def f_BPL_DM(par0,par1,par2,par3,par4):
    eta = pow(10., par0)
    gamma = par1
    normalization_sec = par2
    DM_mass = par3
    sigma_v_normalization = math.pow(10,par4)
    chisq = 0
    for t in range(24, len(epos), 1):
        model_PWN = PWN.flux(epos[t], eta, b_0, d, E_c, gamma, T)
        model_DM = np.power(epos[t],3.)*sigma_v_normalization*DM.DM_spectrum(epos[t], DM_mass, DMchannel)[0]
        model_sec = PWN.flux_secondary(epos[t], normalization_sec)
        model_tot = model_PWN + model_sec + model_DM
        chisq = chisq + np.power((model_tot - pos[t]) / errortot_pos[t], 2.)
    return chisq

# Define an empty array of chisquare values from the the cross section array
chisquare_arr = np.zeros((len(channel_arr), len(sigma_exp_arr)))

for i, DMchannel in enumerate(channel_arr):
    for j, sigma_exp in enumerate(sigma_exp_arr):
        m = Minuit(f_BPL_DM, par0=0.0, par1=1.5, par2=1.2, par3=200., par4=sigma_exp)
        m.errors['par0'] = 1e-4
        m.errors['par1'] = 1e-4
        m.errors['par2'] = 1e-4
        m.errors['par3'] = 1e-4
        m.errors['par4'] = 1e-4
        m.limits['par0'] = (-10.0, 10.)
        m.limits['par1'] = (0., 5.)
        m.limits['par2'] = (0.1, 10.)
        m.limits['par3'] = (1e1, 1e4)
        m.limits['par4'] = (-3, 6)
        m.errordef = 1
        m.migrad()
        # print('value', m.values)
        # print('error', m.errors)
        # print('fval', m.fval)
        chisquare_arr[i,j] = m.fval

print(chisquare_arr)

# Define function that finds the upperlimit
def upperlimit(chisq_arr):
    '''
    Upperlimit defined as the sigma_v that causes a worsening of the chisquare values of >= 2.7 from the
    minimum. This function takes in an array of sigma_v's, a corresponding array of chisquare values (which
    is two dimensional accounting for both the dark matter channel as well as sigma_v's, finds the minimum of
    chisquare value for each channel and returns an array of chisquare value minimums
    '''
    # chi_arr is the array of chisquare values for single channel
    min_chisq = min(chisq_arr)
    lim_index = np.argwhere(chisq_arr >= min_chisq + 2.7)[0,0]
    return lim_index

# So we eventually end up with one upper limit for each channel?