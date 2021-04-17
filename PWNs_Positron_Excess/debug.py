'''
    Tao You
    4/14/2021
    --This file calculates a simple multichannel fit of positron product from dark matter annihilation to the AMS-02 data
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
channel_arr = np.array([16, 10, 13])

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


##############################################################

# Now, we produce plots based upon our best fit values

DMchannel = 16

par0=3.978098565255518
par1=3.125231085769756
par2=1.1255795776199276
par3=259.40975349834264
par4=1.4217021875190277

# Preparing for the plot

model_vec_tot = np.zeros(len(epos))
model_vec_sec = np.zeros(len(epos))
model_vec_pulsar = np.zeros(len(epos))
model_vec_dm = np.zeros(len(epos))

for t in range(len(epos)):
    model_vec_pulsar[t] = PWN.flux(epos[t], np.power(10, par0), b_0, d, E_c, par1, T)
    model_vec_sec[t] = PWN.flux_secondary(epos[t], par2)
    model_vec_dm[t] = np.power(epos[t],3.)*math.pow(10,par4)*DM.DM_spectrum(epos[t], par3,DMchannel)[0]
    model_vec_tot[t] = model_vec_pulsar[t] + model_vec_sec[t] + model_vec_dm[t]

fig = pl.figure(figsize=(8, 6))
pl.rcParams['font.size'] = '18'
pl.plot(epos, model_vec_tot, lw=1.3, ls='-', color="blue", label='PWN+SEC+DM')
pl.plot(epos, model_vec_pulsar, lw=2.0, ls='-.', color="green", label='PWN')
pl.plot(epos, model_vec_sec, lw=2.0, ls='--', color="red", label='Secondary')
pl.plot(epos, model_vec_dm, lw=2.0, ls=':', color="cyan", label='DM')
pl.errorbar(epos, pos, xerr=[x_errordown_pos, x_errorup_pos], yerr=errortot_pos, fmt='.', color="black",
            label="AMS-02 $e^+$")
pl.ylabel(r'$E^3 \Phi_e$ [GeV$^2$/cm$^2$/s/sr]', fontsize=18)
pl.xlabel(r'$E_e$ [GeV]', fontsize=18)
pl.axis([1., 5.e3, 5e-6, 1e-2])
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.tick_params('both', length=7, width=2, which='major')
pl.tick_params('both', length=5, width=2, which='minor')
pl.grid(True)
pl.yscale('log')
pl.xscale('log')
pl.legend(loc=2, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=2)
fig.tight_layout(pad=0.5)
# pl.savefig("ATNF_pulsar_flux.pdf")
pl.savefig("DM_multichannel_%s.png"%DMchannel)
