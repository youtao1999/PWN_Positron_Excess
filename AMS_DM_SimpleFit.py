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
directory = {r'$b\bar{b}$': 16, r'$e^+e^-$': 7, r'$\mu^+\mu^-$': 10, r'$\tau^+\tau^-$': 13, r'$t^+t^-$': 17, r'$\omega^+\omega^-$': 20, r'c\bar{c}': 14,
                 r'$h^+h^-$': 26}
channel_name_dict = {y: x for x, y in directory.items()}
channel_arr = np.array([16, 7, 10, 13, 17, 20, 14, 26])

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

##############################################################

# Now, we produce plots based upon our best fit values

# Check to see if the output file already exists
if os.path.isdir("Multichannel dark matter annihilation fitting results"):
    shutil.rmtree("Multichannel dark matter annihilation fitting results")

# make output directory
Output = "Multichannel dark matter annihilation fitting results"
os.mkdir(Output)
os.chdir(Output)

for DMchannel in channel_arr:
    m = Minuit(f_BPL_DM, par0=0.0, par1=1.5, par2=1.2, par3=200., par4=1.)
    m.errors['par0'] = 1e-4
    m.errors['par1'] = 1e-4
    m.errors['par2'] = 1e-4
    m.errors['par3'] = 1e-4
    m.errors['par4'] = 1e-4
    m.limits['par0'] = (-10.0, 10.)
    m.limits['par1'] = (0., 5.)
    m.limits['par2'] = (0.1, 10.)
    m.limits['par3'] = (1e1, 1e4)
    m.limits['par4'] = (-3,6)
    m.errordef = 1
    m.migrad()
    print('value', m.values)
    print('error', m.errors)
    print('fval', m.fval)

    # Preparing for the plot

    model_vec_tot = np.zeros(len(epos))
    model_vec_sec = np.zeros(len(epos))
    model_vec_pulsar = np.zeros(len(epos))
    model_vec_dm = np.zeros(len(epos))

    for t in range(len(epos)):
        model_vec_pulsar[t] = PWN.flux(epos[t], np.power(10, m.values["par0"]), b_0, d, E_c, m.values["par1"], T)
        model_vec_sec[t] = PWN.flux_secondary(epos[t], m.values["par2"])
        model_vec_dm[t] = np.power(epos[t],3.)*math.pow(10,m.values["par4"])*DM.DM_spectrum(epos[t], m.values["par3"],DMchannel)[0]
        model_vec_tot[t] = model_vec_pulsar[t] + model_vec_sec[t] + model_vec_dm[t]

    fig = pl.figure(figsize=(8, 6))
    pl.rcParams['font.size'] = '18'
    pl.plot(epos, model_vec_tot, lw=1.3, ls='-', color="blue", label='PWN+SEC+DM')
    pl.plot(epos, model_vec_pulsar, lw=2.0, ls='-.', color="green", label='PWN')
    pl.plot(epos, model_vec_sec, lw=2.0, ls='--', color="red", label='Secondary')
    pl.plot(epos, model_vec_dm, lw=2.0, ls=':', color="cyan", label=channel_name_dict[DMchannel])
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

# exit "Output" directory
os.chdir("../")