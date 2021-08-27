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
from scipy.interpolate import interp1d
import matplotlib.pyplot as pl
import time_package

directory = {'bbbar_16': 16, 'ee-_7': 7, 'mumu_10': 10, 'tautau_13': 13, 'tt_17': 17, 'WW_20': 20, 'cc': 14, 'HH': 26}

# necessary functions
def get_keys_from_value(d, val):
    # the following function is copied and pasted from the website: https://note.nkmk.me/en/python-dict-get-key-from-value/
    return [k for k, v in d.items() if v == val]

# Determine the channels we would like to fit
#e 7, mu 10, tau 13, bb 16, tt 17, WW 20, ZZ 23, gamma 25, h 26
channel_arr = np.array([14])

# Make array of cross sections and masses
mass_arr = np.logspace(1.0, 4.0, 30) # GeV
sigma_arr = np.logspace(-29.0 ,-20.0 ,60)
sigma_exp_arr = np.log10(sigma_arr)

# make general output directory for all the channels
os.chdir('../')
os.mkdir('Upperlimit data')
os.chdir('PWN_Positron_Excess')

def sigma_vs_chisq(channel, sigma_exp_arr, mass_arr):
    # this function outputs data directory that contains all the chisq vs cross section for every single mass for the
    # given dark matter channel

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
        flux_array = np.loadtxt("fluxtot_gamma%d.txt" % i)
        z[i, :] = flux_array[:, 1]

    os.chdir('../Upperlimit data')
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
            model_DM = np.power(epos[t],3.)*sigma_v_normalization*DM.DM_spectrum(epos[t], DM_mass, channel)[0]/np.power(10.,-26)
            model_sec = PWN.flux_secondary(epos[t], normalization_sec)
            model_tot = model_PWN + model_sec + model_DM
            chisq = chisq + np.power((model_tot - pos[t]) / errortot_pos[t], 2.)
        return chisq

    # Define an empty array of chisquare values from the the cross section array
    chisquare_arr = np.zeros((len(mass_arr), len(sigma_exp_arr)))
    upperlim_arr = np.zeros(len(mass_arr))

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
            m.limits['par4'] = (1., 100000.)
            m.fixed['par3'] = True
            m.fixed['par4'] = True
            m.errordef = 1
            m.migrad()
            print('value', m.values)
            print('error', m.errors)
            print('fval', m.fval)
            chisquare_arr[j, k] = m.fval

    print(chisquare_arr)

    # Output files
    # Exit current directory which contains codes -- for the purpose of backup to GitHub -- we don't want to back up the data
    # since it is too big for GitHub, therefore we are going to exit the current directory and establish a new one in parallel
    # with the repo so that everytime we commit to GitHub, the data stays local

    os.mkdir(get_keys_from_value(directory, channel)[0])
    os.chdir(get_keys_from_value(directory, channel)[0])
    os.mkdir("sigma_v vs chisquare")
    os.chdir("sigma_v vs chisquare")

    for j, mass in enumerate(chisquare_arr[i]):
        outF = open("sigma_v_vs_chisquare_mass=%d.txt" % mass_arr[j], "w")
        for k, chisq in enumerate(chisquare_arr[i, j]):
           outF.write("%.3f %.3e \n"%(sigma_exp_arr[k], chisq))
        outF.close()
    os.chdir("../../")
    # ending working directory is upperlimit
    return chisquare_arr

def upperlimit(channel, mass_arr):

    # this function takes in all the chisq vs sigma data calculated from the previous function and calculates and plots
    # one single upperlimit chisq value for each mass and store it in the "Cross section upperlimit vs dark matter mass"
    # folder
    def upperlim_sigma(chi_vec, sigma_vec):
        likelihood_scan = -chi_vec
        like_zero = likelihood_scan
        index_max = np.argsort(-likelihood_scan)[0]
        likelihood_max = likelihood_scan[index_max]
        TS = -2. * (like_zero - likelihood_max)
        f_ul = interp1d(likelihood_scan[index_max:], sigma_vec[index_max:], kind='linear')
        if likelihood_max - 2.71 > -(likelihood_max - likelihood_scan[index_max:].min()):
            sigmavUL = f_ul(likelihood_max - 2.71)
        else:
            print('WARNING')
            sigmavUL = -(likelihood_max - likelihood_scan[index_max:].min())
            print(sigmavUL)
        return sigmavUL

    # entering output directory in order to extract sigma vs chisq data
    channel_file_name = get_keys_from_value(directory, channel)[0]
    os.chdir(channel_file_name + '/sigma_v vs chisquare')

    # initializing upperlimit array
    upperlim_arr = np.zeros(len(mass_arr))

    # calculate upperlim_arr
    for j, mass in enumerate(mass_arr):
        table = np.loadtxt("sigma_v_vs_chisquare_mass=%d.txt" % mass_arr[j])
        sigma_vec = table[:,0]
        chi_vec = table[:,1]
        upperlim_arr[j] = upperlim_sigma(chi_vec, sigma_vec)

    # Exit data directory
    os.chdir("../")

    # Output files
    # Check to see if the output file already exists
    if os.path.isdir("Cross section upperlimit vs dark matter mass"):
        shutil.rmtree("Cross section upperlimit vs dark matter mass")

    # make output directory
    Output = "Cross section upperlimit vs dark matter mass"
    os.mkdir(Output)
    os.chdir(Output)

    # Produce output data
    outF = open("Sigma_v_upperlim_vs_mass_channel%d.txt" % channel, "w")
    for u, upperlim in enumerate(upperlim_arr):
        outF.write("%.3f %.3f \n" % (mass_arr[u], upperlim))
    outF.close()

    # Produce sigma_v vs chisq plot

    # Boundary index for the plot
    upperbound = 16
    lowerbound = 11
    fig = pl.figure(figsize=(8, 6))
    pl.rcParams['font.size'] = '18'
    pl.plot(mass_arr[1:], np.power(10, upperlim_arr[1:]), lw=1.3, ls='-', color="blue", label='Channel = %d' % channel)
    pl.xlabel('$Mass/GeV$', fontsize=18)
    pl.ylabel(r'$\sigma_v$', fontsize=18)
    # pl.axis([np.power(10, sigma_v)[lowerbound], np.power(10, sigma_v)[upperbound], chisq[lowerbound], chisq[upperbound]])
    pl.xticks(fontsize=18)
    pl.yticks(fontsize=18)
    pl.tick_params('both', length=7, width=2, which='major')
    pl.tick_params('both', length=5, width=2, which='minor')
    pl.grid(True)
    pl.yscale('log')
    pl.xscale('log')
    pl.legend(loc=2, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=2)
    fig.tight_layout(pad=0.5)
    pl.savefig("sigma_upperlim_vs_mass-channel%d.png" % channel)

    # exit "Output" directory
    os.chdir("../")

    return upperlim_arr

# the actual call for the function that produces the data
for channel in channel_arr:
    single_channel_sigma_vs_chsiq = sigma_vs_chisq(channel, sigma_exp_arr, mass_arr)
    single_channel_upperlim_arr = upperlimit(channel, mass_arr)

# exit upperlimit data directory
os.chdir('../')