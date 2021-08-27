'''
For each channel, this python script loops through the chisq vs cross section data for each mass,
calculates the delta chisq for this mass, and from this delta chisq calculates the statistical significance
for this mass value. Plot the statistical significance vs mass for each channel.

work flow:
    1. Loop through each chisq vs cross section data file
    2. Calculate delta chisq for each mass
    3. Calculate statistical significance for each mass
    4. Store the data of mass vs statistical significance for this channel
    5. Output the plot for mass vs statistical significance for this channel, do the same for all channels.

Some additional things that need to be taken care of:

1. Fix the Middlebury VPN login issue so that I can remote acccess 525 machines
2. Once the VPN is fixed, then learn the screening code from Chris and start recalculating the upperlimit values
3. Start thinking about making code directories and output directories separate, so that the code directory can be
smoothly backed up and accessed through GitHub without any files becoming too large, and the output data directory needs
to be linux friendly since the output directory needs to be stored on linux machines on 525.
'''

import matplotlib.pyplot as pl
import numpy as np
import os
from scipy import stats
import re

# this code copied and pasted from user136036's answer to stackoverflow:
# https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir
# serves to sort the list of datafiles in alphanumeric order
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

# necessary functions
def get_keys_from_value(d, val):
    # the following function is copied and pasted from the website: https://note.nkmk.me/en/python-dict-get-key-from-value/
    return [k for k, v in d.items() if v == val]

channel_arr = np.array([7, 10, 13, 17, 20])
mass_arr = np.logspace(1.0, 4.0, 30) # GeV

directory = {'bbbar_16': 16, 'ee-_7': 7, 'mumu_10': 10, 'tautau_13': 13, 'tt_17': 17, 'WW_20': 20}

os.chdir('../Upperlimit data')

def deltachisq(channel, mass_arr):
    # this function takes in the chisq vs sigma_v data for each mass value, calculates the deltachisq for this mass
    # output data directory called "statistical significance" containing the data in txt files as well as the plot
    # against mass
    channel_file_name = get_keys_from_value(directory, channel)[0]
    os.chdir(channel_file_name + '/sigma_v vs chisquare')
    listOfDataFiles = os.listdir()
    sig_array = np.zeros(len(listOfDataFiles))
    listOfDataFiles = sorted_alphanumeric(listOfDataFiles)
    for fileIndex, dataFile in enumerate(listOfDataFiles):
        data = np.loadtxt(dataFile)
        sigma_v = data[:, 0]
        chisq = data[:, 1]
        deltachisq = abs(chisq[0] - min(chisq))
        significance = stats.norm.isf(1 - stats.chi2.cdf(deltachisq, 2))
        sig_array[fileIndex] = significance
    # exit sigma v vs chisquare directory, back to channel name directory
    os.chdir('../')

    # make output directory
    os.mkdir('statistical significance')
    os.chdir('statistical significance')
    # Produce output data
    outF = open("stat_sig_vs_mass_channel%d.txt", "w")
    for i, significance in enumerate(sig_array):
        outF.write("%.3f %.3f \n"%(mass_arr[i], significance))
    outF.close()

    # reversing key value pairs in dictionary to label the plots
    channel_name_dict = {y: x for x, y in directory.items()}

    directory = {'bbbar_16': 16, 'ee-_7': 7, 'mumu_10': 10, 'tautau_13': 13, 'tt_17': 17, 'WW_20': 20}

    # Boundary index for the plot
    fig = pl.figure(figsize=(8, 6))
    pl.rcParams['font.size'] = '18'
    pl.plot(mass_arr, sig_array, lw=1.3, ls='-', color="blue", label='Channel = ' + channel_name_dict[channel])
    pl.xlabel('$Mass/GeV$', fontsize=18)
    pl.ylabel(r'$\sigma$', fontsize=18)
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
    pl.savefig("statistical_significance_vs_mass-channel%d.png"%channel)

    # exit statistical significance directory, back to channel filename directory
    os.chdir('../')

    # display current directory
    print(os.getcwd())

    # exit channel filename directory, back to upperlimit data directory
    os.chdir('../')

    return sig_array

# Loop through all the channels to produce the results
for channel in channel_arr:
    deltachisq_arr = deltachisq(channel, mass_arr)

# exit Upperlimit data directory, switch back to PWNs_positron_excess
os.chdir('../PWN_Positron_Excess')

#r'$b\bar{b}$'
# r'$c\bar{c}$'
# r'$e^+e^-$’
# r’$\tau^+\tau^-$’
# r’$\mu^+\mu^-$’
# Linearize y-axis and make the range from 0-5
