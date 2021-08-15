'''
    Tao You
    5/13/2021
    --This file contains upperlimit try-outs for modeling dark matter contribution
    to AMS positron excess
'''

import matplotlib.pyplot as pl
from scipy.interpolate import interp1d
import numpy as np
import os
import shutil

# Specify channel plot
channel = 14

# stats.norm.isf( 1 - stats.chi2.cdf(6., 2) )
# 6. is the delta chisq between the null hypothesis and the minimum chisq
def upperlim_sigma(chi_vec, sigma_vec):
    likelihood_scan = -chi_vec
    like_zero = likelihood_scan
    index_max = np.argsort(-likelihood_scan)[0]
    likelihood_max = likelihood_scan[index_max]
    TS =  -2.*(like_zero-likelihood_max)
    f_ul = interp1d(likelihood_scan[index_max:], sigma_vec[index_max:], kind='linear')
    if likelihood_max-2.71 > -(likelihood_max-likelihood_scan[index_max:].min()):
        sigmavUL = f_ul(likelihood_max-2.71)
    else:
        print('WARNING')
        sigmavUL = -(likelihood_max-likelihood_scan[index_max:].min())
        print(sigmavUL)
    return sigmavUL

os.chdir("sigma_v vs chisquare")
os.chdir("sigma_v vs chisquare%d"%channel)

mass_arr = np.logspace(1.0, 4.0, 30) # GeV
upperlim_arr = np.zeros(len(mass_arr))

for j, mass in enumerate(mass_arr):
    table = np.loadtxt("sigma_v_vs_chisquare_mass=%d.txt" % mass_arr[j])
    sigma_vec = table[:,0]
    chi_vec = table[:,1]
    upperlim_arr[j] = upperlim_sigma(chi_vec, sigma_vec)

# Exit data directory
os.chdir("../../")
# Output files
# Check to see if the output file already exists
if os.path.isdir("Cross section upperlimit vs dark matter mass"):
    shutil.rmtree("Cross section upperlimit vs dark matter mass")

# make output directory
Output = "Cross section upperlimit vs dark matter mass"
os.mkdir(Output)
os.chdir(Output)

# Produce output data
outF = open("Sigma_v_upperlim_vs_mass_channel%d.txt"%channel, "w")
for u, upperlim in enumerate(upperlim_arr):
    outF.write("%.3f %.3f \n"%(mass_arr[u], upperlim))
outF.close()

# Produce sigma_v vs chisq plot

# Boundary index for the plot
upperbound = 16
lowerbound = 11
fig = pl.figure(figsize=(8, 6))
pl.rcParams['font.size'] = '18'
pl.plot(mass_arr[1:], np.power(10,upperlim_arr[1:]), lw=1.3, ls='-', color="blue", label='Channel = %d'%channel)
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
pl.savefig("sigma_upperlim_vs_mass-channel%d.png"%channel)

# exit "Output" directory
os.chdir("../")
