import matplotlib.pyplot as pl
import numpy as np
import os

# Enter into data directory
os.chdir("Cross section upperlimit vs dark matter mass")

# Specify channel plot
channel = 16
# Produce sigma_v vs chisq plot
# load data
data = np.loadtxt("Sigma_v_upperlim_vs_mass_channel%d.txt"%channel)
mass = data[:,0]
sigma_v = data[:,1]

# Boundary index for the plot
upperbound = 16
lowerbound = 11
fig = pl.figure(figsize=(8, 6))
pl.rcParams['font.size'] = '18'
pl.plot(mass, sigma_v, lw=1.3, ls='-', color="blue", label='Channel = 16')
pl.ylabel(r'$\chi^2$', fontsize=18)
pl.xlabel(r'$\sigma_v$', fontsize=18)
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