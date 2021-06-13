import matplotlib.pyplot as pl
import numpy as np
import os
# from AMS_DM_Upperlimits import mass_arr
# Specify channel plot
channel = 16

# Enter into data directory
os.chdir("sigma_v vs chisquare")
os.chdir("sigma_v vs chisquare%d"%channel)

# Produce sigma_v vs chisq plot
mass_arr = np.logspace(1.0, 4.0, 30) # GeV

for mass in mass_arr:
    data = np.loadtxt("sigma_v_vs_chisquare_mass=%d.txt"%mass)
    sigma_v = data[:,0]
    chisq = data[:,1]

    # Boundary index for the plot
    upperbound = 16
    lowerbound = 11
    fig = pl.figure(figsize=(8, 6))
    pl.rcParams['font.size'] = '18'
    pl.plot(np.power(10,sigma_v), chisq, lw=1.3, ls='-', color="blue", label='Mass = 100. GeV')
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
    pl.savefig("sigma_v_vs_chisquare_mass=%d.png"%mass)

# exit "Output" directory
os.chdir("../")
os.chdir("../")
