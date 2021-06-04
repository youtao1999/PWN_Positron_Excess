import matplotlib.pyplot as pl
import numpy as np
import os

# Enter into data directory
os.chdir("sigma_v vs chisquare")

# Specify channel plot
channel = 13
# Produce sigma_v vs chisq plot
# load data
data = np.loadtxt("sigma_v_vs_chisquare%d.txt"%channel)
sigma_v = data[:,0]
chisq = data[:,1]

print(sigma_v, chisq)

fig = pl.figure(figsize=(8, 6))
pl.rcParams['font.size'] = '18'
pl.plot(np.power(10, sigma_v), chisq, lw=1.3, ls='-', color="blue", label='Mass = 100. GeV')
pl.ylabel(r'$\chi^2$', fontsize=18)
pl.xlabel(r'$\sigma_v$', fontsize=18)
pl.axis([np.power(10, sigma_v)[0], np.power(10, sigma_v)[-1], chisq[0], chisq[-1]])
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.tick_params('both', length=7, width=2, which='major')
pl.tick_params('both', length=5, width=2, which='minor')
pl.grid(True)
pl.yscale('log')
pl.xscale('log')
pl.legend(loc=2, prop={'size': 16}, numpoints=1, scatterpoints=1, ncol=2)
fig.tight_layout(pad=0.5)
pl.savefig("sigma_v_vs_chisquare%d.png"%channel)

# exit "Output" directory
os.chdir("../")