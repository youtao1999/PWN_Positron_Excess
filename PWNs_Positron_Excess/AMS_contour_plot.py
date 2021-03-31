'''
    Tao You
    3/31/2021
    --This file take in given tabular data from a txt file (a contour grid of chisquare values depending on distance and
    ages of pulsars, smooth out the data and produce contour plot.
'''

import numpy as np
import matplotlib.pyplot as pl

# load data
table = np.loadtxt("Tabular data of chi-square contour plot (complete).txt")

# smooth out data
chisquare = table

# define necessary constants
kpc = 3.09e18
kyr = 1000*365*24*3600

# creating mesh grid
dist_grid = np.logspace(-1., 1.,100) #units in terms of kpc
age_grid = np.logspace(1., 4.,100) #units in terms of kyr
distance_vec = kpc*dist_grid #units in terms of m (for calculation)
age_vec = kyr*age_grid #units in terms of s (for calculation)

print(type(chisquare))
# Contour plot
fig = pl.figure(figsize=(8,6))
pl.rcParams['font.size'] = '18'
#dlog = ( log10(5.*np.nanmin(chisquare))-log10(np.nanmin(chisquare)) )/50.
#scale_vec = np.power( 10. , np.arange( log10(np.nanmin(chisquare)),log10(5.*np.nanmin(chisquare)), dlog ) )
#scale_cb = np.power( 10. , np.arange( log10(np.nanmin(chisquare)),log10(5.*np.nanmin(chisquare)), dlog*10. ) )
dlog = ( 5.*np.min(chisquare)-np.min(chisquare) )/100.
scale_vec = np.arange( np.min(chisquare),5.*np.min(chisquare), dlog )
scale_cb = np.arange( np.min(chisquare),5.*np.min(chisquare), dlog*10.)
pl.contourf(age_grid, dist_grid, chisquare, levels=scale_vec, cmap='hot')
pl.colorbar(ticks=scale_cb)
#plt.colorbar()
pl.ylabel(r'$d$ [kpc]', fontsize=18)
pl.xlabel(r'$T$ [kyr]', fontsize=18)
pl.axis([age_grid[0],age_grid[len(age_grid)-1],dist_grid[0],dist_grid[len(dist_grid)-1]])
pl.xticks(fontsize=16)
pl.yticks(fontsize=16)
pl.grid(True)
pl.yscale('log')
pl.xscale('log')
pl.legend(loc=2,prop={'size':15},numpoints=1, scatterpoints=1, ncol=2)
fig.tight_layout(pad=0.5)
pl.savefig("Contour_dage.pdf")
pl.savefig("Contour_dage.png")
print(np.shape(scale_cb))