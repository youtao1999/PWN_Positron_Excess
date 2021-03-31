'''
    Tao You
    3/30/2021
    --This file takes in a mesh grid of pulsars of difference distances and ages, fit our model to each given distance
    and age, then produce a table of the resulting chisquare values, and finally produce a contour plot based upon these
    values.
'''

import AMS_model as model
print("############################")
print("Calculating chisquare values for given distance and age mesh grid")
print("############################")
print("")

import matplotlib.pyplot as pl
import numpy as np

# define necessary constants
kpc = 3.09e18
kyr = 1000*365*24*3600

# creating mesh grid
dist_grid = np.logspace(-1., 1.,100) #units in terms of kpc
age_grid = np.logspace(1., 4.,100) #units in terms of kyr
distance_vec = kpc*dist_grid #units in terms of m (for calculation)
age_vec = kyr*age_grid #units in terms of s (for calculation)
chisquare = np.zeros((len(distance_vec),len(age_vec)))

# make output directory
Output = "Tabular data of chi-square contour plot.txt"
outF = open(Output, "w")

# write output file
for t in range(len(distance_vec)):
    for u in range(len(age_vec)):
        chisquare[t,u] = model.function_calculatechi(distance_vec[t],age_vec[u])
        outF.write("%.3f "%(chisquare[t,u]))
    outF.write("\n")
outF.close()

############################
# Contour plot
fig = pl.figure(figsize=(8,6))
pl.rcParams['font.size'] = '18'
#dlog = ( log10(5.*np.nanmin(chisquare))-log10(np.nanmin(chisquare)) )/50.
#scale_vec = np.power( 10. , np.arange( log10(np.nanmin(chisquare)),log10(5.*np.nanmin(chisquare)), dlog ) )
#scale_cb = np.power( 10. , np.arange( log10(np.nanmin(chisquare)),log10(5.*np.nanmin(chisquare)), dlog*10. ) )
dlog = ( 5.*np.nanmin(chisquare)-np.nanmin(chisquare) )/100.
scale_vec = np.arange( np.nanmin(chisquare),5.*np.nanmin(chisquare), dlog )
scale_cb = np.arange( np.nanmin(chisquare),5.*np.nanmin(chisquare), dlog*10.)
#print scale_vec
pl.contourf(age_grid, dist_grid, chisquare, 100, levels=list(scale_vec), cmap='hot')
#im = plt.imshow(table, interpolation='nearest', cmap='hot')
#plt.contour(ra_vec, dec_vec, table, colors='black')
pl.colorbar(ticks=scale_cb)
#plt.colorbar()
pl.ylabel(r'$d$ [kpc]', fontsize=18)
pl.xlabel(r'$T$ [kyr]', fontsize=18)
pl.axis([age_vec[0],age_vec[len(age_vec)-1],distance_vec[0],distance_vec[len(distance_vec)-1]])
pl.xticks(fontsize=16)
pl.yticks(fontsize=16)
pl.grid(True)
pl.yscale('log')
pl.xscale('log') 
pl.legend(loc=2,prop={'size':15},numpoints=1, scatterpoints=1, ncol=2)
fig.tight_layout(pad=0.5)
pl.savefig("Contour_dage.pdf")
pl.savefig("Contour_dage.png")