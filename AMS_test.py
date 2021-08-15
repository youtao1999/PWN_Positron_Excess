'''
    Tao You
    3/30/2021
    --This file merely produces the plot of given AMS-02 data
'''

print("############################")
print("Plotting given data")
print("############################")
print("")

import matplotlib.pyplot as pl
import numpy as np

#normalization of the secondary production
normalization_sec = 1.2

#take the AMS-02 data
#folder_data = r'''C:\Users\Sissi\ Chen\Desktop\Positron\'''
tabledata = np.loadtxt('positron_ams02_19.dat')
pos=tabledata[:,6]*np.power(tabledata[:,2],3.)*tabledata[:,13]/1.e4 #rescale the flux into 1/GeV/cm$^2$/s/sr
epos= tabledata[:,2] #flux in 1/GeV/m$^2$/s/sr
x_errorup_pos_val = tabledata[:,1]
x_errordown_pos_val = tabledata[:,0]
x_errorup_pos = np.subtract(x_errorup_pos_val, epos)
x_errordown_pos = np.subtract(epos, x_errordown_pos_val)
errortot_pos = np.power(tabledata[:,2],3.)*np.sqrt(tabledata[:,7]*tabledata[:,7]+tabledata[:,12]*tabledata[:,12])*tabledata[:,13]/1.e4

#Exctract secondary production
#folder_data = r'''C:\Users\Sissi\ Chen\Desktop\Positron\'''
tabledata_sec = np.loadtxt('positrons_med_sec.dat')
epos_sec=tabledata_sec[:,0]
pos_sec= normalization_sec*tabledata_sec[:,1]

# total flux figure
fig = pl.figure(figsize=(8,6))
pl.rcParams['font.size'] = '16'
pl.errorbar(epos, pos*np.power(epos,-3.), xerr= [x_errordown_pos, x_errorup_pos],yerr=errortot_pos*np.power(epos,-3.),fmt='.', color="black",label="AMS-02 $e^+$")
pl.plot(epos_sec, pos_sec*np.power(epos_sec,-3.), lw=2.0, ls='--', color="red", label='Secondary data')
pl.ylabel(r'$\Phi_e$ [1/GeV/cm$^2$/s/sr]', fontsize=18)
pl.xlabel(r'$E_e$ [GeV]', fontsize=18)
pl.axis([0.1,1e3,1e-10,1e-3])
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.tick_params('both', length=7, width=2, which='major')
pl.tick_params('both', length=5, width=2, which='minor')
pl.grid(True)
pl.yscale('log')
pl.xscale('log')
pl.legend(loc=1,prop={'size':16},numpoints=1, scatterpoints=1, ncol=1)
fig.tight_layout(pad=0.5)
pl.savefig("data_totalflux.pdf")