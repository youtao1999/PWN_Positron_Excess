from scipy.interpolate import interp2d
from math import *
from iminuit import Minuit
import numpy as np
import AMS_model as PWN
import os
import matplotlib.pyplot as pl
# constants
pigreco = 3.141592653589793
cluce = 29979245800.  # cm/s
Ee = 1e-9  # eV in GeV
kpc = 3.09e21
kyr = 1000. * 365. * 24. * 3600.
E_c = 1000  # cutoff energy in GeV
tau_0 = 10. * kyr  # kyr
eta = 1.0  # flux conversion rate
delta = 0.408
k_0 = 0.0967 * pow(kpc, 2.) / (kyr * 1.e3)
alpha = 2.0
b_0 = 1e-16
ergGeV = 624.15097

# Extract energy range from AMS-02 data

tabledata = np.loadtxt('positron_ams02_19.dat')
epos = tabledata[: 0]
pos=tabledata[:,6]*np.power(tabledata[:,2], 3.)*tabledata[:,13]/1.e4 #rescale the flux into 1/GeV/cm$^2$/s/sr
epos= tabledata[:,2] #flux in 1/GeV/m$^2$/s/sr

# take the AMS-02 data
# folder_data = r'''C:\Users\Sissi\ Chen\Desktop\Positron\'''
x_errorup_pos_val = tabledata[:, 1]
x_errordown_pos_val = tabledata[:, 0]
x_errorup_pos = np.subtract(x_errorup_pos_val, epos)
x_errordown_pos = np.subtract(epos, x_errordown_pos_val)
errortot_pos = np.power(tabledata[:, 2], 3.) * np.sqrt(
    tabledata[:, 7] * tabledata[:, 7] + tabledata[:, 12] * tabledata[:, 12]) * tabledata[:, 13] / 1.e4

# Exctract secondary production
# folder_data = r'''C:\Users\Sissi\ Chen\Desktop\Positron\'''
tabledata_sec = np.loadtxt('positrons_med_sec.dat')
epos_sec = tabledata_sec[:, 0]
pos_sec = tabledata_sec[:, 1]

# necessary functions to calculate positron energy flux as a function of energy

table = np.loadtxt("ATNF_Catalog.txt")
DIST = table[:, 1]  # in kpc
AGE = table[:, 2] / 1e3  # in kyr
EDOT = table[:, 3]

# Define function that calculates the total flux of all the ATNF pulsars for a given energy

def total_flux(E, gamma, eta):
    flux_tot = 0
    for i in range(len(AGE)):
        if AGE[i] <= 1.e4 and AGE[i] > 50. and DIST[i] <= 10. and DIST[i] > 0. and EDOT[i] > 0.:
            flux_tot += PWN.flux_general(E, eta, b_0, DIST[i] * kpc, E_c, gamma, AGE[i] * kyr,EDOT[i])
            # if flux_general(E, eta, b_0, DIST[i]*kpc, E_c, gamma, AGE[i]*kyr, EDOT[i])>1e-3:
            #    print(i,DIST[i],AGE[i],EDOT[i],flux_general(E, eta, b_0, DIST[i]*kpc, E_c, gamma, AGE[i]*kyr, EDOT[i]))
    return flux_tot


# testing for different spectral indices and efficiencies
print("############################")
print("Fitting the flux to the AMS-02 data with free parameters as the spectral index and efficiency, assuming uniform for all pulsars.")
print("############################")
print("")

# Interpolating 2D x = epos, y = gamma_array, z = totalflux
gamma_array = np.arange(1.4, 2.4, 0.1)
x = epos
y = gamma_array
z = np.zeros((len(gamma_array), len(epos)))

for i in range(len(gamma_array)):
    # os.chdir(Output)
    flux_array = np.loadtxt("fluxtot_gamma%d.txt" % i)
    z[i, :] = flux_array[:, 1]
    # os.chdir("../")

# %d integer number
# #%f floating
# #%e exponential notation 1e10
# #%.1f floating nymber with 1 decimal number
funcinterpolate = interp2d(x, y, z)


#####################################

# Model curve fit

def f_BPL(par0, par1, par2):
    eta = pow(10., par0)
    gamma = par1
    normalization_sec = par2
    chisq = 0
    for t in range(26, len(epos), 1):
        model = eta * funcinterpolate(epos[t], gamma)
        model_sec = PWN.flux_secondary(epos[t], normalization_sec)
        model_tot = model + model_sec
        chisq = chisq + np.power((model_tot - pos[t]) / errortot_pos[t], 2.)
    return chisq

m = Minuit(f_BPL, par0=-1.0, par1=1.5, par2=1.5)
m.errors['par0'] = 1e-4
m.errors['par1'] = 1e-4
m.errors['par2'] = 1e-4
m.limits['par0'] = (-5., 1.4)
m.limits['par1'] = (1.4, 2.4)
m.limits['par2'] = (0., 5.)
m.errordef = 1
# m.print_param()#or call print_initial_param
m.migrad()
# print('parameters', m.parameters)
# print('args', m.args)
print('value', m.values)
print('error', m.errors)
print('fval', m.fval)
# print('current state', f(*m.args)
# print('covariance', m.covariance)
# print('matrix()', m.matrix()) #covariance
# print('matrix(correlation=True)', m.matrix(correlation=True)) #correlation
# m.print_matrix() #correlation

###########################
# Output files
model_vec_tot = np.zeros(len(epos))
model_vec_sec = np.zeros(len(epos))
model_vec_pulsar = np.zeros(len(epos))

for t in range(len(epos)):
    model_vec_pulsar[t] = pow(10., m.values["par0"]) * funcinterpolate(epos[t], m.values["par1"])
    model_vec_sec[t] = PWN.flux_secondary(epos[t], m.values["par2"])
    model_vec_tot[t] = model_vec_pulsar[t] + model_vec_sec[t]

# Check to see if the output file already exists
if os.path.isfile("ATNF_model_data"):
    os.remove("ATNF_model_data")

# make output file
outF = open("ATNF_model_data.txt", "w")
for t in range(len(epos)):
    outF.write("{:e} {:.10f} {:.10f} {:.10f} \n".format(epos[t], model_vec_pulsar[t], model_vec_sec[t], model_vec_tot[t]))
outF.close()


fig = pl.figure(figsize=(8,6))
pl.rcParams['font.size'] = '18'
pl.plot(epos, model_vec_tot,lw=1.3,ls='-',color="blue", label='PWN+SEC')
pl.plot(epos, model_vec_pulsar, lw=2.0, ls='-.', color="green", label='PWN')
pl.plot(epos, model_vec_sec, lw=2.0, ls='--', color="red", label='Secondary')
pl.errorbar(epos, pos, xerr= [x_errordown_pos, x_errorup_pos],yerr=errortot_pos,fmt='.', color="black",label="AMS-02 $e^+$")
#pl.text(8.,4.5e-3, r'$T=10^4$ kyr', fontsize=16, color='black')
#pl.text(1e4,8.4e-4,r'$T=10$ kyr',fontsize=16, color='red')
pl.ylabel(r'$E^3 \Phi_e$ [GeV$^2$/cm$^2$/s/sr]', fontsize=18)
pl.xlabel(r'$E_e$ [GeV]', fontsize=18)
pl.axis([1.,5.e3,5e-5,1.e-2])
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.tick_params('both', length=7, width=2, which='major')
pl.tick_params('both', length=5, width=2, which='minor')
pl.grid(True)
pl.yscale('log')
pl.xscale('log')
pl.legend(loc=2,prop={'size':16},numpoints=1, scatterpoints=1, ncol=2)
#fig.suptitle('The Effect of Pulsar Age on Positron Flux', fontsize=18)
fig.tight_layout(pad=0.5)
pl.savefig("ATNF_pulsar_flux_debug.png")

