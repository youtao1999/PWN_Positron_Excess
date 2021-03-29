print("############################")
print("Making plot for analysis of Cosmic Ray He")
print("############################")
print("")
import astropy.io.fits as pyfits
from scipy.integrate import quad
from math import *
import matplotlib.pyplot as pl
from scipy.interpolate import interp1d
import numpy as np
from iminuit import Minuit
import matplotlib.pyplot as plt
import scipy
import numpy as np
from IPython.display import Latex
from scipy.integrate import quad
import math

#constants
pigreco = 3.141592653589793 
cluce = 29979245800. #cm/s
Ee = 1e-9 # eV in GeV

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
pl.errorbar(epos, pos*np.power(epos,-3.), xerr= [x_errordown_pos, x_errorup_pos],yerr=errortot_pos*np.power(epos,-3.),fmt='.', color="black",label="AMS-02 $e^+$")
pl.plot(epos_sec, pos_sec*np.power(epos_sec,-3.), lw=2.0, ls='--', color="red", label='Secondary data')
pl.ylabel(r'$\Phi_e$ [1/GeV/cm$^2$/s/sr]', fontsize=18)
pl.xlabel(r'$E_e$ [GeV]', fontsize=18)
pl.axis([0.1,1e3,1e-10,1e-3], fontsize=18)
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

# define necessary constants

kpc = 3.09e18
kyr = 1000*365*24*3600
E_range = np.power(10.,np.arange(math.log10(1.),math.log10(1e5)+0.1,0.1)) #GeV
#d=[0.2,0.5,1.0,2.0]
#T=[30.,100.,200.,500.]
d = 0.5*kpc #kpc
T = 100*kyr #age in kyrs
E_c = 1000 #cutoff energy in GeV
gamma = 1.8
tau_0 = 10*kyr #kyr
E_dot = 1e35 #spin-down luminosity in erg/s
eta = 0.1 # flux conversion rate
delta = 0.408
k_0 = kpc**2*0.0967/(1.0e6*365*24*3600)
alpha = 2
b_0 = 1e-16

# necessary functions to calculate positron energy flux as a function of energy
def integrand(E, delta, k_0, alpha):
    return 4*k_0/b_0*E**(delta-alpha)

def E_0_func(E, alpha, b_0, T):
    # define E_0 as a function of E
    return (np.exp((-alpha+1)*np.log(E))+b_0*T*(-alpha+1))**(1/(-alpha+1))

def Q(E, Q_0, gamma, E_c):
    # define the source term 
    E = pow(10.,E)
    return Q_0*(E)**(-gamma)*np.exp(-E/E_c)*np.log(10.)*E

def b(b_0, E):
    # define energy loss
    return b_0*E**2
    
def Q_0_func(eta,gamma):
    #eta = 0.1 # flux conversion rate
    tau_0 = 10*kyr #kyr
    E_dot = 1e35 #spin-down luminosity in erg/s
    E_tot = quad(Q,math.log(0.1,10),math.log(1.e4,10),args=(1.,gamma,E_c))[0]
    return eta*tau_0*E_dot*(1+T/tau_0)**2/E_tot

def Q_0_func_general(T,eta,gamma):
    #eta = 0.1 # flux conversion rate
    tau_0 = 10*kyr #kyr
    E_dot = 1e35 #spin-down luminosity in erg/s
    E_tot = quad(Q,math.log(0.1,10),math.log(1.e4,10),args=(1.,gamma,E_c))[0]
    return eta*tau_0*E_dot*(1+T/tau_0)**2/E_tot

# define the model function that we aim to approximate

def flux(E, eta, b_0, d, E_c, gamma, T):
    # define positron flux
    E_0 = E_0_func(E, alpha, b_0, T)
    E_max = math.exp( (1./(1.-alpha))*math.log((alpha-1.)*T*b_0) )
    if E<E_max:
        Q_0 = Q_0_func(eta,gamma)
        lamda = math.sqrt(quad(integrand, E, E_0, args = (delta, k_0, alpha))[0])
        flux = b(b_0, E_0)/b(b_0, E)/(np.pi*lamda**2)**1.5*np.exp(-(abs(d))**2/(lamda**2))*Q(math.log10(E_0), Q_0, gamma, E_c)
        return flux*pow(E,3.)
    else:
        return 0
    
def flux_general(E, eta, b_0, d, E_c, gamma, T):
    # define positron flux
    E_0 = E_0_func(E, alpha, b_0, T)
    E_max = math.exp( (1./(1.-alpha))*math.log((alpha-1.)*T*b_0) )
    if E<E_max:
        Q_0 = Q_0_func_general(T,eta,gamma)
        lamda = math.sqrt(quad(integrand, E, E_0, args = (delta, k_0, alpha))[0])
        flux = b(b_0, E_0)/b(b_0, E)/(np.pi*lamda**2)**1.5*np.exp(-(abs(d))**2/(lamda**2))*Q(math.log10(E_0), Q_0, gamma, E_c)
        return flux*pow(E,3.)
    else:
        return 0

def flux_secondary(Ee,normalization_sec):
    epos_sec=tabledata_sec[:,0]
    pos_sec= normalization_sec*tabledata_sec[:,1]
    func_interpolator = interp1d(epos_sec,pos_sec)
    return func_interpolator(Ee)

# define function to be minimized

def chisqr(E,flux,fluxerr,Q_0,b_0,d,E_c,gamma,T):
    chisqrv = 0
    for t in range(0,len(E),1):
        model = flux(Q_0, b_0, d, E, E_c, gamma, T)
        chisqrv = chisqrv + np.power((model-flux[t])/fluxerr[t],2.)
    return chisqrv

######################################

#def f(par0,par1,par2): for a fixed age and distance
def f_BPL(par0,par1,par2):
    eta = pow(10.,par0)
    gamma = par1
    normalization_sec = par2
    chisq = 0
    for t in range(24,len(epos),1):
        model = flux(epos[t],eta,b_0,d,E_c,gamma,T)
        model_sec = flux_secondary(epos[t],normalization_sec)
        model_tot = model+model_sec
        chisq = chisq + np.power((model_tot-pos[t])/errortot_pos[t],2.)
    return chisq

def function_calculatechi(distance,age):
    
    #def f(par0,par1,par2):
    def f_BPL(par0,par1,par2):
        eta = pow(10.,par0)
        gamma = par1
        normalization_sec = par2
        chisq = 0
        for t in range(24,len(epos),1):
            model = flux(epos[t],eta,b_0,distance,E_c,gamma,age)
            model_sec = flux_secondary(epos[t],normalization_sec)
            model_tot = model+model_sec
            chisq = chisq + np.power((model_tot-pos[t])/errortot_pos[t],2.)
        return chisq
    
    #######################################

    ############################

    # Model curve fit

    m=Minuit(f_BPL, par0=0.0, error_par0=1e-4, limit_par0=(-10.0,10.), par1=1.5, error_par1=1e-4, limit_par1=(0.,5.), par2=1.2, error_par2=1e-4, limit_par2=(0.1,10.), print_level=1,errordef=1)
    m.print_param()#or call print_initial_param
    m.migrad()
    #print('parameters', m.parameters)
    #print('args', m.args)
    print('value', m.values)
    print('error', m.errors)
    print('fval', m.fval)
    
    return m.fval
######################################

######################################

# Model curve fit

m=Minuit(f_BPL, par0=0.0, error_par0=1e-4, limit_par0=(-10.0,10.), par1=1.5, error_par1=1e-4, limit_par1=(0.,5.), par2=1.2, error_par2=1e-4, limit_par2=(0.1,10.), print_level=1,errordef=1)
m.print_param()#or call print_initial_param
m.migrad()
#print('parameters', m.parameters)
#print('args', m.args)
print('value', m.values)
print('error', m.errors)
print('fval', m.fval)
#print('current state', f(*m.args))
#print('covariance', m.covariance)
#print('matrix()', m.matrix()) #covariance
#print('matrix(correlation=True)', m.matrix(correlation=True)) #correlation
m.print_matrix() #correlation

############################

# Preparing for plot

energy_vec = np.power(10.,np.arange(0.,3.5,0.1))

model_vec_tot = np.zeros(len(energy_vec))
model_vec_sec = np.zeros(len(energy_vec))
model_vec_pulsar = np.zeros(len(energy_vec))

for t in range(len(energy_vec)):
    model_vec_pulsar[t]=flux(energy_vec[t],pow(10.,m.values["par0"]),b_0,d,E_c,m.values["par1"],T)
    model_vec_sec[t]=flux_secondary(energy_vec[t],m.values["par2"])
    model_vec_tot[t]=model_vec_pulsar[t]+model_vec_sec[t]

############################

fig = pl.figure(figsize=(8,6))
pl.plot(energy_vec,model_vec_tot,lw=1.3,ls='-',color="blue", label='PWN+SEC')
pl.plot(energy_vec, model_vec_pulsar, lw=2.0, ls='-.', color="green", label='PWN')
pl.plot(energy_vec, model_vec_sec, lw=2.0, ls='--', color="red", label='Secondary')
pl.errorbar(epos, pos, xerr= [x_errordown_pos, x_errorup_pos],yerr=errortot_pos,fmt='.', color="black",label="AMS-02 $e^+$")
#pl.text(8.,4.5e-3, r'$T=10^4$ kyr', fontsize=16, color='black')
#pl.text(1e4,8.4e-4,r'$T=10$ kyr',fontsize=16, color='red')
pl.ylabel(r'$E^3 \Phi_e$ [GeV$^2$/cm$^2$/s/sr]', fontsize=18)
pl.xlabel(r'$E_e$ [GeV]', fontsize=18)
pl.axis([1.,5.e3,5e-5,1.e-2], fontsize=18)
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
pl.savefig("pulsarflux_example_test.pdf")

##########################

# Contour plot

distance_vec = np.logspace(-1., 1.,100)
age_vec = np.logspace(1., 4.,100)
chisquare = np.zeros((len(distance_vec),len(age_vec)))

for t in range(len(distance_vec)):
    for u in range(len(age_vec)):
        chisquare[t,u] = function_calculatechi(distance_vec[t],age_vec[u])
            
############################
            
fig = pl.figure(figsize=(8,6))
#dlog = ( log10(5.*chisquare.min())-log10(chisquare.min()) )/50.
#scale_vec = np.power( 10. , np.arange( log10(chisquare.min()),log10(5.*chisquare.min()), dlog ) )
#scale_cb = np.power( 10. , np.arange( log10(chisquare.min()),log10(5.*chisquare.min()), dlog*10. ) )
dlog = ( 5.*chisquare.min()-chisquare.min() )/100.
scale_vec = np.arange( chisquare.min(),5.*chisquare.min(), dlog ) 
scale_cb = np.arange( chisquare.min(),5.*chisquare.min(), dlog*10.  )
#print scale_vec
pl.contourf(age_vec, distance_vec, chisquare, 100, levels=list(scale_vec), cmap='hot')
#im = plt.imshow(table, interpolation='nearest', cmap='hot')
#plt.contour(ra_vec, dec_vec, table, colors='black')
pl.colorbar(ticks=scale_cb)
#plt.colorbar()
pl.ylabel(r'$d$ [kpc]', fontsize=18)
pl.xlabel(r'$T$ [kyr]', fontsize=18)
pl.axis([age_vec[0],age_vec[len(age_vec)-1],distance_vec[0],distance_vec[len(distance_vec)-1]], fontsize=14)
pl.xticks(fontsize=16)
pl.yticks(fontsize=16)
pl.grid(True)
pl.yscale('log')
pl.xscale('log') 
pl.legend(loc=2,prop={'size':15},numpoints=1, scatterpoints=1, ncol=2)
fig.tight_layout(pad=0.5)
pl.savefig("Contour_dage.pdf")
pl.savefig("Contour_dage.png")

# make table for ATNF-catalog pulsar data

table = np.loadtxt("ATNF-Catalog.txt")
DIST = table[:,0] #in kpc
AGE = table[:,1]/1e3 #in kyr
EDOT = table[:,2]

flux_tot = 0
for i, t in enumerate(AGE):
    if AGE[i] <= 1e4 and DIST[i] <= 10:
        flux_tot += flux_general(E, eta, b_0, DIST[i] E_c, gamma, AGE[i]) # we dont need Edot to calculate the flux?
    