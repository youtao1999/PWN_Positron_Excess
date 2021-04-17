'''
    Tao You
    3/30/2021
    --This package contains all the necessary function calls for modeling of PWN positron fluxes and for fitting the model
    to AMS-02 data
'''

from math import *
from scipy.interpolate import interp1d
from iminuit import Minuit
import numpy as np
from scipy.integrate import quad
import math
import numba
from numba import jit

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

@jit(nopython=True)
def integrand(E, delta, k_0, alpha):
    return 4 * k_0 / b_0 * E ** (delta - alpha)

def E_0_func(E, alpha, b_0, T):
    # define E_0 as a function of E
    return (np.exp((-alpha + 1) * np.log(E)) + b_0 * T * (-alpha + 1)) ** (1 / (-alpha + 1))

def Q(E, Q_0, gamma, E_c):
    # define the source term
    E = pow(10., E)
    return Q_0 * (E) ** (-gamma) * np.exp(-E / E_c) * np.log(10.) * E

def b(b_0, E):
    # define energy loss
    return b_0 * E ** 2

def Q_0_func(eta, gamma):
    # eta = 0.1 # flux conversion rate
    tau_0 = 10 * kyr  # kyr
    E_dot = 1e35  # spin-down luminosity in erg/s
    E_tot = quad(Q, math.log(0.1, 10), math.log(1.e4, 10), args=(1., gamma, E_c))[0]
    return eta * tau_0 * E_dot * (1 + T / tau_0) ** 2 / E_tot

def Q_0_func_general(T, eta, gamma):
    # eta = 0.1 # flux conversion rate
    tau_0 = 10 * kyr  # kyr
    E_dot = 1e35  # spin-down luminosity in erg/s
    E_tot = quad(Q, math.log(0.1, 10), math.log(1.e4, 10), args=(1., gamma, E_c))[0]
    return eta * tau_0 * E_dot * (1 + T / tau_0) ** 2 / E_tot

# define the model function that we aim to approximate

def flux(E, eta, b_0, d, E_c, gamma, T):
    # define positron flux
    E_0 = E_0_func(E, alpha, b_0, T)
    E_max = math.exp((1. / (1. - alpha)) * math.log((alpha - 1.) * T * b_0))
    if E < E_max:
        Q_0 = Q_0_func(eta, gamma)
        lamda = math.sqrt(quad(integrand, E, E_0, args=(delta, k_0, alpha))[0])
        flux = b(b_0, E_0) / b(b_0, E) / (np.pi * lamda ** 2) ** 1.5 * np.exp(-(abs(d)) ** 2 / (lamda ** 2)) * Q(
            math.log10(E_0), Q_0, gamma, E_c)
        return flux * pow(E, 3.)
    else:
        return 0


def flux_general(E, eta, b_0, d, E_c, gamma, T) -> object:
    # define positron flux
    E_0 = E_0_func(E, alpha, b_0, T)
    E_max = math.exp((1. / (1. - alpha)) * math.log((alpha - 1.) * T * b_0))
    if E < E_max:
        Q_0 = Q_0_func_general(T, eta, gamma)
        lamda = math.sqrt(quad(integrand, E, E_0, args=(delta, k_0, alpha))[0])
        flux = b(b_0, E_0) / b(b_0, E) / (np.pi * lamda ** 2) ** 1.5 * np.exp(-(abs(d)) ** 2 / (lamda ** 2)) * Q(
            math.log10(E_0), Q_0, gamma, E_c)
        return flux * pow(E, 3.)
    else:
        return 0


def flux_secondary(Ee, normalization_sec):
    epos_sec = tabledata_sec[:, 0]
    pos_sec = normalization_sec * tabledata_sec[:, 1]
    func_interpolator = interp1d(epos_sec, pos_sec)
    return func_interpolator(Ee)


# define function to be minimized

def chisqr(E, flux, fluxerr, Q_0, b_0, d, E_c, gamma, T):
    chisqrv = 0
    for t in range(0, len(E), 1):
        model = flux(Q_0, b_0, d, E, E_c, gamma, T)
        chisqrv = chisqrv + np.power((model - flux[t]) / fluxerr[t], 2.)
    return chisqrv


######################################

# def f(par0,par1,par2): for a fixed age and distance
def f_BPL(par0, par1, par2):
    eta = pow(10., par0)
    gamma = par1
    normalization_sec = par2
    chisq = 0
    for t in range(24, len(epos), 1):
        model = flux(epos[t], eta, b_0, d, E_c, gamma, T)
        model_sec = flux_secondary(epos[t], normalization_sec)
        model_tot = model + model_sec
        chisq = chisq + np.power((model_tot - pos[t]) / errortot_pos[t], 2.)
    return chisq


def function_calculatechi(distance, age):
    # def f(par0,par1,par2):
    def f_BPL(par0, par1, par2):
        eta = pow(10., par0)
        gamma = par1
        normalization_sec = par2
        chisq = 0
        for t in range(24, len(epos), 1):
            model = flux(epos[t], eta, b_0, distance, E_c, gamma, age)
            model_sec = flux_secondary(epos[t], normalization_sec)
            model_tot = model + model_sec
            chisq = chisq + np.power((model_tot - pos[t]) / errortot_pos[t], 2.)
        return chisq

    #######################################

    ############################

    # Model curve fit

    m = Minuit(f_BPL, par0=0.0, par1=1.5, par2=1.2)
    m.errors['par0'] = 1e-4
    m.errors['par1'] = 1e-4
    m.errors['par2'] = 1e-4
    m.limits['par0'] = (-10.0, 10.)
    m.limits['par1'] = (0., 5.)
    m.limits['par2'] = (0.1, 10.)
    m.errordef = 1
    m.migrad()
    # print('parameters', m.parameters)
    # print('args', m.args)
    print('value', m.values)
    print('error', m.errors)
    print('fval', m.fval)
    return m.fval

##############################################
# Function calls for fitting the total positron flux of ATNF Catalog pulsars

table = np.loadtxt("ATNF_Catalog.txt")
DIST = table[:,1] #in kpc
AGE = table[:,2]/1e3 #in kyr
EDOT = table[:,3]

# Define function that calculates the total flux of all the ATNF pulsars for a given energy

def total_flux(E, gamma, eta):
    flux_tot = 0
    for i in range(len(AGE)):
        if AGE[i] <= 1.e4 and AGE[i] >50. and DIST[i] <= 10. and DIST[i] >0. and EDOT[i]>0.:
            flux_tot += flux_general(E, eta, b_0, DIST[i]*kpc, E_c, gamma, AGE[i]*kyr, EDOT[i]) # we dont need Edot to calculate the flux?
            #if flux_general(E, eta, b_0, DIST[i]*kpc, E_c, gamma, AGE[i]*kyr, EDOT[i])>1e-3:
            #    print(i,DIST[i],AGE[i],EDOT[i],flux_general(E, eta, b_0, DIST[i]*kpc, E_c, gamma, AGE[i]*kyr, EDOT[i]))
    return flux_tot