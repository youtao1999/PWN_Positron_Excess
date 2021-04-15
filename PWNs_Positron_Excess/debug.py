from scipy.integrate import quad
from math import *
import matplotlib.pyplot as pl
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import numpy as np

# Constants
pigreco = 3.141592653589793
sigmat = 0.665 * pow(10., -24.)  # cm^2
cluce = 29979245800.  # cm/s
ht = 6.582119 * pow(10., -16.) * pow(10., -9.)  # GEV*S
hp = 4.13566 * 1e-15 * 1e-9  # GEV*S
htc = ht * cluce
kbol = 8.61734 * pow(10., -5.) * pow(10., -9.)
echarge = 4.8032068e-10  # esu
h_erg = 6.6260755e-27  # erg*s
emass_g = 9.1093897e-28  # g
Gauss = pow(6.24142 * pow(10., 11.) * pow(10., -9.), 1. / 2.)
mc = 0.000511  # GeV
kpc = pow(10., 3.) * 3.0856775 * pow(10., 18.)  # cm
erg = 6.2415097e2  # GeV
kyr = 24. * 60. * 60. * 365. * pow(10., 3.)  # s
kb = 8.61734 * pow(10., -5.) * pow(10., -9.)  # GeV
me = 0.000511  # GeV
srdeg = pow(180. / np.pi, 2.)
amu = 1.66054 * 1e3 * 1e-27  # g
alphaf = 7.29735e-3
rodot = 8.33 * kpc  # cm
rhodot = 0.4  # GeV/cm^3

table = np.loadtxt('DMfluxes_positrons_cleaned_NFW_MED_MF2.txt')


###DENSITY###

def DMdensity_funcdist(dist, DMdensity):
    rhos = DMdensity[0]
    rs = DMdensity[1]
    gammas = DMdensity[2]
    profile = DMdensity[3]

    rho = 0.
    if profile == 'gNFW':
        # rho = (rhos/rhodot)/( pow(dist/rs,gammas)*pow(1.+dist/rs,3.-gammas) )
        rho = (rhos) / (pow(dist / rs, gammas) * pow(1. + dist / rs, 3. - gammas))
    if profile == 'NFW':
        # rho = (rhos/rhodot)/( pow(dist/rs,1.0)*pow(1.+dist/rs,3.-1.0) )
        rho = (rhos) / (pow(dist / rs, 1.0) * pow(1. + dist / rs, 3. - 1.0))
    elif profile == 'Einasto':
        # rho = (rhos/rhodot)*exp( (-2./gammas)*(pow(dist/rs,gammas) -1.) )
        rho = (rhos) * exp((-2. / gammas) * (pow(dist / rs, gammas) - 1.))
    elif profile == 'Burkert':
        # rho = (rhos/rhodot)/(pow(1.+dist/rs,2.-gammas)*(1.+ pow(dist/rs,3.-gammas)))
        rho = (rhos) / (pow(1. + dist / rs, 2. - gammas) * (1. + pow(dist / rs, 3. - gammas)))

    return rho


def DMdensity_funcintegral(dist, DMdensity):
    rhos = DMdensity[0]
    rs = DMdensity[1]
    gammas = DMdensity[2]
    profile = DMdensity[3]

    rho = 0.
    if profile == 'gNFW':
        # rho = (rhos/rhodot)/( pow(dist/rs,gammas)*pow(1.+dist/rs,3.-gammas) )
        rho = (rhos) / (pow(dist / rs, gammas) * pow(1. + dist / rs, 3. - gammas))
    if profile == 'NFW':
        # rho = (rhos/rhodot)/( pow(dist/rs,1.0)*pow(1.+dist/rs,3.-1.0) )
        rho = (rhos) / (pow(dist / rs, 1.0) * pow(1. + dist / rs, 3. - 1.0))
    elif profile == 'Einasto':
        # rho = (rhos/rhodot)*exp( (-2./gammas)*(pow(dist/rs,gammas) -1.) )
        rho = (rhos) * exp((-2. / gammas) * (pow(dist / rs, gammas) - 1.))
    elif profile == 'Burkert':
        # rho = (rhos/rhodot)/(pow(1.+dist/rs,2.-gammas)*(1.+ pow(dist/rs,3.-gammas)))
        rho = (rhos) / (pow(1. + dist / rs, 2. - gammas) * (1. + pow(dist / rs, 3. - gammas)))

    return rho * pow(dist, 2.)


def totmass(DMdensity, distmax):
    return 4. * np.pi * quad(DMdensity_funcintegral, 0., distmax, args=(DMdensity))[0]
    # https://www.khanacademy.org/math/multivariable-calculus/integrating-multivariable-functions/triple-integrals-a/a/triple-integrals-in-spherical-coordinates


def func_interpolate(varval, variablevec, funcvec):
    result = 0.
    if varval < variablevec[0]:
        result = 0.
    elif varval > variablevec[len(variablevec) - 1]:
        result = 0.
    else:
        Log10E_bin = log10(variablevec[1]) - log10(variablevec[0])
        nbin = (log10(varval) - log10(variablevec[0])) / Log10E_bin
        binval = int(nbin)
        # print(fluxDM[binval],fluxDM[binval+1],EnergyDM[binval],EnergyDM[binval+1])
        result = pow(10., log10(funcvec[binval]) + (log10(funcvec[binval + 1]) - log10(funcvec[binval])) * (
                    log10(varval) - log10(variablevec[binval])) / (
                                 log10(variablevec[binval + 1]) - log10(variablevec[binval])))
        # print(Energy,EnergyDM[binval],EnergyDM[binval+1],fluxDM[binval],fluxDM[binval+1],result)
    # print('Interpolate',varval,variablevec[binval],variablevec[binval+1],funcvec[binval],funcvec[binval+1],result)
    return result


def exctractcirellitable_fluxpositron(DMmass, DMchannel):
    '''
    This function returns the energy spectrum in 1/GeV for the particle production from DM annihilation.
    DMmass: dark matter mass in GeV
    DMchannel: dark matter annihilation channel
        #e 7, mu 10, tau 13, bb 16, tt 17, WW 20, ZZ 23, gamma 25, h 26
    particle: particle produced from the DM annihilation ('gammas' or 'positrons')
    EWcorr: electroweak corrections ('Yes' or 'No')
    '''
    # mDM      halo   prop   MF    Log[10,E]   eL            eR            e             \[Mu]L        \[Mu]R        \[Mu]         \[Tau]L       \[Tau]R       \[Tau]        q             c             b             t             WL            WT            W             ZL            ZT            Z             g             \[Gamma]      h             \[Nu]e        \[Nu]\[Mu]    \[Nu]\[Tau]   V->e          V->\[Mu]      V->\[Tau]

    listenergies = 61
    energy_vec = np.arange(-1., 5.1, 0.1)

    energy = np.zeros(listenergies)
    fluxDM = np.zeros(listenergies)

    massvec = []
    for t in range(len(table)):
        if t % listenergies == 0:
            massvec.append(table[t, 0])
    massvec = np.array(massvec)

    flux = []
    for t in range(len(table)):
        # print(table[t,DMchannel])
        flux.append(table[t, DMchannel])

    f = interp2d(massvec, energy_vec, flux, kind='linear')

    for t in range(len(energy_vec)):
        fluxDM[t] = f(DMmass, energy_vec[t])

    return np.power(10., energy_vec), pow(DMmass, -2.) * fluxDM / (log(10.) * np.power(10., energy_vec))
    # return np.power(10.,energy_vec),fluxDM


def DM_spectrum(Energy, DMmass, DMchannel):
    # this function creates the flux, annihilation cross section 10^-26;
    EnergyDM, fluxDM = exctractcirellitable_fluxpositron(DMmass, DMchannel)
    for t in range(len(EnergyDM)):
        if fluxDM[t] > 0.:
            fluxDM[t] = fluxDM[t]
        else:
            fluxDM[t] = 1e-30

    dNdE = 0.
    dNdE_int = 0.
    if Energy < DMmass:
        f = interp1d(EnergyDM, fluxDM, kind='linear')
        dNdE = func_interpolate(Energy, EnergyDM, fluxDM)
        return dNdE, f(Energy)
    else:
        return dNdE, dNdE_int


# mDM      halo   prop   MF    Log[10,E]   eL            eR            e             \[Mu]L        \[Mu]R        \[Mu]         \[Tau]L       \[Tau]R       \[Tau]        q             c             b             t             WL            WT            W             ZL            ZT            Z             g             \[Gamma]      h             \[Nu]e        \[Nu]\[Mu]    \[Nu]\[Tau]   V->e          V->\[Mu]      V->\[Tau]
# table = np.loadtxt('AfterPropagation_Ann_positrons.dat')
# fwrite = open('DMfluxes_positrons_cleaned_NFW_MED_MF2.txt', 'w')
# for t in range(len(table)):
# for t in range(1000):
# if table[t,1]==100 and table[t,2]==200 and table[t,3]==200:
# for u in range(len(table[t])):
# fwrite.write("%.3e  "%table[t,u])
# fwrite.write(" \n")
# fwrite.close()

Mdm = 200.
f_vec = np.power(10., np.arange(log10(1e-1), log10(Mdm), 0.02))
dNdE1_vec = np.zeros(len(f_vec))
dNdE2_vec = np.zeros(len(f_vec))
dNdE3_vec = np.zeros(len(f_vec))
dNdE4_vec = np.zeros(len(f_vec))
dNdE5_vec = np.zeros(len(f_vec))
dNdE6_vec = np.zeros(len(f_vec))
dNdE7_vec = np.zeros(len(f_vec))
dNdE8_vec = np.zeros(len(f_vec))
dNdE9_vec = np.zeros(len(f_vec))

# e 7, mu 10, tau 13, bb 16, tt 17, WW 20, ZZ 23, gamma 25, h 26

for t in range(len(f_vec)):
    dNdE1_vec[t] = np.power(f_vec[t], 3.) * DM_spectrum(f_vec[t], Mdm, 7)[0]
    dNdE2_vec[t] = np.power(f_vec[t], 3.) * DM_spectrum(f_vec[t], Mdm, 10)[0]
    dNdE3_vec[t] = np.power(f_vec[t], 3.) * DM_spectrum(f_vec[t], Mdm, 13)[0]
    dNdE4_vec[t] = np.power(f_vec[t], 3.) * DM_spectrum(f_vec[t], Mdm, 16)[0]
    dNdE5_vec[t] = np.power(f_vec[t], 3.) * DM_spectrum(f_vec[t], Mdm, 17)[0]
    dNdE6_vec[t] = np.power(f_vec[t], 3.) * DM_spectrum(f_vec[t], Mdm, 20)[0]
    dNdE7_vec[t] = np.power(f_vec[t], 3.) * DM_spectrum(f_vec[t], Mdm, 23)[0]
    dNdE8_vec[t] = np.power(f_vec[t], 3.) * DM_spectrum(f_vec[t], Mdm, 25)[0]
    dNdE9_vec[t] = np.power(f_vec[t], 3.) * DM_spectrum(f_vec[t], Mdm, 26)[0]

print((DM_spectrum(f_vec[t], Mdm, 7)))[0]

