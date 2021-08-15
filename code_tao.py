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

#extract data.
table = np.loadtxt('He_AMS_PRL119.251101.2017_ekin_000.txt')
print(np.shape(table))
energy_vec = table[:,0]
energylow_vec = table[:,1]
energyup_vec = table[:,2]
flux_vec = table[:,3]
fluxerrlow_vec = table[:,4]
fluxerrup_vec = table[:,5]


'''
def model_func(E,norm,index):
    model = norm*np.power(E,-index)
    return model
'''

def model_func_BPL(E,norm,gamma1,gamma2,Eb):
    if E<Eb:
        return norm*pow(E,-gamma1)
    else:
        return norm*pow(Eb,-gamma1+gamma2)*pow(E,-gamma2)

def chisqr(E,flux,fluxerr,norm,index1, index2, Eb):
    chisqrv = 0
    for t in range(0,len(E),1):
        model = model_func_BPL(E[t],norm,index1, index2, Eb)
        chisqrv = chisqrv + np.power((model-flux[t])/fluxerr[t],2.)
    return chisqrv
print(chisqr(energy_vec,flux_vec,fluxerrlow_vec,10000.,1, 2.7, energy_vec[35]),len(energy_vec[25:len(energy_vec)-1]))

#def f(par0,par1):
def f_BPL(par0,par1,par2,par3):
    chisq = 0
    for t in range(35,len(energy_vec),1):
        model = model_func_BPL(energy_vec[t],par0,par1,par2,par3)
        chisq = chisq + np.power((model-flux_vec[t])/fluxerrup_vec[t],2.)
    return chisq

model_vec = np.zeros(len(energy_vec))
for t in range(len(energy_vec)):
    print(t,energy_vec[t],model_vec[t],model_func_BPL(energy_vec[t],10000.,1, 2.7, energy_vec[35]))
    model_vec[t] = model_func_BPL(energy_vec[t],10000.,1, 2.7, energy_vec[35])

############################
# Produce CRplot.pdf

fig = pl.figure(figsize=(8,6))
pl.errorbar(energy_vec, flux_vec, yerr=[fluxerrlow_vec,fluxerrup_vec], fmt="*", color="black",label=r'AMS-02 data')
pl.plot(energy_vec, model_vec, ls='--', color="blue", lw=2.0, label=r'Model')
pl.ylabel(r'$\phi$', fontsize=18)
pl.xlabel(r'$E$ [GeV]', fontsize=18)
#pl.axis([vecangle_bin[0],vecangle_bin[len(vecangle_bin)-1],0.001,0.57], fontsize=18)
pl.axis([energy_vec[0],energy_vec[len(energy_vec)-1],0.8*flux_vec.min(),1.2*flux_vec.max()], fontsize=18)
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.tick_params('both', length=7, width=2, which='major')
pl.tick_params('both', length=5, width=2, which='minor')
pl.grid(True)
pl.yscale('log')
pl.xscale('log')
pl.legend(loc=1,prop={'size':18},numpoints=1, scatterpoints=1, ncol=1)
fig.tight_layout(pad=0.5)
#pl.savefig(folder_plots+"Phitheta_IC_geminga_Hawcrange_gammae%.1f_D%.1e_W%.1e_try.pdf"%(Dval,normalization_bestfit*W0/erg,gamma_e))
pl.savefig('CRplot.pdf')

############################
# Produce CRplot_rescaled.pdf

fig = pl.figure(figsize=(8,6))
pl.errorbar(energy_vec, flux_vec*np.power(energy_vec,2.7), yerr=[fluxerrlow_vec*np.power(energy_vec,2.7),fluxerrup_vec*np.power(energy_vec,2.7)], fmt="*", color="black",label=r'AMS-02 data')
pl.plot(energy_vec, model_vec, ls='--', color="blue", lw=2.0, label=r'Model')
pl.ylabel(r'$\phi$ [1/GeV/s/cm$^2$/sr]', fontsize=18)
pl.xlabel(r'$E$ [GeV]', fontsize=18)
pl.axis([energy_vec.min(),energy_vec.max()*1.20,20.0*(flux_vec*np.power(energy_vec,2.7)).min(),1.2*(flux_vec*np.power(energy_vec,2.7)).max()], fontsize=18)
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.tick_params('both', length=7, width=2, which='major')
pl.tick_params('both', length=5, width=2, which='minor')
pl.grid(True)
pl.yscale('log')
pl.xscale('log')
pl.legend(loc=4,prop={'size':18},numpoints=1, scatterpoints=1, ncol=1)
fig.tight_layout(pad=0.5)
pl.savefig('CRplot_rescaled.pdf')

############################
# Produce CRplotratios.pdf

fig = pl.figure(figsize=(8,6))
pl.errorbar(energy_vec, flux_vec/model_vec, yerr=[fluxerrlow_vec/model_vec,fluxerrup_vec/model_vec], fmt="*", color="black")
#pl.plot(energy_vec, model_vec, ls='--', color="black", lw=2.0, label=r'Model')
pl.ylabel(r'$\phi$', fontsize=18)
pl.xlabel(r'$E$ [GeV]', fontsize=18)
#pl.axis([vecangle_bin[0],vecangle_bin[len(vecangle_bin)-1],0.001,0.57], fontsize=18)
pl.axis([10.,energy_vec[len(energy_vec)-1],0.5,1.4], fontsize=18)
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.tick_params('both', length=7, width=2, which='major')
pl.tick_params('both', length=5, width=2, which='minor')
pl.grid(True)
pl.yscale('linear')
pl.xscale('log')
pl.legend(loc=1,prop={'size':18},numpoints=1, scatterpoints=1, ncol=1)
fig.tight_layout(pad=0.5)
#pl.savefig(folder_plots+"Phitheta_IC_geminga_Hawcrange_gammae%.1f_D%.1e_W%.1e_try.pdf"%(Dval,normalization_bestfit*W0/erg,gamma_e))
pl.savefig('CRplotratios.pdf')


#def f(par0,par1):
def f(par0,par1):
    for chisq in range(0,len(energy_vec),1):
        model = model_func(energy_vec[t],par0,par1)
        chisq = chisq + np.power((model-flux_vec[t])/fluxerrup_vec[t],2.)
    return chisq

m=Minuit(f_BPL, par0=600., error_par0=1e-4, limit_par0=(1e1,1e6), par1=2.8, error_par1=1e-4, limit_par1=(0.,6.), par2 = 2.6, error_par2 = 1e-4, limit_par2 = (0.,6.), par3 = 200, limit_par3 = (10, 10000), error_par3 = 0.01, print_level=1,errordef=1)
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
#print(m.get_fmin())
#print(m.get_fmin().is_valid)
normalization_bestfit = m.values["par0"]
normalization_error = m.errors["par0"]
index_bestfit = m.values["par1"]
index_error = m.errors["par1"]
index2_bestfit = m.values["par2"]
index2_error = m.errors["par2"]
energy_break = m.values["par3"]
#D_bestfit = m.values["par1"]

print(normalization_bestfit,normalization_error,index_bestfit,index_error,m.fval/(len(energy_vec)-25-2))


model_vec = np.zeros(len(energy_vec))
for t in range(len(energy_vec)):
    model_vec[t] = model_func_BPL(energy_vec[t],normalization_bestfit,index_bestfit,index2_bestfit,energy_break)

fig = pl.figure(figsize=(8,6))
pl.errorbar(energy_vec, flux_vec, yerr=[fluxerrlow_vec,fluxerrup_vec], fmt="*", color="black",label=r'AMS-02 data')
pl.plot(energy_vec, model_vec, ls='--', color="black", lw=2.0, label=r'Model')
pl.ylabel(r'$\phi$', fontsize=18)
pl.xlabel(r'$E$ [MeV]', fontsize=18)
#pl.axis([vecangle_bin[0],vecangle_bin[len(vecangle_bin)-1],0.001,0.57], fontsize=18)
pl.axis([energy_vec[0],energy_vec[len(energy_vec)-1],0.8*flux_vec.min(),1.2*flux_vec.max()], fontsize=18)
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.tick_params('both', length=7, width=2, which='major')
pl.tick_params('both', length=5, width=2, which='minor')
pl.grid(True)
pl.yscale('log')
pl.xscale('log')
pl.legend(loc=1,prop={'size':18},numpoints=1, scatterpoints=1, ncol=1)
fig.tight_layout(pad=0.5)
#pl.savefig(folder_plots+"Phitheta_IC_geminga_Hawcrange_gammae%.1f_D%.1e_W%.1e_try.pdf"%(Dval,normalization_bestfit*W0/erg,gamma_e))
pl.savefig('CRplotfit.pdf')


fig = pl.figure(figsize=(8,6))
pl.errorbar(energy_vec, flux_vec/model_vec, yerr=[fluxerrlow_vec/model_vec,fluxerrup_vec/model_vec], fmt="*", color="black")
#pl.plot(energy_vec, model_vec, ls='--', color="black", lw=2.0, label=r'Model')
pl.ylabel(r'$\phi$', fontsize=18)
pl.xlabel(r'$E$ [MeV]', fontsize=18)
#pl.axis([vecangle_bin[0],vecangle_bin[len(vecangle_bin)-1],0.001,0.57], fontsize=18)
pl.axis([10.,energy_vec[len(energy_vec)-1],0.80,1.2], fontsize=18)
pl.xticks(fontsize=18)
pl.yticks(fontsize=18)
pl.tick_params('both', length=7, width=2, which='major')
pl.tick_params('both', length=5, width=2, which='minor')
pl.grid(True)
pl.yscale('linear')
pl.xscale('log')
pl.legend(loc=1,prop={'size':18},numpoints=1, scatterpoints=1, ncol=1)
fig.tight_layout(pad=0.5)
#pl.savefig(folder_plots+"Phitheta_IC_geminga_Hawcrange_gammae%.1f_D%.1e_W%.1e_try.pdf"%(Dval,normalization_bestfit*W0/erg,gamma_e))
pl.savefig('CRplotratiosfit.pdf')