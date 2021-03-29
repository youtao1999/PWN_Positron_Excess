import matplotlib.pyplot as plt
import scipy
import numpy as np
from IPython.display import Latex

# Define experimental range

Edot_0 = float(1e35)
tau_0 = [1, 10, 100]
n_range = [2, 3, 4]
t_range = np.arange(0, 1000, 0.1)

# Plot with Edot as a function of time for Edot_0 = 1e35 erg/s, 
# tau_0 = [1, 10, 100], n = 3
fig = plt.figure(figsize = (8,6))
plt.ylabel(r'$\frac{dE}{dt}$', fontsize = 18)
plt.xlabel(r'$t$ [kyr]')
for tau in tau_0:
    y = np.zeros(len(t_range))
    for i, t in enumerate(t_range):
        y[i] = Edot_0*np.power(float(1+t/tau),float(-(n_range[1]+1)/(n_range[1]-1)))
    plt.plot(t_range, y)
    plt.legend(loc = 'best', bbox_to_anchor=(1, 1), ncol = 1)
plt.xscale('log')
plt.yscale('log')
plt.savefig('spin-down luminosity with tau_0 variation.pdf')

# Plot with Edot as a function of time for Edot_0 = 1e35 erg/s, 
# tau_0 = 10, n = [2, 3, 4]
fig = plt.figure(figsize = (8,6))
plt.ylabel(r'$\frac{dE}{dt}$', fontsize = 18)
plt.xlabel(r'$t$ [kyr]')
for n in n_range:
    y = np.zeros(len(t_range))
    for i, t in enumerate(t_range):
        y[i] = Edot_0*np.power(float(1+t/tau_0[1]),float(-(n+1)/(n-1)))
    plt.plot(t_range, y)
    plt.legend(loc = 'best', bbox_to_anchor=(1, 1), ncol = 1)
plt.xscale('log')
plt.yscale('log')
plt.savefig('spin-down luminosity with n variation.pdf')


# Plot energy losses for synchrotron radiation for different values of 
# magnetic field B = [1, 3, 10] in micro-Gauss

E_range = (0, 10000, 0.1)
B_range = [1, 3, 10]

fig = plt.figure(figsize = (8,6))
plt.ylabel(r'$-\frac{dE}{dt}$', fontsize = 18)
plt.xlabel(r'$E$ [GeV]')
for B in B_range:
    y = np.zeros(len(E_range))
    for i, E in enumerate(E_range):
        y[i] = 2.53*10**(-18)*np.power(B,2)*np.power(E,2)
    plt.plot(E_range, y)
    plt.legend(loc = 'best', bbox_to_anchor=(1, 1), ncol = 1)
plt.xscale('log')
plt.yscale('log')
plt.savefig('energy loss with B variation.pdf')
