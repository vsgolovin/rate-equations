#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:49:16 2020

@author: vsgolovin
"""

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from ldmodel import CurrentPulse, LDModel
from solver import solve
from utilities import rect_pulse, process_results

# epitaxial design parameters
epi = dict()
epi['lambda'] = 1060e-7
epi['n_eff'] = 3.44
epi['d_a'] = 90e-7
epi['Gamma'] = 0.87e-2
epi['ni'] = 2e6
epi['eta_inj'] = 0.99
epi['alpha_i'] = 1.0
epi['gain'] = lambda n, S: np.max((2140*np.log(n/1.8e18), -1e4))
epi['beta_sp'] = 1e-3
epi['srh_n1'] = epi['ni'] 
epi['srh_p1'] = epi['ni']
epi['srh_tau_n'] = 5e-9
epi['srh_tau_p'] = 5e-9
epi['rad_B'] = 1e-10
epi['aug_Cn'] = 5e-30
epi['aug_Cp'] = 5e-30

# laser diode model parameters
L = 3000e-4
w = 100e-4
R1 = 0.95
R2 = 0.05
m = 60  # number of grid points
ldm = LDModel(L, w, R1, R2, epi, m)

# current pulse
J_fun = partial(rect_pulse, t_max=8e-9, width=10e-9, t_rise=1e-9, t_fall=1e-9,
                amplitude=20/(L*w), constant=0)
z1_fun = lambda t: 0
z2_fun = lambda t: 3000e-4
pulse = CurrentPulse(J_fun, z1_fun, z2_fun)
ldm.add_current_pulse(pulse)

# initial condition
y0 = np.zeros(m*3)
y0[:m] = epi['ni']
y0[m:] = 0

# solving the initial value problem
t_fin = 16e-9
timesteps = 3200
sol = solve(ldm, y0, t_fin, timesteps)
J, I, P1, P2 = process_results(ldm, sol)
P = P1 + P2
t = sol.t*1e9
z = ldm.get_grid()*1e4
y = sol.y
n_avg = y[:m, :].mean(axis=0)

#%% plotting the results
plt.close('all')
fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True)
fig.set_size_inches((6, 10))
ax0, ax1, ax2, ax3 = axs

ax0.plot(t, I)
ax0.set_ylabel('$I$ (A)')

ax1.contourf(t, z, J, antialiased=False, cmap=plt.cm.inferno)
ax1.set_ylim(0, L*1e4)
ax1.set_ylabel('$z$ ($\mu$m)')

ax2.plot(t, n_avg)
ax2.set_ylabel(r'$\bar{n}$ (cm$^{-3}$)')

ax3.plot(t, P)
ax3.set_ylabel('$P$ (W)')
ax3.set_xlim(0, t_fin*1e9)
ax3.set_xlabel('$t$ (ns)')
