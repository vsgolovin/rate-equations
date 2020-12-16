# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:11:37 2020

@author: vsgolovin
"""

import numpy as np
import constants as const

class CurrentPulse:

    def __init__(self, J_t, z1_t, z2_t):
        """
        Current pulse in the laser. This class allows for changing current
        pulse location along the laser cavity in time.

        Parameters
        ----------
        J_t : function
            Current density as a function of time.
        z1_t : function
            Current pulse left (small z) boundary as a function of time.
        z2_t : function
            Current pulse right (large z) boundary as a function of time.

        """
        self.J_fun = J_t
        self.z1_fun = z1_t
        self.z2_fun = z2_t

    def calculate(self, t, z):
        "Calculate current density at time `t` and location `z`."
        z1 = self.z1_fun(t)
        z2 = self.z2_fun(t)
        if z<z1 or z>z2:
            return 0
        J = self.J_fun(t)
        return J

class LDModel:

    def __init__(self, L, w, R1, R2, epi, npoints=50):
        """
        Class for storing all the laser diode model parameters.

        Parameters
        ----------
        L : number
            Resonator length (cm).
        w : number
            Stripe width (cm).
        R1 : float
            First mirror reflectivity (z=0, 0<R1<1).
        R2 : float
            Second mirror reflectivity (z=L, 0<R2<1).
        epi : dict
            Epitaxial design parameters.

        """
        # size and mirrors' reflectivities
        self.L = L
        self.w = w
        self.R1 = R1
        self.R2 = R2

        # epitaxial design parameters
        self.lam = epi['lambda']
        self.n_eff = epi['n_eff']
        self.d_a = epi['d_a']
        self.Gamma = epi['Gamma']
        self.ni = epi['ni']
        self.eta_inj = epi['eta_inj']
        self.alpha_i = epi['alpha_i']
        self.gain = epi['gain']
        self.beta_sp = epi['beta_sp']
        self.srh_n1 = epi['srh_n1']
        self.srh_p1 = epi['srh_p1']
        self.srh_tau_n = epi['srh_tau_n']
        self.srh_tau_p = epi['srh_tau_p']
        self.rad_B = epi['rad_B']
        self.aug_Cn = epi['aug_Cn']
        self.aug_Cp = epi['aug_Cp']

        # additional parameters
        self.d_wg = self.d_a / self.Gamma
        self.vg = const.c / self.n_eff

        self.J_funs = list()  # current pulses
        self.z_grid = np.linspace(0, self.L, npoints)  # grid

    def add_current_pulse(self, pulse):
        "Add a 'CurrentPulse` object to the list of current pulses."
        self.J_funs.append(pulse)

    def total_current_density(self, t):
        "Calculate total current density at every grid point at time `t`."
        assert len(self.J_funs)>0
        J_total = np.zeros_like(self.z_grid)
        for J_obj in self.J_funs:
            for i, zi in enumerate(self.z_grid):
                Ji = J_obj.calculate(t, zi)
                J_total[i] += Ji
        return J_total

    def total_current(self, t):
        "Calculate total current through the device at time `t`."
        J_total = self.total_current_density(t)
        J_avg = (J_total[:-1]+J_total[1:])/2
        dz = self.z_grid[1:]-self.z_grid[:-1]
        I_total = np.sum(J_avg*dz)*self.w
        return I_total

    def calculate_power(self, sol):
        """
        Calculate output power from both sides of the laser using 

        Parameters
        ----------
        sol : scipy.OdeSolution
            Laser diode model parameters.

        Returns
        -------
        P1 : float
            Output power from mirror R1.
        P2 : float
            Output power from mirror R2.

        """
        N = sol.y
        m = len(self.z_grid)
        assert len(sol.y) == m*3
        Sout1 = N[2*m]
        Sout2 = N[2*m-1]
        Eph = const.h*const.c/self.lam
        Swg = self.d_wg*self.w
        P1 = (1-self.R1)*Sout1*self.vg*Swg*Eph
        P2 = (1-self.R2)*Sout2*self.vg*Swg*Eph
        return P1, P2
