# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:51:29 2019

@author: vs_golovin
"""

import numpy as np
import constants as const

class Laser_Diode():
    """
    Class for storing diode laser properties.

    Parameters:
    ----------
    epi : dict
        Epitaxial design properties.
    L : number
        Resonator length (cm).
    w : number
        Stripe width (cm).
    R1 : number
        Back mirror reflectivity (0<R1<1).
    R2 : number
        Front mirror reflectivity (0<R1<1).
    beta_sp : number
        Spontaneous emission factor (unitless).
    """
    def __init__(self, epi, L, w, R1, R2, beta_sp=1e-5):

        # structural properties
        self.L = L
        self.w = w
        self.R1 = R1
        self.R2 = R2

        # epitaxial design properties
        self.epi_name = epi['name']
        self.lam = epi['wavelength']
        self.index = epi['index']
        self.d_a = epi['d_a']
        self.Gamma = epi['Gamma']
        self.g_0 = epi['g_0']
        self.n_tr = epi['n_tr']
        self.n_i = epi['n_i']
        self.n_gmin = epi['n_gmin']
        self.eps_gc = epi['eps_gc']
        self.rec_B = epi['B']
        self.rec_C = epi['C']
        self.eta_inj = epi['eta_inj']
        self.alpha_i = epi['alpha_i']

        # spontaneous emission factor
        self.beta_sp = beta_sp

        # calculating some useful parameters
        self._calculate_params()


    def _calculate_params(self):
        """
        Update calculated parameters.
        """
        self.alpha_m = 1/(2*self.L)*np.log(1/(self.R1*self.R2))
        self.V_a  = self.d_a*self.w*self.L
        self.V_wg = self.V_a/self.Gamma


    def set_d_a(self, d_a):
        """
        Change active region thickness.
        """
        self.d_a = d_a
        self._calculate_params()


    def set_Gamma(self, Gamma):
        """
        Change active region optical confinement factor.
        """
        self.Gamma = Gamma
        self._calculate_params()


    def set_L(self, L):
        """
        Change resonator length.
        """
        self.L = L
        self._calculate_params()


    def gain(self, n, S):
        """
        Material gain for given free electron density `n`
        and photon density `S`.
        """
        g = self.g_0*np.log(np.max((n, self.n_gmin)) / self.n_tr)
        g /= (1+self.eps_gc*S)
        return g


    def estimate_Ith(self):
        """
        Calculate an estimate of threshold current.

        Returns
        -------
        I_th : number
            Estimated threshold current (A).

        """
        g_th = (self.alpha_i+self.alpha_m)/self.Gamma  # threshold gain
        n_th = self.n_tr*np.exp(g_th/self.g_0)  # threshold carrier density
        I_th = const.q_elem*self.V_a/self.eta_inj*(
                self.rec_B*(n_th**2-self.n_i**2)
               +self.rec_C*(n_th**3-self.n_i**3)
               )
        return I_th


    def calculate_power(self, S):
        """
        Calculate output power corresponding to photon density `S`.

        Parameters
        ----------
        S : number or numpy.ndarray
            0D photon density (cm-3).

        Returns
        -------
        P : number or numpy.ndarray
            Output power (W).

        """
        E_ph = const.h*const.c/self.lam
        v_g = const.c/self.index
        return S*self.V_wg*self.alpha_m*v_g*E_ph
        
