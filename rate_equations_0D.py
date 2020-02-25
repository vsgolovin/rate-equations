# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:32:22 2019

@author: vs_golovin
"""

import numpy as np
import constants as const

def re_residuals_0D(t, N, I_fun, ld):
    """
    Parameters:
    t : number
        Time (s).
    N : (number, number)
        Electron and photon densities (cm-3).
    I_fun : function
        Current (A) as a function of time.
    ld : laser_diode.Laser_Diode
        Laser properties.
    """
    n, S = N
    V_a = ld.w*ld.L*ld.d_a
    v_g = const.c/ld.index
    
    F1 = (ld.eta_inj*I_fun(t)/(const.q_elem*V_a) - ld.rec_B*(n**2-ld.n_i**2)
          - ld.rec_C*(n**3-ld.n_i**3) - v_g*ld.gain(n, S)*S)
    F2 = (v_g*(ld.Gamma*ld.gain(n, S) - ld.alpha_i - ld.alpha_m)*S
          + ld.beta_sp*ld.rec_B*(n**2-ld.n_i**2))
    
    return np.array([F1, F2])


def re_jacobian_0D(t, N, I_fun, ld):
    """
    Parameters:
    t : number
        Time (s).
    N : (number, number)
        Electron and photon densities (cm-3).
    I_fun : function
        Current (A) as a function of time.
    ld : laser_diode.Laser_Diode
        Laser properties.
    """
    n, S = N
    v_g = const.c/ld.index
    gain = ld.gain(n, S)
    
    F11 = -2*ld.rec_B*n - 3*ld.rec_C*n**2 - v_g*ld.g_0/n*S/(1+ld.eps_gc*S)
    F12 = -v_g*gain/(1+ld.eps_gc*S)
    F21 = v_g*ld.Gamma*ld.g_0/n*S/(1+ld.eps_gc*S) + ld.beta_sp*2*ld.rec_B*n
    F22 = v_g*(ld.Gamma*gain/(1+ld.eps_gc*S) - ld.alpha_i - ld.alpha_m)
    
    return np.array([[F11, F12], [F21, F22]])
