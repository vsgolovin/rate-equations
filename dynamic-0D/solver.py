# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:32:22 2019

@author: vs_golovin
"""

from functools import partial
import numpy as np
from scipy.integrate import solve_ivp
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


def solve_RE(ld, I_fun, t_fin=10e-9, timesteps=3000, method='Radau', rtol=5e-5):
    """
    Solve 0D rate equations using `scipy.integrate.solve_ivp`.

    Parameters
    ----------
    ld : laser_diode.Laser_Diode
        Diode laser properties.
    I_fun : function
        Pump current (A) vs time (s). Must have an `amplitude` argument.
    t_fin : number, optional
        Endpoint of integration interval (s). The default is 10e-9.
    timesteps : integer, optional
        Number of grid points for time. The default is 3000.
    method : string or `scipy.integrate.OdeSolver`, optional
        Integration method to be used by `scipy.integrate.solve_ivp`.
        The default is 'Radau'.
    rtol : number
        Relative tolerance. The default is 5e-5.

    Returns
    -------
    sol : `scipy.integrate.OdeSolution`
         Solution.

    """
    res = partial(re_residuals_0D, I_fun=I_fun, ld=ld)
    jac = partial(re_jacobian_0D, I_fun=I_fun, ld=ld)
    sol = solve_ivp(fun=res, t_span=(0, t_fin), y0=[ld.n_i, 0], method=method,
                    jac=jac, t_eval=np.linspace(0, t_fin, timesteps), rtol=rtol)
    return sol