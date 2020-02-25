#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 12:58:57 2019

@author: vs_golovin
"""

import numpy as np
from scipy.integrate import solve_ivp
from functools import partial
from rate_equations_0D import re_residuals_0D, re_jacobian_0D


def solve_RE(ld, I_fun, t_fin=10e-9, timesteps=3000, method='Radau', rtol=5e-5):
    """
    Solve 0D rate equations using `scipy.integrate.solve_ivp`.

    Parameters
    ----------
    ld : laser_diode.Laser_Diode
        Diode laser properties.
    I_func : function
        Pump current (A) vs time (s). Must have an `amplitude` argument.
    t_fin : number, optional
        Endpoint of integration interval (s). The default is 10e-9.
    timesteps : TYPE, optional
        Number of grid points for time. The default is 3000.
    method : string or `scipy.integrate.OdeSolver`, optional
        Integration method to be used by `scipy.integrate.solve_ivp`.
        The default is 'DOP853'.
    rtol : number
        Relative tolerance. The default is 1e-4.

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