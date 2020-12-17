# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 18:42:18 2020

@author: vsgolovin
"""

import numpy as np

def rect_pulse(t, t_max, width, t_rise, t_fall, amplitude, constant=0):
    """
    Generate a rectangular pulse.
    Uses sigmoid function for smoothing.

    Parameters
    ----------
    t : number
        current time
    t_max : number
        midpoint
    width : number
        pulse width (FWHM)
    t_rise : number
        rise time (5% - 95%)
    t_fall : number
        fall time (95% - 5%)
    amplitude : number
        pretty self-explanatory
    constant : number
        base level
    t_rise + t_fall <= width

    """
    f1 = 1/ (1 + np.exp( -(t-(t_max-width/2)) / (t_rise/6) ))
    f2 = 1/ (1 + np.exp( -(t-(t_max+width/2)) / (t_fall/6) ))
    return amplitude*(f1-f2)+constant

def process_results(ldm, sol):
    """
    Calculate current density, total current and output power using simulation
    results `sol` and model parameters `ldm`.

    Parameters
    ----------
    ldm : ldmodel.LDModel
        Laser diode model parameters.
    sol : scipy.integrate.OdeSolution
        Simulation results.

    Below 'm' denotes the number of `ldm` grid nodes, `n` denotes the number of
    time points (`n = len(sol.t)`).

    Returns
    -------
    J : numpy.ndarray
        Current density at every grid node.
        `J.shape = (m, n)`
    I : numpy.ndarray
        Total current through laser.
        `I.shape = (n, )`
    P1 : numpy.ndarray
        Output power from mirror R1.
        `P1.shape = (n, )`
    P2 : numpy.ndarray
        Output power from mirror R2.
        `P2.shape = (n, )`

    """
    # checking solution shape and adding shortcut names
    t = sol.t
    n = len(t)
    z = ldm.get_grid()
    m = len(z)
    _m, _n = sol.y.shape
    assert (_m==3*m) and (_n==n)

    # calculating current density and total current
    J = np.zeros((m, n))
    I = np.zeros(n)
    for i, ti in enumerate(t):
        Ji = ldm.total_current_density(ti)
        J[:, i] = Ji
        Ji_avg = (Ji[:-1]+Ji[1:])/2
        dz = z[1:]-z[:-1]
        I[i] = np.sum(Ji_avg*dz*ldm.w)

    # calculating output power
    P1, P2 = ldm.calculate_power(sol)

    return J, I, P1, P2