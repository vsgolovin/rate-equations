#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 21:57:16 2020

Contains a function for evaluating the largest possible current suitable
for "short pulse" operation. Whether the output pulse is considered short is
decided using one of implemented methods.

@author: vs_golovin
"""

from functools import partial
import numpy as np
from solver import solve_RE
from utilities import find_d2ydx2_zeros, find_boundaries


def d2ydx2(x, y, y_min=0.05, nroots=2):
    """
    Calculates the number of second derivative roots and compares it to
    `nroots`. Normally single oscillation waveform corresponds to `nroots=2`.
    
    Parameters
    ----------
    x : numpy.ndarray
        x values of the waveform
    y : numpy.ndarray
        y values of the waveform
    y_min : number
        Minimum value of y (in relative units) to be considered.
    nroots :
        The desired number of second derivative roots.
    
    Returns
    -------
        flag : boolean
            `True` if the number of second derivative roots is equal to
            `nroots`, `False` otherwise.
        nroots_calc :
            Calculated number of second derivative roots.
    """
    rx, ry = find_d2ydx2_zeros(x, y, y_min=y_min)
    nroots_calc = len(ry)
    if nroots_calc==nroots:
        return True, nroots
    else:
        return False, nroots_calc


def pulse_width(x, y, width, level=0.90, y_min=1e-6):
    """
    Calculates pulse width using `utilities.find_boundaries` and compares
    the result to `width`.
    
    Parameters
    ----------
    x : numpy.ndarray
        x values of the waveform
    y : numpy.ndarray
        y values of the waveform
    width : number
        Target pulse width (seconds).
    level : number
        Minimum value of the integral on the interval to be found relative to
        the total integral (0 < level < 1).
    y_min : number
        Cutoff value of y (relative units). If it is nonzero only the part of
        the waveform between first and last y values larger than `y_min` will
        be considered.
    
    Returns
    -------
    flag : boolean
        `True` if calculated pulse width is smaller than `width`, `False`
        otherwise.
    width_calc : number
        Calculated pulse width (seconds).
    
    """
    x1, x2 = find_boundaries(x, y, level=level, y_min=y_min)
    width_calc = x2 - x1
    if width_calc<=width:
        return True, width_calc
    else:
        return False, width_calc


METHODS = {'2nd_deriv' : d2ydx2,
           'width' : pulse_width}


def max_current(ld, I_fun, t_fin, timesteps=2000, int_method='Radau',
                rtol=5e-5, method='width', P_min=10.0, I_max=None,
                I_start=1.0, dI_start=0.2, dI_final=1e-3, k=2.5, **options):
    """
    Find maximum possible current that still corresponds to a short output
    pulse. Whether the pulse is short is decided using `method`. It is assumed
    that the range of suitable currents is continuous and its length is larger
    than `dI_start`, otherwise correct results are not guaranteed.

    Parameters
    ----------
    ld : laser_diode.Laser_Diode
        An object containing all the laser diode parameters.
    I_fun : function
        A function describing current pulse. Must have an `amplitude` parameter.
    t_fin : number
        Final point of time. Numerical integration is performed over range
        (0, `t_fin`).
    timesteps : number
        Number of points of the used time grid.
    int_method : string or scipy.integrate.OdeSolver
        Used method of numerical integration.
    rtol : number
        Relative tolerance, a parameter passed to `int_method`.
    method : string
        One of implemented methods to decide whether calculated waveform 
        can be considered short.
    P_min : number
        Minimum peak output power (in Watts) that should be reached before
        ruling out short pulse operation. In other words, if the largest
        calculated power exceeds `P_min` and not a single obtained optical
        pulse can be considered short, than it is assumed that short pulse
        operation is impossible.
    I_max : number
        An upper limit of possible current pulse amplitude (Amperes).
    I_start : number or None
        Starting (minimum) pulse current amplitude in terms of threshold
        current, which is calculated using `ld.estimate_Ith` method.
    dI_start : numberMultiplier
        Starting (maximum) step for increasing current pulse amplitude
        (Amperes). Should be smaller than the range of current amplitudes
        suitable for short pulse operation.
    dI_final : number
        Final (minimum) step for increasing current pulse amplitude (Amperes).
        Defines the precision of calculated maximum current.
    k : number
        Coefficient used while decreasing current amplitude step.
        `dI_new = dI_old / k`

    Returns
    -------
    I_best : number
        Maximum current amplitude that ensures short output pulse.
        `-1` if unable to find any such current.
    sol_best : object
        Return value of `scipy.integrate.solve_ivp`. Important properties:
            t : numpy.ndarray
                Time points
            y : numpy.ndarray
                Solution of time dependent rate equations.
                `y[0]` -- carrier density, `y[1]` -- photon density.
        For more information check `scipy.integrate.solve_ivp` documentation.
    results : numpy.ndarray
        An array of all current amplitudes (zeroth row) and all characteristic
        values used to classify pulse as short or not, i.e. pulse lengths
        (first row).
    """

    def calc_all(I):
        """
        A shorthand for carrying out numerical integration and some other
        operations.
        """
        pulse = partial(I_fun, amplitude=I)
        sol = solve_RE(ld, pulse, t_fin=t_fin, timesteps=timesteps,
                       method=int_method, rtol=rtol)
        assert sol.success
        P = ld.calculate_power(sol.y[1])
        flag, value = method(sol.t, P, **options)
        return sol, P.max(), flag, value

    # some tests
    assert dI_final < dI_start
    assert k > 1

    # select the method describing requirements for the output pulse
    if method in METHODS:
        method = METHODS[method]
    else:
        raise ValueError("Unknown `method` {}. Please select one of {}."\
                         .format(method, tuple(METHODS.keys())))

    # minimum current and initalization
    I_th = ld.estimate_Ith()
    I = I_th * I_start
    sol, P, success, value = calc_all(I)
    results = np.array([[I], [value]])
    if I_max and I>I_max:
        return -1, sol, results

    # finding suitable current (steps of dI_start)
    dI = dI_start
    P_best = P
    while (not success) and P<P_min and (not I_max or I_max and (I+dI)<I_max):
        I += dI
        sol, P, success, value = calc_all(I)
        results = np.append(results, [[I], [value]], axis=1)
        if success:
            sol_best = sol
            P_best = P
    if not success:
        return -1, sol, results
    
    # finding maximum current
    while dI>dI_final:
        if not I_max or (I+dI) <= I_max:
            I += dI
        else:
            dI /= k
            continue
        sol, P, flag, value = calc_all(I)
        results = np.append(results, [[I], [value]], axis=1)
        if flag:
            assert P>=P_best
            I_best = I
            sol_best = sol
            P_best = P
        else:
            I -= dI
            dI /= k

    return I_best, sol_best, results
