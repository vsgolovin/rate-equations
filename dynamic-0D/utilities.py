# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 18:17:31 2019

@author: vs_golovin
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


def rect_pulse(t, t_max, length, t_rise, t_fall, amplitude, constant=0):
    """
    Generate a rectangular pulse.
    Uses Fermi function for smoothing.
    
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
    t_rise + t_fall <= length
    """
    f1 = 1/ (1 + np.exp( -(t-(t_max-length/2)) / (t_rise/6) ))
    f2 = 1/ (1 + np.exp( -(t-(t_max+length/2)) / (t_fall/6) ))
    return amplitude*(f1-f2)+constant


def find_d2ydx2_zeros(x, y, y_min=0.05):
    """
    Find zeros of the second derivative (d2y/dx2). Only consider parts of the
    curve where `y >= y_min`.

    Parameters
    ----------
    x : `numpy.ndarray`
        x values.
    y : `numpy.ndarray`
        y values.
    y_min : float [0, 1], optional
        Minimum value of y (in relative units) to be considered.
        The default is 0.05.

    Returns
    -------
    roots_x : `numpy.ndarray`
        x coordinates of roots.
    roots_y : `numpy.ndarray`
        y coordinates of roots.

    """
    yn = y/np.max(y)
    ind = (yn >= y_min)
    x = x[ind]
    y = y[ind]

    d2y = np.zeros_like(y)
    d2y[1:-1] = y[:-2]-2*y[1:-1]+y[2:]
    roots_x = []
    roots_y = []
    for i in range(2, len(y)-1):
        if d2y[i]*d2y[i-1] < 0:
            roots_x.append((x[i]+x[i-1])/2)
            roots_y.append((y[i]+y[i-1])/2)
    roots_x = np.array(roots_x)
    roots_y = np.array(roots_y)
    return roots_x, roots_y


def find_FWHM(x, y):
    """
    Find FWHM (full width at half maximum) of the waveform.
    """
    xmax = x[np.argmax(y)]
    yn = y/np.max(y)
    spl = InterpolatedUnivariateSpline(x=x, y=yn-0.5)
    roots = spl.roots()
    x1 = roots[roots<xmax][-1]
    x2 = roots[roots>xmax][0]
    return x2-x1


def find_boundaries(x, y, level=0.9, y_min=1e-6):
    """
    Find boundaries of the shortest interval whose integral is at least
    `level` * total integral of y(x).

    Parameters
    ----------
    x : numpy.ndarray
        x values
    y : numpy.ndarray
        y values (should be non-negative)
    level : number
        Minimum value of the integral on the interval to be found relative to
        the total integral (0 < level < 1).
    y_min : number or None
        Cutoff value of y (relative units). If it is specified only the part of
        the waveform between first and last y values larger than `y_min` will
        be considered.
    """

    # checking passed values
    assert len(x) == len(y)
    assert level>0 and level<1
    if y_min:
        assert y_min>0 and y_min<1

    # cutting off waveform 'tails' with small y
    if y_min:
        yn = y / y.max()
        ix1 = 0
        while yn[ix1] < y_min:
            ix1 += 1
        ix2 = len(yn)-1
        while yn[ix2] < y_min:
            ix2 -= 1
        x = x[ix1 : ix2+1]
        y = y[ix1 : ix2+1]
    
    # finding length of the shortest possible interval
    n = len(y) - 1
    ydx = (y[1:] + y[:-1]) / 2 * (x[1:] - x[:-1])
    integral = np.sum(ydx)
    m = n - 1
    k_best = 0
    while m > 0:
        success = False
        for k in range(n-m):
            ip = np.sum(ydx[k:m+k])
            if ip >= level*integral:
                k_best = k
                success = True
                break
        if not success:
            break
        m -= 1

    # checking (greedily) if there is an interval
    # with a larger integral
    while ydx[m+k_best] > ydx[k_best]:
        k_best += 1

    # boundaries
    x1 = x[k_best]
    x2 = x[m+k_best+1]

    return x1, x2


def tanh_pulse(t, shift, rise_time, fall_time, width, amplitude):
    """
    Generate a pulse using `tanh` function.
    """
    i1 = np.tanh((t-shift)/rise_time)
    i2 = np.tanh((t-shift-width)/fall_time)
    return amplitude*(i1-i2)
