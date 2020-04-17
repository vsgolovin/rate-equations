#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:10:29 2020

@author: vs_golovin
"""

from functools import partial
import numpy as np
from solver import solve_RE
from utilities import find_d2ydx2_zeros


def sp_init(ld, I_fun, t_fin, timesteps, I_init=5.0,
            method='Radau', rtol=5e-5, dI=1.0, dI_min=5e-3,
            I_max=None, P_min=0.1, d2ydx2_y_min=0.03):
    """
    Find an arbitrary current that ensures single-pulse operation.
    """

    def sol_power(current):
        """
        Shorthand for solving rate equations and calculating power.
        Has nothing to do with solar power.
        """
        pulse = partial(I_fun, amplitude=current)
        sol = solve_RE(ld=ld, I_fun=pulse, t_fin=t_fin, timesteps=timesteps,
                       method=method, rtol=rtol)
        assert sol.success
        P = ld.calculate_power(sol.y[1])
        return sol, P

    # initializing currents
    I_th = ld.estimate_Ith()
    if I_max and I_th > I_max:
        sol, _ = sol_power(I_max)
        return -1, sol
    I1 = I_th
    I2 = I_init*I_th
    dI = dI*I_th

    # minimum current
    sol1, P1 = sol_power(I1)
    if np.max(P1)>P_min:
        print("Function sp_init: P_min = %fW" % np.max(P1))
        return -1, sol1

    # maximum current
    sol2, P2 = sol_power(I2)

    # if I2 is too small
    reached_max = False
    while not reached_max and P2.max()<P_min:
        I2 += dI
        if I_max and I2>I_max:
            reached_max = True
            I2 = I_max
        sol2, P2 = sol_power(I2)
        if reached_max and P2.max()<P_min:
            return -1, sol2

    # single pulse at I2
    rx2, ry2 = find_d2ydx2_zeros(x=sol2.t, y=P2, y_min=d2ydx2_y_min)
    nr2 = len(ry2)
    if nr2==2:
        return I2, sol2

    # find current for single-pulse operation using bisection method
    while (I2-I1)>dI_min:
        I_mid = (I1+I2)/2
        sol, P = sol_power(I_mid)
        if P.max()<P_min:
            I1 = I_mid
            continue
        rx, ry = find_d2ydx2_zeros(x=sol.t, y=P, y_min=d2ydx2_y_min)
        nr = len(ry)
        if nr == 2:
            return I_mid, sol
        elif nr > 2:
            I2 = I_mid
            continue
        else:
            raise Exception("Function sp_init: I = %f, nr = %d" % (I_mid, nr))

    # could not find current
    return -1, sol


def sp_max_current(ld, I_fun, t_fin, timesteps=2000, method='Radau', rtol=5e-5,
                   dI_max=1.0, dI_min=1e-3, k=2.5, init_I=5.0, init_dI=1.0,
                   init_dI_min=5e-3, I_max=None, P_min=0.1, d2ydx2_y_min=0.03):
    """
    Find an arbitrary current that ensures single-pulse operation.
    """

    # find an initial value using sp_init()
    I, sol = sp_init(ld=ld, I_fun=I_fun, t_fin=t_fin, timesteps=timesteps,
                     I_init=init_I, method=method, rtol=rtol, dI=init_dI,
                     dI_min=init_dI_min, I_max=I_max, P_min=P_min,
                     d2ydx2_y_min=d2ydx2_y_min)
    if I<0:  # could not achieve single-pulse operation
        return I, sol

    # finding maximum current
    dI = dI_max
    while dI>=dI_min:
        I_new = I + dI
        pulse = partial(I_fun, amplitude=I_new)
        sol_new = solve_RE(ld=ld, I_fun=pulse, t_fin=t_fin, timesteps=timesteps,
                           method=method, rtol=rtol)
        assert sol_new.success
        rx, ry = find_d2ydx2_zeros(x=sol_new.t, y=sol_new.y[1],
                                   y_min=d2ydx2_y_min)
        nr = len(ry)  # number of inflection points
        if nr==2:     # single pulse
            I = I_new
            assert np.max(sol.y[1])<np.max(sol_new.y[1])
            sol = sol_new
        elif nr>2:    # trailing oscillations (or small bumps)
            dI /= k
        else:         # 0 or 1 root -- something went wrong
            raise Exception("Function sp_init: I = %f, nr = %d" % (I, nr))

    return I, sol
