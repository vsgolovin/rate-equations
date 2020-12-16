# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:41:25 2020

@author: vsgolovin
"""

from functools import partial
import numpy as np
from scipy.integrate import solve_ivp
from recombination import srh, rad, auger
import constants as const

def residual(t, N, ldm):
    """
    Calculate residuals of 1D rate equations. Namely, an equation for electron
    density, an equation for forward-propagating photon density and an equation
    for backward-propagating photon density.

    Parameters
    ----------
    t : number
        Current time.
    N : numpy.ndarray
        Electron density and densities of photons moving forward and backward
        along the resonator axis z.
        of photons moving backward along z.
    ldm : ldmodel.LDModel
        Laser diode model parameters.

    Returns
    -------
    F : numpy.ndarray
        Residual of 1D rate equations for electron and photon densities.
        Has the same shape as `N`.

    """
    # unpacking N
    m = len(ldm.z_grid)  # number of grid nodes
    assert len(N)==m*3
    n  = N[   :m]    # electron density
    Sf = N[m  :2*m]  # forward-propagating photon density
    Sb = N[2*m:]     # backward-propagating p.d.
    S  = Sf+Sb

    # calculating values/vectors used in several equations
    dz = ldm.L / (m-1)  # assuming uniform mesh
    assert np.abs((ldm.z_grid[1]-ldm.z_grid[0]) - dz)/dz < 1e-6
    J = ldm.total_current_density(t)
    R_srh = srh(n=n, p=n, n1=ldm.srh_n1, p1=ldm.srh_p1, tau_n=ldm.srh_tau_n,
                tau_p=ldm.srh_tau_p)
    R_rad = rad(n=n, p=n, n0=ldm.ni, p0=ldm.ni, B=ldm.rad_B)
    R_aug = auger(n=n, p=n, n0=ldm.ni, p0=ldm.ni, Cn=ldm.aug_Cn, Cp=ldm.aug_Cp)
    R_rec = R_srh+R_rad+R_aug
    gain = np.array([ldm.gain(ni, Si) for ni, Si in zip(n, S)])

    # calculating residuals
    # electron density equation
    F1 = ldm.eta_inj*J/(const.q*ldm.d_a) - R_rec - ldm.vg*gain*S
    # forward-propagating photons
    F2 = np.zeros_like(F1)
    F2[0]  = (-ldm.vg*(-ldm.R1*Sb[0]+Sf[0])/(1*dz)
              +ldm.vg*Sf[0]*(ldm.Gamma*gain[0]-ldm.alpha_i)
              +ldm.Gamma*ldm.beta_sp*R_rad[0]/2)
    F2[1:] = (-ldm.vg*(-Sf[:-1]+Sf[1:])/(1*dz)
              +ldm.vg*Sf[1:]*(ldm.Gamma*gain[1:]-ldm.alpha_i)
              +ldm.Gamma*ldm.beta_sp*R_rad[1:]/2)
    # backward-propagating photons
    F3 = np.zeros_like(F2)
    F3[:-1] = ( ldm.vg*(-Sb[:-1]+Sb[1:])/(1*dz)
               +ldm.vg*Sb[:-1]*(ldm.Gamma*gain[:-1]-ldm.alpha_i)
               +ldm.Gamma*ldm.beta_sp*R_rad[:-1]/2)
    F3[-1]  = ( ldm.vg*(-Sb[-1]+ldm.R2*Sf[-1])/(1*dz)
               +ldm.vg*Sb[-1]*(ldm.Gamma*gain[-1]-ldm.alpha_i)
               +ldm.Gamma*ldm.beta_sp*R_rad[-1]/2)
    F = np.concatenate([F1, F2, F3])

    return F

def solve(ldm, y0, t_fin=5e-9, timesteps=1000):
    """
    Solve laser diode 1D rate equations. Uses `scipy.integrate.solve_ivp`.

    Parameters
    ----------
    ldm : ldmodel.LDModel
        Laser diode model parameters.
    y0 : numpy.ndarray
        Initial condition.
    
    """
    res = partial(residual, ldm=ldm)
    sol = solve_ivp(fun=res, t_span=(0, t_fin), y0=y0, vectorized=False,
                    t_eval=np.linspace(0, t_fin, timesteps))
    assert sol.success
    return sol