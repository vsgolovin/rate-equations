# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:53:36 2020

A collection of functions for calculating recombination rates.

@author: vsgolovin
"""

def srh(n, p, n1, p1, tau_n, tau_p):
    """
    Shockley-Read-Hall recombination rate.
    """
    F = (n*p - n1*p1) / ((n+n1)*tau_p + (p+p1)*tau_n)
    return F

def rad(n, p, n0, p0, B):
    """
    Radiative recombination rate.
    """
    F = B*(n*p-n0*p0)
    return F

def auger(n, p, n0, p0, Cn, Cp):
    """
    Auger recombination rate.
    """
    F = (Cn*n+Cp*p) * (n*p-n0*p0)
    return F
