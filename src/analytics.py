# src/analytics.py
# Analytical reference solutions — provided in full.
# You do not need to implement these, but you should understand the equations.
import numpy as np


def timoshenko_deflection(x, L, h, P, E, nu):
    """
    Vertical deflection v(x, y=0) along the neutral axis of a tip-loaded cantilever.
    From Timoshenko & Goodier, Theory of Elasticity.
    Includes both bending and shear deformation contributions.
    """
    I = h**3 / 12
    G = E / (2 * (1 + nu))
    return (P / (6 * E * I)) * (3 * L * x**2 - x**3) + \
           (P / (2 * G * I)) * (L - x) * (h**2 / 4)


def euler_bernoulli_deflection(x, L, P, E, h):
    """Vertical deflection from classical beam theory (no shear deformation)."""
    I = h**3 / 12
    return P * x**2 * (3 * L - x) / (6 * E * I)


def timoshenko_sigma_xx(x, y, L, P, h):
    """Normal stress sigma_xx(x, y) in a tip-loaded cantilever from elasticity theory."""
    I = h**3 / 12
    return -P * (L - x) * y / I


def timoshenko_tau_xy(y, P, h):
    """Shear stress tau_xy(y) across the beam depth from elasticity theory."""
    I = h**3 / 12
    return (P / (2 * I)) * (h**2 / 4 - y**2)


def kirsch_stress_polar(r, theta, R, sigma_inf):
    """
    Kirsch solution in polar coordinates.
    Returns (sigma_rr, sigma_tt, tau_rt).
    At r = R, theta = pi/2: sigma_tt = 3 * sigma_inf.
    """
    rho = R / r
    sigma_rr = (sigma_inf / 2) * ((1 - rho**2) + (1 - 4*rho**2 + 3*rho**4) * np.cos(2*theta))
    sigma_tt = (sigma_inf / 2) * ((1 + rho**2) - (1 + 3*rho**4) * np.cos(2*theta))
    tau_rt   = -(sigma_inf / 2) * (1 + 2*rho**2 - 3*rho**4) * np.sin(2*theta)
    return sigma_rr, sigma_tt, tau_rt


def kirsch_stress_cartesian(r, theta, R, sigma_inf):
    """Kirsch solution transformed to Cartesian (sigma_xx, sigma_yy, tau_xy)."""
    sigma_rr, sigma_tt, tau_rt = kirsch_stress_polar(r, theta, R, sigma_inf)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos2 = cos_t**2
    sin2 = sin_t**2
    sincos = sin_t * cos_t

    sigma_xx = sigma_rr * cos2 + sigma_tt * sin2 - 2 * tau_rt * sincos
    sigma_yy = sigma_rr * sin2 + sigma_tt * cos2 + 2 * tau_rt * sincos
    tau_xy   = (sigma_rr - sigma_tt) * sincos + tau_rt * (cos2 - sin2)
    return sigma_xx, sigma_yy, tau_xy
