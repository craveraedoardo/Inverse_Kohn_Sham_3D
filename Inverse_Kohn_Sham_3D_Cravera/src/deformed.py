"""
deformed.py
===========
Axially deformed harmonic oscillator, used to build deformed target densities
(thesis, Sections 5.4 and 6.2).

An axially symmetric oscillator has omega_x = omega_y = omega_perp != omega_z.
The deformation is parametrised by delta at CONSTANT VOLUME,
omega_perp^2 * omega_z = omega_0^3 (omega_0 being the spherical frequency), so
that targets with different delta have the same overall size:

    omega_z    = omega_0 * (1 - 2*delta/3),
    omega_perp = omega_0 * (1 +   delta/3),

after which both frequencies are rescaled to enforce the volume condition
exactly.

    delta > 0  ->  prolate (cigar shape, elongated along z)
    delta < 0  ->  oblate  (disc shape)
    delta = 0  ->  spherical

The functions return the oscillator lengths (b_perp, b_z), to be passed to
``eigenfunctions3D.HO3D``, and the coefficients of the anisotropic reference
potential

    V(x, y, z) = (m/2) * [ omega_perp^2 (x^2 + y^2) + omega_z^2 z^2 ].
"""
import numpy as np
from consts3D import hbc, m, nuclear_omega_x


def deformed_frequencies(A, delta):
    """Return (omega_perp, omega_z) in fm^-1 for deformation delta at constant volume."""
    w0 = nuclear_omega_x(A)
    wz = w0 * (1.0 - 2.0 * delta / 3.0)
    wp = w0 * (1.0 + delta / 3.0)
    s = (w0**3 / (wp**2 * wz))**(1.0 / 3.0)   # rescale to constant volume
    return wp * s, wz * s


def deformed_lengths(A, delta):
    """Return the oscillator lengths (b_perp, b_z) in fm for deformation delta."""
    wp, wz = deformed_frequencies(A, delta)
    bp = np.sqrt(hbc / (m * wp))
    bz = np.sqrt(hbc / (m * wz))
    return bp, bz


def deformed_nu(A, delta):
    """Coefficients of the anisotropic trap V = nu_perp (x^2 + y^2) + nu_z z^2.

    nu = m*omega^2/2 with omega in fm^-1, so V is in MeV for coordinates in fm.
    Returns (nu_perp, nu_z).
    """
    wp, wz = deformed_frequencies(A, delta)
    return 0.5 * m * wp**2, 0.5 * m * wz**2
