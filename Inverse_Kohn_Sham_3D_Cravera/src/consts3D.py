"""
consts3D.py
===========
Physical constants and harmonic-oscillator (HO) parameters.

All lengths are in fm and all energies in MeV. The oscillator frequency is
stored as a wave number, omega [fm^-1] = (hbar*omega [MeV]) / (hbar*c), so that
the usual expressions keep their textbook form with hbar = c = 1 in mixed
MeV-fm units.

The trap frequency follows the standard nuclear systematics

    hbar*omega(A) = 41 * A^(-1/3) MeV,

and the oscillator is isotropic: the x, y and z functions are identical and
are kept separate only so that the rest of the code can address each axis
explicitly (the axially deformed case is handled in ``deformed.py``).
"""
import numpy as np

hbc = 197.327053  # hbar*c [MeV fm]
m = 938.91869     # nucleon mass [MeV] (average of proton and neutron)

# Kinetic prefactor alpha = hbar^2 / (2m) [MeV fm^2]; it multiplies |grad phi|^2
# in the kinetic-energy functional.
a = hbc**2 / (2 * m)


def nuclear_omega_x(A):
    """Oscillator frequency along x, as a wave number [fm^-1], for mass number A."""
    return 41. / (hbc * np.power(A, 1. / 3))


def b_x(A):
    """Oscillator length b = sqrt(hbar / (m*omega)) along x [fm]."""
    return np.sqrt(hbc / (m * nuclear_omega_x(A)))


def nuclearNu_x(A):
    """Coefficient nu = m*omega^2/2 of the trap V = nu*x^2 along x [MeV fm^-2]."""
    return 0.5 * m * nuclear_omega_x(A)**2


def nuclear_omega_y(A):
    """Oscillator frequency along y [fm^-1] (isotropic: equal to the x value)."""
    return 41. / (hbc * np.power(A, 1. / 3))


def b_y(A):
    """Oscillator length along y [fm]."""
    return np.sqrt(hbc / (m * nuclear_omega_y(A)))


def nuclearNu_y(A):
    """Coefficient nu = m*omega^2/2 of the trap along y [MeV fm^-2]."""
    return 0.5 * m * nuclear_omega_y(A)**2


def nuclear_omega_z(A):
    """Oscillator frequency along z [fm^-1] (isotropic: equal to the x value)."""
    return 41. / (hbc * np.power(A, 1. / 3))


def b_z(A):
    """Oscillator length along z [fm]."""
    return np.sqrt(hbc / (m * nuclear_omega_z(A)))


def nuclearNu_z(A):
    """Coefficient nu = m*omega^2/2 of the trap along z [MeV fm^-2]."""
    return 0.5 * m * nuclear_omega_z(A)**2
