"""
eigenfunctions3D.py
===================
Cartesian eigenfunctions of the 3D harmonic oscillator, returned as callables
that can be evaluated on any tensor-product grid (uniform grid or Lagrange
mesh).

The single-particle state (nx, ny, nz) is the product of three 1D functions

    phi(x, y, z) = h_nx(x/bx) h_ny(y/by) h_nz(z/bz),
    h_n(u)       = H_n(u) exp(-u^2/2),

with H_n the physicists' Hermite polynomial. Different lengths (bx, by, bz)
give the anisotropic (deformed) oscillator.
"""
import numpy as np
from numpy.polynomial.hermite import Hermite


def HO3D(nx, ny, nz, bx, by, bz, dV):
    """Return a callable psi(x, y, z) for the HO eigenfunction (nx, ny, nz).

    Parameters
    ----------
    nx, ny, nz : int
        Oscillator quanta along each axis.
    bx, by, bz : float
        Oscillator lengths [fm].
    dV : float
        Volume element used to normalise the function on the grid,
        sum(|psi|^2) * dV = 1.

    Notes
    -----
    ``psi`` takes the three 1D coordinate arrays (not a meshgrid) and returns
    an array of shape (len(x), len(y), len(z)).

    The normalisation with a CONSTANT dV is exact only on a uniform grid. On
    the Lagrange mesh the integration weights vary from node to node, so the
    callers re-normalise the result with the mesh weights
    (``phi /= sqrt(sum(w3D * phi**2))``); see e.g. ``build_target_density``
    in the test scripts and ``iKS_problem.getStartingPoint``.
    """

    def f1d(x, n, b):
        Hn = Hermite.basis(n)
        h = Hn(x / b) * np.exp(-(x / b)**2 / 2)
        return h

    def psi(x, y, z):
        # Separable product, broadcast to a (Nx, Ny, Nz) array.
        fx = f1d(x, nx, bx)[:, None, None]
        fy = f1d(y, ny, by)[None, :, None]
        fz = f1d(z, nz, bz)[None, None, :]

        psi_raw = fx * fy * fz

        norm = np.sum(psi_raw**2) * dV
        return psi_raw / np.sqrt(norm)
    return psi
