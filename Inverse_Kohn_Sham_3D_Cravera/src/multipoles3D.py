"""
multipoles3D.py
===============
Multipole moments for the 3D IKS solver (thesis, Section 6.1): real solid
harmonics, a diagnostic evaluator of Q_lm[rho], and the coefficients of the
optional multipole constraints passed to IPOPT.

Convention
----------
Real solid harmonics in Racah normalisation,

    R_lm(r) = sqrt(4 pi / (2l+1)) * r^l * Y_lm(theta, phi)   (real combinations),

which are plain polynomials in the Cartesian coordinates (no special
functions, no complex arithmetic):

    R_00  = 1
    R_10  = z               R_1,1  = x            R_1,-1 = y
    R_20  = (2z^2 - x^2 - y^2)/2
    R_2,1 = sqrt(3) x z     R_2,-1 = sqrt(3) y z
    R_2,2 = sqrt(3)(x^2 - y^2)/2                 R_2,-2 = sqrt(3) x y
    R_30  = (5z^3 - 3 z r^2)/2
    R_40  = (35 z^4 - 30 z^2 r^2 + 3 r^4)/8

The physical moment is Q_lm = int d^3r rho(r) R_lm(r), in fm^l (Eq. 6.2).
(The "standard" mass quadrupole Q_2 = int rho (2z^2 - x^2 - y^2) is 2*Q_20.)

Why the constraint is redundant in this code
--------------------------------------------
The variables are f_k = phi_k / sqrt(rho_target) and the density constraint
is imposed POINTWISE on the live nodes:

    c_r(f) = sum_k occ_k f_kr^2 - 1 = 0.

Hence rho_KS(r) = rho_target(r) * (1 + c_r), and the multipole constraint

    C_lm(f) = Q_lm[rho_KS] - Q_lm[rho_target]
            = sum_{r in live} w_r rho_r R_lm(r) * c_r(f)
            = a . c                                               (Eq. 6.5)

is an EXACT LINEAR COMBINATION of the density constraints already present,
with coefficients a_r = w_r rho_r R_lm(r). It adds no information: its
Jacobian row lies in the span of the existing rows (LICQ is violated) and the
multipliers are no longer unique.

Operational consequence (verifiable): with lam_r the multiplier of the
pointwise constraint and mu_j that of the multipole constraint, stationarity
only fixes the combination

    lam~_r = lam_r + sum_j mu_j a_r^(j)                           (Eq. 6.6)

and since V_KS = lam~ / (rho w), a potential extracted naively as
lam / (rho w) is SHIFTED with respect to the run without the constraint by

    Delta V(r) = - sum_j mu_j R_lm^(j)(r) / scale_j               (Eq. 6.7)

For l = 0 this is a constant (absorbed by the gauge anchoring); for l >= 1
it is a genuine polynomial deformation of the extracted potential.
``gauge_correction`` below returns the field needed to undo it.

Normalisation of the constraint
-------------------------------
The coefficients carry dimension fm^l: for l = 2 they are O(b^2), whereas
IPOPT works best with O(1) constraints. R_lm is therefore divided by
scale = b^l (b = oscillator length, Eq. 6.3). The scale is returned because
it is needed by the gauge correction above.
"""

import numpy as np

SQRT3 = np.sqrt(3.0)


# ─────────────────────────────────────────────────────────────────────────────
# Real solid harmonics
# ─────────────────────────────────────────────────────────────────────────────

def solid_harmonic(l, m, X, Y, Z):
    """R_lm(x, y, z), real solid harmonic (Racah normalisation), on 3D arrays.

    Implemented: all l = 0, 1, 2, plus (3, 0) and (4, 0). These are the
    channels needed for axially symmetric or triaxial densities; higher
    orders can be added by inserting the corresponding polynomial in the table.
    """
    key = (int(l), int(m))
    R2 = X**2 + Y**2 + Z**2

    # Lazy table: only the requested polynomial is evaluated.
    table = {
        (0,  0): lambda: np.ones_like(X),
        (1,  0): lambda: Z,
        (1,  1): lambda: X,
        (1, -1): lambda: Y,
        (2,  0): lambda: (2.0 * Z**2 - X**2 - Y**2) / 2.0,
        (2,  1): lambda: SQRT3 * X * Z,
        (2, -1): lambda: SQRT3 * Y * Z,
        (2,  2): lambda: SQRT3 * (X**2 - Y**2) / 2.0,
        (2, -2): lambda: SQRT3 * X * Y,
        (3,  0): lambda: (5.0 * Z**3 - 3.0 * Z * R2) / 2.0,
        (4,  0): lambda: (35.0 * Z**4 - 30.0 * Z**2 * R2 + 3.0 * R2**2) / 8.0,
    }
    if key not in table:
        raise NotImplementedError(
            f"Armonica solida (l={l}, m={m}) non implementata. "
            f"Disponibili: {sorted(table.keys())}")
    return table[key]()


def parse_lm_list(lm_list):
    """Normalise a list of constraints into a list of validated (l, m) tuples.

    ``None`` gives an empty list; |m| > l raises ValueError.
    """
    if lm_list is None:
        return []
    out = []
    for item in lm_list:
        l, m = int(item[0]), int(item[1])
        if abs(m) > l:
            raise ValueError(f"|m| > l non ammesso: (l={l}, m={m})")
        out.append((l, m))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic evaluator (independent of the optimiser)
# ─────────────────────────────────────────────────────────────────────────────

def multipole_moments(rho, grid, w3D, lm_list, mask=None):
    """Q_lm = int rho R_lm dV with the quadrature of the grid (in fm^l).

    Parameters
    ----------
    rho     : array (Nx, Ny, Nz)
    grid    : tuple (x, y, z) of 1D node arrays
    w3D     : array (Nx, Ny, Nz) of quadrature weights
    lm_list : list of (l, m)
    mask    : optional 0/1 array; if given, it restricts the integral. Use the
              SAME mask for rho_KS and rho_target, so that discretisation
              errors cancel to first order in the difference.

    Returns
    -------
    dict {(l, m): value}
    """
    x, y, z = grid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    w = w3D if mask is None else w3D * mask
    return {(l, m): float(np.sum(w * rho * solid_harmonic(l, m, X, Y, Z)))
            for (l, m) in parse_lm_list(lm_list)}


# ─────────────────────────────────────────────────────────────────────────────
# Constraint coefficients for the optimiser
# ─────────────────────────────────────────────────────────────────────────────

def build_constraint_coefficients(lm_list, grid, w3D, tab_rho, live_nodes,
                                  length_scale=1.0):
    """Coefficients a_r of the multipole constraints, restricted to the live nodes.

        C_lm(f) = sum_{r in live} a_r * (sum_k occ_k f_kr^2 - 1)
        a_r     = w_r * rho_r * R_lm(r) / length_scale^l

    Parameters
    ----------
    live_nodes   : flat indices of the nodes where the density constraint is active
    length_scale : b in the normalisation scale = b^l (Eq. 6.3)

    Returns
    -------
    coefs  : array (n_mp, n_live)  coefficients on the live nodes
    scales : array (n_mp,)         length_scale**l used for each constraint
    """
    lm = parse_lm_list(lm_list)
    if len(lm) == 0:
        return np.zeros((0, len(live_nodes))), np.zeros(0)

    x, y, z = grid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    base = (w3D * tab_rho).ravel()[live_nodes]

    coefs, scales = [], []
    for (l, m) in lm:
        s = float(length_scale) ** l
        R = solid_harmonic(l, m, X, Y, Z).ravel()[live_nodes]
        coefs.append(base * R / s)
        scales.append(s)
    return np.asarray(coefs), np.asarray(scales)


def gauge_correction(grid, lm_list, mu, scales):
    """Polynomial field Delta V(r) = sum_j mu_j R_lm^(j)(r) / scale_j (Eq. 6.7).

    It must be ADDED BACK to the potential extracted as lam / (rho w) to return
    to the gauge of the run without multipole constraints. ``mu`` are the
    multipliers of the multipole constraints (the last n_mp entries of IPOPT's
    ``mult_g``) and ``scales`` the values returned by
    ``build_constraint_coefficients``.

    Returns an array (Nx, Ny, Nz) in MeV (the same units as the multipliers).
    """
    lm = parse_lm_list(lm_list)
    if len(lm) == 0:
        x, y, z = grid
        return np.zeros((len(x), len(y), len(z)))
    x, y, z = grid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    dV = np.zeros_like(X, dtype=float)
    for j, (l, m) in enumerate(lm):
        dV += float(mu[j]) * solid_harmonic(l, m, X, Y, Z) / float(scales[j])
    return dV
