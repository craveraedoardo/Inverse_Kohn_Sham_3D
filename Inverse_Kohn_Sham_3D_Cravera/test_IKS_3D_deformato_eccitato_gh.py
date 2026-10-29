"""
test_IKS_3D_deformato_eccitato_gh.py
====================================
Excited configuration of an axially deformed potential: does the CV-IKS
method reconstruct it? Thesis: Section 6.2 (Tables 6.3 and 6.4).

Question
--------
In an oscillator deformed along z the single-particle levels reorder. With
    E(nx, ny, nz) = (nx + 1/2) hw_perp + (ny + 1/2) hw_perp + (nz + 1/2) hw_z
for delta > 0 (prolate, w_z < w_perp) the level (001) drops below (100) and
(010), which were degenerate with it at delta = 0. Hence, for A = 4:

    CONFIG "fondamentale" (ground)   {(000), (001)}  density elongated along z, Q20 > 0
    CONFIG "eccitata"     (excited)  {(000), (100)}  density elongated along x, Q22 != 0

Both configurations live in the SAME potential, and the single-particle gap
is hbar (w_perp - w_z), exactly the gap that opens at the level crossing.
Building the target from one or the other asks the CV-IKS to reconstruct the
same anisotropic potential from a ground-state or from an excited-state density.

What is at stake
----------------
The Levy constrained search defines
    T_s[rho] = min_{Phi -> rho} <Phi| T |Phi>,
i.e. the MINIMUM over all determinants that reproduce rho. For the aufbau
configuration that minimum is reached by the determinant that generated the
density. For an excited configuration there is no such guarantee: if another
determinant with the same density and lower kinetic energy exists, the
solver will find it, and the reconstructed potential will NOT be the
generating one even though the density is reproduced perfectly.

The test does not assume either outcome. The decisive diagnostics are the
comparison between T_alpha of the solution and the exact T of the generating
configuration, and the overlap matrix S with the generating orbitals:

  * T_alpha ~ T_exact(config) and S unitary -> the solver found the same
    determinant (up to a U(N_occ) rotation); the reconstructed potential is
    compared with the anisotropic generating one;
  * T_alpha << T_exact(config) or S not unitary -> the solver found a
    DIFFERENT determinant with the same density and lower kinetic energy; a
    limit of the formulation, not a numerical error.

The gap between the two configurations is ~10 MeV at delta = 0.4, whereas the
known discrepancy between T_alpha and the true kinetic energy due to the
discrete product rule is ~0.2 MeV on the Lagrange mesh: the two effects are
two orders of magnitude apart and cannot be confused.

Starting point (CFG["START"])
-----------------------------
  * "aufbau"      : the solver's getStartingPoint (isotropic orbitals of the
                    spherical aufbau set, fixed orientation along x);
  * "costruzione" : the generating orbitals (uses the answer: only tells
                    whether the generating determinant is a local minimum);
  * "orientato"   : the ORIENTED START of thesis Section 6.2 — configuration
                    and oscillator lengths deduced from the multipole moments
                    of the target only (kurtosis per axis, Eq. 6.10, then
                    second moments, Eq. 6.11). No privileged information.
The ``scan`` mode measures the basin of attraction of the oriented start by
perturbing the estimated lengths (isotropically, b -> b(1+eps), and in shape,
b_perp -> b_perp(1-eta), b_z -> b_z(1+eta)).

Technical notes
---------------
1. The internal mesh of the optimiser is built with b_x(A) = b_y(A) = b_z(A)
   (consts3D), so it is ISOTROPIC even for a deformed system. The target is
   therefore tabulated on the same isotropic mesh; it is the ORBITALS that
   are anisotropic, with lengths (b_perp, b_perp, b_z) from deformed.py. The
   isotropic mesh is less efficient for a deformed density, but both
   configurations are compared on the same grid.
2. The optimiser takes the set of orbitals from getOrbitalSet(Z, basis), which
   for A = 4 always returns {(000),(100)}. It only uses their number and
   occupations, which are the same for both configurations, so no change is
   needed. The default starting point does derive from that set, though,
   hence the START options above.
3. Lagrange mesh only. On the uniform grid the minimum of the discrete
   functional is spurious and dual_inf_tol acts as a regularisation
   parameter: a competing determinant would be indistinguishable from an artefact.

Usage
-----
    python test_IKS_3D_deformato_eccitato_gh.py            # delta = +0.40, aufbau start
    python test_IKS_3D_deformato_eccitato_gh.py 0.6        # another delta
    python test_IKS_3D_deformato_eccitato_gh.py warm       # start from the generating orbitals
    python test_IKS_3D_deformato_eccitato_gh.py orient     # oriented start (Table 6.4)
    python test_IKS_3D_deformato_eccitato_gh.py scan       # basin scan of the oriented start

Output: test_3D_data_deformato_eccitato/IKS_Z4_deformato_eccitato_delta+0.40.pdf
"""

import sys
import os
import re
import time
import tempfile
from contextlib import contextmanager

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the modules in src/ importable regardless of the working directory.
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from consts3D import a as ALPHA, hbc, b_x, b_y, b_z
from deformed import deformed_frequencies, deformed_lengths, deformed_nu
from orbitals3D import uncoupledHObasis
from optimizer3D_lm import iKS_problem
from lagrange_mesh_3D import LagrangeMesh3D
from eigenfunctions3D import HO3D
from multipoles3D import multipole_moments
from plot_helpers import physical_radius, potential_ylim

# ─────────────────────────────────────────────────────────────────────────────
# Physical case and run configuration
# ─────────────────────────────────────────────────────────────────────────────
N_PART = 4
OCC    = 2

CFG = dict(
    DELTA      = 0.40,    # deformation at constant volume; > 0 prolate
    N_GH       = 28,      # nodes per axis
    MESH_SCALE = 1.0,     # mesh scale s
    BETA       = 1e-3,    # regularisation weight
    TOL        = 1e-8,    # IPOPT tol and constr_viol_tol
    DUAL_TOL   = 1e-8,    # dual_inf_tol (actually passed to IPOPT)
    # IPOPT also stops at an "acceptable level" after acceptable_iter
    # consecutive iterations satisfying looser tolerances (default 15).
    # Raising max_iter has no effect on that termination: with max_iter=1000
    # the run still stops at 99 iterations. With 0 the criterion is disabled
    # and the solver continues to the tight tolerances or to max_iter.
    MAX_ITER   = 500,
    ACCEPTABLE_ITER = 0,
    # Starting point. getStartingPoint builds f from the ISOTROPIC oscillator
    # orbitals of the aufbau set {(000),(100)}: for a target elongated along z
    # this is a distant point. Starting from the GENERATING orbitals instead
    # answers a precise question: is the generating determinant a local
    # minimum of the discrete functional? If the solver stays there, the
    # problem was the start; if it moves away, that determinant is not a minimum.
    # START selects the starting point:
    #   "aufbau"     : the code's getStartingPoint (isotropic orbitals of the
    #                  aufbau set, orientation fixed along x)
    #   "costruzione": the orbitals that generated the target (uses the
    #                  answer: only establishes whether the physical
    #                  determinant is a local minimum, not usable in practice)
    #   "orientato"  : configuration and lengths deduced from the MULTIPOLE
    #                  MOMENTS of the target, without privileged information
    START = "aufbau",
    # Perturbation of the lengths estimated by the oriented start: b -> b(1+eps).
    # Used to measure the WIDTH OF THE BASIN of attraction. On the synthetic
    # target the estimate of b is exact, so the oriented start coincides with
    # the generating orbitals and convergence says nothing about robustness:
    # perturbing simulates the approximate estimate one would get for a target
    # not generated by an oscillator.
    B_PERTURB = 0.0,
    # ANISOTROPIC perturbation: b_perp -> b_perp(1-eta), b_z -> b_z(1+eta).
    # Changes the SHAPE of the starting determinant, not its scale. It is the
    # more severe test: the isotropic perturbation leaves the relative
    # elongation of the start intact, this one alters it and can invert it.
    B_PERTURB_ANISO = 0.0,
    # Basin scans (``scan`` mode): the first varies the scale, the second the
    # shape. The ranges are wide because with the initial values (-0.35, +0.50)
    # convergence held at ALL points and the edge of the basin was not reached.
    B_SCAN = (0.0, 0.10, 0.25, 0.50, 0.80, 1.20,
              -0.10, -0.25, -0.50, -0.65, -0.80),
    ETA_SCAN = (0.0, 0.10, 0.25, 0.40, 0.60,
                -0.10, -0.25, -0.40, -0.60),
)

# The two configurations, in the same deformed potential.
CONFIGS = {
    "fondamentale": [(0, 0, 0), (0, 0, 1)],
    "eccitata":     [(0, 0, 0), (1, 0, 0)],
}

OUTPUT_DIR = "test_3D_data_deformato_eccitato"

RHO_FLOOR_FRAC = 1e-10    # mask for the potential extraction
RHO_PHYS_FRAC  = 1e-2     # physical region of the potential metrics
RATIO_TOL      = 0.05     # tolerance on T_direct/T_alpha (informative flag)
VW_MARGIN      = 0.99     # T_alpha >= VW_MARGIN * T_vW
RHO_DIAG       = 1e-10    # mask for T_alpha, T_beta, T_direct
RHO_LOG        = 1e-14    # floor inside log(sqrt(rho))
# T_vW = alpha * int |grad sqrt(rho)|^2 carries no rho weight that suppresses
# the tails: including nodes where sqrt(rho) is below the resolution of the
# basis, the GLOBAL derivative matrix of the Lagrange mesh produces spurious
# values there and the integral diverges. Measured on this cell: with the mask
# rho >= 1e-10 the ground configuration gives T_vW = 294 MeV against an exact
# T_s of 98.4, i.e. it VIOLATES the bound; with rho >= 1e-6 it gives 74.0 MeV,
# correct. The effect does not depend on the floor on rho (checked with
# floors 1e-14, 1e-30 and none) but only on the extent of the integration
# region. T_alpha and T_direct are not affected because they carry a factor
# rho or phi that suppresses the tails.
RHO_VW         = 1e-6     # conservative mask, for T_vW only


# ─────────────────────────────────────────────────────────────────────────────
# Analytic energies of the anisotropic oscillator
# ─────────────────────────────────────────────────────────────────────────────
def sp_energy(nx, ny, nz, hw_perp, hw_z):
    """Single-particle energy of the axially deformed oscillator [MeV]."""
    return (nx + 0.5) * hw_perp + (ny + 0.5) * hw_perp + (nz + 0.5) * hw_z


def config_energies(A, delta, states):
    """(E_total, T_total) of the configuration ``states``, in MeV.

    T = E/2 by the virial theorem for the harmonic oscillator.
    """
    wp, wz = deformed_frequencies(A, delta)
    hw_perp, hw_z = hbc * wp, hbc * wz
    E = sum(OCC * sp_energy(*s, hw_perp, hw_z) for s in states)
    return E, 0.5 * E


# ─────────────────────────────────────────────────────────────────────────────
# Capture of the IPOPT log
# ─────────────────────────────────────────────────────────────────────────────
@contextmanager
def capture_stdout_fd():
    """Capture everything written to file descriptor 1 (including IPOPT's C-level
    output) into box["text"], echoing it to stdout at the end."""
    box = {"text": ""}
    sys.stdout.flush()
    saved = os.dup(1)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 1)
    try:
        yield box
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        tmp.seek(0)
        box["text"] = tmp.read().decode("utf-8", "replace")
        tmp.close()
        sys.stdout.write(box["text"])
        sys.stdout.flush()


def parse_ipopt_text(txt):
    """Extract iterations, dependent constraints, number of variables, final dual
    infeasibility, complementarity and exit message from an IPOPT log."""
    out = dict(n_iter=-1, n_dep=0, n_var=-1, dual_inf=float('nan'),
               compl=float('nan'), exit_msg="")
    m = re.findall(r"Number of Iterations\.*:\s*(\d+)", txt)
    if m:
        out['n_iter'] = int(m[-1])
    m = re.findall(r"Detected\s+(\d+)\s+linearly dependent equality constraints", txt)
    if m:
        out['n_dep'] = sum(int(v) for v in m)
    m = re.findall(r"Total number of variables\.*:\s*(\d+)", txt)
    if m:
        out['n_var'] = int(m[-1])
    m = re.findall(r"Dual infeasibility\.*:\s*\S+\s+(\S+)", txt)
    if m:
        out['dual_inf'] = float(m[-1])
    m = re.findall(r"Complementarity\.*:\s*\S+\s+(\S+)", txt)
    if m:
        out['compl'] = float(m[-1])
    m = re.findall(r"EXIT:\s*(.+)", txt)
    if m:
        out['exit_msg'] = m[-1].strip()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Building the case
# ─────────────────────────────────────────────────────────────────────────────
def build_mesh(A, cfg):
    """ISOTROPIC Gauss-Hermite mesh, identical to the one inside the solver."""
    N, s = cfg["N_GH"], cfg["MESH_SCALE"]
    return LagrangeMesh3D(N, N, N, s * b_x(A), s * b_y(A), s * b_z(A))


def build_target(A, delta, states, mesh):
    """Target density and generating (anisotropic) orbitals on the given mesh."""
    bp, bz = deformed_lengths(A, delta)
    phis = []
    for (nx, ny, nz) in states:
        p = HO3D(nx, ny, nz, bp, bp, bz, mesh.dV)(mesh.x, mesh.y, mesh.z)
        p /= np.sqrt(mesh.integrate(p**2))
        phis.append(p)
    rho = sum(OCC * p**2 for p in phis)
    return rho, phis


def analytic_potential(A, delta, grid):
    """V(x,y,z) = nu_perp (x^2+y^2) + nu_z z^2, the exact KS potential."""
    nu_p, nu_z = deformed_nu(A, delta)
    X, Y, Z = np.meshgrid(*grid, indexing='ij')
    return nu_p * (X**2 + Y**2) + nu_z * Z**2


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def make_grad(mesh):
    """Return a function u -> (du/dx, du/dy, du/dz) on the given mesh."""
    return lambda u: (mesh.d_dx(u), mesh.d_dy(u), mesh.d_dz(u))


def kinetic_decomposition(grads, w3D, rho, f_list, phi_list, occs, beta):
    """T_alpha, T_beta, T_direct on the RHO_DIAG mask; T_vW on the conservative
    RHO_VW mask, for the reason explained above. Also returns the ratio
    T_direct/T_alpha and an informative 'physical' flag."""
    mask = (rho >= RHO_DIAG * rho.max())
    s = np.sqrt(np.maximum(rho, RHO_LOG * rho.max()))
    glog = grads(np.log(s))
    nglog = sum(g**2 for g in glog)

    T_a = T_b = T_d = 0.0
    for k, occ in enumerate(occs):
        f = f_list[k]
        gf = grads(f)
        gm2 = sum(g**2 for g in gf)
        dot = sum(gl * g for gl, g in zip(glog, gf))
        T_a += occ * float(np.sum(
            w3D * ALPHA * rho * (gm2 + f**2 * nglog + 2.0 * f * dot) * mask))
        T_b += occ * float(np.sum(w3D * beta * gm2 * mask))
        gphi = grads(phi_list[k])
        T_d += occ * float(np.sum(
            w3D * ALPHA * sum(g**2 for g in gphi) * mask))

    gs = grads(s)
    mask_vw = (rho >= RHO_VW * rho.max())
    T_vw = float(np.sum(w3D * ALPHA * sum(g**2 for g in gs) * mask_vw))
    ratio = T_d / T_a if T_a != 0.0 else float('nan')
    physical = bool(abs(ratio - 1.0) < RATIO_TOL and T_a >= VW_MARGIN * T_vw)
    return dict(T_alpha=T_a, T_beta=T_b, T_direct=T_d, T_vW=T_vw,
                ratio=ratio, physical=physical)


def rho_weighted_rmse(dV, rho):
    """Density-weighted RMSE and max |dV| on rho >= RHO_PHYS_FRAC * rho_max (Eq. 4.13)."""
    mask = np.isfinite(dV) & (rho >= RHO_PHYS_FRAC * rho.max())
    if not np.any(mask):
        return float('nan'), float('nan')
    w = rho[mask]
    d = dV[mask]
    return (float(np.sqrt(np.sum(w * d**2) / np.sum(w))),
            float(np.max(np.abs(d))))


def overlap_matrix(phi_a, phi_b, w3D):
    """S_ij = int phi_a_i phi_b_j dV."""
    n = len(phi_a)
    return np.array([[float(np.sum(w3D * phi_a[i] * phi_b[j]))
                      for j in range(n)] for i in range(n)])


def grid_adequacy(A, delta, cfg):
    """T_direct of the GENERATING orbitals vs the analytic value.

    Does not involve the solver: it checks that the grid can represent the
    target. If the error here is large, the inversion cannot do better and
    the fault is not its own.
    """
    mesh = build_mesh(A, cfg)
    grads = make_grad(mesh)
    print("\n  --- Adeguatezza della griglia (senza solutore) ---")
    print(f"    {'config':<14} {'T analitica':>12} {'T_direct':>12} {'err':>8}"
          f" {'T_vW':>12} {'bound':>8}")
    for name, states in CONFIGS.items():
        _, T_an = config_energies(A, delta, states)
        rho, phis = build_target(A, delta, states, mesh)
        mk = rho >= RHO_DIAG * rho.max()
        Td = sum(OCC * float(np.sum(mesh.w3D * ALPHA
                 * sum(g**2 for g in grads(p)) * mk)) for p in phis)
        s_ = np.sqrt(np.maximum(rho, RHO_LOG * rho.max()))
        mk_vw = rho >= RHO_VW * rho.max()
        TvW = float(np.sum(mesh.w3D * ALPHA
                           * sum(g**2 for g in grads(s_)) * mk_vw))
        ok = "ok" if TvW < T_an else "VIOLATO"
        print(f"    {name:<14} {T_an:>12.4f} {Td:>12.4f} "
              f"{100*(Td-T_an)/T_an:>7.2f}% {TvW:>12.4f} {ok:>8}")


def second_moments_from_multipoles(rho, grid, w3D):
    """Second moments int rho x^2, y^2, z^2 obtained from the multipoles of degree 2.

    The three diagonal second moments and the three multipoles (M2, Q20, Q22)
    carry the same information. With
        M2  = int rho r^2 = A + B + C
        Q20 = int rho (2z^2 - x^2 - y^2)/2 = (2C - A - B)/2
        Q22 = sqrt(3) int rho (x^2 - y^2)/2 = sqrt(3)(A - B)/2
    one inverts:
        C = (M2 + 2 Q20)/3,   A - B = 2 Q22/sqrt(3),   A + B = M2 - C.
    Q21, Q2-1, Q2-2 are computed as well: they measure the OFF-diagonal
    moments. If they are not negligible, the principal axes of the density do
    not coincide with the Cartesian ones and the construction below — which
    chooses among the Cartesian states (100), (010), (001) — does not apply.
    Returns (moments [A, B, C], M2, Q dict, max off-diagonal |Q2m|).
    """
    X, Y, Z = np.meshgrid(*grid, indexing='ij')
    M2 = float(np.sum(w3D * rho * (X**2 + Y**2 + Z**2)))
    Q = multipole_moments(rho, grid, w3D,
                          [(2, 0), (2, 2), (2, 1), (2, -1), (2, -2)])
    C = (M2 + 2.0 * Q[(2, 0)]) / 3.0
    AB = M2 - C
    dAB = 2.0 * Q[(2, 2)] / np.sqrt(3.0)
    A = 0.5 * (AB + dAB)
    B = 0.5 * (AB - dAB)
    off = max(abs(Q[(2, 1)]), abs(Q[(2, -1)]), abs(Q[(2, -2)]))
    return np.array([A, B, C]), M2, Q, off


def axis_kurtosis(rho, grid, w3D):
    """Normalised kurtosis along each axis: k_i = N int rho u_i^4 / (int rho u_i^2)^2 (Eq. 6.10).

    It is the right discriminant to decide WHICH axis carries the excitation.
    The second moment alone is not enough: it mixes two effects, the
    oscillator length (set by the potential) and the occupation. For strong
    deformation the former dominates and a choice based on second moments is
    wrong — at delta = +0.80 it picks z even for a configuration excited along x.

    The kurtosis is instead INDEPENDENT of b_i, being a dimensionless ratio:
        pure n = 0 state             k = 3
        n = 0 / n = 1 mixture (50/50) k = 2.25
    The axis with the smallest kurtosis carries the excitation. The degree-4
    content of the density is that of the l = 4 multipoles (Q40 and partners):
    orienting the start correctly therefore requires multipoles up to fourth
    order, not just the quadrupole.
    """
    X, Y, Z = np.meshgrid(*grid, indexing='ij')
    N = float(np.sum(w3D * rho))
    k = np.zeros(3)
    for i, U in enumerate((X, Y, Z)):
        m2 = float(np.sum(w3D * rho * U**2))
        m4 = float(np.sum(w3D * rho * U**4))
        k[i] = N * m4 / m2**2 if m2 > 0 else np.inf
    return k


def oriented_start_config(rho, grid, w3D, n_orb, occ):
    """Configuration and oscillator lengths deduced from the multipoles (oriented start).

    Step 1 — which orbitals to occupy. (000) is always occupied; the remaining
    single excitations go along the axes with the smallest KURTOSIS (see
    axis_kurtosis), because the excitation flattens the profile along its own
    axis independently of the width.

    Step 2 — the lengths. Once the occupations are known, the second moment
    along axis i is
        int rho u_i^2 = b_i^2 * sum_k occ_k (n_{k,i} + 1/2)      (Eq. 6.11)
    which gives b_i. For a target generated by oscillator orbitals the
    estimate is exact; otherwise it is an approximation, good enough for a start.

    Neither step uses information on the potential or on the configuration
    that generated the target: both read only moments of the density.
    Returns (states, b, info).
    """
    mom, M2, Q, off = second_moments_from_multipoles(rho, grid, w3D)
    kurt = axis_kurtosis(rho, grid, w3D)

    order = np.argsort(kurt)                      # axes by increasing kurtosis
    states = [(0, 0, 0)]
    for ax in order[:n_orb - 1]:
        st = [0, 0, 0]
        st[int(ax)] = 1
        states.append(tuple(st))

    b = np.zeros(3)
    for i in range(3):
        w = sum(occ * (st[i] + 0.5) for st in states)
        b[i] = np.sqrt(mom[i] / w)

    # relative kurtosis margin between the last chosen axis and the first excluded one
    srt = np.sort(kurt)
    margine = ((srt[n_orb - 1] - srt[n_orb - 2]) / srt[n_orb - 2]
               if n_orb - 1 < 3 and srt[n_orb - 2] > 0 else 1.0)
    info = dict(moments=mom, M2=M2, Q=Q, off_diag=off, kurtosis=kurt,
                margine=margine, order=order)
    return states, b, info


def start_vector_from_orbitals(prob, rho, phis):
    """Starting vector f = phi/sqrt(rho) with the convention of getStartingPoint:
    floor 1e-14 on rho and plateau value on the dead nodes, so as not to create
    the jump that the global derivative matrix would propagate."""
    rho_safe = np.maximum(rho, 1e-14 * rho.max())
    dead = rho < 1e-10 * rho.max()
    occ_tot = sum(o.occupation for o in prob.orbital_set)
    plateau = 1.0 / np.sqrt(occ_tot)
    st = np.zeros((len(phis),) + rho.shape)
    for j, p in enumerate(phis):
        f = p / np.sqrt(rho_safe)
        f[dead] = plateau
        st[j] = f
    return st.ravel()


# ─────────────────────────────────────────────────────────────────────────────
# One run
# ─────────────────────────────────────────────────────────────────────────────
def run_config(name, states, A, delta, cfg):
    """Build the target of configuration ``states``, run the IKS problem with the
    starting point selected by cfg["START"] and collect all the diagnostics
    (IPOPT log, kinetic decomposition, potential error, overlap matrix S)."""
    mesh = build_mesh(A, cfg)
    rho_target, phis_costr = build_target(A, delta, states, mesh)
    grid, w3D = mesh.grid, mesh.w3D
    x, y, z = grid
    n_orb = len(states)          # needed already by the starting-point block
    N = cfg["N_GH"]

    prob = iKS_problem(
        Z=N_PART, N=0, data=[x, y, z, rho_target],
        basis=uncoupledHObasis(), reg_parameter=cfg["BETA"],
        output_folder=f"def_{name}",
        exact_hess=True, max_iter=int(cfg["MAX_ITER"]),
        rel_tol=cfg["TOL"], constr_viol=cfg["TOL"],
        dual_inf_tol=cfg["DUAL_TOL"],
        lb=[-2, -2, -2], ub=[2, 2, 2], hx=0.1,
        use_lagrange_mesh=True, Nx=N, Ny=N, Nz=N,
        mesh_scale=cfg["MESH_SCALE"])
    prob.add_option(b"mu_init", 0.01)
    prob.add_option(b"acceptable_iter", int(cfg["ACCEPTABLE_ITER"]))

    # Starting point: replace getStartingPoint on the instance (solve() always calls it)
    start_info = None
    if cfg["START"] == "costruzione":
        st0 = start_vector_from_orbitals(prob, rho_target, phis_costr)
        prob.getStartingPoint = lambda: st0
    elif cfg["START"] == "orientato":
        st_states, b_est, start_info = oriented_start_config(
            rho_target, grid, w3D, n_orb, OCC)
        eps = float(cfg.get("B_PERTURB", 0.0))
        eta = float(cfg.get("B_PERTURB_ANISO", 0.0))
        b_est = b_est * (1.0 + eps)
        if eta:
            b_est = b_est * np.array([1.0 - eta, 1.0 - eta, 1.0 + eta])
        phis0 = []
        for stt in st_states:
            q = HO3D(*stt, b_est[0], b_est[1], b_est[2], mesh.dV)(x, y, z)
            q /= np.sqrt(mesh.integrate(q**2))
            phis0.append(q)
        st0 = start_vector_from_orbitals(prob, rho_target, phis0)
        prob.getStartingPoint = lambda: st0
        start_info.update(states=st_states, b=b_est, perturb=eps,
                          perturb_aniso=eta)

    # T_alpha at the actual starting point, to see where the descent starts from
    st_vec = prob.getStartingPoint().reshape(len(states), *rho_target.shape)
    s_tmp = np.sqrt(np.maximum(rho_target, RHO_LOG * rho_target.max()))
    dec0 = kinetic_decomposition(make_grad(mesh), w3D, rho_target,
                                 [st_vec[k] for k in range(len(states))],
                                 [st_vec[k] * s_tmp for k in range(len(states))],
                                 [OCC] * len(states), cfg["BETA"])

    t0 = time.time()
    failed = False
    results = None
    with capture_stdout_fd() as cap:          # IPOPT log parsed below
        try:
            results, _info = prob.solve()
        except Exception as exc:
            print(f"  !! solve fallito: {type(exc).__name__}: {exc}")
            failed = True
    elapsed = time.time() - t0
    diag = parse_ipopt_text(cap["text"])

    out = dict(name=name, states=states, grid=grid, w3D=w3D,
               rho_target=rho_target, phis_costr=phis_costr,
               elapsed=elapsed, failed=failed, mesh=mesh, T_alpha_start=dec0['T_alpha'],
               start=cfg["START"], start_info=start_info,
               n_constr=prob.n_constr, n_live=prob.n_live, **diag)
    if failed:
        return out

    phi = results['u']
    f_sol = np.asarray(results['x']).reshape(n_orb, *rho_target.shape)
    rho_calc = sum(OCC * phi[k]**2 for k in range(n_orb))

    # Potential V = lambda/(rho w), gauge V = 0 at the density maximum
    lag = np.asarray(results['lagrange'])
    lam_full = np.full(prob.n_points, np.nan)
    lam_full[prob.live_nodes] = lag[:prob.n_live]
    lam = lam_full.reshape(prob.n_points_x, prob.n_points_y, prob.n_points_z)

    mask = (rho_target >= RHO_FLOOR_FRAC * rho_target.max())
    with np.errstate(divide='ignore', invalid='ignore'):
        V = np.where(mask, lam / (rho_target * w3D), np.nan)

    icx, icy, icz = np.unravel_index(np.nanargmax(rho_target), rho_target.shape)
    V = V - V[icx, icy, icz]
    V_ana = analytic_potential(A, delta, grid)
    V_ana = V_ana - V_ana[icx, icy, icz]

    dec = kinetic_decomposition(make_grad(mesh), w3D, rho_target,
                                [f_sol[k] for k in range(n_orb)],
                                [phi[k] for k in range(n_orb)],
                                [OCC] * n_orb, cfg["BETA"])
    rw, mx = rho_weighted_rmse(V - V_ana, rho_target)
    S = overlap_matrix(phis_costr, [phi[k] for k in range(n_orb)], w3D)

    out.update(status=results['status'], obj=float(results['obj']),
               density_err=float(np.sqrt(np.sum(w3D * (rho_calc - rho_target)**2))),
               norm_calc=float(np.sum(w3D * rho_calc)),
               V=V, V_ana=V_ana, rmse=rw, maxerr=mx, S=S,
               center=(icx, icy, icz), rho_calc=rho_calc, **dec)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def report(r, A, delta, T_self, T_other, other_name):
    """Print the full report of one run (convergence, kinetic energies, oriented
    start details, overlap matrix, potential error)."""
    W = 96
    print("\n" + "=" * W)
    print(f"  CONFIGURAZIONE '{r['name']}'  —  stati {r['states']}")
    print("=" * W)
    if r['failed']:
        print("  Il solve e' fallito: nessuna metrica disponibile.")
        return

    Q = multipole_moments(r['rho_target'], r['grid'], r['w3D'],
                          [(2, 0), (2, 2), (4, 0)])
    print(f"  Q_20 = {Q[(2,0)]:+.4f} fm^2    Q_22 = {Q[(2,2)]:+.4f} fm^2"
          f"    Q_40 = {Q[(4,0)]:+.4e} fm^4")
    print(f"  vincoli m = {r['n_constr']}   nodi vivi = {r['n_live']}   "
          f"n_x = {r['n_var']}")
    warn = ("" if r['exit_msg'].startswith("Optimal")
            else "   <<< NON convergente")
    print(f"  iterazioni = {r['n_iter']}   tempo = {r['elapsed']:.1f} s   "
          f"exit: {r['exit_msg']}{warn}")
    print(f"  dual inf = {r['dual_inf']:.2e}   complementarita' = {r['compl']:.2e}")
    print(f"  |rho_calc - rho_target|_2 = {r['density_err']:.3e}   "
          f"norma = {r['norm_calc']:.6f}  (attesa {N_PART})")

    print("\n  --- Cinetica ---")
    print(f"  T_esatta di QUESTA configurazione        = {T_self:10.4f} MeV")
    print(f"  T_esatta della configurazione '{other_name}' = {T_other:10.4f} MeV")
    lbl = {"aufbau": "aufbau isotropo", "costruzione": "orbitali di costruzione",
           "orientato": "orientato dai multipoli"}[r['start']]
    print(f"  T_alpha nel punto di PARTENZA            = {r['T_alpha_start']:10.4f} MeV"
          f"   (avvio: {lbl})")
    if r['start_info'] is not None:
        si = r['start_info']
        m = si['moments']
        k = si['kurtosis']
        print(f"    momenti secondi dai multipoli (l=2): int rho x^2={m[0]:.4f}  "
              f"y^2={m[1]:.4f}  z^2={m[2]:.4f} fm^2")
        print(f"    curtosi per asse (contenuto l=4): {k[0]:.4f}  {k[1]:.4f}  "
              f"{k[2]:.4f}   (3 = gaussiana pura, 2.25 = miscela n=0/n=1)")
        print(f"    fuori diagonale max|Q_2m| = {si['off_diag']:.2e} "
              f"(assi principali cartesiani se ~0)")
        print(f"    configurazione dedotta: {si['states']}   "
              f"(vera: {r['states']})   "
              + ("COINCIDE" if list(si['states']) == list(r['states'])
                 else "DIVERSA"))
        pert = si.get('perturb', 0.0)
        pana = si.get('perturb_aniso', 0.0)
        tag = ""
        if pert:
            tag += f"   [scala {100*pert:+.0f}%]"
        if pana:
            tag += f"   [forma {100*pana:+.0f}%]"
        print(f"    b usate: ({si['b'][0]:.4f}, {si['b'][1]:.4f}, "
              f"{si['b'][2]:.4f}) fm" + tag)
        print(f"    margine di curtosi fra asse scelto ed escluso: "
              f"{100*si['margine']:.1f}%"
              + ("   [scelta robusta]" if si['margine'] > 0.05
                 else "   [scelta INCERTA: assi quasi equivalenti]"))
    print(f"  T_alpha (soluzione)                      = {r['T_alpha']:10.4f} MeV"
          f"   ({100*(r['T_alpha']-T_self)/T_self:+.2f}% vs questa config)")
    print(f"  T_beta                                   = {r['T_beta']:10.4f} MeV")
    print(f"  T_vW  (limite inferiore)                 = {r['T_vW']:10.4f} MeV")
    print(f"  T_direct = alpha * int |grad phi|^2       = {r['T_direct']:10.4f} MeV")
    # Physicality verdict. In synthetic tests the exact value is known, so the
    # primary criterion is the comparison with it, not the ratio
    # T_direct/T_alpha. The 'warm' run showed that the two are DECOUPLED: the
    # ground configuration started from the generating orbitals has
    # ratio = 198.8 and at the same time T_alpha correct to 0.05% and the best
    # potential of all runs (RMSEw 2.71 MeV). The ratio measures the presence
    # of sub-grid structure in the orbitals, to which T_direct is
    # hypersensitive, NOT the quality of the solution. It remains a useful but
    # informative indicator.
    err_T = 100.0 * (r['T_alpha'] - T_self) / T_self
    fisica = (abs(err_T) < 1.0) and (r['T_alpha'] >= VW_MARGIN * r['T_vW'])
    verdetto = "fisica" if fisica else "NON fisica"
    note = []
    if abs(err_T) >= 1.0:
        note.append(f"T_alpha scostata di {err_T:+.2f}%")
    if r['T_alpha'] < VW_MARGIN * r['T_vW']:
        note.append("T_alpha < T_vW")
    if note:
        verdetto += " (" + ", ".join(note) + ")"
    print(f"  verdetto: {verdetto}")
    print(f"  T_direct/T_alpha = {r['ratio']:.4f}   "
          "(indice di contaminazione sub-griglia, informativo)")

    print("\n  --- Determinante trovato vs determinante di costruzione ---")
    S = r['S']
    print("  matrice di sovrapposizione S_ij = int phi_costr_i phi_sol_j dV:")
    for row in S:
        print("      [" + "  ".join(f"{v:+8.5f}" for v in row) + "]")
    off = np.max(np.abs(S - np.diag(np.diag(S))))
    dia = np.min(np.abs(np.diag(S)))
    print(f"  max|S_ij (i!=j)| = {off:.4e}     min|S_ii| = {dia:.6f}")

    print("\n  --- Potenziale ricostruito vs potenziale anisotropo esatto ---")
    print(f"  RMSEw = {r['rmse']:.4e} MeV     max = {r['maxerr']:.4e} MeV")


def classify_overlap(S):
    """Classify the overlap matrix S_ij = int phi_gen_i phi_sol_j.

    The right criterion is UNITARITY, not diagonality: if S is unitary the two
    sets of orbitals span the SAME subspace, hence the same Slater
    determinant, and differ only by a rotation in the U(N_occ) gauge, which
    leaves both rho and T_s unchanged (thesis Remark 2.7). A diagonal S is the
    special case in which the rotation is the identity.
    Returns (||S S^T - I||_max, max off-diagonal, min |diagonal|, label).
    """
    n = S.shape[0]
    unit_err = float(np.max(np.abs(S @ S.T - np.eye(n))))
    off = float(np.max(np.abs(S - np.diag(np.diag(S)))))
    dia = float(np.min(np.abs(np.diag(S))))
    if unit_err < 5e-2:
        kind = "stesso determinante" + (", stessi orbitali" if off < 5e-2
                                        else ", ruotati in U(N_occ)")
    else:
        kind = "determinante DIVERSO"
    return unit_err, off, dia, kind


def verdict(res, T_exact, W=96):
    """Print a one-paragraph reading of each configuration's outcome."""
    print("\n" + "=" * W)
    print("  LETTURA")
    print("=" * W)
    for name, r in res.items():
        if r['failed']:
            print(f"  {name:<14} run fallito")
            continue
        dT = r['T_alpha'] - T_exact[name]
        unit_err, off, dia, kind = classify_overlap(r['S'])
        print(f"  {name:<14} dT_alpha = {dT:+9.4f} MeV   "
              f"||S S^T - I|| = {unit_err:.3e}   max|S_offdiag| = {off:.3e}   "
              f"RMSEw V = {r['rmse']:.4e} MeV")
        print(f"      -> {kind}")
        if r['exit_msg'] and not r['exit_msg'].startswith("Optimal"):
            print(f"      !! IPOPT non ha raggiunto la convergenza piena "
                  f"({r['exit_msg']}): il punto non e' stazionario e le")
            print(f"         quantita' sopra non vanno interpretate come "
                  f"proprieta' della soluzione.")
        elif "DIVERSO" in kind:
            print(f"         Il minimo di Levy non e' il determinante di "
                  f"costruzione: il potenziale ricostruito non e' quello di")
            print(f"         partenza pur riproducendo la densita'.")
        elif abs(dT) < 1.0:
            print(f"         T_alpha compatibile con la configurazione di "
                  f"costruzione: l'RMSE misura l'accuratezza dell'inversione")
            print(f"         sul potenziale anisotropo.")
        else:
            print(f"         T_alpha scostata dal valore della configurazione: "
                  f"verificare convergenza e bound di von Weizsaecker.")
    print("\n  Riferimento: lo scarto noto fra T_alpha e la cinetica vera dovuto")
    print("  alla regola del prodotto discreta e' ~0.2 MeV su Lagrange mesh. Uno")
    print("  scarto di quell'ordine non indica un cambio di determinante; uno")
    print("  scarto confrontabile col gap fra le configurazioni si'.")
    print()


def make_figure(res, A, delta, path):
    """2x3 figure: potential along x and z for each configuration, target
    density profiles, and bar chart of the potential error."""
    ok = {k: v for k, v in res.items() if not v['failed']}
    if not ok:
        return
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    fig.suptitle(f"IKS 3D — Z={N_PART}, oscillatore deformato "
                 f"$\\delta={delta:+.2f}$ (Lagrange mesh): "
                 f"fondamentale vs eccitata", fontweight="bold", fontsize=13)
    colors = {"fondamentale": "#0891b2", "eccitata": "#9333ea"}

    for col, (name, r) in enumerate(ok.items()):
        x = r['grid'][0]
        z = r['grid'][2]
        icx, icy, icz = r['center']
        R = physical_radius(r['rho_target'][:, icy, icz], x, frac=1e-3)
        Rlim = 1.3 * R

        ax = axes[0, col]
        ax.plot(x, r['V_ana'][:, icy, icz], color="#dc2626", lw=2.2,
                label="V analitico")
        ax.plot(x, r['V'][:, icy, icz], "--", lw=1.6, color=colors[name],
                label="V IKS")
        ax.set_xlim(-Rlim, Rlim)
        yl = potential_ylim(r['V_ana'][:, icy, icz], x, R)
        if yl:
            ax.set_ylim(*yl)
        ax.set_xlabel("x [fm]"); ax.set_ylabel("V [MeV]")
        ax.set_title(f"{name}: V lungo x (RMSEw={r['rmse']:.2e})")
        ax.legend(fontsize=8); ax.grid(alpha=0.4)

        ax = axes[1, col]
        ax.plot(z, r['V_ana'][icx, icy, :], color="#dc2626", lw=2.2,
                label="V analitico")
        ax.plot(z, r['V'][icx, icy, :], "--", lw=1.6, color=colors[name],
                label="V IKS")
        ax.set_xlim(-Rlim, Rlim)
        yl = potential_ylim(r['V_ana'][icx, icy, :], z, R)
        if yl:
            ax.set_ylim(*yl)
        ax.set_xlabel("z [fm]"); ax.set_ylabel("V [MeV]")
        ax.set_title(f"{name}: V lungo z")
        ax.legend(fontsize=8); ax.grid(alpha=0.4)

    ax = axes[0, 2]
    for name, r in ok.items():
        x = r['grid'][0]
        icx, icy, icz = r['center']
        ax.plot(x, r['rho_target'][:, icy, icz], lw=1.8, color=colors[name],
                label=f"{name} (lungo x)")
        ax.plot(r['grid'][2], r['rho_target'][icx, icy, :], "--", lw=1.6,
                color=colors[name], alpha=0.6, label=f"{name} (lungo z)")
    ax.set_xlabel("coordinata [fm]"); ax.set_ylabel(r"$\rho$ [fm$^{-3}$]")
    ax.set_title("Densita' target: profili lungo x e z")
    ax.legend(fontsize=8); ax.grid(alpha=0.4)

    ax = axes[1, 2]
    names = list(ok.keys())
    vals = [ok[n]['rmse'] for n in names]
    ax.bar(names, vals, color=[colors[n] for n in names])
    ax.set_yscale("log")
    ax.set_ylabel(r"RMSE$_w$ del potenziale [MeV]")
    ax.set_title("Errore sul potenziale ricostruito")
    ax.grid(alpha=0.4, axis="y")

    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Figura salvata: {path}")


def basin_scan(A, delta, cfg):
    """Width of the basin of attraction of the oriented start.

    For each eps the run is restarted with lengths b(1+eps) (and then with the
    shape perturbation eta) and it is recorded whether it converges and
    whether the determinant found is still the generating one. The largest
    |eps| for which convergence holds measures how imprecise the start can be
    — i.e. how usable the procedure is on targets not generated by an
    oscillator, where the estimate of b is necessarily approximate.
    """
    W = 96
    print("\n" + "=" * W)
    print("  SCANSIONE DEL BACINO — avvio orientato con lunghezze perturbate")
    print("=" * W)
    saved = cfg["START"], cfg["B_PERTURB"], cfg["B_PERTURB_ANISO"]
    cfg["START"] = "orientato"
    results = {}
    assi = (("SCALA  (b -> b(1+eps): la forma del determinante non cambia)",
             "B_PERTURB", "B_PERTURB_ANISO", cfg["B_SCAN"]),
            ("FORMA  (b_perp -> b_perp(1-eta), b_z -> b_z(1+eta))",
             "B_PERTURB_ANISO", "B_PERTURB", cfg["ETA_SCAN"]))
    for titolo, chiave, azzera, valori in assi:
        print(f"\n  ══ perturbazione di {titolo}")
        for name, states in CONFIGS.items():
            _, T_an = config_energies(A, delta, states)
            print(f"\n  {name}   (T esatta = {T_an:.4f} MeV)")
            print(f"    {'par':>7} {'iter':>6} {'esito':<26} {'T_alpha':>10} "
                  f"{'err%':>8} {'||SS^T-I||':>11} {'RMSEw':>10}")
            rows = []
            for v in valori:
                cfg[chiave] = v
                cfg[azzera] = 0.0
                r = run_config(name, states, A, delta, cfg)
                if r['failed']:
                    print(f"    {v:>+7.2f} {'--':>6} {'solve fallito':<26}")
                    rows.append((v, None))
                    continue
                unit_err, off, dia, kind = classify_overlap(r['S'])
                err = 100.0 * (r['T_alpha'] - T_an) / T_an
                print(f"    {v:>+7.2f} {r['n_iter']:>6} "
                      f"{r['exit_msg'][:26]:<26} {r['T_alpha']:>10.4f} "
                      f"{err:>7.2f}% {unit_err:>11.2e} {r['rmse']:>10.4f}")
                rows.append((v, dict(iter=r['n_iter'], exit=r['exit_msg'],
                                     T=r['T_alpha'], err=err, unit=unit_err,
                                     rmse=r['rmse'])))
            results[(titolo, name)] = rows
            ok = [e for e, d in rows if d and d['exit'].startswith("Optimal")
                  and abs(d['err']) < 1.0 and d['unit'] < 5e-2]
            if ok:
                bordo = (min(ok) > min(valori)) or (max(ok) < max(valori))
                nota = "" if bordo else "   [BORDO NON RAGGIUNTO: estendere]"
                print(f"    -> convergenza corretta in "
                      f"[{min(ok):+.2f}, {max(ok):+.2f}]{nota}")
            else:
                print("    -> nessun valore con convergenza corretta")
    cfg["START"], cfg["B_PERTURB"], cfg["B_PERTURB_ANISO"] = saved
    return results


# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Parse the command line (delta, warm/orient/scan), print the setup and run."""
    args = [a.lower() for a in sys.argv[1:]]
    if any(a.startswith("w") or a.startswith("c") for a in args):
        CFG["START"] = "costruzione"
    if any(a.startswith("o") for a in args):
        CFG["START"] = "orientato"
    do_scan = any(a.startswith("s") for a in args)
    for a in args:
        try:
            CFG["DELTA"] = float(a)
        except ValueError:
            pass
    delta = CFG["DELTA"]
    A = N_PART
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wp, wz = deformed_frequencies(A, delta)
    hw_perp, hw_z = hbc * wp, hbc * wz
    bp, bz = deformed_lengths(A, delta)
    nu_p, nu_z = deformed_nu(A, delta)

    W = 96
    print()
    print("=" * W)
    print(f"  IKS 3D — Z={N_PART} — oscillatore deformato, stato fondamentale "
          f"vs eccitato   [Lagrange mesh]")
    print("=" * W)
    print(f"  delta = {delta:+.3f}  ({'prolato' if delta > 0 else 'oblato' if delta < 0 else 'sferico'},"
          f" volume costante)")
    print(f"  hw_perp = {hw_perp:.4f} MeV     hw_z = {hw_z:.4f} MeV"
          f"     gap = {hw_perp - hw_z:+.4f} MeV")
    print(f"  b_perp  = {bp:.4f} fm          b_z  = {bz:.4f} fm")
    print(f"  nu_perp = {nu_p:.4f}            nu_z = {nu_z:.4f}"
          "   [V = nu_perp (x^2+y^2) + nu_z z^2]")
    print(f"  N_GH = {CFG['N_GH']}, mesh_scale = {CFG['MESH_SCALE']} "
          f"(mesh ISOTROPA, orbitali anisotropi)")
    print(f"  beta = {CFG['BETA']:.1e}, tol = {CFG['TOL']:.0e}, "
          f"dual_inf_tol = {CFG['DUAL_TOL']:.0e}, "
          f"acceptable_iter = {CFG['ACCEPTABLE_ITER']}")
    print(f"  punto di partenza: " + {
        "aufbau": "aufbau isotropo di getStartingPoint",
        "costruzione": "orbitali di COSTRUZIONE (usa la risposta)",
        "orientato": "ORIENTATO dai momenti multipolari del target",
    }[CFG["START"]])

    print("\n  Configurazioni e ordinamento dei livelli:")
    T_exact = {}
    for name, states in CONFIGS.items():
        E, T = config_energies(A, delta, states)
        T_exact[name] = T
        sp = ", ".join(f"{s}: {sp_energy(*s, hw_perp, hw_z):.3f}" for s in states)
        print(f"    {name:<14} {states}   E = {E:.4f} MeV   T = {T:.4f} MeV")
        print(f"    {'':<14} energie di s.p. [MeV]: {sp}")
    dT = T_exact["eccitata"] - T_exact["fondamentale"]
    print(f"    gap cinetico fra le due configurazioni: {dT:+.4f} MeV")
    if dT <= 0:
        print("    ATTENZIONE: con questo delta la configurazione etichettata")
        print("    'eccitata' non e' sopra quella 'fondamentale'. Per delta < 0")
        print("    (oblato) i ruoli si scambiano: rinominare o cambiare segno.")

    grid_adequacy(A, delta, CFG)          # sanity check of the mesh, no solver

    if do_scan:
        basin_scan(A, delta, CFG)
        return

    res = {}
    for name, states in CONFIGS.items():
        print(f"\n  ── run: {name} " + "─" * max(0, 60 - len(name)))
        res[name] = run_config(name, states, A, delta, CFG)

    other = {"fondamentale": "eccitata", "eccitata": "fondamentale"}
    for name in CONFIGS:
        report(res[name], A, delta, T_exact[name],
               T_exact[other[name]], other[name])

    verdict(res, T_exact)
    make_figure(res, A, delta,
                os.path.join(OUTPUT_DIR,
                             f"IKS_Z{N_PART}_deformato_eccitato_"
                             f"delta{delta:+.2f}.pdf"))


if __name__ == "__main__":
    main()