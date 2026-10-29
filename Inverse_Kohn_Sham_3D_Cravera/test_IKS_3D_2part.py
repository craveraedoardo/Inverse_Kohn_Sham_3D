"""
test_IKS_3D_2part.py
====================
Validation of the IKS solver for A = 2 (two neutrons, a single orbital (000))
on the Gauss-Hermite Lagrange mesh. Thesis: Section 5.1 (Table 5.1, Fig. 5.1).

Why A = 2 is the ideal validation case
--------------------------------------
With one orbital the density is a pure Gaussian, rho = 2|phi_000|^2, and the
problem is degenerate: the density constraint 2f^2 = 1 fixes f = 1/sqrt(2)
everywhere (no degrees of freedom left). There is no real optimisation, but
precisely for this reason the value of the objective is PREDICTABLE: for the
ground state the von Weizsaecker functional coincides with the exact kinetic
energy of the oscillator,

    T = occ * (3/2) * (hbar*omega / 2) = 2 * (3/4) hbar*omega.

That number is the benchmark the code has to reproduce. The potential is also
known exactly (the harmonic trap), so the reconstructed potential can be
compared node by node.

What the script does
--------------------
1. Convergence scan in the number of nodes per axis N: the objective is
   evaluated on the exact target density, without optimising. This isolates
   the discretisation error from the optimisation error at no cost.
   Two lessons from the development of the code are visible here:
     - with the Lagrange derivative matrix on Gauss-Hermite nodes, N = 8 is
       grossly insufficient (kinetic energy overestimated by ~7x);
     - the energy converges to the analytic value only for N >~ 20 and ONLY if
       the non-physical nodes (rho << rho_max) are masked out: the derivative
       matrix is dense, so outer nodes with tiny sqrt(rho) but huge derivative
       contaminate the integral.
2. Full IKS run (exact Hessian) at the largest N, with the optimal mesh scale
   s = 0.794 (Section 5.5), and extraction of the potential.
3. Figure with the convergence of T_alpha, the density and the potential.

Usage
-----
    python test_IKS_3D_2part.py

Output: test_3D_data_2part/IKS_2particles_validation.pdf
(the solver also writes test_3D_data/test_2_particles/ and ipopt.out).

This module is also imported by ``compare_grid.py`` and ``h_scan_gh_Z2.py``,
which call ``run_iks`` and ``build_target_density`` directly.
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Typographic style of the figures: sizes chosen to be readable when
# projected (presentation) as well as in print (thesis).
# Change them here to restyle every figure of the script.
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.size":          17,   # generic text
    "axes.labelsize":     21,   # axis and colour-bar labels
    "axes.titlesize":     22,   # panel titles
    "figure.titlesize":   26,   # suptitle
    "xtick.labelsize":    17,
    "ytick.labelsize":    17,
    "legend.fontsize":    17,
    "legend.title_fontsize": 17,
    "axes.labelpad":      6.0,
    "axes.linewidth":     1.2,
    "xtick.major.width":  1.2,
    "ytick.major.width":  1.2,
    "xtick.major.size":   5.0,
    "ytick.major.size":   5.0,
    "lines.linewidth":    2.0,
    "lines.markersize":   7.0,
    "grid.linewidth":     0.9,
})


# Make the modules in src/ importable regardless of the working directory.
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_helpers import physical_radius, potential_ylim

from consts3D import (a as ALPHA, hbc, nuclear_omega_x,
                      nuclearNu_x, nuclearNu_y, nuclearNu_z,
                      b_x, b_y, b_z)
from orbitals3D import uncoupledHObasis, getOrbitalSet
from optimizer3D_lm import iKS_problem
from lagrange_mesh_3D import LagrangeMesh3D
from eigenfunctions3D import HO3D

OUTPUT_DIR = "test_3D_data_2part"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Threshold of the physical-node mask (same as in the optimiser, Eq. 3.13)
RHO_FLOOR_FRAC = 1e-10
REG            = 1e-3          # default regularisation weight beta


# ──────────────────────────────────────────────────────────────────────────────
# Analytic references
# ──────────────────────────────────────────────────────────────────────────────
def kinetic_reference(A, occ):
    """Exact kinetic energy of the 3D HO ground state (for occ particles) [MeV]."""
    hw = hbc * nuclear_omega_x(A)          # hbar*omega in MeV
    return occ * 0.75 * hw                  # occ * (3/2) * (hbar*omega/2)


def build_target_density(mesh, A, n_part):
    """Target rho = sum_k occ_k |phi_k|^2, with phi_k normalised with the mesh weights.

    The orbitals are always built on the PHYSICAL oscillator length b_x(A),
    whatever the mesh scale: the density is fixed, only the nodes move.
    Returns (rho, orbitals).
    """
    x, y, z = mesh.x, mesh.y, mesh.z
    rho = np.zeros((mesh.Nx, mesh.Ny, mesh.Nz))
    orbitals = list(getOrbitalSet(n_part, uncoupledHObasis()))
    for orb in orbitals:
        wf   = HO3D(orb.nx, orb.ny, orb.nz, b_x(A), b_y(A), b_z(A), mesh.dV)
        phi  = wf(x, y, z)
        phi /= np.sqrt(mesh.integrate(phi**2))   # quadrature normalisation (Eq. 4.7)
        rho += orb.occupation * phi**2
    return rho, orbitals


# ──────────────────────────────────────────────────────────────────────────────
# Convergence scan: T(N) vs analytic value
# ──────────────────────────────────────────────────────────────────────────────
def convergence_scan(A, n_part, occ, N_list):
    """For each N evaluate the objective (von Weizsaecker functional) on the exact
    target density, with the physical-node mask, and compare it with the
    analytic kinetic energy.

    This isolates the behaviour of the FUNCTIONAL from the optimisation: since
    for A = 2 the solution f = 1/sqrt(2) is known, evaluating the objective on
    the true density tells whether the code reproduces the physics.

    Returns (E_ref, rows) with rows = [(N, T_vW, err_%, n_live, n_total), ...].
    """
    E_ref = kinetic_reference(A, occ)
    rows  = []
    for N in N_list:
        mesh = LagrangeMesh3D(N, N, N, b_x(A), b_y(A), b_z(A))
        rho, _ = build_target_density(mesh, A, n_part)

        mask = (rho >= RHO_FLOOR_FRAC * rho.max())
        sqrt_rho = np.sqrt(rho)
        grad_sq  = mesh.grad(sqrt_rho)
        # T_vW = alpha * int |grad sqrt(rho)|^2, restricted to the live nodes
        integrand = np.sum(grad_sq**2, axis=-1) * mask
        T_vW = ALPHA * mesh.integrate(integrand)

        n_live = int(mask.sum())
        err_pct = 100.0 * (T_vW - E_ref) / E_ref
        rows.append((N, T_vW, err_pct, n_live, rho.size))
    return E_ref, rows


# ──────────────────────────────────────────────────────────────────────────────
# Full IKS run at a given N (exact Hessian)
# ──────────────────────────────────────────────────────────────────────────────
def run_iks(A, n_part, N, occ, beta=None, mesh_scale=1.0):
    """Solve the IKS problem on an N^3 Lagrange mesh and extract the potential.

    Parameters
    ----------
    A, n_part : mass number and number of particles (both 2 here)
    N         : nodes per axis
    occ       : occupation of the orbital (unused, kept for the signature
                shared with the other test scripts)
    beta      : regularisation weight (default REG)
    mesh_scale: dimensionless mesh scale s, b_mesh = s * b_x(A)

    Returns a dict with the solver status, objective, timing, density error,
    potential error and the arrays needed for the figures.
    """
    beta = REG if beta is None else beta
    # mesh_scale s: b_mesh = s*b_x(A). It scales ONLY the geometry of the
    # mesh (nodes/weights/D); the physics (HO3D in build_target_density, V_ana)
    # stays on b_x(A). It MUST coincide with the mesh built inside iKS_problem
    # (same s), so that the density is tabulated on the solver's nodes.
    mesh = LagrangeMesh3D(N, N, N, mesh_scale * b_x(A),
                          mesh_scale * b_y(A), mesh_scale * b_z(A))
    x, y, z = mesh.x, mesh.y, mesh.z
    rho_target, orbitals = build_target_density(mesh, A, n_part)
    n_orb = len(orbitals)

    # lb/ub are ignored on the Lagrange mesh; Z=2, N=0 describes two neutrons
    # (only the particle count matters, see iKS_problem).
    data = [x, y, z, rho_target]
    prob = iKS_problem(
        Z=2, N=0, lb=[-2, -2, -2], ub=[2, 2, 2], data=data,
        basis=uncoupledHObasis(), reg_parameter=beta,
        output_folder="test_2_particles",
        exact_hess=True, max_iter=500,
        rel_tol=1e-8, constr_viol=1e-8, dual_inf_tol=1.0,
        use_lagrange_mesh=True, Nx=N, Ny=N, Nz=N, mesh_scale=mesh_scale,
    )
    prob.add_option(b"mu_init", 0.01)

    t0 = time.time()
    results, info = prob.solve()
    elapsed = time.time() - t0

    obj      = float(results.get('obj', float('nan')))
    status   = results.get('status', 'N/A')
    phi      = results['u']
    # Reconstructed density and its deviation from the target
    rho_calc = np.sum([orbitals[k].occupation * phi[k]**2
                       for k in range(n_orb)], axis=0)
    norm_calc   = mesh.integrate(rho_calc)
    density_err = np.sqrt(mesh.integrate((rho_calc - rho_target)**2))

    # ── Potential extraction (thesis Section 4.4) ────────────────────────────
    # IPOPT's multipliers lambda balance the gradient of the objective, which
    # carries the quadrature weight w INSIDE the operator (D^T(.w) and local
    # terms .w), whereas the density constraint (sum occ f^2 - 1) carries no
    # weight. The multiplier therefore absorbs one extra factor w, and the
    # correct formula is
    #       V_KS = lambda / (rho * w)                               (Eq. 4.8)
    # (with a constant weight w = dV it reduces to the classical lambda/rho).
    # NOTE: density constraints exist only on the physical nodes (live_nodes),
    # so lambda_density = results['lagrange'][:n_live] must be mapped back onto
    # the right nodes (NaN elsewhere, so that they drop out of the statistics).
    lam_live = results['lagrange'][:prob.n_live]
    lam_full = np.full(mesh.n_points, np.nan)
    lam_full[prob.live_nodes] = lam_live
    lam = lam_full.reshape(mesh.Nx, mesh.Ny, mesh.Nz)

    w3D  = mesh.w3D
    mask = (rho_target >= RHO_FLOOR_FRAC * rho_target.max())
    with np.errstate(divide='ignore', invalid='ignore'):
        V_iks = lam / (rho_target * w3D)
    V_iks = np.where(mask, V_iks, np.nan)
    # Gauge: V = 0 at the centre (node of maximum density)
    icx, icy, icz = np.unravel_index(np.nanargmax(rho_target), rho_target.shape)
    V_iks -= V_iks[icx, icy, icz]

    # Exact reference: the harmonic trap, in the same gauge as V_iks
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    V_ana = (nuclearNu_x(A) * X**2 + nuclearNu_y(A) * Y**2 + nuclearNu_z(A) * Z**2)
    V_ana -= V_ana[icx, icy, icz]

    # RMSE of the potential along the x axis through the centre, |x| < 3b.
    # (The density-weighted RMSE of Eq. 4.13 is computed by compare_grid.py
    # and h_scan_gh_Z2.py from the fields returned below.)
    b_ho   = b_x(A)
    sel    = (np.abs(x) < 3 * b_ho)
    dVx    = (V_iks[:, icy, icz] - V_ana[:, icy, icz])
    valid  = sel & np.isfinite(dVx)
    pot_err = np.sqrt(np.mean(dVx[valid]**2)) if np.any(valid) else float('nan')

    return dict(
        N=N, mesh=mesh, status=status, obj=obj, elapsed=elapsed,
        norm_calc=norm_calc, density_err=density_err, pot_err=pot_err,
        V_iks=V_iks, V_ana=V_ana, rho_target=rho_target, rho_calc=rho_calc,
        center=(icx, icy, icz), grid=(x, y, z),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────
def make_figure(E_ref, scan_rows, iks_res, fig_path):
    """2x2 figure: T(N) and its relative error; density and potential along x."""
    Ns   = [r[0] for r in scan_rows]
    Es   = [r[1] for r in scan_rows]

    x = iks_res['grid'][0]
    icx, icy, icz = iks_res['center']
    iy, iz = icy, icz

    # Plot window: where the density is significant
    rho_x = iks_res['rho_target'][:, iy, iz]
    R = physical_radius(rho_x, x, frac=1e-3)
    Rlim = 1.15 * R

    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5), constrained_layout=True)
    fig.suptitle("A=2 : validazione vs analitico (Lagrange mesh)",
                 fontweight="bold")

    # (0,0) convergence of T_alpha
    ax = axes[0, 0]
    ax.axhline(E_ref, color="#dc2626", lw=2, ls="--",
               label=f"T$_\\alpha$ analitico = {E_ref:.2f} MeV")
    ax.plot(Ns, Es, "o-", color="#2563eb", label="T$_{vW}$(N) (densità target)")
    ax.set_yscale("log")
    ax.set_xlabel("N$_{\\rm GH}$ (nodi per asse)"); ax.set_ylabel("T$_\\alpha$ [MeV]")
    ax.set_title("Energia cinetica")
    ax.legend(); ax.grid(alpha=0.4)

    # (0,1) relative error
    ax = axes[0, 1]
    err = [abs(r[2]) for r in scan_rows]
    ax.semilogy(Ns, err, "s-", color="#9333ea")
    ax.set_xlabel("N$_{\\rm GH}$ (nodi per asse)"); ax.set_ylabel("|errore| [%]")
    ax.set_title("Errore relativo su T$_\\alpha$"); ax.grid(alpha=0.4)

    # (1,0) 1D density cut
    ax = axes[1, 0]
    ax.plot(x, iks_res['rho_target'][:, iy, iz], color="#dc2626", lw=2, label="ρ target")
    ax.plot(x, iks_res['rho_calc'][:, iy, iz], "--", color="#16a34a", lw=1.8, label="ρ IKS")
    ax.set_xlim(-Rlim, Rlim)
    ax.set_xlabel("x [fm]"); ax.set_ylabel("ρ [fm⁻³]")
    ax.set_title(f"Densità (N={iks_res['N']}, norma={iks_res['norm_calc']:.3f})")
    ax.legend(); ax.grid(alpha=0.4)

    # (1,1) 1D potential cut, physical region
    ax = axes[1, 1]
    V_iks_x = iks_res['V_iks'][:, iy, iz]
    ax.plot(x, iks_res['V_ana'][:, iy, iz], color="#dc2626", lw=2, label="V analitico (HO)")
    ax.plot(x, V_iks_x, "--", color="#2563eb", lw=1.8, label="V IKS")
    ax.set_xlim(-Rlim, Rlim)
    yl = potential_ylim(V_iks_x, x, R)
    if yl:
        ax.set_ylim(*yl)
    ax.set_xlabel("x [fm]"); ax.set_ylabel("V [MeV]")
    ax.set_title(f"Potenziale lungo x\n(RMSE={iks_res['pot_err']:.2e} MeV)")
    ax.legend(); ax.grid(alpha=0.4)

    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figura salvata: {fig_path}")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║     TEST IKS 3D — N=2 : validazione vs analitico  [Lagrange Mesh]   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    A = 2; n_part = 2
    orb0 = list(getOrbitalSet(n_part, uncoupledHObasis()))[0]
    occ  = orb0.occupation
    print(f"  A={A}, 1 orbitale {(orb0.nx, orb0.ny, orb0.nz)}, occ={occ}")

    # ── Convergence scan (no optimisation) ───────────────────────────────────
    N_list = [8, 12, 16, 20, 24, 30, 41]
    E_ref, scan_rows = convergence_scan(A, n_part, occ, N_list)
    print(f"\n  E_kin analitico (ground state) = {E_ref:.4f} MeV\n")
    print("  Convergenza del funzionale (densità target esatta, maschera nodi fisici):")
    print(f"  {'N':>4}  {'E_kin [MeV]':>14}  {'errore':>10}  {'nodi vivi':>14}")
    for (N, E, errp, live, tot) in scan_rows:
        print(f"  {N:>4}  {E:>14.3f}  {errp:>9.2f}%  {live:>6}/{tot:<6}")

    # ── Full IKS run at the largest N of the scan, optimal scale s = 0.794 ───
    N_run = N_list[-1]
    print(f"\n  Run IKS completo (Hessiana esatta) a N={N_run} …")
    iks_res = run_iks(A, n_part, N_run, occ, mesh_scale=0.794)
    print(f"\n  Status        : {iks_res['status']}")
    print(f"  E_kin (IKS)   : {iks_res['obj']:.4f} MeV   (rif. {E_ref:.4f})")
    print(f"  Δ rel E_kin   : {100*(iks_res['obj']-E_ref)/E_ref:+.2f} %")
    print(f"  Δρ L2         : {iks_res['density_err']:.3e}")
    print(f"  Norma ρ_calc  : {iks_res['norm_calc']:.4f}  (atteso {n_part})")
    print(f"  V RMSE (|x|<3b): {iks_res['pot_err']:.3e} MeV")
    print(f"  Tempo         : {iks_res['elapsed']:.1f} s")

    make_figure(E_ref, scan_rows, iks_res,
                os.path.join(OUTPUT_DIR, "IKS_2particles_validation.pdf"))
    print("\n  Fatto.\n")


if __name__ == "__main__":
    main()
