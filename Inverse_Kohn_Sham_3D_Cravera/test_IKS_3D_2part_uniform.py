"""
test_IKS_3D_2part_uniform.py
============================
Validation of the IKS solver for A = 2 on the UNIFORM GRID (4th-order finite
differences, findiff), to be compared with the Gauss-Hermite version
``test_IKS_3D_2part.py``. Thesis: Section 5.1 (Table 5.1, Fig. 5.2).

Purpose of the comparison
-------------------------
The uniform grid and the Lagrange mesh are two discretisations of the same
problem. Expected differences:
  * GAUSS-HERMITE: nodes clustered where the density is significant, so few
    points per axis (N ~ 24) suffice, but the outer nodes with
    sqrt(rho) -> 0 must be masked.
  * UNIFORM: equispaced points, so a box [-L, L] and a spacing h have to be
    chosen; fewer "dead node" problems, but many points end up in the tails
    where rho ~ 0.

For A = 2 (pure Gaussian ground state) the analytic reference is
    T = 2 * (3/4) hbar*omega ~= 48.81 MeV.
Since f = 1/sqrt(2) everywhere, the regulariser plays no role here and the
value of beta (REG) is irrelevant.

Usage
-----
    python test_IKS_3D_2part_uniform.py

Output: test_3D_data_2part_uniform/IKS_2particles_uniform.pdf

This module is also imported by ``compare_grid.py``.
"""

import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from findiff import FinDiff

# --------------------------------------------------------------------------
# Typographic style of the figures (same as test_IKS_3D_2part.py).
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.size":          17,   # generic text
    "axes.labelsize":     21,   # axis labels
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
                      nuclearNu_x, nuclearNu_y, nuclearNu_z, b_x)
from orbitals3D import uncoupledHObasis, getOrbitalSet
from optimizer3D_lm import iKS_problem
from eigenfunctions3D import HO3D

OUTPUT_DIR = "test_3D_data_2part_uniform"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REG     = 0.1         # regularisation weight beta (irrelevant for A = 2)
N_GRID  = 41          # points per axis of the full run
L_BOX   = 8.0         # half-width of the box [fm]


def kinetic_reference(A, occ):
    """Exact kinetic energy of the 3D HO ground state (for occ particles) [MeV]."""
    return occ * 0.75 * hbc * nuclear_omega_x(A)


def build_uniform_grid(n, L):
    """n equispaced points on [-L, L]; returns (xs, h)."""
    xs = np.linspace(-L, L, n)
    h  = xs[1] - xs[0]
    return xs, h


def build_target_density_uniform(xs, h, A, n_part):
    """Target rho on the uniform grid; returns (rho, phi_norm, orbitals)."""
    orbitals = list(getOrbitalSet(n_part, uncoupledHObasis()))
    phi_norm = []
    for o in orbitals:
        p = HO3D(o.nx, o.ny, o.nz, b_x(A), b_x(A), b_x(A), h**3)(xs, xs, xs)
        p /= np.sqrt(np.sum(p**2) * h**3)
        phi_norm.append(p)
    rho = sum(orbitals[k].occupation * phi_norm[k]**2 for k in range(len(orbitals)))
    return rho, phi_norm, orbitals


def kinetic_functional_uniform(xs, h, rho, phi_norm, orbitals, beta):
    """T_alpha (physical kinetic part of the objective) on the exact target,
    uniform grid with findiff derivatives. No mask is applied here.
    ``beta`` is unused (T_beta is not included)."""
    s = np.sqrt(np.maximum(rho, 1e-300))
    dx = FinDiff(0, h, 1, acc=4); dy = FinDiff(1, h, 1, acc=4); dz = FinDiff(2, h, 1, acc=4)
    glog = np.stack([dx(np.log(s)), dy(np.log(s)), dz(np.log(s))], axis=-1)
    nglog = np.sum(glog**2, axis=-1)
    T_a = 0.0
    for k, o in enumerate(orbitals):
        f  = phi_norm[k] / s
        gf = np.stack([dx(f), dy(f), dz(f)], axis=-1)
        gm2 = np.sum(gf**2, axis=-1); dot = np.sum(glog * gf, axis=-1)
        # alpha*rho*(|grad f|^2 + f^2|g|^2 + 2 f grad f.g) = alpha*|grad phi|^2
        T_a += o.occupation * np.sum(ALPHA * rho * (gm2 + f**2*nglog + 2*f*dot)) * h**3
    return T_a


def convergence_scan_uniform(A, n_part, occ, grids, beta):
    """T_alpha on the exact target for each (n, L) in ``grids`` (no optimisation).

    Returns (E_ref, rows) with rows = [(n, L, h, T_alpha, err_%), ...].
    """
    E_ref = kinetic_reference(A, occ)
    rows = []
    for (n, L) in grids:
        xs, h = build_uniform_grid(n, L)
        rho, phi_norm, orbitals = build_target_density_uniform(xs, h, A, n_part)
        T_a = kinetic_functional_uniform(xs, h, rho, phi_norm, orbitals, beta)
        rows.append((n, L, h, T_a, 100*(T_a-E_ref)/E_ref))
    return E_ref, rows


def run_iks_uniform(A, n_part, n, L, beta):
    """Solve the IKS problem on an n^3 uniform grid on [-L, L]^3 and extract V.

    Returns a dict with the solver status, objective, timing, density and
    potential errors and the arrays needed for the figures.
    """
    xs, h = build_uniform_grid(n, L)
    rho_target, phi_norm, orbitals = build_target_density_uniform(xs, h, A, n_part)
    n_orb = len(orbitals)

    prob = iKS_problem(
        Z=A, N=0, lb=[-L, -L, -L], ub=[L, L, L],
        data=[xs, xs, xs, rho_target], hx=h, hy=h, hz=h,
        basis=uncoupledHObasis(), reg_parameter=beta,
        output_folder="test_2_uniform",
        exact_hess=True, max_iter=500,
        rel_tol=1e-8, constr_viol=1e-8, dual_inf_tol=1.0,
        use_lagrange_mesh=False,
    )
    prob.add_option(b"mu_init", 0.01)

    t0 = time.time()
    results, info = prob.solve()
    elapsed = time.time() - t0

    obj      = float(results.get('obj', float('nan')))
    status   = results.get('status', 'N/A')
    phi      = results['u']
    rho_calc = np.sum([orbitals[k].occupation * phi[k]**2 for k in range(n_orb)], axis=0)
    norm_calc   = np.sum(rho_calc) * h**3
    density_err = np.sqrt(np.sum((rho_calc - rho_target)**2) * h**3)

    # Potential: V = lambda/(rho w), with w = dV constant on the uniform grid.
    # The density multipliers exist only on the live nodes (NaN elsewhere).
    lam_live = results['lagrange'][:prob.n_live]
    lam_full = np.full(prob.n_points, np.nan)
    lam_full[prob.live_nodes] = lam_live
    lam = lam_full.reshape(n, n, n)
    w3D = prob.w3D
    mask = (rho_target >= 1e-10 * rho_target.max())
    with np.errstate(divide='ignore', invalid='ignore'):
        V_iks = lam / (rho_target * w3D)
    V_iks = np.where(mask, V_iks, np.nan)
    # Gauge: V = 0 at the central node (n odd -> the origin is a grid point)
    ic = n // 2
    V_iks -= V_iks[ic, ic, ic]

    # Exact reference (harmonic trap) in the same gauge
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    V_ana = nuclearNu_x(A)*X**2 + nuclearNu_y(A)*Y**2 + nuclearNu_z(A)*Z**2
    V_ana -= V_ana[ic, ic, ic]
    # RMSE along the x axis through the centre, |x| < 3b
    b_ho = b_x(A)
    sel = (np.abs(xs) < 3*b_ho)
    dVx = (V_iks[:, ic, ic] - V_ana[:, ic, ic])
    valid = sel & np.isfinite(dVx)
    pot_err = np.sqrt(np.mean(dVx[valid]**2)) if np.any(valid) else float('nan')

    return dict(n=n, L=L, h=h, status=status, obj=obj, elapsed=elapsed,
                norm_calc=norm_calc, density_err=density_err, pot_err=pot_err,
                V_iks=V_iks, V_ana=V_ana, rho_target=rho_target, rho_calc=rho_calc,
                xs=xs, ic=ic)


def make_figure(E_ref, scan_rows, res, fig_path):
    """2x2 figure: T(n) and its relative error; density and potential along x."""
    ns = [r[0] for r in scan_rows]; Ts = [r[3] for r in scan_rows]
    xs = res['xs']; ic = res['ic']
    rho_x = res['rho_target'][:, ic, ic]
    R = physical_radius(rho_x, xs, frac=1e-3)
    Rlim = 1.15 * R

    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5), constrained_layout=True)
    fig.suptitle("A=2 GRIGLIA UNIFORME (FinDiff)", fontweight="bold")

    # (0,0) convergence of T_alpha
    ax = axes[0, 0]
    ax.axhline(E_ref, color="#dc2626", lw=2, ls="--",
               label=f"T$_\\alpha$ analitico = {E_ref:.2f} MeV")
    ax.plot(ns, Ts, "o-", color="#0891b2", label="T$_\\alpha$(n)")
    ax.set_xlabel("n (punti/asse)"); ax.set_ylabel("T$_\\alpha$ [MeV]")
    ax.set_title("Energia cinetica"); ax.legend(); ax.grid(alpha=0.4)

    # (0,1) relative error
    ax = axes[0, 1]
    err = [abs(r[4]) for r in scan_rows]
    ax.semilogy(ns, err, "s-", color="#9333ea")
    ax.set_xlabel("n (punti/asse)"); ax.set_ylabel("|errore| [%]")
    ax.set_title("Errore relativo su T$_\\alpha$"); ax.grid(alpha=0.4)

    # (1,0) density along x
    ax = axes[1, 0]
    ax.plot(xs, res['rho_target'][:, ic, ic], color="#dc2626", lw=2, label="ρ target")
    ax.plot(xs, res['rho_calc'][:, ic, ic], "--", color="#16a34a", lw=1.8, label="ρ IKS")
    ax.set_xlim(-Rlim, Rlim)
    ax.set_xlabel("x [fm]"); ax.set_ylabel("ρ [fm⁻³]")
    ax.set_title(f"Densità (n={res['n']}, norma={res['norm_calc']:.3f})")
    ax.legend(); ax.grid(alpha=0.4)

    # (1,1) potential along x, physical region
    ax = axes[1, 1]
    V_iks_x = res['V_iks'][:, ic, ic]
    ax.plot(xs, res['V_ana'][:, ic, ic], color="#dc2626", lw=2, label="V analitico (HO)")
    ax.plot(xs, V_iks_x, "--", color="#0891b2", lw=1.8, label="V IKS")
    ax.set_xlim(-Rlim, Rlim)
    yl = potential_ylim(V_iks_x, xs, R)
    if yl:
        ax.set_ylim(*yl)
    ax.set_xlabel("x [fm]"); ax.set_ylabel("V [MeV]")
    ax.set_title(f"Potenziale lungo x\n(RMSE={res['pot_err']:.2e} MeV)")
    ax.legend(); ax.grid(alpha=0.4)

    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figura salvata: {fig_path}")


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║     TEST IKS 3D — Z=2 : GRIGLIA UNIFORME (FinDiff)                  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    A = 2; n_part = 2
    orb0 = list(getOrbitalSet(n_part, uncoupledHObasis()))[0]
    occ = orb0.occupation
    print(f"  A={A}, 1 orbitale {(orb0.nx, orb0.ny, orb0.nz)}, occ={occ}")

    # Convergence scan on (points per axis, half-width of the box)
    grids = [(21, 6.0), (31, 7.0), (41, 8.0), (51, 8.0), (61, 9.0)]
    E_ref, scan_rows = convergence_scan_uniform(A, n_part, occ, grids, REG)
    print(f"\n  E_kin analitico = {E_ref:.4f} MeV\n")
    print("  Convergenza cinetica (griglia uniforme, densità target esatta):")
    print(f"  {'n':>4}  {'L[fm]':>6}  {'h[fm]':>6}  {'T_α [MeV]':>12}  {'errore':>9}")
    for (n, L, h, Ta, errp) in scan_rows:
        print(f"  {n:>4}  {L:>6.1f}  {h:>6.3f}  {Ta:>12.4f}  {errp:>8.2f}%")

    # Full IKS run
    print(f"\n  Run IKS completo (uniforme, n={N_GRID}, L={L_BOX}) …")
    res = run_iks_uniform(A, n_part, N_GRID, L_BOX, REG)
    print(f"\n  Status        : {res['status']}")
    print(f"  E_kin (IKS)   : {res['obj']:.4f} MeV   (rif. {E_ref:.4f}, "
          f"Δ {100*(res['obj']-E_ref)/E_ref:+.2f}%)")
    print(f"  Δρ L2         : {res['density_err']:.3e}")
    print(f"  Norma ρ_calc  : {res['norm_calc']:.4f}  (atteso {n_part})")
    print(f"  V RMSE (|x|<3b): {res['pot_err']:.3e} MeV")
    print(f"  Tempo         : {res['elapsed']:.1f} s")

    make_figure(E_ref, scan_rows, res,
                os.path.join(OUTPUT_DIR, "IKS_2particles_uniform.pdf"))
    print("\n  Confronto con Gauss-Hermite: vedi test_IKS_3D_2part.py")
    print("  (GH: 0.34% a N=24=13824 punti; qui: griglia uniforme n³ punti)")
    print("\n  Fatto.\n")


if __name__ == "__main__":
    main()
