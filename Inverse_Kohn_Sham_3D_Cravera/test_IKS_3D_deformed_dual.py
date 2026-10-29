"""
test_IKS_3D_deformed_dual.py
============================
Self-contained test for
  1. densities from an ANISOTROPIC (axially deformed) harmonic oscillator;
  2. a scan of the IPOPT option dual_inf_tol, measuring the accuracy
     (T_alpha and RMSE of the potential) as a function of the tolerance on
     the dual infeasibility.
Thesis: Section 5.3, "The effect of the dual tolerance" (Figs. 5.9 and 5.10).

Both scans run on the Lagrange mesh (Gauss-Hermite) and on the uniform grid,
for comparison. The script re-implements the IKS runs it needs, reusing the
infrastructure (consts3D, eigenfunctions3D, orbitals3D, optimizer3D_lm,
lagrange_mesh_3D) and deformed.py for the anisotropic oscillator.

Main result (thesis): on the uniform grid both indicators are STEP functions
of dual_inf_tol (a loose tolerance stops the solver early, before it reaches
the spurious minimum of the discrete functional); on the Lagrange mesh they
are flat, because the runs are fully converged anyway.

NOTE: the occupied orbitals follow the spherical aufbau order of
getOrbitalSet ({(000),(100)}) with the deformed oscillator lengths.

Usage
-----
  python test_IKS_3D_deformed_dual.py --dual-scan      # dual_inf_tol scan (GH + uniform)
  python test_IKS_3D_deformed_dual.py --deform-demo    # deformed-density reconstruction
  python test_IKS_3D_deformed_dual.py                  # both

Output: test_3D_data_deformed_dual/{dual_scan_gh, dual_scan_unif, deform_demo}.pdf
"""
import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the modules in src/ importable regardless of the working directory.
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_helpers import (physical_radius, potential_ylim,
                          masked_plane, symmetric_vlim, potential_error)
from consts3D import (a as ALPHA, hbc, nuclear_omega_x,
                      nuclearNu_x, nuclearNu_y, nuclearNu_z,
                      b_x, b_y, b_z)
from orbitals3D import uncoupledHObasis, getOrbitalSet
from optimizer3D_lm import iKS_problem
from lagrange_mesh_3D import LagrangeMesh3D
from eigenfunctions3D import HO3D
from deformed import deformed_lengths, deformed_nu, deformed_frequencies

OUTPUT_DIR = "test_3D_data_deformed_dual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

A_DEFAULT   = 4
N_PART      = 4
N_RUN_GH    = 50       # nodes per axis, Lagrange mesh
N_RUN_UNIF  = 51       # points per axis, uniform grid
L_BOX       = 7.5      # half-width of the uniform box [fm]
REG         = 1e-3      # beta


# ──────────────────────────────────────────────────────────────────────────────
# Target density (spherical if delta = 0, anisotropic otherwise)
# ──────────────────────────────────────────────────────────────────────────────
def _lengths(A, delta):
    """Oscillator lengths (bx, by, bz): spherical or deformed."""
    if abs(delta) > 1e-12:
        bp, bz = deformed_lengths(A, delta)
        return bp, bp, bz
    return b_x(A), b_y(A), b_z(A)


def _reference_potential(A, delta, X, Y, Z):
    """Reference harmonic potential (isotropic or anisotropic) on the meshgrid X, Y, Z."""
    if abs(delta) > 1e-12:
        nu_p, nu_z = deformed_nu(A, delta)
        return nu_p * (X**2 + Y**2) + nu_z * Z**2
    return nuclearNu_x(A) * X**2 + nuclearNu_y(A) * Y**2 + nuclearNu_z(A) * Z**2


def _orbitals_density(eval_wf, A, n_part, delta, norm_fn):
    """Normalised orbitals and density for a given wave-function evaluator.

    eval_wf(orbital, bx, by, bz) returns the orbital on the grid; norm_fn(p)
    its squared norm. Returns (rho, phi_list, orbitals)."""
    bx, by, bz = _lengths(A, delta)
    orbitals = list(getOrbitalSet(n_part, uncoupledHObasis()))
    phi = []
    for o in orbitals:
        p = eval_wf(o, bx, by, bz)
        p = p / np.sqrt(norm_fn(p))
        phi.append(p)
    rho = sum(orbitals[k].occupation * phi[k]**2 for k in range(len(orbitals)))
    return rho, phi, orbitals


def _install_deformed_start(prob, A, delta, phi_norm):
    """Non-invasive override of the solver's starting point for deformed densities.

    getStartingPoint() in the solver always builds SPHERICAL orbitals; for a
    deformed density this gives an inconsistent f = phi_spherical/sqrt(rho_deformed).
    Here prob.getStartingPoint is replaced, on the instance, by a version that
    uses the (already deformed) orbitals that generated the target density.
    The solver file is not modified."""
    if abs(delta) <= 1e-12:
        return  # spherical: the default start is fine
    rho_safe = np.maximum(prob.tab_rho, 1e-14 * prob.tab_rho.max())
    dead = prob.tab_rho < 1e-10 * prob.tab_rho.max()
    occ_tot = sum(o.occupation for o in prob.orbital_set)
    plateau = getattr(prob, "_plateau_value", None) or (1.0 / np.sqrt(occ_tot))
    st = np.zeros((prob.n_orbitals, prob.n_points_x, prob.n_points_y, prob.n_points_z))
    for j, p in enumerate(phi_norm):
        f = p / np.sqrt(rho_safe)
        f[dead] = plateau
        st[j] = f
    flat = st.ravel()
    prob.getStartingPoint = lambda: flat   # override on the instance


# ──────────────────────────────────────────────────────────────────────────────
# Run IKS — Lagrange mesh (Gauss–Hermite)
# ──────────────────────────────────────────────────────────────────────────────
def run_gh(A, n_part, N, beta, delta=0.0, dual_inf_tol=1.0):
    """Full IKS run on the (s = 1) Lagrange mesh; returns T_alpha, potentials and errors."""
    mesh = LagrangeMesh3D(N, N, N, b_x(A), b_y(A), b_z(A))
    x, y, z = mesh.x, mesh.y, mesh.z

    rho_target, phi_norm, orbitals = _orbitals_density(
        lambda o, bx, by, bz: HO3D(o.nx, o.ny, o.nz, bx, by, bz, mesh.dV)(x, y, z),
        A, n_part, delta, norm_fn=lambda p: mesh.integrate(p**2))
    n_orb = len(orbitals)

    prob = iKS_problem(
        Z=A, N=0, lb=[-2, -2, -2], ub=[2, 2, 2],
        data=[x, y, z, rho_target],
        basis=uncoupledHObasis(), reg_parameter=beta,
        output_folder="test_deformed_dual_gh",
        exact_hess=True, max_iter=500,
        rel_tol=1e-4, constr_viol=1e-4, dual_inf_tol=dual_inf_tol,
        use_lagrange_mesh=True, Nx=N, Ny=N, Nz=N,
    )
    prob.add_option(b"mu_init", 0.01)
    _install_deformed_start(prob, A, delta, phi_norm)

    t0 = time.time()
    results, info = prob.solve()
    elapsed = time.time() - t0

    # kinetic energy T_alpha from the decomposition of the functional at the solution
    f_sol = results['x'].reshape(n_orb, mesh.Nx, mesh.Ny, mesh.Nz)
    T_a, _ = _decompose_gh(mesh, rho_target, [f_sol[k] for k in range(n_orb)],
                           orbitals, beta)

    # potential: V = lambda/(rho w) on the live nodes, gauge V = 0 at the density maximum
    lam_live = results['lagrange'][:prob.n_live]
    lam_full = np.full(mesh.n_points, np.nan)
    lam_full[prob.live_nodes] = lam_live
    lam = lam_full.reshape(mesh.Nx, mesh.Ny, mesh.Nz)
    mask = rho_target >= 1e-10 * rho_target.max()
    with np.errstate(divide='ignore', invalid='ignore'):
        V_iks = np.where(mask, lam / (rho_target * mesh.w3D), np.nan)
    ic = np.unravel_index(np.nanargmax(rho_target), rho_target.shape)
    V_iks -= V_iks[ic]

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    V_ref = _reference_potential(A, delta, X, Y, Z); V_ref -= V_ref[ic]
    pe = potential_error(V_iks, V_ref, rho_target, x, frac=1e-2)

    return dict(T_alpha=T_a, V_iks=V_iks, V_ref=V_ref, rho_target=rho_target,
                pot_err=pe, elapsed=elapsed, center=ic, grid=(x, y, z),
                delta=delta, grid_kind="gh")


def _decompose_gh(mesh, rho, f_list, orbitals, beta):
    """T_alpha (expanded form) and T_beta on the Lagrange mesh. Same decomposition
    as the validation tests: uses grad log sqrt(rho) and masks the physical
    region (dead nodes, where f = plateau, are excluded from the integral)."""
    mask = rho >= 1e-10 * rho.max()
    s = np.sqrt(np.maximum(rho, 1e-14 * rho.max()))
    glog = mesh.grad(np.log(s))
    nglog = np.sum(glog**2, axis=-1)
    T_a = T_b = 0.0
    for k, o in enumerate(orbitals):
        f = f_list[k]
        gf = mesh.grad(f)
        gm2 = np.sum(gf**2, axis=-1)
        dot = np.sum(glog * gf, axis=-1)
        T_a += o.occupation * mesh.integrate(ALPHA * rho * (gm2 + f**2*nglog + 2*f*dot) * mask)
        T_b += o.occupation * mesh.integrate(beta * gm2 * mask)
    return T_a, T_b


# ──────────────────────────────────────────────────────────────────────────────
# IKS run — uniform grid
# ──────────────────────────────────────────────────────────────────────────────
def run_unif(A, n_part, n, L, beta, delta=0.0, dual_inf_tol=1.0):
    """Full IKS run on the n^3 uniform grid on [-L, L]^3 (tol = 1, constr_viol = 1e-3)."""
    from findiff import FinDiff
    xs = np.linspace(-L, L, n); h = xs[1] - xs[0]

    rho_target, phi_norm, orbitals = _orbitals_density(
        lambda o, bx, by, bz: HO3D(o.nx, o.ny, o.nz, bx, by, bz, h**3)(xs, xs, xs),
        A, n_part, delta, norm_fn=lambda p: np.sum(p**2) * h**3)
    n_orb = len(orbitals)

    prob = iKS_problem(
        Z=A, N=0, lb=[-L, -L, -L], ub=[L, L, L],
        data=[xs, xs, xs, rho_target], hx=h, hy=h, hz=h,
        basis=uncoupledHObasis(), reg_parameter=beta,
        output_folder="test_deformed_dual_unif",
        exact_hess=True, max_iter=500,
        rel_tol=1, constr_viol=1e-3, dual_inf_tol=dual_inf_tol,
        use_lagrange_mesh=False,
    )
    prob.add_option(b"mu_init", 0.01)
    prob.add_option(b"mumps_mem_percent", 20)
    _install_deformed_start(prob, A, delta, phi_norm)

    t0 = time.time()
    results, info = prob.solve()
    elapsed = time.time() - t0

    ic = n // 2
    f_sol = results['x'].reshape(n_orb, n, n, n)
    T_a, _ = _decompose_unif(xs, h, rho_target,
                             [f_sol[k] for k in range(n_orb)], orbitals, beta)

    lam_live = results['lagrange'][:prob.n_live]
    lam_full = np.full(rho_target.size, np.nan)
    lam_full[prob.live_nodes] = lam_live
    lam = lam_full.reshape(n, n, n)
    w3D = prob.w3D if hasattr(prob, "w3D") else np.full((n, n, n), h**3)
    mask = rho_target >= 1e-10 * rho_target.max()
    with np.errstate(divide='ignore', invalid='ignore'):
        V_iks = np.where(mask, lam / (rho_target * w3D), np.nan)
    V_iks -= V_iks[ic, ic, ic]

    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    V_ref = _reference_potential(A, delta, X, Y, Z); V_ref -= V_ref[ic, ic, ic]
    pe = potential_error(V_iks, V_ref, rho_target, xs, frac=1e-2)

    return dict(T_alpha=T_a, V_iks=V_iks, V_ref=V_ref, rho_target=rho_target,
                pot_err=pe, elapsed=elapsed, center=(ic, ic, ic), grid=(xs, xs, xs),
                delta=delta, grid_kind="unif")


def _decompose_unif(xs, h, rho, f_list, orbitals, beta):
    """T_alpha and T_beta on the uniform grid (findiff derivatives, acc=4)."""
    from findiff import FinDiff
    mask = rho >= 1e-10 * rho.max()
    s = np.sqrt(np.maximum(rho, 1e-14 * rho.max()))
    dx = FinDiff(0, h, 1, acc=4); dy = FinDiff(1, h, 1, acc=4); dz = FinDiff(2, h, 1, acc=4)
    glog = np.stack([dx(np.log(s)), dy(np.log(s)), dz(np.log(s))], axis=-1)
    nglog = np.sum(glog**2, axis=-1)
    T_a = T_b = 0.0
    for k, o in enumerate(orbitals):
        f = f_list[k]
        gf = np.stack([dx(f), dy(f), dz(f)], axis=-1)
        gm2 = np.sum(gf**2, axis=-1); dot = np.sum(glog * gf, axis=-1)
        T_a += o.occupation * np.sum(ALPHA * rho * (gm2 + f**2 * nglog + 2*f*dot) * mask) * h**3
        T_b += o.occupation * np.sum(beta * gm2 * mask) * h**3
    return T_a, T_b


# ──────────────────────────────────────────────────────────────────────────────
# dual_inf_tol scan
# ──────────────────────────────────────────────────────────────────────────────
def dual_scan(run_fn, A, n_part, res_arg, beta, dual_list, delta=0.0):
    """Run run_fn(A, n_part, *res_arg, beta, delta, dual_inf_tol) for each tolerance.

    res_arg is (N,) for GH or (n, L) for the uniform grid, passed as a tuple.
    The kinetic reference is computed for the specific deformation delta.
    Returns rows (dual_inf_tol, T_alpha, err_%, RMSE_rho V, max|dV|)."""
    E_ref = _analytic_Ekin(A, n_part, delta=delta)
    rows = []
    for dual in dual_list:
        r = run_fn(A, n_part, *res_arg, beta, delta=delta, dual_inf_tol=dual)
        Ta = r['T_alpha']
        err = 100.0 * (Ta - E_ref) / E_ref if np.isfinite(Ta) else np.nan
        pe = r['pot_err']
        rows.append((dual, Ta, err, pe['rmse_pesato'], pe['max_abs']))
    return rows


def make_dual_figure(rows_by_delta, title, fig_path):
    """Two panels: RMSE of the potential and kinetic error vs dual_inf_tol,
    one curve per deformation delta."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle(title, fontweight="bold", fontsize=12)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(rows_by_delta), 1)))

    ax = axes[0]
    for (delta, rows), c in zip(rows_by_delta.items(), colors):
        duals = [r[0] for r in rows]; Vr = [r[3] for r in rows]
        ax.loglog(duals, Vr, "o-", color=c, label=f"δ={delta:+.2f}")
    ax.set_xlabel("dual_inf_tol"); ax.set_ylabel("RMSE V [MeV]")
    ax.set_title("Errore del potenziale vs dual_inf_tol")
    ax.legend(fontsize=8); ax.grid(alpha=0.4, which="both")

    ax = axes[1]
    for (delta, rows), c in zip(rows_by_delta.items(), colors):
        duals = [r[0] for r in rows]; errs = [abs(r[2]) for r in rows]
        ax.loglog(duals, np.array(errs) + 1e-6, "s-", color=c, label=f"δ={delta:+.2f}")
    ax.set_xlabel("dual_inf_tol"); ax.set_ylabel("|errore T$_α$| [%]")
    ax.set_title("Errore cinetico vs dual_inf_tol")
    ax.legend(fontsize=8); ax.grid(alpha=0.4, which="both")

    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figura dual-scan salvata: {fig_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Demo: reconstruction of deformed densities (2D figures)
# ──────────────────────────────────────────────────────────────────────────────
def deform_demo(A, n_part, deltas, fig_path):
    """Target density and reconstructed potential on the x-z plane for several
    delta, to visualise the shape (cigar/disc) and the reconstruction."""
    nrow = len(deltas)
    fig, axes = plt.subplots(nrow, 3, figsize=(13, 4 * nrow),
                             constrained_layout=True, squeeze=False)
    fig.suptitle("Ricostruzione IKS su densità deformata (Lagrange mesh)",
                 fontweight="bold", fontsize=12)
    for i, delta in enumerate(deltas):
        r = run_gh(A, n_part, N_RUN_GH, REG, delta=delta, dual_inf_tol=1e-4)
        x, y, z = r['grid']; ic = r['center']
        R = physical_radius(r['rho_target'][:, ic[1], ic[2]], x, frac=1e-3)
        # x-z plane (at y = y_c)
        # density
        ax = axes[i][0]
        rho_xz, ext = _plane_xz(r['rho_target'], x, z, ic, 1.15 * R)
        im = ax.imshow(rho_xz, origin="lower", extent=ext, aspect="equal", cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"δ={delta:+.2f}: ρ(x,z)"); ax.set_xlabel("x [fm]"); ax.set_ylabel("z [fm]")
        # reconstructed potential
        ax = axes[i][1]
        V_xz, ext = _plane_xz(r['V_iks'], x, z, ic, R)
        vlo, vhi = symmetric_vlim(V_xz)
        im = ax.imshow(V_xz, origin="lower", extent=ext, aspect="equal",
                       cmap="RdBu_r", vmin=vlo, vmax=vhi)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("V$_{KS}$(x,z) IKS"); ax.set_xlabel("x [fm]"); ax.set_ylabel("z [fm]")
        # reference potential
        ax = axes[i][2]
        Vr_xz, ext = _plane_xz(r['V_ref'], x, z, ic, R)
        im = ax.imshow(Vr_xz, origin="lower", extent=ext, aspect="equal",
                       cmap="RdBu_r", vmin=vlo, vmax=vhi)
        fig.colorbar(im, ax=ax, fraction=0.046)
        rmse = r['pot_err']['rmse_pesato']
        ax.set_title(f"V riferimento (RMSE={rmse:.2f})")
        ax.set_xlabel("x [fm]"); ax.set_ylabel("z [fm]")
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figura deform-demo salvata: {fig_path}")


def _plane_xz(arr3d, x, z, ic, R):
    """Slice on the x-z plane at y = y_c, cropped and masked at radius R."""
    icx, icy, icz = ic
    sx = np.abs(x) <= R
    sz = np.abs(z) <= R
    xx, zz = x[sx], z[sz]
    plane = arr3d[np.ix_(sx, [icy], sz)][:, 0, :].T   # (z, x)
    XX, ZZ = np.meshgrid(xx, zz)
    radial = np.sqrt(XX**2 + ZZ**2) <= R
    plane = np.where(radial, plane, np.nan)
    return plane, [xx.min(), xx.max(), zz.min(), zz.max()]


# ──────────────────────────────────────────────────────────────────────────────
def _analytic_Ekin(A, n_part, delta=0.0):
    """Analytic kinetic energy of the (possibly anisotropic) oscillator.

    For an oscillator eigenstate the energy is
        E = hbar omega_perp (n_x + n_y + 1) + hbar omega_z (n_z + 1/2),
    and by the virial theorem T = E/2 (also in the anisotropic case). For
    delta = 0 it reduces to the spherical T = 1/2 sum q (n + 3/2) hbar omega_0.
    """
    orbitals = list(getOrbitalSet(n_part, uncoupledHObasis()))
    if abs(delta) > 1e-12:
        wp, wz = deformed_frequencies(A, delta)
        hwp, hwz = hbc * wp, hbc * wz
        E_tot = sum(o.occupation * (hwp * (o.nx + o.ny + 1) + hwz * (o.nz + 0.5))
                    for o in orbitals)
    else:
        hw = hbc * nuclear_omega_x(A)
        E_tot = sum(o.occupation * hw * (o.nx + o.ny + o.nz + 1.5) for o in orbitals)
    return 0.5 * E_tot


def main():
    print("\n  Test deformazione + scan dual_inf_tol (file dedicato)\n")
    A, n_part = A_DEFAULT, N_PART
    print(f"  E_kin analitico di riferimento (Z={A}):")
    for d in (0.0, 0.4, -0.4):
        print(f"    δ={d:+.2f}:  T = {_analytic_Ekin(A, n_part, delta=d):.4f} MeV")
    print()

    do_dual   = ("--dual-scan" in sys.argv) or (len(sys.argv) == 1)
    do_demo   = ("--deform-demo" in sys.argv) or (len(sys.argv) == 1)

    dual_list  = [1e6, 1e3, 1.0, 1e-2, 1e-4, 1e-6]
    delta_list = [0.0, 0.4]      # sferico, prolato

    if do_dual:
        # Lagrange mesh (thesis Fig. 5.10)
        print("  [Lagrange mesh] scan dual_inf_tol:")
        rows_gh = {}
        for delta in delta_list:
            print(f"    δ={delta:+.2f} (T_rif={_analytic_Ekin(A, n_part, delta=delta):.3f} MeV):")
            rows = dual_scan(run_gh, A, n_part, (N_RUN_GH,), REG, dual_list, delta=delta)
            for (dual, Ta, errp, vr, vmax) in rows:
                print(f"      dual={dual:>8.0e}:  T_α={Ta:8.3f} ({errp:+.2f}%),  "
                      f"RMSE V={vr:8.3f},  max|ΔV|={vmax:8.2f}")
            rows_gh[delta] = rows
        make_dual_figure(rows_gh, "IKS 3D Z=4 (Lagrange mesh): scan dual_inf_tol",
                         os.path.join(OUTPUT_DIR, "dual_scan_gh.pdf"))

        # uniform grid (thesis Fig. 5.9)
        print("\n  [Griglia uniforme] scan dual_inf_tol:")
        rows_un = {}
        for delta in delta_list:
            print(f"    δ={delta:+.2f} (T_rif={_analytic_Ekin(A, n_part, delta=delta):.3f} MeV):")
            rows = dual_scan(run_unif, A, n_part, (N_RUN_UNIF, L_BOX), REG, dual_list, delta=delta)
            for (dual, Ta, errp, vr, vmax) in rows:
                print(f"      dual={dual:>8.0e}:  T_α={Ta:8.3f} ({errp:+.2f}%),  "
                      f"RMSE V={vr:8.3f},  max|ΔV|={vmax:8.2f}")
            rows_un[delta] = rows
        make_dual_figure(rows_un, "IKS 3D Z=4 (griglia uniforme): scan dual_inf_tol",
                         os.path.join(OUTPUT_DIR, "dual_scan_unif.pdf"))
        print("\n  NB: dual_inf_tol agisce solo con IPOPT reale; con un mock le")
        print("  curve risultano piatte (le opzioni del solver vengono ignorate).")

    if do_demo:
        print("\n  [Demo] ricostruzione su densità deformata (Lagrange mesh):")
        deform_demo(A, n_part, [0.0, 0.4, -0.4],
                    os.path.join(OUTPUT_DIR, "deform_demo.pdf"))

    print("\n  Fatto.\n")


if __name__ == "__main__":
    main()