"""
test_IKS_3D_8part_multipoles.py
===============================
A = 8: multipole-moment constraints on a CLOSED shell with a spherical density.
Thesis: Section 6.1 (Tables 6.1 and 6.2, row "A = 8 GH").

Why A = 8
---------
The occupied orbitals are {(000), (100), (010), (001)}: the sum
x^2 + y^2 + z^2 is isotropic, so the density is spherical and the l > 0
moments vanish to machine precision, as for A = 2. Unlike A = 2, however, the
problem is not degenerate:

  * n_orb = 4 -> n_x = 4*n_live, against n_c = n_live + 10. The flip to
    fixed_variable_treatment = relax_bounds must NOT happen: the n_x column
    must stay the same between baseline and constrained runs (impossible to
    check at A = 2).
  * 10 orthonormality constraints (``pairs``), so the cross block of the
    Hessian and the slicing [density] + [pairs] + [multipoles] are fully
    exercised.
  * the solution is not fixed by the constraints alone: there is a genuine
    optimisation, so "dE base ~ 0" becomes a meaningful statement (at A = 2
    the solution was f = 1/sqrt(2) everywhere regardless).
  * the unitary gauge U(4) is large: the overlap diagnostic shows whether the
    runs converge to different points of the same orbit at identical density —
    something a constraint built from rho cannot prevent.

The reference potential is the isotropic oscillator, exact because the
target is built from pure HO orbitals.

Parameters: N_GH = 28 and beta = 1e-3 are the best point found in the
convergence studies for A = 8. The uniform grid at these values is much
worse on the potential: this is expected and does not affect the test, which
compares each grid with ITS OWN baseline.

Usage
-----
    python test_IKS_3D_8part_multipoles.py                 # GH, both profiles
    python test_IKS_3D_8part_multipoles.py uniform
    python test_IKS_3D_8part_multipoles.py gh keep

Output: test_3D_data_8part_multipole/IKS_8part_multipole_<grid>_<profile>.pdf
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

from consts3D import (hbc, nuclear_omega_x, nuclearNu_x, nuclearNu_y,
                      nuclearNu_z, b_x, b_y, b_z)
from orbitals3D import uncoupledHObasis, getOrbitalSet
from optimizer3D_lm import iKS_problem
from lagrange_mesh_3D import LagrangeMesh3D
from eigenfunctions3D import HO3D
from multipoles3D import multipole_moments, gauge_correction
from plot_helpers import physical_radius, potential_ylim

RHO_FLOOR_FRAC = 1e-10   # mask for the extraction of V (consistent with the optimiser)
RHO_PHYS_FRAC  = 1e-2    # "physical" region for the potential metrics

# Solver profiles: options imposed EXPLICITLY, identical on both grids. The
# code's default enables dependency_detector only on the Lagrange-mesh branch,
# so without this override GH and uniform would not solve the same numerical
# problem.
#   remove : dependent rows removed                    -> mu = 0 by construction
#   keep   : rows kept, singular KKT system regularised -> mu != 0
PROFILES = {
    "remove": dict(dependency_detector=b"mumps", fixed_variable_treatment=None),
    "keep":   dict(dependency_detector=b"none",  fixed_variable_treatment=None),
}


# ─────────────────────────────────────────────────────────────────────────────
# Physical case: A = 8
# ─────────────────────────────────────────────────────────────────────────────
N_PART = 8

CFG = dict(
    N_GH       = 28,      # optimum from the A = 8 convergence scans
    MESH_SCALE = 1.0,     # mesh scale s
    N_UNIFORM  = 41,      # points per axis, uniform grid
    L_BOX      = 9.0,     # half-width of the uniform box [fm]; density wider than at A = 4
    BETA       = 1e-3,    # regularisation weight (both grids)
)

# Constraint configurations; the first one MUST be the baseline (None).
CONFIGS = [
    ("baseline",         None),
    ("l=0",              [(0, 0)]),
    ("l=2 m=0",          [(2, 0)]),
    ("l=2 completo",     [(2, 0), (2, 1), (2, -1), (2, 2), (2, -2)]),
    ("l=0 + l=2 m=0",    [(0, 0), (2, 0)]),
    ("l=4 m=0",          [(4, 0)]),
]

OUTPUT_DIR = "test_3D_data_8part_multipole"


# ─────────────────────────────────────────────────────────────────────────────
# Capture of the IPOPT log
# ─────────────────────────────────────────────────────────────────────────────
# `ipopt.out` is NOT read: IPOPT opens it in truncation mode and flushes it to
# disk only when the application is destroyed, so right after solve() it is
# empty. The same lines, however, go to stdout from the C library: they are
# captured at file-descriptor level (redirecting sys.stdout is not enough,
# since the output does not go through Python).

@contextmanager
def capture_stdout_fd():
    """Capture everything written to fd 1 into box["text"], echoing it afterwards."""
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
    """Extract iterations, dependent constraints, DOF flip, number of variables,
    dual infeasibility, complementarity and exit message from an IPOPT log."""
    out = dict(n_iter=-1, n_dep=0, dof_flip=False, n_var=-1,
               dual_inf=float('nan'), compl=float('nan'), exit_msg="")
    m = re.findall(r"Number of Iterations\.*:\s*(\d+)", txt)
    if m:
        out['n_iter'] = int(m[-1])
    m = re.findall(r"Detected\s+(\d+)\s+linearly dependent equality constraints", txt)
    if m:
        out['n_dep'] = sum(int(v) for v in m)
    out['dof_flip'] = ("Too few degrees of freedom" in txt)
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
# References and construction of the case
# ─────────────────────────────────────────────────────────────────────────────
def kinetic_reference(A, orbitals):
    """T = sum_k occ_k (N_k + 3/2) hbar*omega / 2  (HO virial theorem) [MeV]."""
    hw = hbc * nuclear_omega_x(A)
    return sum(o.occupation * (o.N + 1.5) * hw / 2.0 for o in orbitals)


def build_case(grid_kind, A, n_part, cfg):
    """Build mesh/grid, target rho, orbitals and the grid-specific kwargs of iKS_problem.

    Returns ((x, y, z), w3D, rho, orbitals, kwargs)."""
    orbitals = list(getOrbitalSet(n_part, uncoupledHObasis()))

    if grid_kind == "gh":
        N, s = cfg["N_GH"], cfg["MESH_SCALE"]
        mesh = LagrangeMesh3D(N, N, N, s * b_x(A), s * b_y(A), s * b_z(A))
        x, y, z = mesh.x, mesh.y, mesh.z
        w3D = mesh.w3D
        rho = np.zeros((N, N, N))
        for o in orbitals:
            p = HO3D(o.nx, o.ny, o.nz, b_x(A), b_y(A), b_z(A), mesh.dV)(x, y, z)
            p /= np.sqrt(mesh.integrate(p**2))
            rho += o.occupation * p**2
        kwargs = dict(lb=[-2, -2, -2], ub=[2, 2, 2], hx=0.1,
                      use_lagrange_mesh=True, Nx=N, Ny=N, Nz=N, mesh_scale=s)
    else:
        n, L = cfg["N_UNIFORM"], cfg["L_BOX"]
        xs = np.linspace(-L, L, n)
        h  = xs[1] - xs[0]
        x = y = z = xs
        w3D = np.full((n,) * 3, h**3)
        rho = np.zeros((n,) * 3)
        for o in orbitals:
            p = HO3D(o.nx, o.ny, o.nz, b_x(A), b_y(A), b_z(A), h**3)(xs, xs, xs)
            p /= np.sqrt(np.sum(p**2) * h**3)
            rho += o.occupation * p**2
        kwargs = dict(lb=[-L] * 3, ub=[L] * 3, hx=h, hy=h, hz=h,
                      use_lagrange_mesh=False)

    return (x, y, z), w3D, rho, orbitals, kwargs


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def rho_weighted_rmse(dV, rho):
    """(density-weighted RMSE, max absolute value) on the physical region.

    The rho weight is the primary metric of the thesis. A raw maximum over the
    whole domain would land on an edge node where rho*w is tiny and
    lam/(rho w) blows up, returning meaningless numbers.
    """
    mask = np.isfinite(dV) & (rho >= RHO_PHYS_FRAC * rho.max())
    if not np.any(mask):
        return float('nan'), float('nan')
    w = rho[mask]
    d = dV[mask]
    return (float(np.sqrt(np.sum(w * d**2) / np.sum(w))),
            float(np.max(np.abs(d))))


def orbital_overlap(phi_a, phi_b, w3D):
    """Overlap matrix S_ij = int phi_a_i phi_b_j dV and its metrics.

    If the two runs converge to the SAME point, S = +-I. If they converge to
    different points of the same U(N_occ) orbit (same density, mixed
    orbitals), S is a non-trivial unitary matrix: proof that a constraint
    built from rho does not fix that gauge.

    Returns (max off-diagonal |S_ij|, min |S_ii|).
    """
    n = len(phi_a)
    S = np.array([[float(np.sum(w3D * phi_a[i] * phi_b[j])) for j in range(n)]
                  for i in range(n)])
    off = S - np.diag(np.diag(S))
    return float(np.max(np.abs(off))), float(np.min(np.abs(np.diag(S))))


# ─────────────────────────────────────────────────────────────────────────────
# A single run
# ─────────────────────────────────────────────────────────────────────────────
def run_case(label, lm_list, profile, grid_kind, A, n_part, cfg, beta):
    """One IKS run with multipole constraints ``lm_list`` and the given solver profile.

    Returns a dict with IPOPT diagnostics, multipole multipliers mu, and naive
    and gauge-corrected potentials with their errors.
    """
    grid, w3D, rho_target, orbitals, kwargs = build_case(grid_kind, A, n_part, cfg)
    x, y, z = grid
    n_orb = len(orbitals)
    opts = PROFILES[profile]

    tag = (f"Z{n_part}_{profile}_{label}"
           .replace(' ', '_').replace('=', '').replace('+', ''))
    prob = iKS_problem(
        Z=n_part, N=0, data=[x, y, z, rho_target],
        basis=uncoupledHObasis(), reg_parameter=beta,
        output_folder=f"mp_{tag}",
        exact_hess=True, max_iter=500,
        rel_tol=1e-8, constr_viol=1e-8, dual_inf_tol=1.0,
        multipole_constraints=lm_list, **kwargs)

    # Options forced AFTER the constructor: they override those of
    # setSolverOptions, which differ between the two grid branches.
    prob.add_option(b"mu_init", 0.01)
    prob.add_option(b"dependency_detector", opts["dependency_detector"])
    if opts["fixed_variable_treatment"] is not None:
        prob.add_option(b"fixed_variable_treatment", opts["fixed_variable_treatment"])

    t0 = time.time()
    failed = False
    results = None
    with capture_stdout_fd() as cap:
        try:
            results, _info = prob.solve()
        except Exception as exc:                   # MUMPS Error=-3 and the like
            print(f"    !! solve fallito: {type(exc).__name__}: {exc}")
            failed = True
    elapsed = time.time() - t0
    diag = parse_ipopt_text(cap["text"])

    base_out = dict(label=label, profile=profile, lm=prob.multipole_constraints,
                    elapsed=elapsed, n_constr=prob.n_constr, n_live=prob.n_live,
                    n_pairs=len(prob.pairs), n_orb=n_orb,
                    grid=grid, w3D=w3D, rho_target=rho_target, failed=failed,
                    **diag)
    if failed:
        return dict(base_out, status="FAILED", obj=float('nan'),
                    density_err=float('nan'), norm_calc=float('nan'),
                    mu=np.zeros(prob.n_multipole),
                    scales=np.asarray(prob.mp_scales),
                    rmse_w=float('nan'), rmse_w_corr=float('nan'),
                    phi=None, V_naive=None, V_corr=None, V_ana=None,
                    dV_gauge=None, rho_calc=None, center=(0, 0, 0))

    phi = results['u']
    rho_calc = np.sum([orbitals[k].occupation * phi[k]**2 for k in range(n_orb)], axis=0)
    density_err = float(np.sqrt(np.sum(w3D * (rho_calc - rho_target)**2)))
    norm_calc   = float(np.sum(w3D * rho_calc))

    # Multipliers: [density (live)] + [pairs] + [multipoles]
    lag      = np.asarray(results['lagrange'])
    lam_live = lag[:prob.n_live]
    mu       = lag[prob.n_live + len(prob.pairs):]

    lam_full = np.full(prob.n_points, np.nan)
    lam_full[prob.live_nodes] = lam_live
    lam = lam_full.reshape(prob.n_points_x, prob.n_points_y, prob.n_points_z)

    mask = (rho_target >= RHO_FLOOR_FRAC * rho_target.max())
    with np.errstate(divide='ignore', invalid='ignore'):
        V_naive = np.where(mask, lam / (rho_target * w3D), np.nan)

    # Potential in the gauge of the run without constraints (Eq. 6.7)
    dV_gauge = gauge_correction(grid, prob.multipole_constraints, mu, prob.mp_scales)
    V_corr   = V_naive + dV_gauge

    icx, icy, icz = np.unravel_index(np.nanargmax(rho_target), rho_target.shape)
    V_naive = V_naive - V_naive[icx, icy, icz]
    V_corr  = V_corr  - V_corr[icx, icy, icz]

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    V_ana = nuclearNu_x(A) * X**2 + nuclearNu_y(A) * Y**2 + nuclearNu_z(A) * Z**2
    V_ana = V_ana - V_ana[icx, icy, icz]

    rw,  _ = rho_weighted_rmse(V_naive - V_ana, rho_target)
    rwc, _ = rho_weighted_rmse(V_corr  - V_ana, rho_target)

    return dict(base_out, status=results['status'], obj=float(results['obj']),
                density_err=density_err, norm_calc=norm_calc,
                mu=np.asarray(mu), scales=np.asarray(prob.mp_scales),
                rmse_w=rw, rmse_w_corr=rwc, phi=phi,
                V_naive=V_naive, V_corr=V_corr, V_ana=V_ana, dV_gauge=dV_gauge,
                rho_calc=rho_calc, center=(icx, icy, icz))


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def report_profile(res_list, E_ref, profile, Z):
    """Print summary table, multipliers, gauge check (Eq. 6.7) and U(N_occ)
    overlap for one solver profile."""
    base = res_list[0]
    print("\n" + "=" * 110)
    print(f"  Z={Z} — PROFILO '{profile}'   (E_kin analitico = {E_ref:.4f} MeV)")
    print("=" * 110)
    print(f"  {'config':<18} {'m':>6} {'dep':>4} {'n_x':>8} {'iter':>5} {'t[s]':>7} "
          f"{'E_kin':>11} {'dE base':>11} {'|drho|':>10} "
          f"{'RMSEw V':>10} {'RMSEw corr':>11}")
    for r in res_list:
        dE = (r['obj'] - base['obj']) if np.isfinite(r['obj']) else float('nan')
        print(f"  {r['label']:<18} {r['n_constr']:>6} {r['n_dep']:>4} "
              f"{r['n_var']:>8} {r['n_iter']:>5} {r['elapsed']:>7.1f} "
              f"{r['obj']:>11.4f} {dE:>+11.3e} {r['density_err']:>10.2e} "
              f"{r['rmse_w']:>10.3e} {r['rmse_w_corr']:>11.3e}")

    print("\n  Moltiplicatori dei vincoli multipolari:")
    for r in res_list[1:]:
        parts = " ".join(f"mu(l={l},m={m})={v:+.4e}"
                         for (l, m), v in zip(r['lm'], r['mu']))
        print(f"    {r['label']:<18} {parts}")

    if base['failed']:
        print("\n  (baseline fallito: verifiche non eseguibili)")
        return

    print("\n  Verifica della gauge — scarto dal baseline vs predizione")
    print("    (RMSE pesato per rho; 'rapporto' = quanta parte dello scarto e'")
    print("     spiegata dalla correzione -sum_j mu_j R_lm/b^l)")
    icx, icy, icz = base['center']
    for r in res_list[1:]:
        if r['failed']:
            print(f"    {r['label']:<18} run fallito")
            continue
        d_obs  = r['V_naive'] - base['V_naive']
        d_pred = -(r['dV_gauge'] - r['dV_gauge'][icx, icy, icz])
        amp, _ = rho_weighted_rmse(d_obs, base['rho_target'])
        res, _ = rho_weighted_rmse(d_obs - d_pred, base['rho_target'])
        ratio = amp / res if res > 0 else float('inf')
        print(f"    {r['label']:<18} scarto={amp:>10.3e} MeV   "
              f"residuo={res:>10.3e} MeV   rapporto={ratio:>10.1f}")

    if base['n_orb'] > 1:
        print("\n  Gauge unitaria U(N_occ) — sovrapposizione con gli orbitali del baseline")
        print("    (S=+-I → stesso punto; fuori diagonale non nulla → punto diverso")
        print("     della stessa orbita, a densita' identica)")
        for r in res_list[1:]:
            if r['failed']:
                continue
            off, dia = orbital_overlap(base['phi'], r['phi'], base['w3D'])
            print(f"    {r['label']:<18} max|S_ij (i!=j)|={off:>10.3e}   "
                  f"min|S_ii|={dia:>10.6f}")


def make_figure(res_list, fig_path, profile, Z):
    """Three panels: naive potentials, gauge-corrected potentials, and observed
    vs predicted shift from the baseline."""
    base = res_list[0]
    if base['failed']:
        return
    x = base['grid'][0]
    icx, icy, icz = base['center']
    R = physical_radius(base['rho_target'][:, icy, icz], x, frac=1e-3)
    Rlim = 1.15 * R
    ok = [r for r in res_list if not r['failed']]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    fig.suptitle(f"IKS 3D — Z={Z} : vincoli multipolari  [profilo '{profile}']",
                 fontweight="bold", fontsize=13)

    ax = axes[0]
    ax.plot(x, base['V_ana'][:, icy, icz], color="#dc2626", lw=2.2, label="V analitico")
    for r in ok:
        ax.plot(x, r['V_naive'][:, icy, icz], "--", lw=1.4, label=r['label'])
    ax.set_xlim(-Rlim, Rlim)
    yl = potential_ylim(base['V_naive'][:, icy, icz], x, R)
    if yl:
        ax.set_ylim(*yl)
    ax.set_xlabel("x [fm]"); ax.set_ylabel("V [MeV]")
    ax.set_title(r"$V=\lambda/(\rho w)$  (gauge non fissata)")
    ax.legend(fontsize=8); ax.grid(alpha=0.4)

    ax = axes[1]
    ax.plot(x, base['V_ana'][:, icy, icz], color="#dc2626", lw=2.2, label="V analitico")
    for r in ok:
        ax.plot(x, r['V_corr'][:, icy, icz], "--", lw=1.4, label=r['label'])
    ax.set_xlim(-Rlim, Rlim)
    if yl:
        ax.set_ylim(*yl)
    ax.set_xlabel("x [fm]"); ax.set_ylabel("V [MeV]")
    ax.set_title(r"$V+\sum_j \mu_j R_{lm}/b^l$  (gauge del baseline)")
    ax.legend(fontsize=8); ax.grid(alpha=0.4)

    ax = axes[2]
    for r in ok[1:]:
        d_obs  = (r['V_naive'] - base['V_naive'])[:, icy, icz]
        d_pred = -(r['dV_gauge'] - r['dV_gauge'][icx, icy, icz])[:, icy, icz]
        line, = ax.plot(x, d_obs, lw=1.6, label=f"{r['label']}: osservato")
        ax.plot(x, d_pred, ":", lw=2.0, color=line.get_color(),
                label=f"{r['label']}: predetto")
    ax.set_xlim(-Rlim, Rlim)
    ax.set_xlabel("x [fm]"); ax.set_ylabel(r"$\Delta V$ [MeV]")
    ax.set_title("Scarto dal baseline vs gauge predetta")
    ax.legend(fontsize=7); ax.grid(alpha=0.4)

    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → Figura salvata: {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
def run_suite(n_part, cfg, configs, output_dir, argv=None):
    """Run the whole suite for a given number of particles.

    cfg     : dict with N_GH, MESH_SCALE, N_UNIFORM, L_BOX, BETA
    configs : list of (label, list of (l, m) or None); the first is the baseline
    argv    : command-line arguments ('uniform', 'remove', 'keep'); default sys.argv
    """
    argv = [a.lower() for a in (argv if argv is not None else sys.argv[1:])]
    grid_kind = "uniform" if any(a.startswith("u") for a in argv) else "gh"
    profiles = [p for p in ("remove", "keep") if p in argv] or ["remove", "keep"]

    os.makedirs(output_dir, exist_ok=True)
    A = n_part
    orbitals = list(getOrbitalSet(n_part, uncoupledHObasis()))
    E_ref = kinetic_reference(A, orbitals)

    print()
    print("=" * 110)
    print(f"  IKS 3D — Z={n_part} — vincoli multipolari   "
          f"[griglia: {grid_kind}] [profili: {', '.join(profiles)}]")
    print("=" * 110)
    print(f"  orbitali occupati: "
          f"{[(o.nx, o.ny, o.nz) for o in orbitals]}  (n_orb={len(orbitals)})")
    print(f"  beta={cfg['BETA']:.1e}   "
          + (f"N_GH={cfg['N_GH']}, mesh_scale={cfg['MESH_SCALE']}"
             if grid_kind == "gh"
             else f"N={cfg['N_UNIFORM']}, L={cfg['L_BOX']} fm"))
    print("  NB: dependency_detector forzato dal test, uguale sulle due griglie.")

    grid, w3D, rho_target, _, _ = build_case(grid_kind, A, n_part, cfg)
    Q = multipole_moments(rho_target, grid, w3D,
                          [(0, 0), (1, 0), (2, 0), (2, 2), (4, 0)])
    print("\n  Momenti della densita' target (quadratura della griglia):")
    for (l, m), v in Q.items():
        print(f"    Q_{l}{m:+d} = {v:+.6e} fm^{l}")

    for profile in profiles:
        res_list = []
        for label, lm in configs:
            print(f"\n  ── [Z={n_part}|{profile}] run: {label} "
                  + "─" * max(0, 46 - len(label)))
            res_list.append(run_case(label, lm, profile, grid_kind,
                                     A, n_part, cfg, cfg['BETA']))
        report_profile(res_list, E_ref, profile, n_part)
        make_figure(res_list, os.path.join(
            output_dir,
            f"IKS_{n_part}part_multipole_{grid_kind}_{profile}.pdf"),
            profile, n_part)

    print("\n" + "=" * 110)
    print("  Come leggere")
    print("=" * 110)
    print("    dep      : righe dichiarate linearmente dipendenti da IPOPT.")
    print("               Deve valere il numero di vincoli multipolari aggiunti.")
    print("    n_x      : variabili viste da IPOPT. Per Z>=4 deve restare")
    print("               costante fra baseline e run vincolati: se salta a N^3")
    print("               e' scattato il flip a relax_bounds (non atteso qui).")
    print("    dE base  : ~0 → soluzione primale invariata, vincolo senza")
    print("               informazione.")
    print("    rapporto : >> 1 conferma che l'unico effetto e' spostare la gauge.")
    print("               Nel profilo 'remove' mu=0 per costruzione e il")
    print("               rapporto non e' significativo: il test vero e' 'keep'.")
    print("    S_ij     : fuori diagonale non nulla → gli orbitali sono ruotati")
    print("               nella gauge U(N_occ) a parita' di densita'.")
    print()


if __name__ == "__main__":
    run_suite(N_PART, CFG, CONFIGS, OUTPUT_DIR)