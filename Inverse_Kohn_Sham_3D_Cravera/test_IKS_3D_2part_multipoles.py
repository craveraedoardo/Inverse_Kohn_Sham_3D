"""
test_IKS_3D_2part_multipoles.py
===============================
A = 2: comparison between runs WITHOUT and WITH constraints on the multipole
moments. Thesis: Section 6.1 (Tables 6.1 and 6.2, row "A = 2 unif.").

What is measured
----------------
With a POINTWISE density constraint, the multipole constraint

    C_j(f) = sum_{r in live} a_jr * (sum_k occ_k f_kr^2 - 1),
    a_jr   = w_r rho_r R_lm(r) / b^l

is an EXACT linear combination of the constraints already present. The test
verifies that:

  (a) the primal solution (f, hence rho and the objective) does not change;
  (b) the cost changes: dense row in the Jacobian, LICQ violated;
  (c) the multipliers are NOT unique: the potential extracted as
      V = lam/(rho w) is shifted by -sum_j mu_j R_lm(r)/b^l with respect to
      the run without the constraint (Eq. 6.7).

SOLVER PROFILES
---------------
``optimizer3D_lm.setSolverOptions`` sets ``dependency_detector = mumps`` ONLY
on the Lagrange-mesh branch. Consequently, on GH the redundant rows are
REMOVED (then mu = 0 identically, the gauge is never exercised and point (c)
stays untested), whereas on the uniform grid they are kept and mu != 0. The
two branches would not be solving the same numerical problem.

Here the option is forced EXPLICITLY by the test, identically on both grids,
and two profiles are run:

    remove : dependency_detector = mumps  -> dependent rows removed
    keep   : dependency_detector = none   -> rows kept, singular KKT system
                                             regularised internally

The ``keep`` profile is the one in which point (c) can be verified on GH too.
Warning: with ``keep`` MUMPS may fail ("Error=-3"); the test catches the
exception and continues, since the failure is itself a result.

Note on the degrees-of-freedom flip
-----------------------------------
For A = 2 the baseline has n_x = n_c (zero degrees of freedom). Any
additional constraint gives n_c > n_x and IPOPT switches from
fixed_variable_treatment = make_parameter to relax_bounds, reintroducing the
dead nodes as variables with nearly coincident bounds. The DOF check happens
BEFORE the removal of dependent rows, so the flip cannot be avoided at A = 2
(the option is left exposed below to verify this). The clean experiment on
this point requires A >= 4, where n_x >> n_c.

Usage
-----
    python test_IKS_3D_2part_multipoles.py                 # GH, both profiles
    python test_IKS_3D_2part_multipoles.py uniform         # uniform grid
    python test_IKS_3D_2part_multipoles.py gh keep         # keep profile only

Output: test_3D_data_2part_multipole/IKS_2part_multipole_<grid>_<profile>.pdf
NOTE: the IPOPT diagnostics are read from the tail of ipopt.out; see the A = 4
script for a more robust capture of the log.
"""

import sys
import os
import re
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

from consts3D import (hbc, nuclear_omega_x, nuclearNu_x, nuclearNu_y,
                      nuclearNu_z, b_x, b_y, b_z)
from orbitals3D import uncoupledHObasis, getOrbitalSet
from optimizer3D_lm import iKS_problem
from lagrange_mesh_3D import LagrangeMesh3D
from eigenfunctions3D import HO3D
from multipoles3D import multipole_moments, gauge_correction
from plot_helpers import physical_radius, potential_ylim

OUTPUT_DIR = "test_3D_data_2part_multipole"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RHO_FLOOR_FRAC = 1e-10   # mask for the extraction of V (consistent with the optimiser)
RHO_PHYS_FRAC  = 1e-2    # "physical" region for the potential metrics
REG            = 1e-3    # beta
N_GH           = 41      # nodes per axis, Lagrange mesh
MESH_SCALE     = 0.794   # mesh scale s (optimum for A = 2, Section 5.5)
N_UNIFORM      = 41      # points per axis, uniform grid
L_BOX          = 8.0     # half-width of the uniform box [fm]
IPOPT_LOG      = "ipopt.out"

# Constraint configurations. The first one MUST be the baseline (None).
CONFIGS = [
    ("baseline",        None),
    ("l=0",             [(0, 0)]),
    ("l=2 m=0",         [(2, 0)]),
    ("l=2 completo",    [(2, 0), (2, 1), (2, -1), (2, 2), (2, -2)]),
    ("l=0 + l=2 m=0",   [(0, 0), (2, 0)]),
]

# Solver profiles: options imposed EXPLICITLY, identical on both grids.
# fixed_variable_treatment=None -> left untouched (IPOPT decides, which is
# what triggers the flip to relax_bounds when n_c > n_x).
PROFILES = {
    "remove": dict(dependency_detector=b"mumps", fixed_variable_treatment=None),
    "keep":   dict(dependency_detector=b"none",  fixed_variable_treatment=None),
}


# ──────────────────────────────────────────────────────────────────────────────
# References and construction of the case
# ──────────────────────────────────────────────────────────────────────────────
def kinetic_reference(A, orbitals):
    """T = sum_k occ_k (N_k + 3/2) hbar*omega / 2  (HO virial theorem) [MeV]."""
    hw = hbc * nuclear_omega_x(A)
    return sum(o.occupation * (o.N + 1.5) * hw / 2.0 for o in orbitals)


def build_case(grid_kind, A, n_part):
    """Build mesh/grid, target rho and the grid-specific kwargs of iKS_problem.

    Returns ((x, y, z), w3D, rho, orbitals, kwargs)."""
    orbitals = list(getOrbitalSet(n_part, uncoupledHObasis()))

    if grid_kind == "gh":
        mesh = LagrangeMesh3D(N_GH, N_GH, N_GH, MESH_SCALE * b_x(A),
                              MESH_SCALE * b_y(A), MESH_SCALE * b_z(A))
        x, y, z = mesh.x, mesh.y, mesh.z
        w3D = mesh.w3D
        rho = np.zeros((N_GH, N_GH, N_GH))
        for o in orbitals:
            p = HO3D(o.nx, o.ny, o.nz, b_x(A), b_y(A), b_z(A), mesh.dV)(x, y, z)
            p /= np.sqrt(mesh.integrate(p**2))
            rho += o.occupation * p**2
        kwargs = dict(lb=[-2, -2, -2], ub=[2, 2, 2], hx=0.1,
                      use_lagrange_mesh=True, Nx=N_GH, Ny=N_GH, Nz=N_GH,
                      mesh_scale=MESH_SCALE)
    else:
        xs = np.linspace(-L_BOX, L_BOX, N_UNIFORM)
        h  = xs[1] - xs[0]
        x = y = z = xs
        w3D = np.full((N_UNIFORM,) * 3, h**3)
        rho = np.zeros((N_UNIFORM,) * 3)
        for o in orbitals:
            p = HO3D(o.nx, o.ny, o.nz, b_x(A), b_y(A), b_z(A), h**3)(xs, xs, xs)
            p /= np.sqrt(np.sum(p**2) * h**3)
            rho += o.occupation * p**2
        kwargs = dict(lb=[-L_BOX] * 3, ub=[L_BOX] * 3, hx=h, hy=h, hz=h,
                      use_lagrange_mesh=False)

    return (x, y, z), w3D, rho, orbitals, kwargs


# ──────────────────────────────────────────────────────────────────────────────
# Reading the IPOPT log
# ──────────────────────────────────────────────────────────────────────────────
def log_offset():
    """Current size of ipopt.out (0 if missing), to read only the next run's part."""
    try:
        return os.path.getsize(IPOPT_LOG)
    except OSError:
        return 0


def parse_ipopt_log(offset):
    """Diagnostics from the part of ipopt.out written by the last run.

    ``offset`` is the size of the file BEFORE the solve: only the tail is read,
    so that values of previous runs are not collected if IPOPT appends.
    """
    out = dict(n_iter=-1, n_dep=0, dof_flip=False,
               dual_inf=float('nan'), compl=float('nan'))
    try:
        with open(IPOPT_LOG) as fh:
            fh.seek(offset)
            txt = fh.read()
    except OSError:
        return out

    m = re.findall(r"Number of Iterations\.*:\s*(\d+)", txt)
    if m:
        out['n_iter'] = int(m[-1])
    m = re.findall(r"Detected (\d+) linearly dependent equality constraints", txt)
    if m:
        out['n_dep'] = sum(int(v) for v in m)
    out['dof_flip'] = ("Too few degrees of freedom" in txt)
    m = re.findall(r"Dual infeasibility\.*:\s*\S+\s+(\S+)", txt)
    if m:
        out['dual_inf'] = float(m[-1])
    m = re.findall(r"Complementarity\.*:\s*\S+\s+(\S+)", txt)
    if m:
        out['compl'] = float(m[-1])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Potential metrics
# ──────────────────────────────────────────────────────────────────────────────
def rho_weighted_rmse(dV, rho):
    """(density-weighted RMSE, max absolute value) on the physical region.

    Taking the MAXIMUM over the whole domain instead would land on an edge
    node where rho*w is tiny and lam/(rho w) blows up, returning ~1e7 MeV with
    no meaning. The rho weight is the primary metric used in the thesis.
    """
    mask = np.isfinite(dV) & (rho >= RHO_PHYS_FRAC * rho.max())
    if not np.any(mask):
        return float('nan'), float('nan')
    w = rho[mask]
    d = dV[mask]
    return (float(np.sqrt(np.sum(w * d**2) / np.sum(w))),
            float(np.max(np.abs(d))))


# ──────────────────────────────────────────────────────────────────────────────
def run_case(label, lm_list, profile, grid_kind, A, n_part):
    """One IKS run with the multipole constraints ``lm_list`` and the solver profile.

    Returns a dict with the IPOPT diagnostics, the multipole multipliers mu,
    the naive potential lam/(rho w), the gauge-corrected one (Eq. 6.7) and
    their errors with respect to the harmonic reference.
    """
    grid, w3D, rho_target, orbitals, kwargs = build_case(grid_kind, A, n_part)
    x, y, z = grid
    n_orb = len(orbitals)
    opts = PROFILES[profile]

    tag = f"{profile}_{label}".replace(' ', '_').replace('=', '').replace('+', '')
    prob = iKS_problem(
        Z=A, N=0, data=[x, y, z, rho_target],
        basis=uncoupledHObasis(), reg_parameter=REG,
        output_folder=f"mp_{tag}",
        exact_hess=True, max_iter=500,
        rel_tol=1e-8, constr_viol=1e-8, dual_inf_tol=1.0,
        multipole_constraints=lm_list, **kwargs)

    # ── Options forced AFTER the constructor: they override those set in
    #    setSolverOptions, which differ between the two grid branches. ──
    prob.add_option(b"mu_init", 0.01)
    prob.add_option(b"dependency_detector", opts["dependency_detector"])
    if opts["fixed_variable_treatment"] is not None:
        prob.add_option(b"fixed_variable_treatment", opts["fixed_variable_treatment"])

    off = log_offset()
    t0 = time.time()
    try:
        results, info = prob.solve()
        failed = False
    except Exception as exc:                       # MUMPS Error=-3 and the like
        print(f"    !! solve fallito: {type(exc).__name__}: {exc}")
        failed = True
        results = None
    elapsed = time.time() - t0
    diag = parse_ipopt_log(off)

    base_out = dict(label=label, profile=profile, lm=prob.multipole_constraints,
                    elapsed=elapsed, n_constr=prob.n_constr, n_live=prob.n_live,
                    grid=grid, w3D=w3D, rho_target=rho_target, failed=failed,
                    **diag)
    if failed:
        return dict(base_out, status="FAILED", obj=float('nan'),
                    density_err=float('nan'), norm_calc=float('nan'),
                    mu=np.zeros(prob.n_multipole),
                    scales=np.asarray(prob.mp_scales),
                    rmse_line=float('nan'), rmse_line_corr=float('nan'),
                    rmse_w=float('nan'), rmse_w_corr=float('nan'),
                    V_naive=None, V_corr=None, V_ana=None, dV_gauge=None,
                    rho_calc=None, center=(0, 0, 0))

    phi = results['u']
    rho_calc = np.sum([orbitals[k].occupation * phi[k]**2 for k in range(n_orb)], axis=0)
    density_err = float(np.sqrt(np.sum(w3D * (rho_calc - rho_target)**2)))
    norm_calc   = float(np.sum(w3D * rho_calc))

    # ── Multipliers: [density (live)] + [pairs] + [multipoles] ────────────────
    lag      = np.asarray(results['lagrange'])
    n_pairs  = len(prob.pairs)
    lam_live = lag[:prob.n_live]
    mu       = lag[prob.n_live + n_pairs:]

    lam_full = np.full(prob.n_points, np.nan)
    lam_full[prob.live_nodes] = lam_live
    lam = lam_full.reshape(prob.n_points_x, prob.n_points_y, prob.n_points_z)

    mask = (rho_target >= RHO_FLOOR_FRAC * rho_target.max())
    with np.errstate(divide='ignore', invalid='ignore'):
        V_naive = np.where(mask, lam / (rho_target * w3D), np.nan)

    # Potential in the gauge of the run without constraint: V0 = V_naive + sum mu_j R/scale
    dV_gauge = gauge_correction(grid, prob.multipole_constraints, mu, prob.mp_scales)
    V_corr   = V_naive + dV_gauge

    icx, icy, icz = np.unravel_index(np.nanargmax(rho_target), rho_target.shape)
    V_naive = V_naive - V_naive[icx, icy, icz]
    V_corr  = V_corr  - V_corr[icx, icy, icz]

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    V_ana = nuclearNu_x(A) * X**2 + nuclearNu_y(A) * Y**2 + nuclearNu_z(A) * Z**2
    V_ana = V_ana - V_ana[icx, icy, icz]

    def rmse_line(V):
        """RMSE along the x axis through the centre, |x| < 3b."""
        sel = (np.abs(x) < 3 * b_x(A))
        d = (V[:, icy, icz] - V_ana[:, icy, icz])
        v = sel & np.isfinite(d)
        return float(np.sqrt(np.mean(d[v]**2))) if np.any(v) else float('nan')

    rw,  _ = rho_weighted_rmse(V_naive - V_ana, rho_target)
    rwc, _ = rho_weighted_rmse(V_corr  - V_ana, rho_target)

    return dict(base_out, status=results['status'], obj=float(results['obj']),
                density_err=density_err, norm_calc=norm_calc,
                mu=np.asarray(mu), scales=np.asarray(prob.mp_scales),
                rmse_line=rmse_line(V_naive), rmse_line_corr=rmse_line(V_corr),
                rmse_w=rw, rmse_w_corr=rwc,
                V_naive=V_naive, V_corr=V_corr, V_ana=V_ana, dV_gauge=dV_gauge,
                rho_calc=rho_calc, center=(icx, icy, icz))


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────
def report_profile(res_list, E_ref, profile):
    """Print the summary table of a profile and the gauge check of Eq. 6.7
    (observed shift from the baseline vs predicted polynomial shift)."""
    base = res_list[0]
    print("\n" + "=" * 104)
    print(f"  PROFILO '{profile}'   (E_kin analitico = {E_ref:.4f} MeV)")
    print("=" * 104)
    print(f"  {'config':<16} {'m':>6} {'dep':>4} {'DOF':>4} {'iter':>5} {'t[s]':>7} "
          f"{'E_kin':>10} {'dE base':>11} {'compl':>10} "
          f"{'RMSEw V':>10} {'RMSEw corr':>11}")
    for r in res_list:
        dE = (r['obj'] - base['obj']) if np.isfinite(r['obj']) else float('nan')
        print(f"  {r['label']:<16} {r['n_constr']:>6} {r['n_dep']:>4} "
              f"{('SI' if r['dof_flip'] else '--'):>4} {r['n_iter']:>5} "
              f"{r['elapsed']:>7.1f} {r['obj']:>10.4f} {dE:>+11.3e} "
              f"{r['compl']:>10.2e} {r['rmse_w']:>10.3e} {r['rmse_w_corr']:>11.3e}")

    print("\n  Moltiplicatori dei vincoli multipolari:")
    for r in res_list[1:]:
        parts = " ".join(f"mu(l={l},m={m})={v:+.4e}"
                         for (l, m), v in zip(r['lm'], r['mu']))
        print(f"    {r['label']:<16} {parts}")

    if base['failed']:
        print("\n  (baseline fallito: verifica di gauge non eseguibile)")
        return

    print("\n  Verifica della gauge — scarto dal baseline vs predizione")
    print("    (RMSE pesato per rho sulla regione fisica; 'rapporto' = quanta")
    print("     parte dello scarto e' spiegata dalla correzione di gauge)")
    icx, icy, icz = base['center']
    for r in res_list[1:]:
        if r['failed']:
            print(f"    {r['label']:<16} run fallito")
            continue
        d_obs  = r['V_naive'] - base['V_naive']
        d_pred = -(r['dV_gauge'] - r['dV_gauge'][icx, icy, icz])
        amp, _ = rho_weighted_rmse(d_obs, base['rho_target'])
        res, _ = rho_weighted_rmse(d_obs - d_pred, base['rho_target'])
        ratio = amp / res if res > 0 else float('inf')
        print(f"    {r['label']:<16} scarto={amp:>10.3e} MeV   "
              f"residuo={res:>10.3e} MeV   rapporto={ratio:>8.1f}")


def make_figure(res_list, fig_path, profile):
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
    fig.suptitle(f"IKS 3D — Z=2 : vincoli multipolari  [profilo '{profile}']",
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


# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = [a.lower() for a in sys.argv[1:]]
    grid_kind = "uniform" if any(a.startswith("u") for a in args) else "gh"
    profiles = [p for p in ("remove", "keep") if p in args] or ["remove", "keep"]

    A, n_part = 2, 2
    orbitals = list(getOrbitalSet(n_part, uncoupledHObasis()))
    E_ref = kinetic_reference(A, orbitals)

    print()
    print("=" * 104)
    print(f"  IKS 3D — Z=2 — vincoli multipolari   [griglia: {grid_kind}] "
          f"[profili: {', '.join(profiles)}]")
    print("=" * 104)
    print("  NB: dependency_detector e' forzato dal test, uguale sulle due griglie.")
    print("      Il default del codice lo imposta solo sul ramo Lagrange mesh.")

    grid, w3D, rho_target, _, _ = build_case(grid_kind, A, n_part)
    Q = multipole_moments(rho_target, grid, w3D,
                          [(0, 0), (1, 0), (2, 0), (2, 2), (4, 0)])
    print("\n  Momenti della densita' target (quadratura della griglia):")
    for (l, m), v in Q.items():
        print(f"    Q_{l}{m:+d} = {v:+.6e} fm^{l}")

    for profile in profiles:
        res_list = []
        for label, lm in CONFIGS:
            print(f"\n  ── [{profile}] run: {label} " + "─" * max(0, 50 - len(label)))
            res_list.append(run_case(label, lm, profile, grid_kind, A, n_part))
        report_profile(res_list, E_ref, profile)
        make_figure(res_list, os.path.join(
            OUTPUT_DIR, f"IKS_2part_multipole_{grid_kind}_{profile}.pdf"), profile)

    print("\n" + "=" * 104)
    print("  Come leggere")
    print("=" * 104)
    print("    dep      : righe dichiarate linearmente dipendenti da IPOPT.")
    print("               Deve valere esattamente il numero di vincoli aggiunti.")
    print("    DOF      : 'SI' = e' scattato il flip a relax_bounds (n_c > n_x).")
    print("               A Z=2 e' inevitabile appena si aggiunge un vincolo.")
    print("    dE base  : ~0 → la soluzione primale non cambia, il vincolo non")
    print("               porta informazione.")
    print("    rapporto : quanta parte dello scarto dal baseline e' spiegata dalla")
    print("               correzione di gauge. >> 1 conferma che l'unico effetto")
    print("               del vincolo e' spostare la gauge del potenziale.")
    print("    Nel profilo 'remove' mu = 0 per costruzione (righe eliminate):")
    print("    la gauge non e' esercitata e il rapporto non e' significativo.")
    print("    Il test vero del punto (c) e' il profilo 'keep'.")
    print()


if __name__ == "__main__":
    main()