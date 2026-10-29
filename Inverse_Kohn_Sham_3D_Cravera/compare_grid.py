"""
compare_grid.py
===============
Comparison of the two discretisations of the 3D IKS problem: Gauss-Hermite
Lagrange mesh vs uniform grid with finite differences. The SAME optimiser
(optimizer3D_lm.py) is used for both, so the grid is the only difference.
Thesis: Section 5.2 (Figs. 5.5 and 5.6).

For A = 2 and A = 4 it produces
  * convergence of the physical kinetic energy T_alpha vs the TOTAL number of
    points (N^3 vs n^3), the true measure of computational cost, and vs the
    number of points PER AXIS at a fixed uniform box (Fig. 5.5);
  * (if IPOPT is available) density and potential profiles along x from full
    runs, and the convergence of the potential error vs the number of points,
    including the comparison at equal N per axis (Fig. 5.6).

The convergence SCANS do not need IPOPT (they evaluate the functional on the
exact target density) and always run. The FULL runs need cyipopt; if it is
not available the script skips them and says so.

The functions of the four validation scripts (test_IKS_3D_{2,4}part[_uniform].py)
are loaded as modules with importlib, so their parameters (e.g. mesh scale,
regularisation) are those defined there.

Usage
-----
    python compare_grid.py              # scans + full runs (if IPOPT is available)
    python compare_grid.py --scan-only

Output: compare_grids_out/{confronto_convergenza, confronto_run_completi,
        confronto_convergenza_potenziale}.pdf
"""

import sys, os
import numpy as np
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the modules in src/ and the test scripts importable.
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from orbitals3D import uncoupledHObasis, getOrbitalSet
from plot_helpers import physical_radius, potential_ylim, potential_error

OUTPUT_DIR = "compare_grids_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Consistent colours for the two grids
C_GH = "#2563eb"     # Gauss-Hermite (blue)
C_UN = "#ea580c"     # uniform (orange)

# Tikhonov regularisation weight (beta), common to all scans/runs.
BETA = 1e-3

# Density threshold of the physical mask, CONSISTENT with the test files
# (RHO_FLOOR_FRAC = 1e-10 in test_IKS_3D_4part.py): nodes with rho < 1e-10 max
# are outside the physical support and V = lambda/(rho w) is undetermined there.
RHO_FLOOR_FRAC = 1e-10


def _load(name, filename):
    """Import a script of this folder as a module (without running its main)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def has_ipopt():
    """True if cyipopt can be imported."""
    return importlib.util.find_spec("cyipopt") is not None


# ──────────────────────────────────────────────────────────────────────────────
# Data collection: convergence scans for both grids
# ──────────────────────────────────────────────────────────────────────────────
def collect_scans():
    """Return a dict with the GH and uniform convergence scans for A = 2 and A = 4.

    Each point is stored as (total points, T_alpha, err_%), so that GH and
    uniform can share the same "number of points" axis. The loaded modules
    are returned under 'mods' for later use.
    """
    gh2 = _load("gh2", "test_IKS_3D_2part.py")
    un2 = _load("un2", "test_IKS_3D_2part_uniform.py")
    gh4 = _load("gh4", "test_IKS_3D_4part.py")
    un4 = _load("un4", "test_IKS_3D_4part_uniform.py")

    occ2 = list(getOrbitalSet(2, uncoupledHObasis()))[0].occupation

    # Z=2
    N_list_gh = [8, 12, 16, 20, 24]
    grids_un  = [(21, 6.0), (31, 7.0), (41, 8.0), (51, 8.0), (61, 9.0)]
    Eg2, gh2_rows = gh2.convergence_scan(2, 2, occ2, N_list_gh)
    Eu2, un2_rows = un2.convergence_scan_uniform(2, 2, occ2, grids_un, BETA)
    # gh rows: (N, T_alpha, err%, live, tot) ; un rows: (n, L, h, T_alpha, err%)
    gh2_pts = [(r[0]**3, r[1], r[2]) for r in gh2_rows]
    un2_pts = [(r[0]**3, r[3], r[4]) for r in un2_rows]

    # Z=4
    N_list_gh4 = [8, 12, 16, 20, 24]
    grids_un4  = [(21, 6.5), (31, 7.5), (41, 8.5), (51, 9.0), (61, 9.5)]
    Eg4, gh4_rows = gh4.convergence_scan(4, 4, N_list_gh4, BETA)
    Eu4, un4_rows = un4.convergence_scan_uniform(4, 4, grids_un4, BETA)
    # gh4 rows: (N, T_alpha, T_beta, err%, live, tot) ; un4 rows: (n, L, h, T_alpha, T_beta, err%)
    gh4_pts = [(r[0]**3, r[1], r[3]) for r in gh4_rows]
    un4_pts = [(r[0]**3, r[3], r[5]) for r in un4_rows]

    return dict(
        E2=Eg2, E4=Eg4,
        gh2=gh2_pts, un2=un2_pts, gh4=gh4_pts, un4=un4_pts,
        mods=(gh2, un2, gh4, un4),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Full runs (require IPOPT)
# ──────────────────────────────────────────────────────────────────────────────
def collect_full_runs(mods):
    """One full IKS run per grid and system, for the profile figure."""
    gh2, un2, gh4, un4 = mods
    occ2 = list(getOrbitalSet(2, uncoupledHObasis()))[0].occupation
    runs = {}
    # A = 2: GH at N = 24, uniform at n = 41
    runs['gh2'] = gh2.run_iks(2, 2, 24, occ2)
    runs['un2'] = un2.run_iks_uniform(2, 2, 41, 8.0, BETA)
    # A = 4: GH at N = 24, uniform at n = 51
    runs['gh4'] = gh4.run_iks(4, 4, 24, BETA)
    runs['un4'] = un4.run_iks_uniform(4, 4, 51, 9.0, BETA)
    return runs


# ──────────────────────────────────────────────────────────────────────────────
# Scan of the POTENTIAL error vs number of points (requires IPOPT)
# ──────────────────────────────────────────────────────────────────────────────
# Unlike the kinetic energy, the potential V = lambda/(rho w) exists only AFTER
# an optimisation run (the Lagrange multipliers are needed). Here a full IKS
# run is launched for each grid size and the robust RMSE of V_KS is measured
# against the exact harmonic reference (the target density is built from pure
# HO orbitals), with the same metric for both grids
# (plot_helpers.potential_error).
def collect_potential_scans(mods,
                            N_list_gh=(16, 20, 24, 28),
                            grids_un2=((31, 7.0), (41, 8.0), (51, 8.0), (61, 9.0)),
                            grids_un4=((31, 7.5), (41, 8.5), (51, 9.0), (61, 9.5))):
    """Return {A: dict(gh=[(points, rmse_u, rmse_w)], un=[...])} from full IKS runs."""
    gh2, un2, gh4, un4 = mods
    occ2 = list(getOrbitalSet(2, uncoupledHObasis()))[0].occupation
    out = {2: dict(gh=[], un=[]), 4: dict(gh=[], un=[])}

    # A = 2 — Gauss-Hermite
    for N in N_list_gh:
        r = gh2.run_iks(2, 2, N, occ2)
        m = potential_error_metrics(r)
        out[2]['gh'].append((N**3, m['rmse_uniforme'], m['rmse_pesato']))
    # A = 2 — uniform
    for (n, L) in grids_un2:
        r = un2.run_iks_uniform(2, 2, n, L, BETA)
        m = potential_error_metrics(r)
        out[2]['un'].append((n**3, m['rmse_uniforme'], m['rmse_pesato']))

    # A = 4 — Gauss-Hermite
    for N in N_list_gh:
        r = gh4.run_iks(4, 4, N, BETA)
        m = potential_error_metrics(r)
        out[4]['gh'].append((N**3, m['rmse_uniforme'], m['rmse_pesato']))
    # A = 4 — uniform
    for (n, L) in grids_un4:
        r = un4.run_iks_uniform(4, 4, n, L, BETA)
        m = potential_error_metrics(r)
        out[4]['un'].append((n**3, m['rmse_uniforme'], m['rmse_pesato']))

    return out


def collect_potential_matched_axis(mods, N_common=(16, 20, 24, 28),
                                   L2_FIXED=6.0, L4_FIXED=7.5):
    """Potential error at EQUAL N per axis (same N on both grids), so that the
    two curves can be overlaid on the same axis — as in the third row of the
    kinetic figure (thesis Fig. 5.6).

    FIXED uniform box (only h changes with N): a growing L would bias the
    comparison by giving the uniform grid an ever coarser spacing. Requires
    one IKS run per N and grid. Returns {A: (N_common, gh_w, un_w, gh_u, un_u)},
    with _w = density-weighted RMSE (primary) and _u = uniform RMSE (secondary).
    """
    gh2, un2, gh4, un4 = mods
    occ2 = list(getOrbitalSet(2, uncoupledHObasis()))[0].occupation
    out = {}

    for Z, (run_gh, run_un, Lfix) in [
            (2, (lambda N: gh2.run_iks(2, 2, N, occ2),
                 lambda N: un2.run_iks_uniform(2, 2, N, L2_FIXED, BETA), L2_FIXED)),
            (4, (lambda N: gh4.run_iks(4, 4, N, BETA),
                 lambda N: un4.run_iks_uniform(4, 4, N, L4_FIXED, BETA), L4_FIXED))]:
        gh_w, un_w, gh_u, un_u = [], [], [], []
        for N in N_common:
            mg = potential_error_metrics(run_gh(N))
            mu = potential_error_metrics(run_un(N))
            gh_w.append(mg['rmse_pesato']);   un_w.append(mu['rmse_pesato'])
            gh_u.append(mg['rmse_uniforme']); un_u.append(mu['rmse_uniforme'])
        out[Z] = (list(N_common), gh_w, un_w, gh_u, un_u)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Extraction of 1D profiles along x (handles the different result structures)
# ──────────────────────────────────────────────────────────────────────────────
def profile_x(run, key):
    """Profile along x through the centre for 'rho_target', 'rho_calc', 'V_iks', 'V_ho'/'V_ana'."""
    if 'grid' in run:                      # GH
        x = run['grid'][0]
        icx, icy, icz = run.get('center', (run['mesh'].Nx//2,)*3)
    else:                                  # uniform
        x = run['xs']; ic = run['ic']; icx = icy = icz = ic
    arr = run.get(key)
    if arr is None:
        # alternative names V_ana / V_ho (A = 2 vs A = 4 scripts)
        arr = run.get('V_ana') if key == 'V_ho' else run.get('V_ho')
    return x, arr[:, icy, icz]


def _grid_axis_and_center(run):
    """x axis and centre indices, for GH (grid/center) or uniform (xs/ic) results."""
    if 'grid' in run:
        x = run['grid'][0]
        icx, icy, icz = run.get('center', (run['mesh'].Nx // 2,) * 3)
    else:
        x = run['xs']; ic = run['ic']; icx = icy = icz = ic
    return x, (icx, icy, icz)


def potential_fields(run):
    """Return (x, center, V_iks, V_ref, rho) as 3D arrays from a full run.

    The reference is the exact harmonic potential: V_ana for A = 2, V_ho for
    A = 4 (same construction m omega^2 r^2 / 2). For densities from pure HO
    orbitals the V_KS that inverts the density IS the harmonic one, so the
    comparison is exact. Both are already anchored to the same gauge as V_iks
    (set to zero at the density maximum in the respective modules).
    """
    x, center = _grid_axis_and_center(run)
    V_iks = run.get('V_iks')
    V_ref = run.get('V_ana', run.get('V_ho'))
    rho   = run.get('rho_target')
    return x, center, V_iks, V_ref, rho


def potential_error_metrics(run):
    """Robust error metrics of V_KS (uniform/weighted RMSE, max) on the physical
    region, computed identically for GH and uniform via
    plot_helpers.potential_error, so that the comparison is fair."""
    x, _, V_iks, V_ref, rho = potential_fields(run)
    return potential_error(V_iks, V_ref, rho, x, frac=1e-2)


# ──────────────────────────────────────────────────────────────────────────────
# Comparison figures
# ──────────────────────────────────────────────────────────────────────────────
def collect_matched_axis(mods):
    """Kinetic scan at common N (same points per axis) for a direct GH vs uniform
    comparison (thesis Fig. 5.5). The uniform box is fixed for all N.
    Returns {A: (N_common, err_gh, err_un)}.
    """
    gh2, un2, gh4, un4 = mods
    occ2 = list(getOrbitalSet(2, uncoupledHObasis()))[0].occupation
    N_common = [16, 20, 24, 28, 32]

    # FIXED uniform box (the same for every N): only the spacing h changes.
    # Sized to contain the density without cutting it (rho(L)/rho_max <~ 1e-8).
    # An L growing with N would bias the comparison, giving the uniform grid an
    # ever coarser h (see the caveat on box sizing in thesis Section 5.2).
    L2_FIXED = 6.0      # A = 2
    L4_FIXED = 7.5      # A = 4: the (100) orbital is wider

    out = {}
    # A = 2
    Eg, gh_rows = gh2.convergence_scan(2, 2, occ2, N_common)
    grids = [(N, L2_FIXED) for N in N_common]
    Eu, un_rows = un2.convergence_scan_uniform(2, 2, occ2, grids, BETA)
    out[2] = (N_common, [abs(r[2]) for r in gh_rows], [abs(r[4]) for r in un_rows])
    # A = 4
    Eg4, gh_rows4 = gh4.convergence_scan(4, 4, N_common, BETA)
    grids4 = [(N, L4_FIXED) for N in N_common]
    Eu4, un_rows4 = un4.convergence_scan_uniform(4, 4, grids4, BETA)
    out[4] = (N_common, [abs(r[3]) for r in gh_rows4], [abs(r[5]) for r in un_rows4])
    return out


def figure_convergence(data, matched, fig_path):
    """3x2 kinetic-energy figure: T_alpha and its error vs total points, and the
    error vs points per axis at a fixed box (columns: A = 2, A = 4)."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 14), constrained_layout=True)
    fig.suptitle("Confronto discretizzazioni IKS 3D — Gauss-Hermite vs Uniforme",
                 fontweight="bold", fontsize=13)

    for col, (Z, E, gh, un) in enumerate([
            (2, data['E2'], data['gh2'], data['un2']),
            (4, data['E4'], data['gh4'], data['un4'])]):
        ghp = np.array(gh); unp = np.array(un)

        # row 0: T_alpha vs total points
        ax = axes[0, col]
        ax.axhline(E, color="#6b7280", lw=1.5, ls="--",
                   label=f"E_kin analitico = {E:.1f}")
        ax.plot(ghp[:, 0], ghp[:, 1], "o-", color=C_GH, label="Gauss-Hermite")
        ax.plot(unp[:, 0], unp[:, 1], "s-", color=C_UN, label="Uniforme")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("punti totali (N³)"); ax.set_ylabel("T$_α$ [MeV]")
        ax.set_title(f"Z={Z}: cinetica vs punti totali")
        ax.legend(fontsize=8); ax.grid(alpha=0.4, which="both")

        # row 1: relative error vs total points (computational cost)
        ax = axes[1, col]
        ax.loglog(ghp[:, 0], np.abs(ghp[:, 2]) + 1e-14, "o-", color=C_GH, label="Gauss-Hermite")
        ax.loglog(unp[:, 0], np.abs(unp[:, 2]) + 1e-14, "s-", color=C_UN, label="Uniforme")
        ax.set_xlabel("punti totali (N³)"); ax.set_ylabel("|errore T$_α$| [%]")
        ax.set_title(f"Z={Z}: errore vs punti totali (costo)")
        ax.legend(fontsize=8); ax.grid(alpha=0.4, which="both")

        # row 2: error vs points PER AXIS, at the SAME N (direct comparison)
        ax = axes[2, col]
        Nc, eg, eu = matched[Z]
        ax.semilogy(Nc, np.array(eg) + 1e-14, "o-", color=C_GH, label="Gauss-Hermite")
        ax.semilogy(Nc, np.array(eu) + 1e-14, "s-", color=C_UN, label="Uniforme")
        ax.set_xlabel("punti per asse (N, stessi per entrambe)")
        ax.set_ylabel("|errore T$_α$| [%]")
        ax.set_title(f"Z={Z}: confronto a parità di N/asse")
        ax.legend(fontsize=8); ax.grid(alpha=0.4, which="both")

    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fig_path}")


def figure_full_runs(runs, fig_path):
    """3x2 figure from the full runs: density, potential and potential error
    along x (columns: A = 2, A = 4), with an inset of the RMSE metrics."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 15), constrained_layout=True)
    fig.suptitle("Confronto run completi — densità, potenziale V$_{KS}$(x) ed errore",
                 fontweight="bold", fontsize=13)

    for col, (Z, gh_key, un_key) in enumerate([(2, 'gh2', 'un2'), (4, 'gh4', 'un4')]):
        rg = runs[gh_key]; ru = runs[un_key]

        # Common physical radius (from the GH target density): beyond it V = lambda/(rho w) diverges.
        xg0, rho_g0 = profile_x(rg, 'rho_target')
        R = physical_radius(rho_g0, xg0, frac=1e-3)
        Rlim = 1.15 * R

        # density
        ax = axes[0, col]
        xg, rho_g = profile_x(rg, 'rho_target')
        _,  rhoc_g = profile_x(rg, 'rho_calc')
        xu, rho_u = profile_x(ru, 'rho_target')
        _,  rhoc_u = profile_x(ru, 'rho_calc')
        ax.plot(xg, rho_g, color="#6b7280", lw=2, label="ρ target (GH)")
        ax.plot(xg, rhoc_g, "--", color=C_GH, lw=1.4, label="ρ IKS (GH)")
        ax.plot(xu, rhoc_u, ":", color=C_UN, lw=1.8, label="ρ IKS (Unif)")
        ax.set_xlabel("x [fm]"); ax.set_ylabel("ρ [fm⁻³]")
        ax.set_title(f"Z={Z}: densità lungo x")
        ax.legend(fontsize=8); ax.grid(alpha=0.4); ax.set_xlim(-Rlim, Rlim)

        # potential (physical region)
        ax = axes[1, col]
        xg, Vg = profile_x(rg, 'V_iks')
        xu, Vu = profile_x(ru, 'V_iks')
        _, Vho = profile_x(rg, 'V_ho')
        # Physical mask (same as the error row and the metric): beyond the region
        # rho >= 1e-2 max, V = lambda/(rho w) is poorly determined and diverges. It
        # is excluded from the plot, not only from the y scale, otherwise the
        # tails dominate (see A = 4).
        _, rho_gxV = profile_x(rg, 'rho_target')
        _, rho_uxV = profile_x(ru, 'rho_target')
        mgV = rho_gxV >= 1e-2 * np.nanmax(rho_gxV)
        muV = rho_uxV >= 1e-2 * np.nanmax(rho_uxV)
        Vg_m = np.where(mgV, Vg, np.nan)
        Vu_m = np.where(muV, Vu, np.nan)
        RcoreV = max(float(np.max(np.abs(xg[mgV]))) if mgV.any() else Rlim,
                     float(np.max(np.abs(xu[muV]))) if muV.any() else Rlim)
        ax.plot(xg, Vho, color="#6b7280", lw=2,
                label="½mω²r²")
        ax.plot(xg, Vg_m, "--", color=C_GH, lw=1.5, label="V$_{KS}$ (GH)")
        ax.plot(xu, Vu_m, ":", color=C_UN, lw=2.0, label="V$_{KS}$ (Unif)")
        ax.set_xlabel("x [fm]"); ax.set_ylabel("V [MeV]")
        ax.set_title(f"Z={Z}: potenziale lungo x (regione fisica)")
        ax.legend(fontsize=8); ax.grid(alpha=0.4); ax.set_xlim(-RcoreV, RcoreV)
        # robust y scale based on both curves in the inner region
        yl_g = potential_ylim(Vg, xg, R)
        yl_u = potential_ylim(Vu, xu, R)
        cand = [v for v in (yl_g, yl_u) if v is not None]
        if cand:
            lo = min(c[0] for c in cand); hi = max(c[1] for c in cand)
            ax.set_ylim(lo, hi)

        # ── potential error ──────────────────────────────────────────────────
        # (a) profile of the signed error V_KS - V_ref along x;
        # (b) bars of the robust RMSE metrics (uniform and density-weighted).
        # Reference: exact harmonic potential (V_ana for A = 2, V_ho for A = 4).
        _, Vref_g = profile_x(rg, 'V_ho')   # falls back to V_ana for A = 2
        _, Vref_u = profile_x(ru, 'V_ho')
        dVg = Vg - Vref_g
        dVu = Vu - Vref_u

        # Physical mask CONSISTENT with the metric and the test files: keep only
        # rho >= frac*max (frac = 1e-2, as in potential_error). It is the "strong"
        # mask: it excludes the whole region where V = lambda/(rho w) is poorly
        # determined, with no ad hoc geometric cut.
        _, rho_gx = profile_x(rg, 'rho_target')
        _, rho_ux = profile_x(ru, 'rho_target')
        FRAC = 1e-2
        mg = rho_gx >= FRAC * np.nanmax(rho_gx)
        mu = rho_ux >= FRAC * np.nanmax(rho_ux)
        dVg_m = np.where(mg, dVg, np.nan)
        dVu_m = np.where(mu, dVu, np.nan)
        # x extent of the physical region (for the axis limits)
        xphys = np.concatenate([np.abs(xg[mg]), np.abs(xu[mu])])
        Rcore = float(np.max(xphys)) if xphys.size else Rlim

        ax = axes[2, col]
        ax.axhline(0.0, color="#9ca3af", lw=1.0, ls="-")
        ax.plot(xg, dVg_m, "--", color=C_GH, lw=1.6, label="ΔV (GH)")
        ax.plot(xu, dVu_m, ":", color=C_UN, lw=2.0, label="ΔV (Unif)")
        ax.set_xlabel("x [fm]"); ax.set_ylabel("V$_{KS}$ − V$_{ref}$ [MeV]")
        ref_lbl = "esatto"  # density from pure HO orbitals => V_KS = exact harmonic
        ax.set_title(f"Z={Z}: errore potenziale lungo x (rif. {ref_lbl})")
        ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.4)
        ax.set_xlim(-Rcore, Rcore)
        # robust y scale based ONLY on the masked physical region
        core = np.concatenate([dVg_m[np.isfinite(dVg_m)],
                               dVu_m[np.isfinite(dVu_m)]])
        if core.size:
            m = max(float(np.percentile(np.abs(core), 98)), 1e-3)
            ax.set_ylim(-1.3 * m, 1.3 * m)

        # inset with the robust RMSE metrics (uniform vs density-weighted)
        eg = potential_error_metrics(rg)
        eu = potential_error_metrics(ru)
        axin = ax.inset_axes([0.66, 0.56, 0.32, 0.34])
        metr = ['rmse\nunif.', 'rmse\npesato']
        gh_vals = [eg['rmse_uniforme'], eg['rmse_pesato']]
        un_vals = [eu['rmse_uniforme'], eu['rmse_pesato']]
        xb = np.arange(len(metr)); bw = 0.38
        axin.bar(xb - bw/2, gh_vals, bw, color=C_GH, label="GH")
        axin.bar(xb + bw/2, un_vals, bw, color=C_UN, label="Unif")
        axin.set_xticks(xb); axin.set_xticklabels(metr, fontsize=6)
        axin.tick_params(axis='y', labelsize=6)
        axin.set_ylabel("MeV", fontsize=6)
        axin.set_title("RMSE V$_{KS}$", fontsize=7)
        axin.legend(fontsize=5, loc="upper right")
        axin.grid(alpha=0.3, axis='y')

    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fig_path}")


def figure_potential_convergence(pot_scans, matched, fig_path):
    """Error of the potential V_KS vs number of points, GH vs uniform, A = 2 and 4.

    Rows:
      0 = density-weighted RMSE vs total points (cost) — primary metric;
      1 = uniform RMSE vs total points (secondary, sensitive to tail nodes);
      2 = comparison at EQUAL N per axis: both curves on the SAME abscissa
          (as in the third row of the kinetic figure), directly overlaid
          (thesis Fig. 5.6). Exact harmonic reference for both.
    """
    fig, axes = plt.subplots(3, 2, figsize=(13, 14), constrained_layout=True)
    fig.suptitle("Errore del potenziale V$_{KS}$ — Gauss-Hermite vs Uniforme",
                 fontweight="bold", fontsize=13)

    for col, Z in enumerate((2, 4)):
        gh = np.array(pot_scans[Z]['gh'])   # (points, rmse_u, rmse_w)
        un = np.array(pot_scans[Z]['un'])
        ref_lbl = "esatto"  # density from pure HO orbitals => V_KS = exact harmonic

        # row 0: density-weighted RMSE (primary metric, robust and monotonic)
        ax = axes[0, col]
        ax.loglog(gh[:, 0], gh[:, 2] + 1e-14, "o-", color=C_GH, label="Gauss-Hermite")
        ax.loglog(un[:, 0], un[:, 2] + 1e-14, "s-", color=C_UN, label="Uniforme")
        ax.set_xlabel("punti totali (N³)")
        ax.set_ylabel("RMSE V$_{KS}$ (pesato ρ) [MeV]")
        ax.set_title(f"Z={Z}: RMSE pesato per ρ vs punti totali (rif. {ref_lbl})")
        ax.legend(fontsize=8); ax.grid(alpha=0.4, which="both")

        # row 1: uniform RMSE (secondary: it also weighs the tail nodes where
        # V = lambda/(rho w) is poorly determined -> may be non-monotonic)
        ax = axes[1, col]
        ax.loglog(gh[:, 0], gh[:, 1] + 1e-14, "o-", color=C_GH, label="Gauss-Hermite")
        ax.loglog(un[:, 0], un[:, 1] + 1e-14, "s-", color=C_UN, label="Uniforme")
        ax.set_xlabel("punti totali (N³)")
        ax.set_ylabel("RMSE V$_{KS}$ (unif.) [MeV]")
        ax.set_title(f"Z={Z}: RMSE uniforme vs punti totali (rif. {ref_lbl})")
        ax.legend(fontsize=8); ax.grid(alpha=0.4, which="both")

        # row 2: comparison at equal N per axis — overlaid curves (weighted RMSE).
        ax = axes[2, col]
        Nc, gh_w, un_w, gh_u, un_u = matched[Z]
        Nc = np.array(Nc)
        ax.semilogy(Nc, np.array(gh_w) + 1e-14, "o-", color=C_GH, label="Gauss-Hermite")
        ax.semilogy(Nc, np.array(un_w) + 1e-14, "s-", color=C_UN, label="Uniforme")
        ax.set_xlabel("punti per asse (N, stessi per entrambe)")
        ax.set_ylabel("RMSE V$_{KS}$ (pesato ρ) [MeV]")
        ax.set_title(f"Z={Z}: confronto a parità di N/asse (rif. {ref_lbl})")
        ax.legend(fontsize=8); ax.grid(alpha=0.4, which="both")

    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fig_path}")


# ──────────────────────────────────────────────────────────────────────────────
def print_summary_table(data):
    """Print the kinetic convergence table (error % on T_alpha) for both grids."""
    print("\n  Tabella di confronto convergenza cinetica (errore % su T_α):")
    for Z, gh, un in [(2, data['gh2'], data['un2']), (4, data['gh4'], data['un4'])]:
        print(f"\n  Z={Z}:")
        print(f"    {'Gauss-Hermite':>28}   |   {'Uniforme':>28}")
        print(f"    {'punti':>10} {'T_α':>9} {'err%':>7}   |   "
              f"{'punti':>10} {'T_α':>9} {'err%':>7}")
        for (g, u) in zip(gh, un):
            print(f"    {g[0]:>10} {g[1]:>9.2f} {g[2]:>6.2f}%   |   "
                  f"{u[0]:>10} {u[1]:>9.2f} {u[2]:>6.2f}%")


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   CONFRONTO GRIGLIE — Gauss-Hermite vs Uniforme  (IKS 3D)           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    print("\n  Raccolta scan di convergenza (non richiede IPOPT)…")
    data = collect_scans()
    matched = collect_matched_axis(data['mods'])
    print_summary_table(data)

    print("\n  Generazione figure…")
    figure_convergence(data, matched, os.path.join(OUTPUT_DIR, "confronto_convergenza.pdf"))

    scan_only = "--scan-only" in sys.argv
    if scan_only:
        print("\n  (--scan-only) Salto i run completi.")
    elif not has_ipopt():
        print("\n  ⚠ cyipopt non disponibile: salto i run completi (densità/potenziale).")
        print("    Lancia in un ambiente con IPOPT per i pannelli di densità e V_KS.")
    else:
        print("\n  Run completi IKS (GH e uniforme, Z=2 e Z=4)…")
        runs = collect_full_runs(data['mods'])
        figure_full_runs(runs, os.path.join(OUTPUT_DIR, "confronto_run_completi.pdf"))
        # summary table of the runs
        print("\n  Run completi — riassunto:")
        print(f"    {'caso':>10} {'T_α [MeV]':>12} {'Δρ L2':>10} "
              f"{'V RMSE unif':>12} {'V RMSE pes.':>12}")
        for key, lbl in [('gh2','GH Z=2'),('un2','Un Z=2'),
                         ('gh4','GH Z=4'),('un4','Un Z=4')]:
            r = runs[key]
            Ta = r.get('T_alpha', r.get('obj', float('nan')))
            # Potential metrics computed IDENTICALLY for all runs
            # (plot_helpers.potential_error), so the comparison between grids is
            # fair and independent of how each module stored its own pot_err.
            em = potential_error_metrics(r)
            print(f"    {lbl:>10} {Ta:>12.3f} {r['density_err']:>10.2e} "
                  f"{em['rmse_uniforme']:>12.2e} {em['rmse_pesato']:>12.2e}")

        # Convergence of the potential error vs points (separate figure).
        # Requires one IKS run per size -> much more expensive than the kinetic scans.
        print("\n  Scan di convergenza del potenziale (un run IKS per taglia)…")
        pot_scans = collect_potential_scans(data['mods'])
        print("  Scan del potenziale a parità di N/asse (per sovrapporre le curve)…")
        pot_matched = collect_potential_matched_axis(data['mods'])
        figure_potential_convergence(
            pot_scans, pot_matched,
            os.path.join(OUTPUT_DIR, "confronto_convergenza_potenziale.pdf"))

    print("\n  Fatto.\n")


if __name__ == "__main__":
    main()