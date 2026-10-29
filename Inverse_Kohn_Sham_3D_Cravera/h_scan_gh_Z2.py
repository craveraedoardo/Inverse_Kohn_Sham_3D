"""
h_scan_gh_Z2.py
===============
Scan of the scale parameter of the Gauss-Hermite Lagrange mesh at fixed N,
for the A = 2 system. Thesis: Section 5.5 (Figs. 5.14 and 5.15, Table 5.5).
It also produces the potential profiles used by reffree_estimator.py
(Section 5.6).

Where efficiency_precision_8part.py scales N at fixed b, this script scales b
at fixed N. Parametrisation: b_mesh = s * b_x(A), with s dimensionless (s = 1
is the historical choice). The physical nodes scale as
x_i = s * b_x(A) * sqrt(2) * t_i, so a small s concentrates the points where
rho > 0 and a large s extends the coverage. The s grid is LOGARITHMIC, so the
ratio r = s_{i+1}/s_i is constant (this simplifies the reference-free
analysis downstream).

DEGENERATE NATURE OF A = 2 — why the scan is informative anyway.
  With a single orbital (000) the constraint 2f^2 = 1 fixes f = 1/sqrt(2)
  everywhere: there is no real optimisation and grad f = 0 (T_beta = 0). No
  coupling between orbitals, no orthonormality constraint. The values of
  N_LIST are the same as those used for A = 8, so that the two scans are
  comparable cell by cell.

Two levels of diagnostics:

  (H0) OFFLINE (needs neither IPOPT nor the test modules) — Fig. 5.14
       For each (N, s):
         - coverage: integral of the target density with the scaled
           quadrature; it should give n_part, and the deviation measures
           whether the mesh "sees" the whole rho (orbitals normalised
           ANALYTICALLY, not with the quadrature, so that the deviation is not
           trivially zero);
         - fraction of dead nodes (rho < 1e-10 rho_max);
         - lost modes = N - rank(diag(sqrt(rho_1D)) D). NOTE: the rank of D
           alone is scale invariant (D ~ 1/b), so the right diagnostic is the
           density-WEIGHTED proxy, which depends on s.

  (H1) ONLINE (needs cyipopt + test_IKS_3D_2part.py) — Fig. 5.15
       For each (N, s): full IKS run -> weighted/uniform RMSE of the potential
       vs the exact harmonic one, wall time, IPOPT outcome, coverage and number
       of live nodes. Points are classified as physical / spurious / failed
       by ``_is_physical`` (thesis Section 5.5: in the large-s arm the RMSE
       collapses to machine precision on a handful of live nodes, which is
       NOT a physical reconstruction).
       Uses test_IKS_3D_2part.run_iks(A, n_part, N, occ, beta, mesh_scale).

Usage
-----
    python h_scan_gh_Z2.py                # full scan (H0 + H1 if IPOPT is available)
    python h_scan_gh_Z2.py --offline      # H0 only (no IPOPT)
    python h_scan_gh_Z2.py --coarse       # reduced s grid (7 points)
    python h_scan_gh_Z2.py --beta 1e-3    # regulariser for H1

Output (h_scan_out_Z2/): H0_offline.{pdf,csv}, H1_rmse_vs_s.{pdf,csv},
V_profiles.npz (input of reffree_estimator.py).
"""

import sys, os, time, re, argparse
import numpy as np
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "src")
for p in (SRC_DIR, HERE):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from lagrange_mesh_3D import derivative_matrix, gauss_hermite_nodes_weights
from consts3D import b_x
from orbitals3D import uncoupledHObasis, getOrbitalSet

# ── Fixed parameters of the A = 2 system ─────────────────────────────────────
Z          = 2                       # mass number (two neutrons)
N_PART     = 2
N_LIST     = [12, 20, 28, 36]        # same as for A = 8 -> comparable scans
GH_TEST    = "test_IKS_3D_2part.py"
OUT_DIR    = "h_scan_out_Z2"
GH_TEST_MOD = "gh2"

C_MAIN = "#2563eb"
C_RANK = "#dc2626"


def s_grid(coarse=False):
    """Logarithmic grid of mesh scales (13 points, or 7 with coarse=True)."""
    n = 7 if coarse else 13
    return np.logspace(-0.6, 0.6, n)   # s in [~0.25, ~4.0], s = 1 included (odd n)


# ──────────────────────────────────────────────────────────────────────────────
# H0 — offline diagnostics
# ──────────────────────────────────────────────────────────────────────────────
def rho_1d_ho(x, orbitals):
    """1D proxy along x: sum of the occupied |h_{nx}(x/b)|^2, normalised to max 1.
    It is used ONLY as a weight for the rank proxy, not as physics: it
    captures where rho ~ 0."""
    from numpy.polynomial.hermite import Hermite
    b = b_x(Z)
    rho = np.zeros_like(x)
    for oo in orbitals:
        h = Hermite.basis(oo.nx)(x / b) * np.exp(-(x / b) ** 2 / 2)
        rho += oo.occupation * h ** 2
    m = rho.max()
    return rho / (m if m > 0 else 1.0)


def numerical_rank(M):
    """Numerical rank from the SVD, with the standard tolerance s_max * max(shape) * eps."""
    sv = np.linalg.svd(M, compute_uv=False)
    if sv.size == 0 or sv[0] == 0.0:
        return 0
    tol = sv[0] * max(M.shape) * np.finfo(float).eps
    return int(np.sum(sv > tol))


def offline_scan(s_vals):
    """H0: coverage, dead-node fraction and lost modes for each (N, s)."""
    from numpy.polynomial.hermite import Hermite
    from math import factorial, sqrt, pi

    orbitals = list(getOrbitalSet(N_PART, uncoupledHObasis()))
    b = b_x(Z)
    rows = []
    for N in N_LIST:
        for s in s_vals:
            b_mesh = s * b
            x, W = gauss_hermite_nodes_weights(N, b_mesh)

            # (a) 3D coverage with ANALYTIC normalisation of the orbitals:
            # ||H_n(x/b) e^{-x^2/2b^2}||^2 = b sqrt(pi) 2^n n!  -> the deviation
            # of sum(w rho) from n_part measures the fidelity of the quadrature.
            w3 = W[:, None, None] * W[None, :, None] * W[None, None, :]

            def phi1d(n, xv):
                h = Hermite.basis(n)(xv / b) * np.exp(-(xv / b) ** 2 / 2)
                return h / sqrt(b * sqrt(pi) * (2.0 ** n) * factorial(n))

            rho3 = np.zeros((N, N, N))
            for oo in orbitals:
                phi = (phi1d(oo.nx, x)[:, None, None]
                       * phi1d(oo.ny, x)[None, :, None]
                       * phi1d(oo.nz, x)[None, None, :])
                rho3 += oo.occupation * phi ** 2
            coverage = float(np.sum(w3 * rho3))
            dead = float(np.mean(rho3 < 1e-10 * rho3.max()))

            # (b) density-weighted 1D rank proxy
            D = derivative_matrix(x)
            r1 = rho_1d_ho(x, orbitals)
            Mw = np.sqrt(np.maximum(r1, 0.0))[:, None] * D
            rank_w = numerical_rank(Mw)

            rows.append(dict(N=N, s=s, coverage=coverage,
                             cov_err=abs(coverage - N_PART) / N_PART,
                             dead_frac=dead,
                             rank_w=rank_w, modes_lost=N - rank_w))
            print(f"  [H0] N={N:2d} s={s:6.3f}  cov={coverage:9.5f} "
                  f"dead={dead:6.3f}  rank(sqrt(rho)D)={rank_w:2d}/{N}")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# H1 — online scan (IPOPT)
# ──────────────────────────────────────────────────────────────────────────────
def _load(name, filename):
    """Import a script of this folder as a module (without running its main)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def has_ipopt():
    """True if cyipopt can be imported."""
    return importlib.util.find_spec("cyipopt") is not None


def _read_ipopt_iterations(path="ipopt.out"):
    """Number of iterations from ipopt.out (NaN if unavailable; see the caveat
    in efficiency_precision_8part.py)."""
    try:
        with open(path, "r", errors="ignore") as f:
            txt = f.read()
    except OSError:
        return float("nan")
    m = re.findall(r"Number of Iterations\.*:\s*(\d+)", txt)
    return int(m[-1]) if m else float("nan")


def online_scan(s_vals, beta=1e-3, save_profiles=None):
    """H1: full IKS run for each (N, s); returns one row of metrics per run.

    save_profiles: if a list, the potential profiles V(x) along the x axis of
    every successful run are appended to it, for the downstream
    reference-free analysis (reffree_estimator.py).
    """
    from plot_helpers import potential_error
    mod = _load(GH_TEST_MOD, GH_TEST)
    # A = 2: run_iks(A, n_part, N, occ, beta, mesh_scale); occ is unused by run_iks.
    rows = []
    for N in N_LIST:
        for s in s_vals:
            print(f"  [H1] N={N} s={s:.3f} ...", flush=True)
            try:
                if os.path.exists("ipopt.out"):
                    os.remove("ipopt.out")
            except OSError:
                pass
            t0 = time.perf_counter()
            row = dict(N=N, s=s, beta=beta)
            try:
                occ = getOrbitalSet(N_PART, uncoupledHObasis())
                run = mod.run_iks(Z, N_PART, N, occ, beta, mesh_scale=s)
                x = run["grid"][0] if "grid" in run else run["xs"]
                met = potential_error(run.get("V_iks"),
                                      run.get("V_ana", run.get("V_ho")),
                                      run.get("rho_target"), x, frac=1e-2)
                # Effective coverage and live nodes: they distinguish a physical
                # reconstruction from an artefact (few central nodes where
                # f = 1/sqrt2 is exact by construction -> RMSE ~ eps, no physics).
                rho_t = run.get("rho_target")
                mesh = run.get("mesh")
                if rho_t is not None and mesh is not None:
                    coverage = float(mesh.integrate(rho_t))
                    n_live = int(np.sum(rho_t >= 1e-10 * rho_t.max()))
                else:
                    coverage, n_live = float("nan"), -1
                row.update(rmse_w=met["rmse_pesato"],
                           rmse_u=met["rmse_uniforme"],
                           coverage=coverage, n_live=n_live,
                           seconds=run.get("elapsed",
                                           time.perf_counter() - t0),
                           iters=_read_ipopt_iterations(),
                           failed=0.0)
                # Save the profile V(x) along the axis for the reference-free
                # analysis (see reffree_estimator.py). It is later interpolated
                # onto a COMMON physical grid there: at different s the nodes
                # sit at different positions, so the V(s) cannot be compared
                # element by element without that step.
                if save_profiles is not None:
                    try:
                        V = run.get("V_iks")
                        ic = len(x) // 2
                        Vx = np.asarray(V)[:, ic, ic]
                        save_profiles.append(dict(N=N, s=s, x=np.asarray(x),
                                                  Vx=Vx))
                    except Exception:
                        pass
            except Exception as e:
                row.update(rmse_w=np.nan, rmse_u=np.nan,
                           coverage=float("nan"), n_live=-1,
                           seconds=time.perf_counter() - t0,
                           iters=np.nan, failed=1.0)
                print(f"      FALLITO: {type(e).__name__}: {e}")
            rows.append(row)
    return rows


# Thresholds of the physical-reliability criterion (see figure_online):
COV_TOL   = 5e-3    # acceptable |coverage - n_part|/n_part (coverage lost at large s)
MIN_LIVE  = 64      # absolute minimum number of live nodes to represent V
MIN_FRAC  = 3e-3    # and at least this fraction of N^3 (scales with the grid)
RMSE_FLOOR = 1e-10  # below this the RMSE is machine noise: an empty
                    # reconstruction (few central nodes where f = 1/sqrt2 is
                    # exact by construction), not physics. A true KS potential is
                    # never accurate at the eps level on a finite mesh.


def _is_physical(row):
    """A converged point is PHYSICALLY reliable only if:
      (1) the quadrature still covers the density (coverage ~ n_part), which
          excludes the collapse at large s where the nodes leave the support;
      (2) enough live nodes remain, in absolute terms AND as a fraction of
          N^3, to represent the potential; AND
      (3) the RMSE is not at machine-noise level — an error ~eps is not a
          success but the signature of the A = 2 degeneracy, where on a few
          central nodes f = 1/sqrt2 is exact BY CONSTRUCTION and nothing is
          being reconstructed. This catches the artefact even when (for small
          N) coverage and live nodes have not collapsed yet.
    The criterion discards noise from below rather than rewarding it. IPOPT
    failures (including out-of-memory) are excluded upstream via row['failed']."""
    if row.get("failed", 1.0) > 0:
        return False
    cov = row.get("coverage", float("nan"))
    nl  = row.get("n_live", -1)
    N   = row.get("N", 0)
    rmse = row.get("rmse_w", float("nan"))
    if np.isfinite(rmse) and rmse < RMSE_FLOOR:
        return False   # machine noise: empty reconstruction
    if not np.isfinite(cov):
        return True    # missing data: do not filter further
    cov_ok  = abs(cov - N_PART) / N_PART <= COV_TOL
    live_ok = (nl >= MIN_LIVE) and (nl >= MIN_FRAC * N ** 3)
    return cov_ok and live_ok


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────
def _csv(path, rows, keys):
    """Save the selected keys of a list of dict rows to a CSV file."""
    arr = np.array([[float(r[k]) for k in keys] for r in rows])
    np.savetxt(path, arr, delimiter=",", header=",".join(keys), comments="")
    print(f"  -> {path}")


def figure_offline(rows, path):
    """H0 figure: coverage error, dead-node fraction and lost modes vs s (thesis Fig. 5.14)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    fig.suptitle("Z=2: diagnostica offline del mesh GH vs scala s "
                 "(b_mesh = s b_x(A))", fontweight="bold")
    for N in N_LIST:
        rr = [r for r in rows if r["N"] == N]
        ss = [r["s"] for r in rr]
        axes[0].semilogx(ss, [r["cov_err"] for r in rr], "o-", label=f"N={N}")
        axes[1].semilogx(ss, [r["dead_frac"] for r in rr], "o-", label=f"N={N}")
        axes[2].semilogx(ss, [r["modes_lost"] for r in rr], "o-", label=f"N={N}")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("|integr. rho - n_part| / n_part")
    axes[0].set_title("Copertura della densita'")
    axes[1].set_ylabel("frazione nodi morti")
    axes[1].set_title("Nodi con rho < 1e-10 rho_max")
    axes[2].set_ylabel("modi persi  N - rank(sqrt(rho) D)")
    axes[2].set_title("Proxy 1D di rank-loss")
    for ax in axes:
        ax.axvline(1.0, color="#9ca3af", ls=":", lw=1.2)
        ax.set_xlabel("s = b_mesh / b_x(A)")
        ax.grid(alpha=0.4, which="both"); ax.legend(fontsize=8)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {path}")


def figure_online(rows, path, off_rows=None):
    """H1 figure (thesis Fig. 5.15), three panels:
      (0) error valley, with PHYSICAL points (filled) distinguished from
          converged-but-spurious points (open, semi-transparent) and failures
          (x); the physically reliable window of s is shaded;
      (1) live nodes vs s (log): why the RMSE at large s is spurious (few
          nodes) and why at small s IPOPT runs out of memory (many nodes);
      (2) three-level status map: physical / spurious / failed.
    ``off_rows`` is accepted but not used.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    fig.suptitle("Z=2: valle fisica del potenziale vs scala s "
                 "(punti vuoti = convergiuti ma non fisici)", fontweight="bold")
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(N_LIST)))

    # --- panel 0: RMSE, physical vs spurious ---
    phys_s_all = []
    for N, c in zip(N_LIST, colors):
        rr = sorted([r for r in rows if r["N"] == N], key=lambda r: r["s"])
        ss = np.array([r["s"] for r in rr])
        ee = np.array([r["rmse_w"] for r in rr])
        ok = np.array([r["failed"] for r in rr]) == 0
        phys = np.array([_is_physical(r) for r in rr])
        good = ok & phys            # converged AND physical
        spur = ok & ~phys           # converged but spurious
        fail = ~ok                  # failed
        # line joining ONLY the physical points
        axes[0].loglog(ss[good], ee[good] + 1e-14, "-", color=c, lw=1.6)
        axes[0].loglog(ss[good], ee[good] + 1e-14, "o", color=c,
                       ms=6, label=f"N={N}")
        axes[0].loglog(ss[spur], ee[spur] + 1e-14, "o", mfc="none",
                       mec=c, ms=6, alpha=0.55)
        if fail.any():
            axes[0].plot(ss[fail], np.full(fail.sum(), 3e-13), "x",
                         ms=8, mew=2, color="#dc2626")
        phys_s_all.extend(ss[good].tolist())

    if phys_s_all:
        lo, hi = min(phys_s_all), max(phys_s_all)
        axes[0].axvspan(lo, hi, color="#16a34a", alpha=0.08,
                        label="finestra fisica")
    axes[0].axvline(1.0, color="#9ca3af", ls=":", lw=1.2)
    axes[0].set_xlabel("s = b_mesh / b_x(A)")
    axes[0].set_ylabel("RMSE V_KS (pesato rho) [MeV]")
    axes[0].set_title("Valle dell'errore — solo i pieni sono fisici")
    axes[0].grid(alpha=0.4, which="both"); axes[0].legend(fontsize=7, ncol=2)

    # --- panel 1: live nodes vs s ---
    for N, c in zip(N_LIST, colors):
        rr = sorted([r for r in rows if r["N"] == N], key=lambda r: r["s"])
        ss = np.array([r["s"] for r in rr])
        nl = np.array([r.get("n_live", np.nan) for r in rr], dtype=float)
        axes[1].loglog(ss, nl, "o-", color=c, ms=5, label=f"N={N}")
    axes[1].axhline(MIN_LIVE, color="#dc2626", ls="--", lw=1.2,
                    label=f"soglia {MIN_LIVE} nodi")
    axes[1].axvline(1.0, color="#9ca3af", ls=":", lw=1.2)
    axes[1].set_xlabel("s"); axes[1].set_ylabel("nodi vivi (rho > 1e-10 rho_max)")
    axes[1].set_title("Perche' i minimi a s grande sono vuoti")
    axes[1].grid(alpha=0.4, which="both"); axes[1].legend(fontsize=7)

    # --- panel 2: three-level status map ---
    # 0 = physical (green), 1 = spurious (yellow), 2 = failed (red)
    svals = sorted({r["s"] for r in rows})
    M = np.full((len(N_LIST), len(svals)), np.nan)
    for i, N in enumerate(N_LIST):
        for j, s in enumerate(svals):
            m = [r for r in rows if r["N"] == N and r["s"] == s]
            if not m:
                continue
            r = m[0]
            M[i, j] = 2 if r["failed"] > 0 else (0 if _is_physical(r) else 1)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#16a34a", "#eab308", "#dc2626"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    im = axes[2].imshow(M, aspect="auto", cmap=cmap, norm=norm,
                        extent=[np.log10(svals[0]), np.log10(svals[-1]),
                                -0.5, len(N_LIST) - 0.5], origin="lower")
    axes[2].set_yticks(range(len(N_LIST)), [f"N={N}" for N in N_LIST])
    axes[2].set_xlabel("log10 s"); axes[2].set_title("Stato del run")
    cb = fig.colorbar(im, ax=axes[2], ticks=[0, 1, 2])
    cb.ax.set_yticklabels(["fisico", "spurio", "fallito"])
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {path}")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Scan della scala del mesh GH per Z=2")
    ap.add_argument("--offline", action="store_true",
                    help="solo diagnostica H0 (niente IPOPT)")
    ap.add_argument("--coarse", action="store_true", help="griglia s a 7 punti")
    ap.add_argument("--beta", type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    s_vals = s_grid(args.coarse)

    print(f"\n=== h-scan GH: Z={Z}, N={N_LIST}, "
          f"s in [{s_vals[0]:.3f}, {s_vals[-1]:.3f}] ({len(s_vals)} punti) ===")
    print("    (sistema degenere: f=1/sqrt2 fisso, T_beta=0 — vedi docstring)\n")

    print("[H0] diagnostica offline ...")
    off = offline_scan(s_vals)
    figure_offline(off, os.path.join(OUT_DIR, "H0_offline.pdf"))
    _csv(os.path.join(OUT_DIR, "H0_offline.csv"), off,
         ["N", "s", "coverage", "cov_err", "dead_frac", "rank_w", "modes_lost"])

    if args.offline:
        print("\n(--offline) Fatto.\n"); return
    if not has_ipopt():
        print("\ncyipopt non disponibile: H1 saltato.\n"); return

    print(f"\n[H1] scan online (beta={args.beta:g}) ...")
    profiles = []
    on = online_scan(s_vals, beta=args.beta, save_profiles=profiles)
    if profiles:
        np.savez(os.path.join(OUT_DIR, "V_profiles.npz"),
                 **{f"run{i}_{k}": v for i, p in enumerate(profiles)
                    for k, v in [("N", p["N"]), ("s", p["s"]),
                                 ("x", p["x"]), ("Vx", p["Vx"])]})
        print(f"  -> {os.path.join(OUT_DIR, 'V_profiles.npz')} "
              f"({len(profiles)} profili, per reffree_estimator.py)")
    figure_online(on, os.path.join(OUT_DIR, "H1_rmse_vs_s.pdf"), off_rows=off)
    _csv(os.path.join(OUT_DIR, "H1_rmse_vs_s.csv"), on,
         ["N", "s", "beta", "rmse_w", "rmse_u", "coverage", "n_live",
          "seconds", "iters", "failed"])
    # Summary of the physical window
    phys = [r for r in on if _is_physical(r)]
    if phys:
        best = min(phys, key=lambda r: r["rmse_w"])
        ss_ok = sorted({r["s"] for r in phys})
        print(f"\n  Finestra fisica s ∈ [{ss_ok[0]:.3f}, {ss_ok[-1]:.3f}]")
        print(f"  Minimo RMSE fisico: {best['rmse_w']:.3e} MeV "
              f"a s={best['s']:.3f}, N={best['N']}")
    print("\nFatto.\n")


if __name__ == "__main__":
    main()