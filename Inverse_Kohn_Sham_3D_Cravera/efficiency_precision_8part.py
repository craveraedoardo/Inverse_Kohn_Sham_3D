"""
efficiency_precision_8part.py
=============================
NUMERICAL EFFICIENCY vs PRECISION of the potential V_KS for the two
discretisations of the 3D IKS problem — Gauss-Hermite Lagrange mesh and
uniform grid (findiff) — for A = 8. Thesis: Section 5.2 (Fig. 5.7 and the
accompanying table of weighted/unweighted RMSE vs N).

A = 8 is the second-level closed-shell test: (000) + the complete p triplet
(100), (010), (001). The filled N = 1 shell makes the density SPHERICAL, so
the V_KS that inverts it IS rigorously the isotropic oscillator
m omega^2 r^2 / 2: the RMSE of the potential against it is an exact metric, as
for A = 2 but with 4 orbitals and a genuine optimisation.

Four tests (the four figures/CSV files produced):

  (T1) Accuracy vs degrees of freedom
       RMSE(V_KS) — density-weighted (primary) and uniform (secondary) — as a
       function of N (points per axis) and N_tot = N^3, for GH and uniform.
       The central comparison: at equal N (same size of the optimisation
       problem), which scheme gives the more accurate V.

  (T2) Accuracy vs real COMPUTATIONAL cost
       Same RMSE with abscissa = wall time of the solve (field 'elapsed'
       returned by the runs) and = IPOPT iterations (parsed from ipopt.out).
       GH has DENSE derivative matrices, the uniform grid BANDED ones, so the
       cost per degree of freedom differs and N is not a faithful proxy.

  (T3) Condition number vs N
       cond_2 of the 1D derivative matrix (Dx for GH; 4th-order FD stencil for
       the uniform grid, as findiff acc=4) as a function of N. NOTE: during
       the thesis work cond_2 turned out to be a misleading indicator (it is
       anti-correlated with the actual solver performance); the number of
       modes lost by the sqrt(rho)-weighted D, computed in h_scan_gh_Z2.py, is
       the more informative diagnostic. T3 is not used in the thesis.

  (T4) Cost <-> precision Pareto front
       Scatter (wall time, weighted RMSE) over an N x beta scan for BOTH
       methods, with the Pareto front highlighted.

The Lagrange-mesh runs use test_IKS_3D_8part.run_iks with its default mesh
scale s = 1.

Usage
-----
    python efficiency_precision_8part.py               # all tests
    python efficiency_precision_8part.py --cond-only   # only T3 (no IPOPT needed)
    python efficiency_precision_8part.py --no-pareto   # skip T4

Output: efficiency_precision_8part_out/{T1,T2,T3,T4}_*.pdf and *.csv
"""

import sys, os, time, re
import numpy as np
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Path setup — identical to the test files, so the modules resolve the same way
# ──────────────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "src")
for p in (SRC_DIR, HERE):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from plot_helpers import potential_error
from lagrange_mesh_3D import derivative_matrix, gauss_hermite_nodes_weights
from consts3D import b_x

OUTPUT_DIR = "efficiency_precision_8part_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

C_GH = "#2563eb"   # Gauss-Hermite (blue)
C_UN = "#ea580c"   # uniform (orange)

Z = 8              # closed shell (N = 0 + complete N = 1); used as the particle number
N_PART = 8

# beta common to both methods (both run functions accept it as an argument).
BETA = 1e-3

# Scan in N (points per axis). FIXED uniform box (only h changes with N), so as
# not to bias the comparison. A = 8 needs somewhat larger N than A = 2 (4
# orbitals, three with a node along different axes -> finer structure).
N_LIST = [18, 20, 24]
L8_FIXED = 8.0     # A = 8: b ~ 1.42 fm -> contains the three p orbitals without edge cuts

# beta scan for the Pareto front (both methods).
BETA_LIST = [1e-2, 1e-3, 1e-4]
N_PARETO = [20, 24, 28, 32]


# ──────────────────────────────────────────────────────────────────────────────
# Loading the test modules
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


def load_test_modules():
    """Load the A = 8 test modules (GH and uniform). Adapt here if the file names change."""
    gh = _load("gh8", "test_IKS_3D_8part.py")
    un = _load("un8", "test_IKS_3D_8part_uniform.py")
    return gh, un


# ──────────────────────────────────────────────────────────────────────────────
# Potential metrics — EXACT reuse of the logic of compare_grid.py
# ──────────────────────────────────────────────────────────────────────────────
def _grid_axis(run):
    """x axis of a run (GH: run['grid'][0]; uniform: run['xs'])."""
    return run['grid'][0] if 'grid' in run else run['xs']


def potential_error_metrics(run):
    """Robust RMSE of V_KS vs the exact harmonic potential, identical for GH and
    uniform. frac = 1e-2 as in the test files; returns rmse_uniforme,
    rmse_pesato, max_abs (see plot_helpers.potential_error)."""
    x     = _grid_axis(run)
    V_iks = run.get('V_iks')
    V_ref = run.get('V_ana', run.get('V_ho'))
    rho   = run.get('rho_target')
    return potential_error(V_iks, V_ref, rho, x, frac=1e-2)


# ──────────────────────────────────────────────────────────────────────────────
# Cost: wall time (from the run's 'elapsed' field) + IPOPT iterations (from ipopt.out)
# ──────────────────────────────────────────────────────────────────────────────
def _read_ipopt_iterations(path="ipopt.out"):
    """Iterations from the last block of ipopt.out (always written by setSolverOptions).

    CAVEAT: IPOPT may flush ipopt.out only when the problem object is
    destroyed, so right after solve() the file can be empty or stale (see the
    note in test_IKS_3D_4part_multipoles.py, which captures stdout instead).
    In that case NaN is returned.
    """
    try:
        with open(path, "r", errors="ignore") as f:
            txt = f.read()
    except OSError:
        return float("nan")
    m = re.findall(r"Number of Iterations\.*:\s*(\d+)", txt)
    return int(m[-1]) if m else float("nan")


def run_and_measure(run_callable, ipopt_out="ipopt.out"):
    """Run run_callable() and return (run, cost).

    The wall time comes from the run's 'elapsed' field (measured around
    solve() in the test files, so it excludes the construction of the target
    and imports); an external timer is the fallback. The iterations are parsed
    from ipopt.out, which is deleted before the run so that only THIS run is
    parsed. cost = dict(seconds, iters, n_tot).
    """
    try:
        if os.path.exists(ipopt_out):
            os.remove(ipopt_out)
    except OSError:
        pass

    t0 = time.perf_counter()
    run = run_callable()
    wall_ext = time.perf_counter() - t0

    seconds = run.get('elapsed', wall_ext) if isinstance(run, dict) else wall_ext
    iters = _read_ipopt_iterations(ipopt_out)
    x = _grid_axis(run) if isinstance(run, dict) else None
    n_tot = int(len(x)) ** 3 if x is not None else float("nan")
    return run, dict(seconds=seconds, iters=iters, n_tot=n_tot)


# ──────────────────────────────────────────────────────────────────────────────
# Runners: (method, N, beta) -> callable. Actual signatures in the A = 8 test files:
#   GH:      run_iks(A=8, n_part=8, N, beta)            [beta is the 4th argument]
#   uniform: run_iks_uniform(A=8, n_part=8, n, L, beta)
# ──────────────────────────────────────────────────────────────────────────────
def run_gh(mods, N, beta=BETA):
    """Callable running the Lagrange-mesh A = 8 problem at N nodes per axis."""
    gh, un = mods
    return lambda: gh.run_iks(Z, N_PART, N, beta)


def run_un(mods, N, beta=BETA):
    """Callable running the uniform-grid A = 8 problem with N points on the fixed box."""
    gh, un = mods
    return lambda: un.run_iks_uniform(Z, N_PART, N, L8_FIXED, beta)


# ──────────────────────────────────────────────────────────────────────────────
# T3 — Conditioning of the derivative matrices (does NOT require IPOPT)
# ──────────────────────────────────────────────────────────────────────────────
def fd_derivative_matrix_uniform(n, h):
    """1D first-derivative matrix on a uniform grid, 4th-order stencil (centred in
    the interior, one-sided at the edges) — consistent with FinDiff(acc=4) used
    by the uniform solver (before antisymmetrisation). For the conditioning
    what matters is the banded structure, not the box size."""
    D = np.zeros((n, n))
    for i in range(2, n - 2):
        D[i, i-2] =  1.0 / (12*h)
        D[i, i-1] = -8.0 / (12*h)
        D[i, i+1] =  8.0 / (12*h)
        D[i, i+2] = -1.0 / (12*h)
    fwd = np.array([-25, 48, -36, 16, -3], dtype=float) / (12*h)
    D[0, 0:5] = fwd
    D[1, 0:5] = np.array([-3, -10, 18, -6, 1], dtype=float) / (12*h)
    D[n-1, n-5:n] = -fwd[::-1]
    D[n-2, n-5:n] = -np.array([1, -6, 18, -10, -3], dtype=float) / (12*h)
    return D


def condition_scan(N_list):
    """cond_2 of the 1D D vs N, GH (Gauss-Hermite nodes, as LagrangeMesh3D.Dx) and uniform."""
    bx = b_x(Z)   # A = 8: oscillator length scaling the GH nodes (s = 1)
    cond_gh, cond_un = [], []
    for N in N_list:
        xg, _ = gauss_hermite_nodes_weights(N, bx)
        cond_gh.append(float(np.linalg.cond(derivative_matrix(xg))))
        h = 2.0 * L8_FIXED / (N - 1)
        cond_un.append(float(np.linalg.cond(fd_derivative_matrix_uniform(N, h))))
    return dict(N=list(N_list), cond_gh=cond_gh, cond_un=cond_un)


def figure_condition(cond, fig_path):
    """T3 figure: cond_2(D) vs N for both schemes."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    N = np.array(cond['N'])
    ax.semilogy(N, cond['cond_gh'], "o-", color=C_GH, label="Gauss-Hermite (D densa)")
    ax.semilogy(N, cond['cond_un'], "s-", color=C_UN, label="Uniforme (FD bandata, acc=4)")
    ax.axhline(1.0/np.finfo(float).eps, color="#9ca3af", ls=":", lw=1.2,
               label="1/ε_macchina (≈4.5e15)")
    ax.set_xlabel("punti per asse N")
    ax.set_ylabel("numero di condizionamento cond₂(D)")
    ax.set_title("Condizionamento della matrice derivata 1D vs N (Z=8)")
    ax.legend(fontsize=9); ax.grid(alpha=0.4, which="both")
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fig_path}")


# ──────────────────────────────────────────────────────────────────────────────
# T1 + T2 — Accuracy vs N and vs cost (requires IPOPT)
# ──────────────────────────────────────────────────────────────────────────────
def collect_accuracy_cost(mods, N_list):
    """Run both schemes for each N; collect RMSE (weighted/uniform), time and iterations."""
    r = dict(N=[], n_tot_gh=[], n_tot_un=[],
             rmse_w_gh=[], rmse_u_gh=[], rmse_w_un=[], rmse_u_un=[],
             sec_gh=[], sec_un=[], it_gh=[], it_un=[])
    for N in N_list:
        print(f"    N={N} …", flush=True)
        run_g, cost_g = run_and_measure(run_gh(mods, N))
        run_u, cost_u = run_and_measure(run_un(mods, N))
        mg = potential_error_metrics(run_g)
        mu = potential_error_metrics(run_u)
        r['N'].append(N)
        r['n_tot_gh'].append(cost_g['n_tot']); r['n_tot_un'].append(cost_u['n_tot'])
        r['rmse_w_gh'].append(mg['rmse_pesato']);  r['rmse_u_gh'].append(mg['rmse_uniforme'])
        r['rmse_w_un'].append(mu['rmse_pesato']);  r['rmse_u_un'].append(mu['rmse_uniforme'])
        r['sec_gh'].append(cost_g['seconds']); r['sec_un'].append(cost_u['seconds'])
        r['it_gh'].append(cost_g['iters']);    r['it_un'].append(cost_u['iters'])
    return r


def _save_csv(path, header, columns):
    """Save equal-length columns to a CSV file with a header line."""
    arr = np.column_stack([np.asarray(c, dtype=float) for c in columns])
    np.savetxt(path, arr, delimiter=",", header=",".join(header), comments="")
    print(f"  → {path}")


def figure_accuracy_vs_N(acc, fig_path):
    """T1 figure: weighted and uniform RMSE vs N per axis and vs N^3 (thesis Fig. 5.7)."""
    N = np.array(acc['N'])
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle("Z=8: accuratezza del potenziale vs gradi di libertà (rif. armonico esatto)",
                 fontweight="bold", fontsize=13)
    ax = axes[0, 0]
    ax.semilogy(N, np.array(acc['rmse_w_gh'])+1e-14, "o-", color=C_GH, label="Gauss-Hermite")
    ax.semilogy(N, np.array(acc['rmse_w_un'])+1e-14, "s-", color=C_UN, label="Uniforme")
    ax.set_xlabel("punti per asse N"); ax.set_ylabel("RMSE V$_{KS}$ (pesato ρ) [MeV]")
    ax.set_title("RMSE pesato vs N/asse"); ax.legend(); ax.grid(alpha=0.4, which="both")
    ax = axes[0, 1]
    ax.semilogy(N, np.array(acc['rmse_u_gh'])+1e-14, "o-", color=C_GH, label="Gauss-Hermite")
    ax.semilogy(N, np.array(acc['rmse_u_un'])+1e-14, "s-", color=C_UN, label="Uniforme")
    ax.set_xlabel("punti per asse N"); ax.set_ylabel("RMSE V$_{KS}$ (unif.) [MeV]")
    ax.set_title("RMSE uniforme vs N/asse"); ax.legend(); ax.grid(alpha=0.4, which="both")
    ax = axes[1, 0]
    ax.loglog(acc['n_tot_gh'], np.array(acc['rmse_w_gh'])+1e-14, "o-", color=C_GH, label="Gauss-Hermite")
    ax.loglog(acc['n_tot_un'], np.array(acc['rmse_w_un'])+1e-14, "s-", color=C_UN, label="Uniforme")
    ax.set_xlabel("punti totali N³"); ax.set_ylabel("RMSE V$_{KS}$ (pesato ρ) [MeV]")
    ax.set_title("RMSE pesato vs punti totali"); ax.legend(); ax.grid(alpha=0.4, which="both")
    ax = axes[1, 1]
    ax.loglog(acc['n_tot_gh'], np.array(acc['rmse_u_gh'])+1e-14, "o-", color=C_GH, label="Gauss-Hermite")
    ax.loglog(acc['n_tot_un'], np.array(acc['rmse_u_un'])+1e-14, "s-", color=C_UN, label="Uniforme")
    ax.set_xlabel("punti totali N³"); ax.set_ylabel("RMSE V$_{KS}$ (unif.) [MeV]")
    ax.set_title("RMSE uniforme vs punti totali"); ax.legend(); ax.grid(alpha=0.4, which="both")
    fig.savefig(fig_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  → {fig_path}")


def figure_accuracy_vs_cost(acc, fig_path):
    """T2 figure: weighted RMSE vs wall time and vs IPOPT iterations."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle("Z=8: accuratezza vs costo computazionale reale",
                 fontweight="bold", fontsize=13)
    ax = axes[0]
    ax.loglog(acc['sec_gh'], np.array(acc['rmse_w_gh'])+1e-14, "o-", color=C_GH, label="Gauss-Hermite")
    ax.loglog(acc['sec_un'], np.array(acc['rmse_w_un'])+1e-14, "s-", color=C_UN, label="Uniforme")
    ax.set_xlabel("wall-time soluzione [s]"); ax.set_ylabel("RMSE V$_{KS}$ (pesato ρ) [MeV]")
    ax.set_title("Accuratezza vs tempo"); ax.legend(); ax.grid(alpha=0.4, which="both")
    for xi, yi, Ni in zip(acc['sec_gh'], acc['rmse_w_gh'], acc['N']):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(f"{Ni}", (xi, yi), fontsize=7, color=C_GH,
                        xytext=(3, 3), textcoords="offset points")
    for xi, yi, Ni in zip(acc['sec_un'], acc['rmse_w_un'], acc['N']):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(f"{Ni}", (xi, yi), fontsize=7, color=C_UN,
                        xytext=(3, -9), textcoords="offset points")
    ax = axes[1]
    it_g = np.array(acc['it_gh'], dtype=float); it_u = np.array(acc['it_un'], dtype=float)
    if np.isfinite(it_g).any() or np.isfinite(it_u).any():
        ax.semilogy(it_g, np.array(acc['rmse_w_gh'])+1e-14, "o-", color=C_GH, label="Gauss-Hermite")
        ax.semilogy(it_u, np.array(acc['rmse_w_un'])+1e-14, "s-", color=C_UN, label="Uniforme")
        ax.set_xlabel("iterazioni IPOPT"); ax.set_ylabel("RMSE V$_{KS}$ (pesato ρ) [MeV]")
        ax.set_title("Accuratezza vs iterazioni"); ax.legend(); ax.grid(alpha=0.4, which="both")
    else:
        ax.text(0.5, 0.5, "iterazioni IPOPT non disponibili\n(ipopt.out non trovato)",
                ha="center", va="center", transform=ax.transAxes); ax.set_axis_off()
    fig.savefig(fig_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  → {fig_path}")


# ──────────────────────────────────────────────────────────────────────────────
# T4 — Cost <-> precision Pareto front
# ──────────────────────────────────────────────────────────────────────────────
def _pareto_front(cost, err):
    """Indices of the Pareto-optimal points (lower cost and lower error)."""
    cost = np.asarray(cost, float); err = np.asarray(err, float)
    ok = np.isfinite(cost) & np.isfinite(err)
    order = np.where(ok)[0][np.argsort(cost[np.where(ok)[0]])]
    front, best = [], np.inf
    for i in order:
        if err[i] < best - 1e-15:
            front.append(i); best = err[i]
    return front


def collect_pareto(mods, N_list, beta_list):
    """N x beta grid of runs for BOTH methods; returns {'gh': [...], 'un': [...]}."""
    pts = dict(gh=[], un=[])
    for beta in beta_list:
        for N in N_list:
            print(f"    [Pareto] N={N}  β={beta:g} …", flush=True)
            run_g, cost_g = run_and_measure(run_gh(mods, N, beta))
            mg = potential_error_metrics(run_g)
            pts['gh'].append((cost_g['seconds'], mg['rmse_pesato'], N, beta))
            run_u, cost_u = run_and_measure(run_un(mods, N, beta))
            mu = potential_error_metrics(run_u)
            pts['un'].append((cost_u['seconds'], mu['rmse_pesato'], N, beta))
    return pts


def figure_pareto(pts, fig_path):
    """T4 figure: scatter of (time, weighted RMSE) with the Pareto fronts."""
    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
    for key, color, lbl, marker in [('gh', C_GH, "Gauss-Hermite (N×β)", "o"),
                                    ('un', C_UN, "Uniforme (N×β)", "s")]:
        arr = np.array(pts[key], dtype=float)
        if arr.size == 0:
            continue
        c, e = arr[:, 0], arr[:, 1]
        ax.scatter(c, e, s=45, color=color, marker=marker, alpha=0.55, label=lbl)
        front = _pareto_front(c, e)
        if front:
            fc, fe = c[front], e[front]; o = np.argsort(fc)
            ax.plot(fc[o], fe[o], "-", color=color, lw=2.0, alpha=0.9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("wall-time soluzione [s]")
    ax.set_ylabel("RMSE V$_{KS}$ (pesato ρ) [MeV]")
    ax.set_title("Z=8: frontiera di Pareto costo↔precisione")
    ax.legend(); ax.grid(alpha=0.4, which="both")
    fig.savefig(fig_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  → {fig_path}")


def print_summary(acc):
    """Print the accuracy-vs-cost table."""
    print("\n  Z=8 — accuratezza vs costo:")
    print(f"    {'N':>4} | {'RMSE_w GH':>11} {'RMSE_w Un':>11} | "
          f"{'t GH[s]':>8} {'t Un[s]':>8} | {'it GH':>6} {'it Un':>6}")
    for i in range(len(acc['N'])):
        print(f"    {acc['N'][i]:>4} | "
              f"{acc['rmse_w_gh'][i]:>11.3e} {acc['rmse_w_un'][i]:>11.3e} | "
              f"{acc['sec_gh'][i]:>8.2f} {acc['sec_un'][i]:>8.2f} | "
              f"{acc['it_gh'][i]:>6} {acc['it_un'][i]:>6}")


def main():
    cond_only = "--cond-only" in sys.argv
    no_pareto = "--no-pareto" in sys.argv

    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  EFFICIENZA vs PRECISIONE del potenziale — GH vs Uniforme (Z=8)     ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    print("\n  [T3] Condizionamento delle matrici derivata vs N …")
    cond = condition_scan(N_LIST)
    figure_condition(cond, os.path.join(OUTPUT_DIR, "T3_condizionamento.pdf"))
    _save_csv(os.path.join(OUTPUT_DIR, "T3_condizionamento.csv"),
              ["N", "cond_gh", "cond_un"], [cond['N'], cond['cond_gh'], cond['cond_un']])

    if cond_only:
        print("\n  (--cond-only) Fatto.\n"); return
    if not has_ipopt():
        print("\n  ⚠ cyipopt non disponibile: T1/T2/T4 richiedono IPOPT. Prodotto solo T3.\n"); return

    mods = load_test_modules()

    print("\n  [T1+T2] Accuratezza e costo vs N …")
    acc = collect_accuracy_cost(mods, N_LIST)
    print_summary(acc)
    figure_accuracy_vs_N(acc, os.path.join(OUTPUT_DIR, "T1_accuratezza_vs_N.pdf"))
    figure_accuracy_vs_cost(acc, os.path.join(OUTPUT_DIR, "T2_accuratezza_vs_costo.pdf"))
    _save_csv(os.path.join(OUTPUT_DIR, "T1_T2.csv"),
              ["N", "n_tot_gh", "n_tot_un", "rmse_w_gh", "rmse_u_gh",
               "rmse_w_un", "rmse_u_un", "sec_gh", "sec_un", "it_gh", "it_un"],
              [acc['N'], acc['n_tot_gh'], acc['n_tot_un'], acc['rmse_w_gh'],
               acc['rmse_u_gh'], acc['rmse_w_un'], acc['rmse_u_un'],
               acc['sec_gh'], acc['sec_un'], acc['it_gh'], acc['it_un']])

    if not no_pareto:
        print("\n  [T4] Frontiera di Pareto …")
        pts = collect_pareto(mods, N_PARETO, BETA_LIST)
        figure_pareto(pts, os.path.join(OUTPUT_DIR, "T4_pareto.pdf"))

    print("\n  Fatto.\n")


if __name__ == "__main__":
    main()